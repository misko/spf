"""Host-side control of the tandem AGC block.

Enables and disables FPGA-driven tandem gain control on a Pluto over libiio,
locally or remotely. The block is reachable at all because it is registered as
an IIO device (``adi_tandem_agc.c``); without that, its registers could only be
mmap'd from the radio's own processor and every action would be an SSH call.

The transaction order below is not arbitrary and must not be rearranged --
each step is commented with the hardware fact that fixes its position. It
mirrors ``runtime/spf_tandem_ctl.c`` step for step; ``test_tandem_agc.py``
asserts the two agree on the register map so they cannot drift silently.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

__all__ = ["TandemMode", "TandemError", "TandemState", "TandemAGC"]

# AD9361 registers, named as the datasheet names them
REG_AGC_CONFIG_2 = 0x0FB       # [1:0] = MAN_GAIN_CTRL_RX1/RX2
REG_AGC_CONFIG_3 = 0x0FC       # [7:5] manual increment step
REG_PEAK_WAIT_TIME = 0x0FE     # [7:5] decrement step, [4:0] peak overload wait
REG_CTRL_OUT_PTR = 0x035
REG_CTRL_OUT_EN = 0x036
REG_MAX_GAIN_INDEX = 0x0FF     # Max Full/LMT Gain Table Index, [6:0]

PIN_CTRL_MASK = 0x03
CTRL_OUT_PAGE_DETECTORS = 0x03
MAX_GAIN_INDEX_MASK = 0x7F

# FPGA block registers, TANDEM_AGC_V1_DESIGN.md §8
REG_ID = 0x00
REG_CTRL = 0x08
REG_STATUS = 0x0C
REG_EPOCH = 0x10
REG_INDEX = 0x14
REG_EXPECT = 0x18
REG_FAULT = 0x2C
ID_MAGIC = 0x54414731           # "TAG1"

STATUS_FPGA_OWNS = 1 << 4

OWNERSHIP_RETRY_LIMIT = 8


class TandemMode(enum.Enum):
    LEGACY = "legacy"
    HOLD = "tandem-hold"
    AUTO = "tandem-auto"


class TandemError(RuntimeError):
    """A tandem transaction failed. The message names the step."""


@dataclasses.dataclass(frozen=True)
class TandemState:
    owns_pins: bool
    pin_control_armed: bool
    epoch: int
    expected_index: int
    rx1_index: int
    rx2_index: int
    fault: int
    device_max_index: int

    @property
    def gain_writable(self) -> bool:
        """While armed the AD9361 accepts host gain writes and DROPS them with
        a success return (measured, E-AGC1). Callers must consult this rather
        than trusting a write's return code."""
        return not self.pin_control_armed


class TandemAGC:
    def __init__(self, ctx: Any):
        """``ctx`` is an iio context. Use :meth:`open` to build one from a URI."""
        self._ctx = ctx
        self._phy = ctx.find_device("ad9361-phy")
        if self._phy is None:
            raise TandemError("no ad9361-phy in this context")
        self._blk = ctx.find_device("tandem-agc")
        if self._blk is None:
            raise TandemError(
                "no tandem-agc device: this firmware has no tandem block, or "
                "its driver did not probe (the driver refuses to register when "
                "the block's ID register does not read TAG1)"
            )

    @classmethod
    def open(cls, uri: str) -> "TandemAGC":
        import iio  # imported lazily so the module is testable without libiio

        return cls(iio.Context(uri))

    # ---------------------------------------------------------------- helpers

    def _rmw(self, reg: int, clear: int, set_: int) -> None:
        """Read-modify-write, always. direct_reg_access writes a whole byte,
        and 0x0FB carries live bits besides [1:0] on shipped builds -- E-AGC1
        found bit 3 set, so a bare write of 0x03 would have cleared it."""
        value = self._phy.reg_read(reg)
        self._phy.reg_write(reg, (value & ~clear) | set_)

    def _gain_mode(self, mode: str) -> None:
        for name in ("voltage0", "voltage1"):
            chan = self._phy.find_channel(name)
            if chan is None:
                raise TandemError(f"no RX channel {name}")
            chan.attrs["gain_control_mode"].value = mode

    def _gain_index(self, channel: str) -> int:
        return int(self._phy.find_channel(channel).attrs["hardwaregain"].value.split()[0])

    def _set_gain_index(self, channel: str, index: int) -> None:
        self._phy.find_channel(channel).attrs["hardwaregain"].value = str(index)

    def max_index(self) -> int:
        """The clamp bound, read from the part. Never hard-coded: the chip
        default is 76 and so is the RTL default, so a constant looks right on
        every radio anyone has tested and stops being right the moment a driver
        loads a shorter gain table."""
        return self._phy.reg_read(REG_MAX_GAIN_INDEX) & MAX_GAIN_INDEX_MASK

    # ----------------------------------------------------------------- status

    def status(self) -> TandemState:
        st = self._blk.reg_read(REG_STATUS)
        return TandemState(
            owns_pins=bool(st & STATUS_FPGA_OWNS),
            pin_control_armed=bool(self._phy.reg_read(REG_AGC_CONFIG_2) & PIN_CTRL_MASK),
            epoch=self._blk.reg_read(REG_EPOCH) & 0xFF,
            expected_index=self._blk.reg_read(REG_EXPECT) & 0xFF,
            rx1_index=self._gain_index("voltage0"),
            rx2_index=self._gain_index("voltage1"),
            fault=self._blk.reg_read(REG_FAULT) & 0xFF,
            device_max_index=self.max_index(),
        )

    # ----------------------------------------------------------------- enable

    def enable(
        self,
        mode: TandemMode = TandemMode.AUTO,
        initial_gain: int = 40,
        harness_baseline: str | None = None,
    ) -> None:
        """Arm tandem gain control. Rolls back on any failure.

        ``harness_baseline`` records this unit's measured D(g,g). It is not
        enforced -- firmware cannot measure harness health -- but its absence
        is worth stating, because E-GSC6 measured a connector-damaged unit at
        0.3x in the high band, i.e. tandem there made phase WORSE than leaving
        it off, while the control unit measured >=7.2x.
        """
        if mode is TandemMode.LEGACY:
            return self.disable()
        if harness_baseline is None:
            import warnings

            warnings.warn(
                "arming tandem with no harness baseline: phase benefit is "
                "per-unit and per-harness, and can be worse than legacy on a "
                "damaged harness",
                stacklevel=2,
            )

        # 1. the block must actually be present
        ident = self._blk.reg_read(REG_ID)
        if ident != ID_MAGIC:
            raise TandemError(f"step 1: FPGA ID 0x{ident:08x} is not TAG1")

        # 2. split gain table reuses the same four pins, so it cannot support
        #    per-channel increment and decrement
        table = self._phy.attrs["gain_table_config"].value if "gain_table_config" in self._phy.attrs else "full"
        if "split" in str(table).lower():
            raise TandemError("step 2: split gain table cannot support tandem")

        # 3. the ENSM must be RX-active BEFORE arming. CTRL_IN edges are
        #    ignored outside RX (measured, E-AGC1 O-1), so a controller armed
        #    outside RX silently does nothing.
        ensm = str(self._phy.attrs["ensm_mode"].value)
        if ensm not in ("fdd", "rx"):
            raise TandemError(f"step 3: ENSM is '{ensm}', not RX-active")

        armed = False
        owned = False
        try:
            # 4. both channels to manual gain
            self._gain_mode("manual")

            # 4b. the clamp bound comes from the part, never a constant (D-8)
            device_max = self.max_index()
            if device_max == 0:
                raise TandemError("step 4b: part reports a zero-length gain table")
            if not 0 <= initial_gain <= device_max:
                raise TandemError(
                    f"step 4b: initial index {initial_gain} exceeds the part's "
                    f"maximum {device_max}"
                )

            # 4c. push the measured bound into the block's index window, so the
            #     RTL's reset default of 76 is overwritten rather than trusted
            self._blk.reg_write(REG_INDEX, (initial_gain << 16) | (device_max << 8))

            # 5. program the common index. LAST point at which software can set
            #    gain -- after step 11 every such write is dropped with success.
            self._set_gain_index("voltage0", initial_gain)
            self._set_gain_index("voltage1", initial_gain)

            # 6. read back and require equality
            rx1, rx2 = self._gain_index("voltage0"), self._gain_index("voltage1")
            if rx1 != rx2 or rx1 != initial_gain:
                raise TandemError(
                    f"step 6: read-back unequal (rx1={rx1} rx2={rx2} wanted {initial_gain})"
                )

            # 7. detector page and output enables
            self._phy.reg_write(REG_CTRL_OUT_PTR, CTRL_OUT_PAGE_DETECTORS)
            self._phy.reg_write(REG_CTRL_OUT_EN, 0xFF)

            # 8. one index per pulse, so the FPGA model is auditable. Both
            #    fields store value-1, and 0x0FE also holds the peak overload
            #    wait time in [4:0] -- read-modify-write or that is destroyed.
            self._rmw(REG_AGC_CONFIG_3, 0xE0, 0x00)
            self._rmw(REG_PEAK_WAIT_TIME, 0xE0, 0x00)

            # 10. hand the pins to the FPGA, held low, BEFORE anything is armed
            self._blk.reg_write(REG_CTRL, 1)
            owned = True
            for _ in range(OWNERSHIP_RETRY_LIMIT + 1):
                if self._blk.reg_read(REG_STATUS) & STATUS_FPGA_OWNS:
                    break
            else:
                raise TandemError("step 10: ownership not acknowledged")

            # 11. ONLY NOW arm pin control
            self._rmw(REG_AGC_CONFIG_2, 0x00, PIN_CTRL_MASK)
            armed = True

            # 12. open the policy gate only after success is established
            if mode is TandemMode.AUTO:
                self._blk.reg_write(REG_CTRL, 2)
        except Exception:
            # Roll back in the reverse of the order that built it up. A
            # half-applied enable is worse than one that never started: the
            # gain would be frozen wherever step 5 left it, host gain writes
            # accepted and silently dropped, and nothing reporting why.
            if armed:
                self._rmw(REG_AGC_CONFIG_2, PIN_CTRL_MASK, 0x00)
            if owned:
                self._blk.reg_write(REG_CTRL, 0)
            raise

    # ---------------------------------------------------------------- disable

    def disable(self, restore_mode: str = "slow_attack") -> None:
        """Release the pins and restore a legacy gain mode."""
        # ask the block to stop and hold the outputs low
        self._blk.reg_write(REG_CTRL, 0)
        # disarm BEFORE releasing the pins: the other order leaves armed pin
        # control over pins the PS may drive or leave floating
        self._rmw(REG_AGC_CONFIG_2, PIN_CTRL_MASK, 0x00)
        self._gain_mode(restore_mode)
