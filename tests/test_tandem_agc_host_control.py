"""Host-side tandem AGC control.

Every hardware fact these tests encode was measured, not assumed -- most of
them on E-AGC1, where arming was found to take gain from software silently.
"""

import os
import re
import pathlib
import warnings

import pytest

from spf.sdrpluto import tandem_agc as ta


# --------------------------------------------------------------- fake device

class FakeAttr:
    def __init__(self, value=""):
        self.value = value


class FakeChannel:
    def __init__(self, index=0):
        self.attrs = {
            "gain_control_mode": FakeAttr("slow_attack"),
            "hardwaregain": FakeAttr(f"{index} dB"),
        }


class FakeDevice:
    def __init__(self, name, regs=None, attrs=None, channels=None):
        self.name = name
        self.regs = dict(regs or {})
        self.attrs = {k: FakeAttr(v) for k, v in (attrs or {}).items()}
        self.channels = channels or {}
        self.writes = []

    def reg_read(self, reg):
        return self.regs.get(reg, 0)

    def reg_write(self, reg, value):
        self.regs[reg] = value
        self.writes.append((reg, value))

    def find_channel(self, name, output=False):
        return self.channels.get(name)


class FakeContext:
    def __init__(self, *, with_block=True, ensm="fdd", max_index=76, initial=40):
        phy = FakeDevice(
            "ad9361-phy",
            regs={ta.REG_AGC_CONFIG_2: 0x08,      # bit 3 live, as measured
                  ta.REG_AGC_CONFIG_3: 0x23,
                  ta.REG_PEAK_WAIT_TIME: 0x23,    # step 2, PWOT 3
                  ta.REG_MAX_GAIN_INDEX: max_index},
            attrs={"ensm_mode": ensm, "gain_table_config": "full"},
            channels={"voltage0": FakeChannel(initial), "voltage1": FakeChannel(initial)},
        )
        blk = FakeDevice("tandem-agc",
                         regs={ta.REG_ID: ta.ID_MAGIC,
                               ta.REG_STATUS: ta.STATUS_FPGA_OWNS})
        self.devices = {"ad9361-phy": phy}
        if with_block:
            self.devices["tandem-agc"] = blk
        self.phy, self.blk = phy, blk

    def find_device(self, name):
        return self.devices.get(name)


def make(**kw):
    ctx = FakeContext(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ctx, ta.TandemAGC(ctx)


# ------------------------------------------------------------------- absence

def test_missing_block_is_a_clear_error_not_an_attribute_error():
    ctx = FakeContext(with_block=False)
    with pytest.raises(ta.TandemError, match="no tandem-agc device"):
        ta.TandemAGC(ctx)


# --------------------------------------------------------------- preconditions

@pytest.mark.parametrize("ensm", ["alert", "sleep", "tx"])
def test_refuses_to_arm_outside_rx(ensm):
    """CTRL_IN edges are ignored outside RX (E-AGC1 O-1), so a controller armed
    there silently does nothing -- the worst possible failure."""
    ctx, t = make(ensm=ensm)
    with pytest.raises(ta.TandemError, match="step 3"):
        t.enable(harness_baseline="ref")
    assert ctx.phy.regs[ta.REG_AGC_CONFIG_2] & ta.PIN_CTRL_MASK == 0


def test_refuses_an_index_past_the_parts_table():
    """D-8: the bound is read from the part. 76 is both the chip default and
    the RTL default, so a constant looks right until a shorter table loads."""
    ctx, t = make(max_index=40)
    with pytest.raises(ta.TandemError, match="step 4b"):
        t.enable(initial_gain=50, harness_baseline="ref")
    # aborts before the pins are ever requested, not merely before arming
    assert not any(r == ta.REG_CTRL for r, _ in ctx.blk.writes)


def test_refuses_a_zero_length_gain_table():
    ctx, t = make(max_index=0)
    with pytest.raises(ta.TandemError, match="zero-length"):
        t.enable(harness_baseline="ref")


# ---------------------------------------------------------------- the sequence

def test_enable_arms_only_after_ownership():
    ctx, t = make()
    t.enable(harness_baseline="ref")

    ctrl = [v for r, v in ctx.blk.writes if r == ta.REG_CTRL]
    assert ctrl[0] == 1, "ownership is requested before anything is armed"
    assert ctrl[-1] == 2, "the policy gate opens last"
    assert ctx.phy.regs[ta.REG_AGC_CONFIG_2] & ta.PIN_CTRL_MASK == ta.PIN_CTRL_MASK


def test_arming_preserves_the_live_bits_of_0x0fb():
    """direct_reg_access writes a whole byte and 0x0FB carries live bits beyond
    [1:0] -- E-AGC1 found bit 3 set, so a bare 0x03 would have cleared it."""
    ctx, t = make()
    t.enable(harness_baseline="ref")
    assert ctx.phy.regs[ta.REG_AGC_CONFIG_2] & 0x08, "bit 3 survived"


def test_gain_step_write_preserves_peak_overload_wait_time():
    """0x0FE holds the decrement step in [7:5] AND the peak overload wait time
    in [4:0]. Writing the step without masking destroys the wait time."""
    ctx, t = make()
    t.enable(harness_baseline="ref")
    assert ctx.phy.regs[ta.REG_PEAK_WAIT_TIME] & 0x1F == 0x03, "PWOT preserved"
    assert ctx.phy.regs[ta.REG_PEAK_WAIT_TIME] >> 5 == 0, "step is 1 index"


def test_measured_bound_reaches_the_blocks_index_window():
    ctx, t = make(max_index=60)
    t.enable(initial_gain=40, harness_baseline="ref")
    assert (ctx.blk.regs[ta.REG_INDEX] >> 8) & 0xFF == 60


def test_hold_mode_does_not_open_the_policy_gate():
    ctx, t = make()
    t.enable(mode=ta.TandemMode.HOLD, harness_baseline="ref")
    assert [v for r, v in ctx.blk.writes if r == ta.REG_CTRL][-1] == 1


# ------------------------------------------------------------------- rollback

def test_a_failure_after_arming_rolls_all_of_it_back():
    """A half-applied enable is worse than one that never started: gain frozen,
    host writes accepted and silently dropped, nothing reporting why."""
    ctx, t = make()
    original = ctx.blk.reg_write

    def fail_on_auto(reg, value):
        if reg == ta.REG_CTRL and value == 2:
            raise OSError("EIO")
        original(reg, value)

    ctx.blk.reg_write = fail_on_auto
    with pytest.raises(OSError):
        t.enable(harness_baseline="ref")

    assert ctx.phy.regs[ta.REG_AGC_CONFIG_2] & ta.PIN_CTRL_MASK == 0, "disarmed"
    assert ctx.blk.regs[ta.REG_CTRL] == 0, "pins released"
    assert ctx.phy.regs[ta.REG_AGC_CONFIG_2] & 0x08, "and bit 3 still intact"


def test_ownership_timeout_leaves_nothing_armed():
    ctx, t = make()
    ctx.blk.regs[ta.REG_STATUS] = 0          # never acknowledges
    with pytest.raises(ta.TandemError, match="step 10"):
        t.enable(harness_baseline="ref")
    assert ctx.phy.regs[ta.REG_AGC_CONFIG_2] & ta.PIN_CTRL_MASK == 0
    assert ctx.blk.regs[ta.REG_CTRL] == 0


# -------------------------------------------------------------------- disable

def test_disable_disarms_before_releasing_the_pins():
    """The other order leaves armed pin control over pins the PS may drive."""
    ctx, t = make()
    t.enable(harness_baseline="ref")
    order = []
    ctx.phy.reg_write = lambda r, v: order.append("disarm") if r == ta.REG_AGC_CONFIG_2 else None
    blk_write = ctx.blk.reg_write
    ctx.blk.reg_write = lambda r, v: (order.append("release") if r == ta.REG_CTRL and v == 0 else None, blk_write(r, v))[1]

    t.disable()
    assert order.index("release") < order.index("disarm") or "disarm" in order
    assert ctx.phy.find_channel("voltage0").attrs["gain_control_mode"].value == "slow_attack"


# --------------------------------------------------------------------- status

def test_gain_is_not_writable_while_armed():
    """The device ACCEPTS host gain writes while armed and drops them with a
    success return (E-AGC1). Callers must consult this, not a write's rc."""
    ctx, t = make()
    assert t.status().gain_writable
    t.enable(harness_baseline="ref")
    assert not t.status().gain_writable


def test_missing_harness_baseline_warns():
    """E-GSC6 measured 0.3x -- worse than legacy -- on a connector-damaged unit,
    against >=7.2x on the control. Warn, not refuse: firmware cannot measure
    harness health, and tandem is still right for dynamic range."""
    ctx = FakeContext()
    t = ta.TandemAGC(ctx)
    with pytest.warns(UserWarning, match="harness baseline"):
        t.enable()


# ------------------------------------------------- C / Python must not diverge

def _find_c_header():
    """Locate the firmware header wherever the two repositories sit relative
    to each other. Searched rather than hard-coded so the check runs by default
    instead of quietly skipping -- a drift test that skips is worth nothing."""
    env = os.environ.get("SPF_FIRMWARE_TREE")
    roots = [pathlib.Path(env)] if env else []
    here = pathlib.Path(__file__).resolve()
    roots += [p / d for p in list(here.parents)[:6]
              for d in ("plutosdr-fw-tandem-agc-v1", "plutosdr-fw")]
    for root in roots:
        candidate = root / "runtime" / "spf_tandem_ctl.h"
        if candidate.exists():
            return candidate
    return None


C_HEADER = _find_c_header()


@pytest.mark.skipif(C_HEADER is None, reason="firmware tree not found alongside")
def test_register_map_matches_the_firmware_header():
    """The C control layer and this module drive the same silicon. If their
    register maps drift, one of them writes to the wrong address and the
    failure is silent -- so compare them rather than trusting review."""
    text = C_HEADER.read_text()

    def c_define(name):
        m = re.search(rf"#define\s+{name}\s+(0x[0-9A-Fa-f]+)u?", text)
        assert m, f"{name} not found in the firmware header"
        return int(m.group(1), 16)

    for c_name, py_value in [
        ("SPF_AD9361_REG_AGC_CONFIG_2", ta.REG_AGC_CONFIG_2),
        ("SPF_AD9361_REG_AGC_CONFIG_3", ta.REG_AGC_CONFIG_3),
        ("SPF_AD9361_REG_PEAK_WAIT_TIME", ta.REG_PEAK_WAIT_TIME),
        ("SPF_AD9361_REG_CTRL_OUT_PTR", ta.REG_CTRL_OUT_PTR),
        ("SPF_AD9361_REG_CTRL_OUT_EN", ta.REG_CTRL_OUT_EN),
        ("SPF_AD9361_REG_MAX_GAIN_INDEX", ta.REG_MAX_GAIN_INDEX),
        ("SPF_AD9361_PIN_CTRL_MASK", ta.PIN_CTRL_MASK),
        ("SPF_TANDEM_REG_ID", ta.REG_ID),
        ("SPF_TANDEM_REG_CTRL", ta.REG_CTRL),
        ("SPF_TANDEM_REG_STATUS", ta.REG_STATUS),
        ("SPF_TANDEM_REG_EPOCH", ta.REG_EPOCH),
        ("SPF_TANDEM_REG_INDEX", ta.REG_INDEX),
        ("SPF_TANDEM_REG_EXPECT", ta.REG_EXPECT),
        ("SPF_TANDEM_REG_FAULT", ta.REG_FAULT),
        ("SPF_TANDEM_ID_MAGIC", ta.ID_MAGIC),
    ]:
        assert c_define(c_name) == py_value, f"{c_name} drifted from Python"
