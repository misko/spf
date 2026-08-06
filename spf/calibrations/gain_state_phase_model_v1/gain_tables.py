"""Audited AD9361 RX gain tables: decode ``(band, requested dB) -> hardware state``.

The three 77-row FULL gain tables were read off the hardware during the 2026-07-30
A-G spectroscopy campaign audit and are byte-identical between the two audited
radios, so the map from a requested gain in dB to a hardware state is **chip data,
not a fitted quantity**. ``gain_tables_audited.json`` carries them, their SHA-256,
and the serials they were verified identical across.

Byte fields (confirmed against ADI's driver ``ad9361_regs.h`` / ``ad9361.c``)::

    byte 0 bits 6:5   LNA_GAIN         4 states
    byte 0 bits 4:0   MIXER_GM_GAIN    16 states
    byte 1 bit  5     TIA_GAIN         2 states
    byte 1 bits 4:0   LPF_GAIN         baseband PGA, 1 dB/LSB
    byte 2 bit  5     RF_DC_CAL        flag -- NOT digital gain
    byte 2 bits 4:0   digital gain     identically 0 on all 231 rows

``RF_DC_CAL`` is set on exactly the rows that begin a new LNA/mixer/TIA state,
which confounds an RF-state phase step with an RF-DC-recalibration step. See the
limitations section of ``README.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TABLES_JSON = Path(__file__).resolve().parent / "gain_tables_audited.json"

BANDS = ("low", "middle", "high")

#: AD9361 driver band selection, keyed on the RX LO (not the RF tone frequency).
LOW_EDGE_HZ = 1_300_000_000
HIGH_EDGE_HZ = 4_000_000_000


def band_for_lo(lo_hz: float) -> str:
    """Return the gain-table band the driver selects for this RX LO.

    The band edges are a hardware property: crossing one re-tables the whole
    gain axis, which is why the same requested dB is a different hardware state
    in different bands.
    """
    if lo_hz <= LOW_EDGE_HZ:
        return "low"
    if lo_hz <= HIGH_EDGE_HZ:
        return "middle"
    return "high"


@dataclass(frozen=True)
class HardwareState:
    """The audited AD9361 RX gain state for one arm."""

    lna: int
    mixer: int
    tia: int
    lpf: int
    rf_dc_cal: int
    row: int
    band: str
    gain_db: int

    @property
    def rf_words(self) -> tuple[int, int, int]:
        """The RF-path words ``(LNA, MIXER, TIA)``.

        Rule 5 of the deployment contract keys off equality of this tuple
        between the two arms: where it is equal, the measured phase difference
        carries no resolvable gain-dependent term and the correction must not
        be applied.
        """
        return (self.lna, self.mixer, self.tia)


class GainTables:
    """The three audited tables, with ADI's first-match gain lookup."""

    def __init__(self, path: Path | str = TABLES_JSON):
        doc = json.loads(Path(path).read_text())
        self.source_sha256: str = doc["source_sha256"]
        self.verified_identical_across_serials: list[str] = list(
            doc["verified_identical_across_serials"]
        )
        self._bands: dict[str, dict] = doc["bands"]
        for name in BANDS:
            if name not in self._bands:
                raise ValueError(f"gain table JSON is missing band {name!r}")

    # ------------------------------------------------------------------ rows
    def gain_range_db(self, band: str) -> tuple[int, int]:
        g = self._bands[band]["gain_db"]
        return int(min(g)), int(max(g))

    def row_for_gain(self, band: str, gain_db: int) -> int:
        """First row whose absolute gain reaches ``gain_db``; -1 if unsupported.

        ``ad9361.c find_table_index()`` scans for the first index whose gain
        reaches the request and then takes the nearest neighbour. On these
        integer 1 dB tables that reduces to first-match.

        Requests **outside** the table -- above the maximum or below the
        minimum -- return -1. Below-minimum is deliberately a refusal rather
        than a clamp to row 0: the driver would clamp, so the realized gain
        would not be the gain the caller asked for, and a correction keyed on
        the requested value would then be describing a cell the hardware never
        visited. This is stricter than the source analysis' ``row_for_gain``,
        which clamps; the difference cannot change any published number because
        no scheduled gain in the source campaign falls below its band minimum
        (see ``PROVENANCE.md``).
        """
        gains = self._bands[band]["gain_db"]
        if not gains or gain_db < gains[0]:
            return -1
        for i, g in enumerate(gains):
            if g >= gain_db:
                return i
        return -1

    @lru_cache(maxsize=4096)
    def state(self, band: str, gain_db: int) -> HardwareState | None:
        """Decode ``(band, requested dB)`` to its audited hardware state.

        Returns ``None`` when the request falls outside the table, which the
        model turns into a fail-closed refusal.
        """
        row = self.row_for_gain(band, int(gain_db))
        if row < 0:
            return None
        b0, b1, b2 = self._bands[band]["bytes"][row]
        return HardwareState(
            lna=(b0 >> 5) & 0x3,
            mixer=b0 & 0x1F,
            tia=(b1 >> 5) & 0x1,
            lpf=b1 & 0x1F,
            rf_dc_cal=(b2 >> 5) & 0x1,
            row=row,
            band=band,
            gain_db=int(gain_db),
        )

    def state_for_lo(self, lo_hz: float, gain_db: int) -> HardwareState | None:
        """Convenience: select the band from the LO, then decode."""
        return self.state(band_for_lo(lo_hz), int(gain_db))

    # ------------------------------------------------------------- diagnostics
    def digital_gain_is_zero_everywhere(self) -> bool:
        """Verify the report's claim that digital gain cannot contribute phase."""
        return all(
            (row[2] & 0x1F) == 0
            for band in self._bands.values()
            for row in band["bytes"]
        )

    def transitions(self, band: str) -> list[dict]:
        """Every adjacent 1 dB step in a band, labelled by which word it changes.

        This is the decoder behind the central mechanism result: a step that
        moves the mixer word costs ~2.664 deg, while a step that only moves the
        baseband LPF word costs ~0.343 deg (at the measurement noise floor).
        """
        gains = self._bands[band]["gain_db"]
        out = []
        for lo, hi in zip(gains, gains[1:]):
            a, b = self.state(band, lo), self.state(band, hi)
            if a is None or b is None:
                continue
            changed = [
                f
                for f in ("lna", "mixer", "tia", "lpf")
                if getattr(a, f) != getattr(b, f)
            ]
            out.append(
                {
                    "band": band,
                    "from_db": lo,
                    "to_db": hi,
                    "changed": changed,
                    "rf_state_changed": bool({"lna", "mixer", "tia"} & set(changed)),
                    "rf_dc_cal_toggled": a.rf_dc_cal != b.rf_dc_cal,
                    "from_state": a,
                    "to_state": b,
                }
            )
        return out


@lru_cache(maxsize=1)
def default_tables() -> GainTables:
    """The committed audited tables (loaded once)."""
    return GainTables()
