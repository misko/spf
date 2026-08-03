"""Shared library: load extracted calibration frames, decode AD9361 gain tables,
and provide circular-statistics helpers.

Phase convention throughout: angle(RX1) - angle(RX2), radians, wrapped to (-pi, pi].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent
EXTRACTED = SCRATCH / "extracted"
AUDIT_JSON = Path(
    "/mnt/qnap01/mouse9911/share/spf_campaigns/"
    "spectroscopy_20260730_full_r2/gain_table_audit.json"
)

TREATED = "104000bac4950008230026001b440a003a"  # historical .17
CONTROL = "1040007c4a94000211000b009186843ef2"  # historical .18
SHORT = {TREATED: "R17", CONTROL: "R18"}

C_LIGHT = 299_792_458.0


# ---------------------------------------------------------------- circular ---
def wrap(x):
    return (np.asarray(x, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def cmean(x, axis=None):
    """Circular mean of angles (radians)."""
    z = np.exp(1j * np.asarray(x, dtype=np.float64))
    return np.angle(np.mean(z, axis=axis))


def cmae_deg(err_rad):
    return float(np.mean(np.abs(np.degrees(wrap(err_rad)))))


def crmse_deg(err_rad):
    return float(np.sqrt(np.mean(np.degrees(wrap(err_rad)) ** 2)))


def cp95_deg(err_rad):
    return float(np.percentile(np.abs(np.degrees(wrap(err_rad))), 95))


def cmax_deg(err_rad):
    return float(np.max(np.abs(np.degrees(wrap(err_rad)))))


def err_stats(err_rad, prefix=""):
    e = np.asarray(err_rad, dtype=np.float64)
    if e.size == 0:
        return {f"{prefix}n": 0}
    return {
        f"{prefix}n": int(e.size),
        f"{prefix}mae_deg": cmae_deg(e),
        f"{prefix}rmse_deg": crmse_deg(e),
        f"{prefix}p95_deg": cp95_deg(e),
        f"{prefix}max_deg": cmax_deg(e),
    }


# -------------------------------------------------------------- gain table ---
def gain_band(lo_hz):
    """AD9361 driver band selection by RX LO frequency."""
    lo_hz = np.asarray(lo_hz, dtype=np.float64)
    band = np.full(lo_hz.shape, 1, dtype=np.int8)  # middle
    band[lo_hz <= 1_300_000_000] = 0  # low
    band[lo_hz > 4_000_000_000] = 2  # high
    return band


BAND_NAMES = {0: "low", 1: "middle", 2: "high"}


@dataclass
class GainTable:
    name: str
    start_hz: int
    end_hz: int
    gain_db: np.ndarray  # (77,)
    b0: np.ndarray
    b1: np.ndarray
    b2: np.ndarray

    @property
    def lna(self):
        return (self.b0 >> 5) & 0x3

    @property
    def mixer(self):
        return self.b0 & 0x1F

    @property
    def tia(self):
        return (self.b1 >> 5) & 0x1

    @property
    def lpf(self):
        """Baseband LPF/PGA gain word, 1 dB per LSB."""
        return self.b1 & 0x1F

    @property
    def rfdc(self):
        """Byte 2 bit 5 is RF_DC_CAL, the per-row RF-DC-calibration flag --
        NOT digital gain. Digital gain (bits 4:0) is zero on all 231 rows, so
        it cannot contribute phase. RF_DC_CAL is set on exactly the rows that
        begin a new LNA/mixer/TIA state, which perfectly confounds an
        'LMT state phase step' with an 'RF-DC correction step'."""
        return (self.b2 >> 5) & 0x1

    @property
    def dig(self):
        """Digital gain, bits 4:0 (identically zero on these tables)."""
        return self.b2 & 0x1F

    def row_for_gain(self, gain_db):
        """First row whose gain_db is >= the request, matching the ADI driver.

        ad9361.c find_table_index() scans for the first index whose absolute
        gain reaches the requested value, then takes the nearest neighbour. On
        these integer 1 dB tables that reduces to first-match. Each table has
        three degenerate rows (0,1,2) at the table minimum, so first- and
        last-match differ there; they carry the same (LNA, MIXER, TIA, LPF)
        words and differ only in the RF_DC_CAL flag, which is not a model
        feature. Requests below the table minimum return -1 (unsupported).
        """
        gain_db = np.asarray(gain_db, dtype=np.int64)
        rows = np.full(gain_db.shape, -1, dtype=np.int64)
        flat = np.ravel(rows)
        for i, g in enumerate(np.ravel(gain_db)):
            hit = np.nonzero(self.gain_db >= g)[0]
            flat[i] = hit[0] if hit.size else -1
        return rows


def load_gain_tables(path: Path = AUDIT_JSON) -> dict[str, dict[str, GainTable]]:
    doc = json.loads(Path(path).read_text())
    out: dict[str, dict[str, GainTable]] = {}
    for radio in doc["radios"]:
        tables = {}
        for band in radio["bands"]:
            rows = band["rows"]
            arr = np.array([r["bytes"] for r in rows], dtype=np.int64)
            tables[band["name"]] = GainTable(
                name=band["name"],
                start_hz=int(band["start_hz"]),
                end_hz=int(band["end_hz"]),
                gain_db=np.array([r["gain_db"] for r in rows], dtype=np.int64),
                b0=arr[:, 0],
                b1=arr[:, 1],
                b2=arr[:, 2],
            )
        out[radio["serial"]] = tables
    return out


# ------------------------------------------------------------------ frames ---
STAGE_ROOTS = {
    "A": ("spectroscopy_20260730_full", "A"),
    "B": ("spectroscopy_20260730_full", "B"),
    "C": ("spectroscopy_20260730_full", "C"),
    "D": ("spectroscopy_20260730_full", "D"),
    "F": ("spectroscopy_20260730_full_r2", "F"),
    "G": ("spectroscopy_20260730_full_r2", "G"),
    "rate_pilot": ("spectroscopy_20260730_full", "rate_pilot"),
    "F_neg": (
        "spectroscopy_20260730_full",
        "F_unsupported_negative_gain_attempt_20260730",
    ),
    "E_tx_0": ("spectroscopy_20260730_full", "E_tx_0"),
    "E_tx_n7": ("spectroscopy_20260730_full", "E_tx_n7"),
    "E_tx_n14": ("spectroscopy_20260730_full", "E_tx_n14"),
    "E_tx_n21": ("spectroscopy_20260730_full", "E_tx_n21"),
    "E_tx_n28": ("spectroscopy_20260730_full", "E_tx_n28"),
    "E_tx_n80": ("spectroscopy_20260730_full", "E_tx_n80"),
}


class Frames:
    """Column-oriented frame table."""

    __slots__ = ("cols",)

    def __init__(self, cols: dict[str, np.ndarray]):
        self.cols = cols

    def __getattr__(self, name):
        try:
            return self.cols[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __len__(self):
        return len(next(iter(self.cols.values())))

    def keys(self):
        return sorted(self.cols)

    def sel(self, mask) -> "Frames":
        mask = np.asarray(mask)
        return Frames({k: v[mask] for k, v in self.cols.items()})

    def copy_with(self, **extra) -> "Frames":
        cols = dict(self.cols)
        cols.update(extra)
        return Frames(cols)


def load_stage(stage: str, serial: str) -> Frames:
    campaign, sdir = STAGE_ROOTS[stage]
    path = EXTRACTED / campaign / sdir / f"{serial}.npz"
    d = np.load(path)
    n = len(d["sweep_completed"])
    g = d["sweep_requested_gain_db"].astype(np.int64)
    lo = d["sweep_lo_frequency_hz"].astype(np.float64)
    cols = {
        "serial": np.full(n, serial, dtype=object),
        "stage": np.full(n, stage, dtype=object),
        "lo_hz": lo,
        "rf_hz": d["sweep_rf_tone_frequency_hz"].astype(np.float64),
        "band": gain_band(lo),
        "g1": g[:, 0],
        "g2": g[:, 1],
        "epoch": d["sweep_epoch"].astype(np.int64),
        "phase": wrap(d["phase_difference_rad"]),
        "completed": d["sweep_completed"].astype(bool),
        "qvalid": d["sweep_quality_valid"].astype(bool),
        "qmask": d["sweep_quality_reason_mask"].astype(np.int64),
        "snr1": d["tone_snr_db"][:, 0].astype(np.float64),
        "snr2": d["tone_snr_db"][:, 1].astype(np.float64),
        "dbfs1": d["tone_dbfs"][:, 0].astype(np.float64),
        "dbfs2": d["tone_dbfs"][:, 1].astype(np.float64),
        "amp_ratio_db": d["amplitude_ratio_db_rx1_over_rx2"].astype(np.float64),
        "coherence": d["coherence"].astype(np.float64),
        "phase_std": d["within_capture_phase_std_rad"].astype(np.float64),
        "tx_gain_db": d["sweep_tx_gain_db"].astype(np.float64),
        "timestamp": d["system_timestamp"].astype(np.float64),
        "sched": d["sweep_schedule_index"].astype(np.int64),
    }
    return Frames(cols)


def load_stages(stages, serials=(TREATED, CONTROL), quality_only=True) -> Frames:
    parts = []
    for stage in stages:
        for serial in serials:
            f = load_stage(stage, serial)
            m = f.completed
            if quality_only:
                m = m & f.qvalid
            parts.append(f.sel(m))
    cols = {}
    for k in parts[0].cols:
        cols[k] = np.concatenate([p.cols[k] for p in parts])
    return Frames(cols)


# ------------------------------------------------------- cell aggregation ---
def cell_key(f: Frames):
    """Unique key per (serial, stage, lo, g1, g2) measurement cell."""
    return np.array(
        [
            f"{s}|{st}|{int(lo)}|{a}|{b}"
            for s, st, lo, a, b in zip(f.serial, f.stage, f.lo_hz, f.g1, f.g2)
        ],
        dtype=object,
    )


def aggregate_cells(f: Frames) -> dict:
    """Circular-average repeats within a cell; returns dict of arrays."""
    keys = cell_key(f)
    uniq, inv = np.unique(keys, return_inverse=True)
    n = len(uniq)
    out = {
        "serial": np.empty(n, dtype=object),
        "stage": np.empty(n, dtype=object),
        "lo_hz": np.zeros(n),
        "band": np.zeros(n, dtype=np.int8),
        "g1": np.zeros(n, dtype=np.int64),
        "g2": np.zeros(n, dtype=np.int64),
        "phase": np.zeros(n),
        "count": np.zeros(n, dtype=np.int64),
        "resultant": np.zeros(n),
        "spread_deg": np.zeros(n),
    }
    for i in range(n):
        m = inv == i
        p = f.phase[m]
        z = np.mean(np.exp(1j * p))
        out["serial"][i] = f.serial[m][0]
        out["stage"][i] = f.stage[m][0]
        out["lo_hz"][i] = f.lo_hz[m][0]
        out["band"][i] = f.band[m][0]
        out["g1"][i] = f.g1[m][0]
        out["g2"][i] = f.g2[m][0]
        out["phase"][i] = np.angle(z)
        out["count"][i] = int(m.sum())
        out["resultant"][i] = float(np.abs(z))
        out["spread_deg"][i] = float(
            np.degrees(np.max(np.abs(wrap(p - np.angle(z))))) if m.sum() > 1 else 0.0
        )
    return out
