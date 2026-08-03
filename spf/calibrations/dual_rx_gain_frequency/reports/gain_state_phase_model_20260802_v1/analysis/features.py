"""Map (band, requested gain dB) -> audited AD9361 hardware state, and build
model design matrices.

All models predict D = phase - anchor(radio, frequency), where the anchor is the
measured equal-reference-gain cell. D stays well inside +-90 deg so ordinary
least squares on the unwrapped value is exact; no circular machinery is needed.
"""

from __future__ import annotations

import numpy as np

import spflib as S

BANDS = ("low", "middle", "high")


class HardwareStates:
    """(band, gain_db) -> (lna, mixer, tia, lpf, dig, row). Tables are
    byte-identical between the two audited radios, so this is universal."""

    def __init__(self):
        tabs = S.load_gain_tables()
        self.tab = tabs[S.TREATED]
        # verify universality
        other = tabs[S.CONTROL]
        for b in BANDS:
            assert np.array_equal(self.tab[b].b0, other[b].b0)
            assert np.array_equal(self.tab[b].b1, other[b].b1)
            assert np.array_equal(self.tab[b].b2, other[b].b2)
        self._cache: dict[tuple[int, int], tuple] = {}

    def state(self, band_idx: int, gain_db: int):
        key = (int(band_idx), int(gain_db))
        if key in self._cache:
            return self._cache[key]
        t = self.tab[BANDS[band_idx]]
        row = int(t.row_for_gain(np.array(gain_db)))
        if row < 0:
            val = None
        else:
            val = (
                int(t.lna[row]),
                int(t.mixer[row]),
                int(t.tia[row]),
                int(t.lpf[row]),
                int(t.dig[row]),
                row,
            )
        self._cache[key] = val
        return val

    def vec(self, band_idx, gain_db, field):
        i = {"lna": 0, "mixer": 1, "tia": 2, "lpf": 3, "dig": 4, "row": 5}[field]
        out = np.zeros(len(band_idx), dtype=np.int64)
        for k, (b, g) in enumerate(zip(band_idx, gain_db)):
            st = self.state(b, g)
            out[k] = -999 if st is None else st[i]
        return out


HW = HardwareStates()


# ---------------------------------------------------------------- anchoring --
def add_anchor(f: S.Frames, ref: int | None = 26, per_epoch: bool = True) -> S.Frames:
    """Attach the measured equal-gain anchor for each (radio, stage, LO[, epoch]).

    The anchor is an INPUT, not a fitted parameter: deployment always measures a
    per-session equal-gain anchor at the operating LO (repo calibration rule 3).

    ref=None uses whatever equal-gain cell that stage scheduled, which lets
    stages with different reference gains be pooled. An antisymmetric model
    D = H(g1) - H(g2) is invariant to that choice, since a constant added to H
    cancels.
    """
    anchor = np.full(len(f), np.nan)
    ep = f.epoch if per_epoch else np.zeros(len(f), dtype=np.int64)
    karr = np.array(
        [f"{a}|{b}|{int(c)}|{int(d)}" for a, b, c, d in
         zip(f.serial, f.stage, f.lo_hz, ep)],
        dtype=object,
    )
    is_ref = (f.g1 == ref) & (f.g2 == ref) if ref is not None else (f.g1 == f.g2)
    for k in np.unique(karr):
        m = karr == k
        mr = m & is_ref
        if mr.sum():
            anchor[m] = S.cmean(f.phase[mr])
    ok = ~np.isnan(anchor)
    g = f.sel(ok)
    D = S.wrap(g.phase - anchor[ok])
    return g.copy_with(anchor=anchor[ok], D=D)


# ----------------------------------------------------------- design matrices --
def _signed_cat(values1, values2, levels, drop=None):
    """Antisymmetric categorical: +1 where the RX1 gain has that level,
    -1 where the RX2 gain has it. Encodes D = H(g1) - H(g2)."""
    cols = []
    names = []
    for lv in levels:
        if drop is not None and lv == drop:
            continue
        cols.append((values1 == lv).astype(float) - (values2 == lv).astype(float))
        names.append(str(lv))
    return np.column_stack(cols) if cols else np.zeros((len(values1), 0)), names


def _arm_cat(values1, values2, levels, drop=None):
    """Arm-specific categorical: separate RX1 and RX2 effects,
    D = Delta1(g1) - Delta2(g2)."""
    cols = []
    names = []
    for lv in levels:
        if drop is not None and lv == drop:
            continue
        cols.append((values1 == lv).astype(float))
        names.append(f"rx1:{lv}")
        cols.append(-(values2 == lv).astype(float))
        names.append(f"rx2:{lv}")
    return np.column_stack(cols) if cols else np.zeros((len(values1), 0)), names


def _key(*arrays):
    return np.array(
        ["|".join(str(x) for x in tup) for tup in zip(*arrays)], dtype=object
    )
