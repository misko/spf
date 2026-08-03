"""The model ladder and its holdout evaluation."""

from __future__ import annotations

import numpy as np

import features as FT
import spflib as S
from models import Design, LadderModel, Term, build_design

TAU_GRID = np.concatenate(
    [np.arange(0.10, 4.0, 0.02), np.arange(4.0, 8.01, 0.05)]
) * 1e-9


def make_ladder(include_freq_lut=True):
    L = []
    add = L.append

    # ---- tier 0: nothing -----------------------------------------------------
    add(LadderModel("L00 zero (per-session anchor only)", [], tier=0,
                    note="no gain correction at all"))

    # ---- tier 1: gain only, frequency-blind ---------------------------------
    add(LadderModel("L01 sym H(g), universal", [Term("g")], tier=1,
                    note="one shared antisymmetric gain LUT for every radio+band"))
    add(LadderModel("L02 sym H(radio,g)", [Term("g", groups=("serial",))], tier=1))
    add(LadderModel("L03 arm d1(g),d2(g) universal",
                    [Term("g", arm_specific=True)], tier=1))
    add(LadderModel("L04 arm d1,d2 per radio",
                    [Term("g", groups=("serial",), arm_specific=True)], tier=1))

    # ---- tier 2: AD9361 hardware state replaces requested dB ----------------
    add(LadderModel("L05 sym H(lna,mixer,tia,lpf) universal",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf")], tier=2,
                    note="additive over audited gain-table state fields"))
    add(LadderModel("L06 sym H(gain-table row) universal", [Term("row")], tier=2,
                    note="row index alone, shared across the three band tables"))
    add(LadderModel("L07 sym H(lna,mixer,tia,lpf) per radio",
                    [Term("lna", groups=("serial",)),
                     Term("mixer", groups=("serial",)),
                     Term("tia", groups=("serial",)),
                     Term("lpf", groups=("serial",))], tier=2))

    # ---- tier 3: band-aware gain LUT ----------------------------------------
    add(LadderModel("L08 sym H(band,g) universal",
                    [Term("g", groups=("band",))], tier=3))
    add(LadderModel("L09 sym H(radio,band,g)",
                    [Term("g", groups=("band", "serial"))], tier=3))
    add(LadderModel("L10 arm d(radio,band,g)",
                    [Term("g", groups=("band", "serial"), arm_specific=True)], tier=3))

    # ---- tier 4: + pure delay ------------------------------------------------
    add(LadderModel("L11 sym H(band,g) + delay(g) universal",
                    [Term("g", groups=("band",)), Term("g", basis="delay")], tier=4,
                    note="lumped phase per band-gain plus a gain-dependent time delay"))
    add(LadderModel("L12 sym H(band,g) + delay(g) per radio",
                    [Term("g", groups=("band", "serial")),
                     Term("g", basis="delay", groups=("serial",))], tier=4))
    add(LadderModel("L13 sym H(state) + delay(lna) universal",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="delay")], tier=4))

    # ---- tier 5: + reflection ripple (the mechanism) ------------------------
    add(LadderModel("L14 sym H(band,g) + 1 ripple, amp per g, universal",
                    [Term("g", groups=("band",)),
                     Term("g", basis="cos0"), Term("g", basis="sin0")],
                    tau_grids=[TAU_GRID], tier=5))
    add(LadderModel("L15 sym H(radio,band,g) + 1 ripple per radio",
                    [Term("g", groups=("band", "serial")),
                     Term("g", basis="cos0", groups=("serial",)),
                     Term("g", basis="sin0", groups=("serial",))],
                    tau_grids=[TAU_GRID], tier=5))
    add(LadderModel("L16 MECHANISTIC: H(state) + ripple amp per LNA state",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="cos0"), Term("lna", basis="sin0")],
                    tau_grids=[TAU_GRID], tier=5,
                    note="ripple amplitude set by the LNA state change only"))
    add(LadderModel("L17 MECHANISTIC + mixer ripple",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="cos0"), Term("lna", basis="sin0"),
                     Term("mixer", basis="cos0"), Term("mixer", basis="sin0")],
                    tau_grids=[TAU_GRID], tier=5))
    add(LadderModel("L18 sym H(band,g) + 2 ripples, amp per g, universal",
                    [Term("g", groups=("band",)),
                     Term("g", basis="cos0"), Term("g", basis="sin0"),
                     Term("g", basis="cos1"), Term("g", basis="sin1")],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))
    add(LadderModel("L19 sym H(band,g)+2 ripples per radio + delay",
                    [Term("g", groups=("band", "serial")),
                     Term("g", basis="delay", groups=("serial",)),
                     Term("g", basis="cos0", groups=("serial",)),
                     Term("g", basis="sin0", groups=("serial",)),
                     Term("g", basis="cos1", groups=("serial",)),
                     Term("g", basis="sin1", groups=("serial",))],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))
    add(LadderModel("L20 L18 + arm-specific ripple correction",
                    [Term("g", groups=("band",)),
                     Term("g", basis="cos0"), Term("g", basis="sin0"),
                     Term("g", basis="cos1"), Term("g", basis="sin1"),
                     Term("g", basis="cos0", groups=("serial",), arm_specific=True),
                     Term("g", basis="sin0", groups=("serial",), arm_specific=True)],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))

    # ---- tier 6: smooth-in-frequency nuisance -------------------------------
    add(LadderModel("L21 sym H(band,g) + quad(f) per band-gain, per radio",
                    [Term("g", groups=("band", "serial")),
                     Term("g", basis="lin", groups=("band", "serial")),
                     Term("g", basis="quad", groups=("band", "serial"))], tier=6))
    add(LadderModel("L22 L19 + quad(f) per band-gain",
                    [Term("g", groups=("band", "serial")),
                     Term("g", basis="lin", groups=("band", "serial")),
                     Term("g", basis="quad", groups=("band", "serial")),
                     Term("g", basis="cos0", groups=("serial",)),
                     Term("g", basis="sin0", groups=("serial",)),
                     Term("g", basis="cos1", groups=("serial",)),
                     Term("g", basis="sin1", groups=("serial",))],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=6))

    # ---- tier 5b: richer mechanistic variants -------------------------------
    add(LadderModel("L25 MECH: H(state) + ripple amp per (band,LNA)",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="cos0", groups=("band",)),
                     Term("lna", basis="sin0", groups=("band",))],
                    tau_grids=[TAU_GRID], tier=5,
                    note="lets the source match |Gamma_s| vary across the band"))
    add(LadderModel("L26 MECH: H(state) + 2 ripples per LNA state",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="cos0"), Term("lna", basis="sin0"),
                     Term("lna", basis="cos1"), Term("lna", basis="sin1")],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))
    add(LadderModel("L27 MECH: + delay(state) + 2 ripples per (band,LNA)",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="delay"), Term("mixer", basis="delay"),
                     Term("lna", basis="cos0", groups=("band",)),
                     Term("lna", basis="sin0", groups=("band",)),
                     Term("lna", basis="cos1", groups=("band",)),
                     Term("lna", basis="sin1", groups=("band",))],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))
    add(LadderModel("L28 MECH L27 + per-radio ripple amplitudes",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf"),
                     Term("lna", basis="delay"), Term("mixer", basis="delay"),
                     Term("lna", basis="cos0", groups=("band", "serial")),
                     Term("lna", basis="sin0", groups=("band", "serial")),
                     Term("lna", basis="cos1", groups=("band", "serial")),
                     Term("lna", basis="sin1", groups=("band", "serial"))],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))

    # ---- tier 5c: MINIMAL mechanistic set -----------------------------------
    # The audited LPF word is a BASEBAND gain: the 1 dB step analysis shows it
    # carries ~0.3 deg (noise floor) while RF-state changes carry ~2.1 deg.
    # Dropping its categorical term is what lets the model predict a requested
    # gain it has never seen.
    add(LadderModel("L30 MIN: H(lna,mixer,tia) only, universal",
                    [Term("lna"), Term("mixer"), Term("tia")], tier=5,
                    note="no baseband LPF term at all"))
    add(LadderModel("L31 MIN + 2 ripples per LNA state, universal",
                    [Term("lna"), Term("mixer"), Term("tia"),
                     Term("lna", basis="cos0"), Term("lna", basis="sin0"),
                     Term("lna", basis="cos1"), Term("lna", basis="sin1")],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))
    add(LadderModel("L32 MIN + 2 ripples per (band,LNA) + delay(lna,mixer)",
                    [Term("lna"), Term("mixer"), Term("tia"),
                     Term("lna", basis="delay"), Term("mixer", basis="delay"),
                     Term("lna", basis="cos0", groups=("band",)),
                     Term("lna", basis="sin0", groups=("band",)),
                     Term("lna", basis="cos1", groups=("band",)),
                     Term("lna", basis="sin1", groups=("band",))],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))
    add(LadderModel("L33 L32 + linear LPF slope (1 param)",
                    [Term("lna"), Term("mixer"), Term("tia"),
                     Term("lpf_lin"),
                     Term("lna", basis="delay"), Term("mixer", basis="delay"),
                     Term("lna", basis="cos0", groups=("band",)),
                     Term("lna", basis="sin0", groups=("band",)),
                     Term("lna", basis="cos1", groups=("band",)),
                     Term("lna", basis="sin1", groups=("band",))],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))
    add(LadderModel("L34 L33 + ripple amps per (band,LNA,radio)",
                    [Term("lna"), Term("mixer"), Term("tia"), Term("lpf_lin"),
                     Term("lna", basis="delay"), Term("mixer", basis="delay"),
                     Term("lna", basis="cos0", groups=("band", "serial")),
                     Term("lna", basis="sin0", groups=("band", "serial")),
                     Term("lna", basis="cos1", groups=("band", "serial")),
                     Term("lna", basis="sin1", groups=("band", "serial"))],
                    tau_grids=[TAU_GRID, TAU_GRID], tier=5))

    # ---- tier 6b: agnostic Fourier-in-frequency comparator ------------------
    fixed = [np.array([t * 1e-9]) for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    add(LadderModel("L29 AGNOSTIC: H(band,g) + 6 fixed-delay Fourier terms per g",
                    [Term("g", groups=("band",))]
                    + [Term("g", basis=b) for i in range(6)
                       for b in (f"cos{i}", f"sin{i}")],
                    tau_grids=fixed, tier=6,
                    note="superset of the ripple model; tests whether the "
                         "mechanism earns its constraints"))

    # ---- tier 7: exact-frequency lookups (accuracy reference) ---------------
    if include_freq_lut:
        add(LadderModel("L23 sym H(radio,f,g)  per-frequency antisym LUT",
                        [Term("g", groups=("serial", "lo_hz"))], tier=7,
                        note="cannot predict an unmeasured frequency by construction"))
        add(LadderModel("L24 arm d(radio,f,g)  per-frequency additive LUT",
                        [Term("g", groups=("serial", "lo_hz"), arm_specific=True)],
                        tier=7, note="the repo's current accuracy reference"))
    return L


# ------------------------------------------------------------------ splits ---
def splits_leave_one_epoch_out(f):
    for e in np.unique(f.epoch):
        te = f.epoch == e
        yield f"epoch={e}", ~te, te


def splits_leave_one_frequency_out(f):
    for lo in np.unique(f.lo_hz):
        te = f.lo_hz == lo
        yield f"lo={lo/1e6:.0f}MHz", ~te, te


def splits_leave_one_radio_out(f):
    for s in np.unique(f.serial):
        te = f.serial == s
        yield f"radio={S.SHORT.get(s,s)[:6]}", ~te, te


def splits_leave_one_band_out(f):
    for b in np.unique(f.band):
        te = f.band == b
        yield f"band={FT.BANDS[b]}", ~te, te


def splits_leave_freq_block_out(f, n_blocks=8):
    """Contiguous frequency blocks held out together: a much harder and more
    realistic 'new operating band' test than single-frequency holdout, which
    leaves immediate neighbours in the training set."""
    los = np.unique(f.lo_hz)
    edges = np.array_split(np.arange(len(los)), n_blocks)
    for k, idx in enumerate(edges):
        block = set(los[idx].tolist())
        te = np.array([x in block for x in f.lo_hz])
        yield f"block{k}({los[idx][0]/1e6:.0f}-{los[idx][-1]/1e6:.0f})", ~te, te


SPLITS = {
    "LOEO leave-one-epoch-out": splits_leave_one_epoch_out,
    "LOFO leave-one-frequency-out": splits_leave_one_frequency_out,
    "LOBLOCK leave-frequency-block-out": splits_leave_freq_block_out,
    "LORO leave-one-radio-out": splits_leave_one_radio_out,
    "LOBAND leave-one-gain-table-band-out": splits_leave_one_band_out,
}


# -------------------------------------------------------------- evaluation ---
def evaluate(f: S.Frames, ladder, split_name, verbose=True):
    y = f.D
    designs = {m.name: build_design(f, m.terms) for m in ladder}
    rows = []
    for m in ladder:
        d = designs[m.name]
        preds = np.zeros(len(f))
        sup = np.zeros(len(f), dtype=bool)
        seen = np.zeros(len(f), dtype=bool)
        npar = []
        taus = []
        for _lbl, tr, te in SPLITS[split_name](f):
            tri = np.nonzero(tr)[0]
            tei = np.nonzero(te)[0]
            if len(tri) == 0 or len(tei) == 0:
                continue
            p, sp, tt, nc = m.fit_eval(d, y, tri, tei)
            preds[tei] = p
            sup[tei] = sp
            seen[tei] = True
            npar.append(nc)
            if tt.size:
                taus.append(tt)
        # fail-closed: an unsupported cell falls back to the anchor, per the
        # repo prediction contract. Never report an extrapolated value there.
        fail_closed = np.where(sup, preds, 0.0)
        err = S.wrap(y[seen] - fail_closed[seen])
        st = S.err_stats(err)
        st_sup = S.err_stats(S.wrap(y[seen & sup] - preds[seen & sup]), prefix="sup_")
        # the raw (un-clamped) error is diagnostic only and must never be
        # confused with the deployable number
        st_raw = S.err_stats(S.wrap(y[seen] - preds[seen]), prefix="raw_")
        # a model that supports nothing must score exactly the anchor-only
        # baseline; this catches a results file written by a stale code path
        if sup[seen].sum() == 0:
            base = S.cmae_deg(y[seen])
            assert abs(st["mae_deg"] - base) < 1e-9, (
                f"{m.name}: zero coverage must fail closed to the "
                f"{base:.4f} deg baseline, got {st['mae_deg']:.4f}"
            )
        # error on the cells a deployed correction actually acts on: the
        # equal-gain anchor cell has an identically-zero residual by
        # construction and dilutes every aggregate
        uneq = seen & (f.g1 != f.g2)
        st_uneq = S.err_stats(
            S.wrap(y[uneq] - fail_closed[uneq]), prefix="unequal_"
        )
        rows.append(
            {
                "model": m.name,
                "tier": m.tier,
                "params": int(np.median(npar)) if npar else 0,
                "coverage": float(sup[seen].mean()),
                **st,
                **st_sup,
                **st_raw,
                **st_uneq,
                "tau_ns": (
                    np.round(np.median(np.array(taus), axis=0) * 1e9, 4).tolist()
                    if taus
                    else []
                ),
                "note": m.note,
            }
        )
        if verbose:
            r = rows[-1]
            print(
                f"  {r['model']:52s} p={r['params']:5d} cov={r['coverage']:5.3f} "
                f"MAE={r['mae_deg']:7.3f} uneqMAE={r['unequal_mae_deg']:7.3f} "
                f"P95={r['p95_deg']:7.3f} max={r['max_deg']:8.3f}"
                + (f"  tau={r['tau_ns']}" if r["tau_ns"] else "")
            )
    return rows
