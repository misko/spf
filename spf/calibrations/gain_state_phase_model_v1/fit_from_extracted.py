"""Refit the gain-state model from extracted campaign scalars.

This is the path to use when new calibration data arrives -- an E-CAL2 LNA fill,
a fifth radio, a denser comb -- and you want coefficients that cover the new
states. It reads only the small ``.npz`` scalar extracts produced by the source
analysis' ``extract.py``, never the multi-GB V7 stores directly.

**Do not refit from a sparse comb without checking its conditioning.** E-CAL3
refitted this model from ten uniformly spaced LOs and got 11.61 deg on the
held-out frequencies -- worse than the 9.06 deg anchor-only baseline, i.e. worse
than shipping no model at all. The cause was not the point count but the
*spacing*: at 600 MHz steps, ``600e6 * (tau1 - tau2) = 0.984`` cycles, so the two
ripple components alias onto each other. That comb's ripple-basis condition
number is 17.92 against a random-10 median of 2.35 -- roughly 1 comb in 2000 is
worse. Random ten-point combs reach 4.30 deg.

``check_comb_conditioning()`` below computes that number, and ``fit()`` warns
when it is bad. Two practical rules follow:

* With the delays **free**, ~16 LOs are needed before a refit reliably beats
  anchor-only. With them **frozen** at fleet values, ~8 suffice, and a
  well-conditioned ten-point comb recovers ~73% of the dense-fit improvement.
  Prefer freezing ``tau_seconds`` from a committed set over refitting them.
* Those figures come from subsampling one dense capture. No bench-time
  reduction is licensed until a sparse comb is captured prospectively.

    python -m spf.calibrations.gain_state_phase_model_v1.fit_from_extracted \
        --extracted /path/to/extracted \
        --stage spectroscopy_20260730_full/A \
        --stage spectroscopy_20260730_full_r2/F \
        --out /tmp/my_model.json

Each ``--stage`` is a ``<campaign>/<stage>`` directory under ``--extracted``
holding one ``<serial>.npz`` per radio.

Holdout scoring is built in because a coefficient set without a held-out number
beside it is not evidence of anything. ``--holdout frequency`` is the honest
default: it answers "predict a frequency this fit never saw", which is the case
the model exists for.

Anchoring, leakage and the reference gain
-----------------------------------------
The model predicts ``D = phase - measured equal-gain anchor``, where the anchor
is the equal-gain cell measured at the *same* radio, stage, LO and epoch. That
mirrors deployment, where the anchor is always a live measurement. Rows with no
anchor available are dropped rather than imputed.

The ripple delays are grid-searched **on the training fold only** in every fold,
so a reported holdout number is not contaminated by delay selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .gain_tables import band_for_lo, default_tables
from .model import GainStatePhaseModel


def load_stage(extracted: Path, rel: str) -> dict[str, np.ndarray]:
    """Load every serial's scalars for one ``<campaign>/<stage>`` directory."""
    d = extracted / rel
    if not d.is_dir():
        raise FileNotFoundError(f"no such stage directory: {d}")
    cols: dict[str, list] = defaultdict(list)
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz extracts in {d}")
    for npz in files:
        z = np.load(npz)
        n = len(z["sweep_completed"])
        g = z["sweep_requested_gain_db"].astype(np.int64)
        cols["serial"].append(np.full(n, npz.stem, dtype=object))
        cols["stage"].append(np.full(n, rel, dtype=object))
        cols["lo_hz"].append(z["sweep_lo_frequency_hz"].astype(float))
        cols["rf_hz"].append(z["sweep_rf_tone_frequency_hz"].astype(float))
        cols["g1"].append(g[:, 0])
        cols["g2"].append(g[:, 1])
        cols["epoch"].append(z["sweep_epoch"].astype(np.int64))
        cols["phase"].append(
            (z["phase_difference_rad"].astype(float) + math.pi) % (2 * math.pi) - math.pi
        )
        cols["qvalid"].append(z["sweep_quality_valid"].astype(bool))
        cols["completed"].append(z["sweep_completed"].astype(bool))
    return {k: np.concatenate(v) for k, v in cols.items()}


def attach_anchor(f: dict[str, np.ndarray], ref_db: int | None) -> dict[str, np.ndarray]:
    """Attach the measured equal-gain anchor per (serial, stage, LO, epoch).

    ``ref_db=None`` accepts whatever equal-gain cell the stage scheduled, which
    lets stages with different reference gains pool. The antisymmetric model is
    invariant to that choice: a constant added to ``H`` cancels in every
    prediction.
    """
    key = [
        f"{s}|{st}|{int(lo)}|{int(e)}"
        for s, st, lo, e in zip(f["serial"], f["stage"], f["lo_hz"], f["epoch"])
    ]
    eq = f["g1"] == f["g2"]
    if ref_db is not None:
        eq &= f["g1"] == ref_db
    anchor: dict[str, list[float]] = defaultdict(list)
    for k, is_eq, p in zip(key, eq, f["phase"]):
        if is_eq:
            anchor[k].append(p)
    # circular mean over however many anchor frames the schedule provided
    mean = {
        k: math.atan2(
            float(np.mean(np.sin(v))), float(np.mean(np.cos(v)))
        )
        for k, v in anchor.items()
    }
    have = np.array([k in mean for k in key])
    a = np.array([mean.get(k, np.nan) for k in key])
    d = (f["phase"] - a + math.pi) % (2 * math.pi) - math.pi
    out = dict(f)
    out["anchor"] = a
    out["D"] = d
    out["has_anchor"] = have
    return out


def circ_stats(err_rad: np.ndarray) -> dict[str, float]:
    e = np.abs((err_rad + math.pi) % (2 * math.pi) - math.pi)
    if not len(e):
        return {"n": 0}
    return {
        "n": int(len(e)),
        "mae_deg": float(np.degrees(e.mean())),
        "rmse_deg": float(np.degrees(np.sqrt((e**2).mean()))),
        "p95_deg": float(np.degrees(np.percentile(e, 95))),
        "max_deg": float(np.degrees(e.max())),
    }


def check_comb_conditioning(
    lo_hz, tau_seconds=(2.56e-9, 0.92e-9)
) -> dict:
    """Condition number of the ripple basis on this LO set.

    The ripple basis is ``[cos(2*pi*f*tau_k), sin(2*pi*f*tau_k)]`` for each
    delay. When the LO spacing makes the two delays indistinguishable the basis
    becomes near-singular and the refit is unidentifiable no matter how good the
    data is -- which is exactly how the E-CAL3 ten-point comb failed.

    Returns the condition number, the per-pair alias fraction (how many whole
    cycles of ``tau_i - tau_j`` fit in one LO step -- near an integer is bad),
    and a verdict. Reference points: the E-CAL3 comb scored 17.92, the median
    random ten-point comb scores 2.35.
    """
    los = np.unique(np.asarray(lo_hz, dtype=float))
    cols = []
    for t in tau_seconds:
        cols.append(np.cos(2 * np.pi * los * t))
        cols.append(np.sin(2 * np.pi * los * t))
    basis = np.column_stack(cols)
    cond = float(np.linalg.cond(basis))

    aliases = []
    if len(los) > 1:
        step = float(np.median(np.diff(los)))
        for i, ti in enumerate(tau_seconds):
            for tj in tau_seconds[i + 1:]:
                cycles = step * abs(ti - tj)
                # Aliasing needs the LO step to advance the beat between the two
                # delays by a whole NON-ZERO number of cycles. Near zero means
                # the beat is oversampled, which is the healthy case -- the dense
                # 50 MHz comb sits at 0.082 cycles/step and is fine.
                nearest = round(cycles)
                aliased = nearest >= 1 and abs(cycles - nearest) < 0.1
                aliases.append(
                    {
                        "delay_pair_ns": [ti * 1e9, tj * 1e9],
                        "cycles_per_lo_step": cycles,
                        "nearest_integer": int(nearest),
                        "distance_from_integer": abs(cycles - nearest),
                        "aliased": bool(aliased),
                    }
                )
    any_aliased = any(a["aliased"] for a in aliases)
    if cond > 10 or any_aliased:
        verdict = "BAD - refit will be unidentifiable; freeze the delays"
    elif cond > 5:
        verdict = "MARGINAL - prefer frozen delays"
    else:
        verdict = "ok"
    return {
        "n_los": int(len(los)),
        "condition_number": cond,
        "aliasing": aliases,
        "verdict": verdict,
    }


def evaluate(f: dict, folds, *, n_ripples: int, ref_name: str) -> dict:
    """Fit per fold on the training rows only and score the held-out rows.

    Unsupported test cells fail closed to the anchor -- their residual is scored
    as the uncorrected ``D``, never as zero and never as an extrapolated value.
    """
    pred = np.zeros(len(f["D"]))
    sup = np.zeros(len(f["D"]), dtype=bool)
    seen = np.zeros(len(f["D"]), dtype=bool)
    for label, tr, te in folds:
        if tr.sum() == 0 or te.sum() == 0:
            continue
        m = GainStatePhaseModel.fit(
            f["lo_hz"][tr], f["g1"][tr], f["g2"][tr], f["D"][tr],
            rf_hz=f["rf_hz"][tr], n_ripples=n_ripples,
        )
        idx = np.nonzero(te)[0]
        for i in idx:
            p = m.predict(
                f["lo_hz"][i], int(f["g1"][i]), int(f["g2"][i]),
                rf_hz=f["rf_hz"][i], apply_rf_state_guard=False,
            )
            pred[i] = p.residual_rad if p.supported else 0.0
            sup[i] = p.supported
        seen[idx] = True
    err = f["D"][seen] - pred[seen]
    uneq = (f["g1"] != f["g2"])[seen]
    return {
        "holdout": ref_name,
        "coverage": float(sup[seen].mean()),
        "all_cells": circ_stats(err),
        "unequal_gain_cells": circ_stats(err[uneq]),
        "baseline_no_correction": circ_stats(f["D"][seen]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extracted", required=True, type=Path)
    ap.add_argument("--stage", action="append", required=True,
                    help="<campaign>/<stage>, repeatable")
    ap.add_argument("--out", type=Path, help="write fitted coefficients here")
    ap.add_argument("--ref-db", type=int, default=None,
                    help="require this exact equal-gain reference (default: any)")
    ap.add_argument("--n-ripples", type=int, default=2)
    ap.add_argument("--holdout", default="frequency",
                    choices=["frequency", "epoch", "radio", "band", "none"])
    ap.add_argument("--quality-only", action="store_true", default=True)
    args = ap.parse_args()

    parts = [load_stage(args.extracted, s) for s in args.stage]
    f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = f["completed"] & (f["qvalid"] if args.quality_only else True)
    f = {k: v[keep] for k, v in f.items()}
    f = attach_anchor(f, args.ref_db)
    f = {k: v[f["has_anchor"]] for k, v in f.items()}

    tab = default_tables()
    band = np.array([band_for_lo(x) for x in f["lo_hz"]])
    print(f"rows={len(f['D'])}  radios={len(set(f['serial']))}  "
          f"LOs={len(np.unique(f['lo_hz']))}  "
          f"gains={sorted(set(f['g1'].tolist()) | set(f['g2'].tolist()))}")
    print(f"baseline D (anchor only): {circ_stats(f['D'])}")

    cond = check_comb_conditioning(f["lo_hz"])
    print(f"\ncomb conditioning: kappa={cond['condition_number']:.2f}  "
          f"({cond['n_los']} LOs)  -> {cond['verdict']}")
    for a in cond["aliasing"]:
        if a["aliased"]:
            print(f"  WARNING: delays {a['delay_pair_ns'][0]:.2f}/"
                  f"{a['delay_pair_ns'][1]:.2f} ns alias at this LO spacing "
                  f"({a['cycles_per_lo_step']:.3f} cycles per step, i.e. ~"
                  f"{a['nearest_integer']} whole cycle(s)). The two ripple "
                  f"components are not separable here.")
    if cond["condition_number"] > 10:
        print("  This comb is worse-conditioned than ~99.9% of random combs of "
              "the same size.\n  Refitting the delays from it will not work -- "
              "freeze them from a committed set.")

    if args.holdout != "none":
        keyed = {"frequency": f["lo_hz"], "epoch": f["epoch"],
                 "radio": f["serial"], "band": band}[args.holdout]
        folds = [(str(v), keyed != v, keyed == v) for v in np.unique(keyed)]
        res = evaluate(f, folds, n_ripples=args.n_ripples,
                       ref_name=f"leave-one-{args.holdout}-out")
        print(json.dumps(res, indent=1))

    model = GainStatePhaseModel.fit(
        f["lo_hz"], f["g1"], f["g2"], f["D"], rf_hz=f["rf_hz"],
        n_ripples=args.n_ripples,
    )
    print(f"\nfull fit: tau = {[round(t*1e9, 3) for t in model.tau_seconds]} ns")
    print(f"columns={model.provenance['n_columns']} rank={model.provenance['rank']}")
    print(f"supported levels: {model.supported_levels}")
    if args.out:
        model.save(args.out, stages=args.stage, n_rows=int(len(f["D"])),
                   holdout=args.holdout)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
