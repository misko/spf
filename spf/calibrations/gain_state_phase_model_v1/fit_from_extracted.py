"""Refit the gain-state model from extracted campaign scalars.

This is the path to use when new calibration data arrives -- an E-CAL2 LNA fill,
a fifth radio, a fresh coarse comb -- and you want coefficients that cover the
new states. It reads only the small ``.npz`` scalar extracts produced by the
source analysis' ``extract.py``, never the multi-GB V7 stores directly.

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
