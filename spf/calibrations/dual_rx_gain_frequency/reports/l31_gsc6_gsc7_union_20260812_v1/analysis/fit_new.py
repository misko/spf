"""Fit the L31-shaped rung on the E-GSC6 + E-GSC7 union, and score it honestly."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SPF_REPO",
                           Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitlib as FL  # noqa: E402
import union as U  # noqa: E402
from spf.calibrations.gain_state_phase_model_v1.gain_tables import (  # noqa: E402
    band_for_lo, default_tables,
)

PROSPECTIVE_RATIO = 1.9   # quoted: rover_applicability_ladder_20260812_v1 section 1.4


def splits_for(frames):
    out = {
        "LOFO": list(FL.folds_leave_one_out(frames, "lo_hz")),
        "LOBLK": list(FL.folds_leave_freq_block_out(frames, 8)),
        "LORO": list(FL.folds_leave_one_out(frames, "serial")),
        "LOEO_partial": list(FL.folds_leave_one_out(frames, "epoch")),
    }
    band = np.array([band_for_lo(x) for x in frames["lo_hz"]], dtype=object)
    f2 = dict(frames)
    f2["band"] = band
    out["LOBAND"] = [
        (lbl, tr, te) for lbl, tr, te in FL.folds_leave_one_out(f2, "band")
    ]
    return out


def score(frames, name, static_fields, n_ripples, results: dict):
    results[name] = {}
    for sname, folds in splits_for(frames).items():
        usable = [(l, tr, te) for l, tr, te in folds
                  if tr.sum() and te.sum()]
        if len(usable) < 2:
            results[name][sname] = {
                "unavailable": f"only {len(usable)} usable fold(s) -- this split "
                               f"does not exist on this dataset"}
            print(f"  {name:18s} {sname:13s} unavailable "
                  f"({len(usable)} usable folds)")
            continue
        r = FL.evaluate(frames, folds, static_fields=static_fields,
                        n_ripples=n_ripples, name=name)
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        # per-source breakdown of the held-out error
        seen, err = r["_seen"], r["_err"]
        for src in ("E-GSC6", "E-GSC7"):
            m = (frames["source"][seen] == src)
            if m.any():
                clean.setdefault("by_source", {})[src] = FL.circ_stats(err[m])
        for radio in ("R17", "R18"):
            m = (frames["radio"][seen] == radio)
            if m.any():
                clean.setdefault("by_radio", {})[radio] = FL.circ_stats(err[m])
        hi = np.array([band_for_lo(x) == "high" for x in frames["lo_hz"][seen]])
        if hi.any():
            clean["high_band_only"] = FL.circ_stats(err[hi])
            clean["high_band_baseline"] = FL.circ_stats(frames["D"][seen][hi])
        results[name][sname] = clean
        print(f"  {name:18s} {sname:13s} mae={clean['all_cells']['mae_deg']:7.4f} "
              f"uneq={clean['unequal_gain_cells']['mae_deg']:7.4f} "
              f"p95={clean['all_cells']['p95_deg']:7.3f} "
              f"cov={clean['coverage']:.3f}")


def paired_test(frames, fields_a, nrip_a, fields_b, nrip_b, split="LOFO"):
    """Paired Wilcoxon on matched rows -- the standard this project holds itself to."""
    from scipy.stats import wilcoxon
    folds = splits_for(frames)[split]
    ra = FL.evaluate(frames, folds, static_fields=fields_a, n_ripples=nrip_a)
    rb = FL.evaluate(frames, folds, static_fields=fields_b, n_ripples=nrip_b)
    seen = ra["_seen"] & rb["_seen"]
    ea = np.abs(FL.wrap(frames["D"][seen] - ra["_pred"][seen]))
    eb = np.abs(FL.wrap(frames["D"][seen] - rb["_pred"][seen]))
    diff = ea - eb
    nz = diff != 0
    stat, p = (wilcoxon(ea[nz], eb[nz]) if nz.sum() else (float("nan"),) * 2)
    return {
        "split": split,
        "n_matched": int(seen.sum()),
        "n_nonzero_diff": int(nz.sum()),
        "mae_a_deg": float(np.degrees(ea.mean())),
        "mae_b_deg": float(np.degrees(eb.mean())),
        "mean_paired_diff_deg": float(np.degrees(diff.mean())),
        "median_paired_diff_deg": float(np.degrees(np.median(diff))),
        "a_better_fraction": float((diff < 0).mean()),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p": float(p),
    }


def diagonal_check(radios=("R17", "R18")):
    """Frame-level, held-out, zero-parameter test of the antisymmetric shape."""
    hd = U.gsc6_heldout_diagonal()
    out = {}
    for radio, blob in hd.items():
        if radio not in radios:
            continue
        cells = blob["cells"]
        d = np.array([c["D_obs_rad"] for c in cells])
        lo = np.array([c["lo_hz"] for c in cells])
        hi = np.array([band_for_lo(x) == "high" for x in lo])
        out[radio] = {
            "serial": blob["serial"],
            "n_cells": len(cells),
            "n_frames": int(sum(c["n"] for c in cells)),
            "all_bands": FL.circ_stats(d),
            "high_band": FL.circ_stats(d[hi]),
            "at_5766": FL.circ_stats(d[lo == 5.766e9]),
        }
    return out


def main(out_path: str):
    tab = default_tables()
    res: dict = {
        "convention": "ANCHORED D = phi - measured equal-gain cell (26 dB), "
                      "fail-closed, degrees",
        "sources": {
            "E-GSC6": "equal_gain_diagonal_20260811_v1/additive_cross_*.json "
                      "(FITTED per-arm coefficients, 2 radios x 24 LOs x 21 gains)",
            "E-GSC7": "e_gsc7_iio_20260812_v1/analysis.json results_5766[].steps_deg "
                      "(FITTED shared-effect steps, 2 radios x 2 transports, "
                      "5766 MHz ONLY, mixer 5..15)",
        },
    }

    variants = {
        "U1_gsc6_plus_gsc7_both_radios": dict(include_gsc7=True,
                                              radios=("R17", "R18")),
        "U2_gsc6_plus_gsc7_R18_only": dict(include_gsc7=True, radios=("R18",)),
        "U3_gsc6_only_both_radios": dict(include_gsc7=False,
                                         radios=("R17", "R18")),
        "U4_gsc6_only_R18": dict(include_gsc7=False, radios=("R18",)),
        "U5_gsc6_plus_gsc7_R17_only": dict(include_gsc7=True, radios=("R17",)),
    }

    res["datasets"] = {}
    res["ladder"] = {}
    for vname, kw in variants.items():
        f = U.build(**kw)
        uneq = f["g1"] != f["g2"]
        hi = np.array([band_for_lo(x) == "high" for x in f["lo_hz"]])
        res["datasets"][vname] = {
            "n_rows": int(len(f["D"])),
            "n_los": int(len(np.unique(f["lo_hz"]))),
            "radios": sorted(set(f["radio"].tolist())),
            "n_gains": int(len(set(f["g1"].tolist()) | set(f["g2"].tolist()))),
            "rows_by_source": {s: int((f["source"] == s).sum())
                               for s in sorted(set(f["source"].tolist()))},
            "baseline_all": FL.circ_stats(f["D"]),
            "baseline_unequal": FL.circ_stats(f["D"][uneq]),
            "baseline_high_band": FL.circ_stats(f["D"][hi]),
        }
        print(f"\n### {vname}: {res['datasets'][vname]['n_rows']} rows, "
              f"{res['datasets'][vname]['n_los']} LOs, "
              f"baseline {res['datasets'][vname]['baseline_all']['mae_deg']:.3f} deg")
        res["ladder"][vname] = {}
        score(f, "L31", FL.L31_FIELDS, 2, res["ladder"][vname])
        score(f, "L30", FL.L30_FIELDS, 0, res["ladder"][vname])
        score(f, "L26", FL.L26_FIELDS, 2, res["ladder"][vname])
        res["ladder"][vname]["L00"] = {
            "LOFO": {"coverage": 0.0,
                     "all_cells": res["datasets"][vname]["baseline_all"],
                     "unequal_gain_cells":
                         res["datasets"][vname]["baseline_unequal"]}
        }

    # --- paired tests, on matched rows -------------------------------------
    f1 = U.build(**variants["U1_gsc6_plus_gsc7_both_radios"])
    print("\npaired tests ...")
    res["paired_tests"] = {
        "L31_vs_L26_LOFO": paired_test(f1, FL.L31_FIELDS, 2, FL.L26_FIELDS, 2),
        "L31_vs_L30_LOFO": paired_test(f1, FL.L31_FIELDS, 2, FL.L30_FIELDS, 0),
        "L31_vs_L26_LOBLK": paired_test(f1, FL.L31_FIELDS, 2, FL.L26_FIELDS, 2,
                                        split="LOBLK"),
    }
    for k, v in res["paired_tests"].items():
        print(f"  {k}: mae_a={v['mae_a_deg']:.4f} mae_b={v['mae_b_deg']:.4f} "
              f"p={v['wilcoxon_p']:.4g} n={v['n_matched']}")

    # --- cross-radio transfer, explicitly ----------------------------------
    res["cross_radio_transfer"] = {}
    for train_r, test_r in (("R18", "R17"), ("R17", "R18")):
        tr = f1["radio"] == train_r
        te = f1["radio"] == test_r
        m = FL.fit_rung(f1["lo_hz"][tr], f1["g1"][tr], f1["g2"][tr],
                        f1["D"][tr], rf_hz=f1["rf_hz"][tr],
                        static_fields=FL.L31_FIELDS, n_ripples=2)
        idx = np.nonzero(te)[0]
        pred = np.zeros(len(idx))
        sup = np.zeros(len(idx), dtype=bool)
        for k, i in enumerate(idx):
            p = m.predict(f1["lo_hz"][i], int(f1["g1"][i]), int(f1["g2"][i]),
                          rf_hz=f1["rf_hz"][i], apply_rf_state_guard=False)
            pred[k] = p.residual_rad if p.supported else 0.0
            sup[k] = p.supported
        err = FL.wrap(f1["D"][te] - np.where(sup, pred, 0.0))
        hi = np.array([band_for_lo(x) == "high" for x in f1["lo_hz"][te]])
        res["cross_radio_transfer"][f"fit_{train_r}_score_{test_r}"] = {
            "coverage": float(sup.mean()),
            "all_bands": FL.circ_stats(err),
            "high_band": FL.circ_stats(err[hi]),
            "baseline_all_bands": FL.circ_stats(f1["D"][te]),
            "baseline_high_band": FL.circ_stats(f1["D"][te][hi]),
        }
        b = res["cross_radio_transfer"][f"fit_{train_r}_score_{test_r}"]
        print(f"  fit {train_r} -> score {test_r}: "
              f"{b['all_bands']['mae_deg']:.3f} deg vs baseline "
              f"{b['baseline_all_bands']['mae_deg']:.3f} deg "
              f"(high band {b['high_band']['mae_deg']:.3f} vs "
              f"{b['baseline_high_band']['mae_deg']:.3f})")

    # --- self-consistency of R18 alone, fitted and scored per fold ---------
    f18 = U.build(include_gsc7=True, radios=("R18",))
    res["paired_tests"]["R18only_L31_vs_L30_LOFO"] = paired_test(
        f18, FL.L31_FIELDS, 2, FL.L30_FIELDS, 0)
    print("  R18-only L31 vs L30 LOFO:",
          json.dumps(res["paired_tests"]["R18only_L31_vs_L30_LOFO"]))

    # --- the honest, frame-level, zero-parameter test ----------------------
    res["gsc6_heldout_diagonal"] = diagonal_check()
    print("\nheld-out diagonal (frame-level, nothing fitted to it):")
    for r, v in res["gsc6_heldout_diagonal"].items():
        print(f"  {r}: all {v['all_bands']['mae_deg']:.3f} deg, "
              f"high {v['high_band']['mae_deg']:.3f} deg, "
              f"5766 {v['at_5766']['mae_deg']:.3f} deg "
              f"({v['n_cells']} cells / {v['n_frames']} frames)")

    # --- prospective expectation -------------------------------------------
    d1 = res["datasets"]["U1_gsc6_plus_gsc7_both_radios"]
    lofo = res["ladder"]["U1_gsc6_plus_gsc7_both_radios"]["L31"]["LOFO"]
    res["prospective"] = {
        "transfer_ratio_quoted": PROSPECTIVE_RATIO,
        "source": "rover_applicability_ladder_20260812_v1 section 1.4 -- E-CAL3 "
                  "measured the committed l26_pooled_v1 at 4.79-4.80 deg against "
                  "a 9.06 deg anchor-only baseline on 103 fresh LOs",
        "retrospective_lofo_mae_deg": lofo["all_cells"]["mae_deg"],
        "retrospective_ratio_vs_baseline":
            d1["baseline_all"]["mae_deg"] / lofo["all_cells"]["mae_deg"],
        "high_band_baseline_mae_deg": d1["baseline_high_band"]["mae_deg"],
        "prospective_expectation_high_band_deg":
            d1["baseline_high_band"]["mae_deg"] / PROSPECTIVE_RATIO,
        "prospective_expectation_all_deg":
            d1["baseline_all"]["mae_deg"] / PROSPECTIVE_RATIO,
    }
    print("\nprospective:", json.dumps(res["prospective"], indent=1))

    # --- the candidate coefficient sets -------------------------------------
    res["fitted_models"] = {}
    for tag, kw in (("pooled_both_radios", dict(radios=("R17", "R18"))),
                    ("R18_only", dict(radios=("R18",)))):
        ff = U.build(include_gsc7=True, **kw)
        model = FL.fit_rung(
            ff["lo_hz"], ff["g1"], ff["g2"], ff["D"], rf_hz=ff["rf_hz"],
            static_fields=FL.L31_FIELDS, n_ripples=2, name="L31", tables=tab,
        )
        res["fitted_models"][tag] = {
            "tau_ns": [t * 1e9 for t in model.tau_seconds],
            "n_rows": model.provenance["n_rows"],
            "n_columns": model.provenance["n_columns"],
            "rank": model.provenance["rank"],
            "supported_levels": {k: [int(x) for x in v]
                                 for k, v in model.supported_levels.items()},
            "families_used": model.families_used,
        }
        np.save(Path(out_path).with_suffix(f".{tag}.npy"),
                np.array([model], dtype=object), allow_pickle=True)
        print(f"\nfitted {tag}:",
              json.dumps(res["fitted_models"][tag], indent=1))

    Path(out_path).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
