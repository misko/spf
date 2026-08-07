"""E-GSC4 -- the adjacent-1 dB hardware-word discriminator, run on the wide
53-LO integer-gain survey instead of the two 2.4 GHz LOs originally designed.

The A-G campaign measured 12 adjacent-1 dB mixer steps and ZERO adjacent-1 dB
LNA steps, so the LNA's role rested on four 9 dB steps and on the ripple. The
wide survey swept every integer gain from -1 to 62 dB at 53 LOs across all
three gain-table bands, so every adjacent transition in every band is present.

Method: reconstruct H(radio, LO, g) from the committed additive fit (see
wide_survey.py for the exact algebra and its caveats), take every adjacent
1 dB step, and classify it by which audited AD9361 word moves, hierarchically
LNA > MIXER > TIA > LPF-only, exactly as REPORT.md 3.3 does. The LPF-only class
is the SAME-DATASET floor, as the E-GSC4 design requires.

These are fitted cell values from a different session; they are independent
corroboration and are never pooled with the campaign's H statistics.
"""

from __future__ import annotations

import json
import itertools

import numpy as np
from scipy import stats as sps

import features as FT
import spflib as S
from wide_survey import WideSurvey


def classify(band, g_from, g_to):
    s1 = FT.HW.state(band, g_from)
    s2 = FT.HW.state(band, g_to)
    if s1 is None or s2 is None:
        return None
    lna1, mix1, tia1, lpf1, _d1, r1 = s1
    lna2, mix2, tia2, lpf2, _d2, r2 = s2
    tab = FT.HW.tab[FT.BANDS[band]]
    rfdc1, rfdc2 = int(tab.rfdc[r1]), int(tab.rfdc[r2])
    if lna1 != lna2:
        cls = "lna"
    elif mix1 != mix2:
        cls = "mixer"
    elif tia1 != tia2:
        cls = "tia"
    elif lpf1 != lpf2:
        cls = "lpf_only"
    else:
        cls = "no_word_change"
    return {
        "class": cls,
        "lna": (lna1, lna2),
        "mixer": (mix1, mix2),
        "tia": (tia1, tia2),
        "lpf": (lpf1, lpf2),
        "rfdc": (rfdc1, rfdc2),
        "row": (r1, r2),
    }


def cluster_bootstrap_ratio(steps, cls_a, cls_b, n=4000, seed=7):
    """95% CI for median|dH|(cls_a) / median|dH|(cls_b), resampling whole
    (radio, LO) clusters -- the effective sample size is clusters, not steps."""
    rng = np.random.default_rng(seed)
    clusters = sorted({s["cluster"] for s in steps})
    by_cluster = {c: [s for s in steps if s["cluster"] == c] for c in clusters}
    out = []
    for _ in range(n):
        pick = rng.choice(len(clusters), len(clusters), replace=True)
        a, b = [], []
        for i in pick:
            for s in by_cluster[clusters[i]]:
                if s["class"] == cls_a:
                    a.append(abs(s["dH_deg"]))
                elif s["class"] == cls_b:
                    b.append(abs(s["dH_deg"]))
        if a and b and np.median(b) > 0:
            out.append(np.median(a) / np.median(b))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)


def summarise(vals):
    v = np.abs(np.asarray(vals, dtype=float))
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median_abs_deg": float(np.median(v)),
        "mean_abs_deg": float(v.mean()),
        "p90_abs_deg": float(np.percentile(v, 90)),
        "max_abs_deg": float(v.max()),
    }


def main(out_path="gsc4_wide_discriminator.json"):
    ws = WideSurvey()
    H = ws.H_deg()
    bands = S.gain_band(ws.freqs)
    gains = ws.gains

    steps = []
    for r, serial in enumerate(ws.serials):
        for fi, f_hz in enumerate(ws.freqs):
            b = int(bands[fi])
            for gi in range(len(gains) - 1):
                if gains[gi + 1] != gains[gi] + 1:
                    continue
                info = classify(b, int(gains[gi]), int(gains[gi + 1]))
                if info is None or info["class"] == "no_word_change":
                    continue
                steps.append(
                    {
                        "serial": serial,
                        "radio": S.SHORT[serial],
                        "lo_hz": float(f_hz),
                        "band": FT.BANDS[b],
                        "cluster": f"{S.SHORT[serial]}|{int(f_hz)}",
                        "g_from": int(gains[gi]),
                        "g_to": int(gains[gi + 1]),
                        "dH_deg": float(H[r, fi, gi + 1] - H[r, fi, gi]),
                        **{k: list(v) if isinstance(v, tuple) else v
                           for k, v in info.items()},
                    }
                )

    classes = ["lna", "mixer", "tia", "lpf_only"]
    by_class = {c: [s["dH_deg"] for s in steps if s["class"] == c] for c in classes}
    stats = {c: summarise(by_class[c]) for c in classes}
    for c in classes:
        stats[c]["n_clusters"] = len(
            {s["cluster"] for s in steps if s["class"] == c}
        )

    floor = stats["lpf_only"]["median_abs_deg"]
    ratios = {}
    for c in ("lna", "mixer", "tia"):
        if stats[c]["n"] == 0:
            continue
        lo, hi, nboot = cluster_bootstrap_ratio(steps, c, "lpf_only")
        u = sps.mannwhitneyu(
            np.abs(by_class[c]), np.abs(by_class["lpf_only"]),
            alternative="two-sided",
        )
        ratios[c] = {
            "median_ratio_vs_lpf_only": stats[c]["median_abs_deg"] / floor,
            "cluster_bootstrap_ci95": [lo, hi],
            "n_bootstrap": nboot,
            "mannwhitney_p": float(u.pvalue),
        }

    # per-band and per-LNA-transition detail
    per_band = {}
    for bname in ("low", "middle", "high"):
        per_band[bname] = {
            c: summarise([s["dH_deg"] for s in steps
                          if s["class"] == c and s["band"] == bname])
            for c in classes
        }

    lna_detail = {}
    for s in steps:
        if s["class"] != "lna":
            continue
        key = (
            f"{s['band']} {s['g_from']}->{s['g_to']} dB  "
            f"LNA {s['lna'][0]}->{s['lna'][1]}  MIX {s['mixer'][0]}->{s['mixer'][1]}"
            f"  TIA {s['tia'][0]}->{s['tia'][1]}  LPF {s['lpf'][0]}->{s['lpf'][1]}"
            f"  RF_DC {s['rfdc'][0]}->{s['rfdc'][1]}"
        )
        lna_detail.setdefault(key, []).append(s["dH_deg"])
    lna_detail_summary = {
        k: {**summarise(v), "signed_median_deg": float(np.median(v))}
        for k, v in sorted(lna_detail.items())
    }

    # the three 2.4 GHz LNA transitions the E-GSC4 design named, at 2412/2467
    named = {}
    for s in steps:
        if s["class"] == "lna" and 2410e6 <= s["lo_hz"] <= 2470e6:
            named.setdefault(
                f"{s['lo_hz']/1e6:g} MHz {s['g_from']}->{s['g_to']} dB", []
            ).append((s["radio"], round(s["dH_deg"], 4)))

    # The cleanest arm of all: an LNA step with MIXER, TIA *and* LPF frozen, so
    # the only other thing moving is the RF_DC_CAL flag -- which REPORT.md 6.2
    # already bounds at <=0.7 deg from the excluded F_neg stage.
    lna_lpf_frozen = [
        s for s in steps
        if s["class"] == "lna" and s["lpf"][0] == s["lpf"][1]
        and s["mixer"][0] == s["mixer"][1] and s["tia"][0] == s["tia"][1]
    ]
    clean = {
        "definition": "LNA word moves; MIXER, TIA and LPF all frozen; only the "
                      "RF_DC_CAL flag co-moves",
        **summarise([s["dH_deg"] for s in lna_lpf_frozen]),
        "n_clusters": len({s["cluster"] for s in lna_lpf_frozen}),
        "transitions": sorted(
            {f"{s['band']} {s['g_from']}->{s['g_to']} dB LNA "
             f"{s['lna'][0]}->{s['lna'][1]}" for s in lna_lpf_frozen}
        ),
    }

    # h_tia identifiability check: does a TIA-only transition exist anywhere?
    tia_rows = [s for s in steps if s["class"] == "tia"]
    tia_detail = {}
    for s in tia_rows:
        key = (f"{s['band']} {s['g_from']}->{s['g_to']} dB  "
               f"TIA {s['tia'][0]}->{s['tia'][1]}  LPF {s['lpf'][0]}->{s['lpf'][1]}"
               f"  RF_DC {s['rfdc'][0]}->{s['rfdc'][1]}")
        tia_detail.setdefault(key, []).append(s["dH_deg"])
    tia_detail = {k: summarise(v) for k, v in sorted(tia_detail.items())}
    CAMPAIGN_FLOOR = (0.355, 0.368)  # REPORT.md 3.3 measured standard error
    h_tia_decision = {
        "rule": "E-GSC1: if h_tia is separately identifiable and its magnitude "
                "sits at or below the campaign's 0.355-0.368 deg measured noise "
                "floor, drop it and re-declare L26 with one fewer family",
        "identifiable_here": len(tia_rows) > 0,
        "median_abs_dH_deg": stats["tia"]["median_abs_deg"],
        "campaign_noise_floor_deg": list(CAMPAIGN_FLOOR),
        "at_or_below_floor": stats["tia"]["median_abs_deg"] <= CAMPAIGN_FLOOR[1],
        "distinguishable_from_same_dataset_lpf_floor": (
            ratios["tia"]["mannwhitney_p"] < 0.05
        ),
        "verdict": None,  # filled below
        "detail": tia_detail,
    }
    h_tia_decision["verdict"] = (
        "DROP h_tia: identifiable but its magnitude is at or below the "
        "campaign's measured per-step noise floor"
        if h_tia_decision["at_or_below_floor"]
        else "KEEP h_tia: magnitude exceeds the campaign noise floor"
    )

    result = {
        "source": "committed model_matrix.json, frequency_specific_additive_gain_per_radio",
        "convention": "ANCHORED H(f,g) = [D(g,26)-D(26,g)]/2 in degrees, "
                      "reconstructed from fitted arm coefficients",
        "caveats": [
            "fitted cell values, not frames; the underlying fit's own residual "
            "is 0.514 deg MAE in-sample / 0.713 deg leave-one-epoch-out, so this "
            "reconstruction has that much measurement noise smoothed out",
            "different session and dates from the A-G campaign; independent "
            "corroboration only, never pooled with the campaign H statistics",
            "no epoch structure in the committed file, so no leave-one-epoch-out "
            "and no directly measured repeatability floor",
        ],
        "n_steps": len(steps),
        "n_clusters": len({s["cluster"] for s in steps}),
        "by_class": stats,
        "vs_same_dataset_lpf_only_floor": ratios,
        "per_band": per_band,
        "lna_transitions_detail": lna_detail_summary,
        "named_2p4ghz_lna_transitions": named,
        "lna_with_lpf_frozen": clean,
        "h_tia_decision": h_tia_decision,
        "tia_only_transitions_exist": len(tia_rows) > 0,
        "published_whole_survey_1db_step_median_abs_deg":
            ws.doc["symmetric_gain_structure"]["adjacent_one_db_step_distribution"][
                "median_absolute_deg"
            ],
    }
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1, default=str)

    print(f"{len(steps)} adjacent 1 dB steps over "
          f"{result['n_clusters']} (radio, LO) clusters\n")
    print(f"{'class':10s} {'n':>6s} {'clusters':>9s} {'median|dH|':>11s} "
          f"{'mean':>8s} {'p90':>8s} {'max':>8s}")
    for c in classes:
        s = stats[c]
        if not s["n"]:
            print(f"{c:10s} {'0':>6s}  -- never occurs --")
            continue
        print(f"{c:10s} {s['n']:6d} {s['n_clusters']:9d} "
              f"{s['median_abs_deg']:11.4f} {s['mean_abs_deg']:8.3f} "
              f"{s['p90_abs_deg']:8.3f} {s['max_abs_deg']:8.3f}")
    print()
    for c, v in ratios.items():
        print(f"{c:10s} / lpf-only floor = {v['median_ratio_vs_lpf_only']:6.3f}x  "
              f"CI95 [{v['cluster_bootstrap_ci95'][0]:.2f}, "
              f"{v['cluster_bootstrap_ci95'][1]:.2f}]  "
              f"MW p={v['mannwhitney_p']:.3g}")
    print("\nLNA transitions present:")
    for k, v in lna_detail_summary.items():
        print(f"  {k}   n={v['n']:3d} median|dH| {v['median_abs_deg']:7.3f} deg "
              f"(signed median {v['signed_median_deg']:+8.3f})")
    print("\nthe three named 2.4 GHz LNA transitions:")
    for k, v in sorted(named.items()):
        print(f"  {k}: {v}")
    print(f"\nLNA step with MIXER, TIA and LPF ALL frozen "
          f"(only RF_DC_CAL co-moves): {clean['transitions']}")
    print(f"  n={clean['n']} clusters={clean['n_clusters']} "
          f"median|dH| {clean['median_abs_deg']:.4f} deg  max {clean['max_abs_deg']:.3f}")
    print(f"\nh_tia: {h_tia_decision['verdict']}")
    for k, v in tia_detail.items():
        print(f"  {k}  n={v['n']} median|dH| {v['median_abs_deg']:.4f} deg")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
