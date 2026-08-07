"""E-CAL5 -- did the chain see a step of known magnitude?

Reuses E-CAL1's estimator unmodified (`load_epoch_h`, `summarize`,
`cluster_bootstrap_ci` from the arm-1 report) so the noise floor measured here is
computed exactly the way it was computed in both E-CAL1 arms, and is therefore
directly comparable.

Steps, from the audited high table:

    5 ->  6   MIXER 1->2 (+ LPF, RF_DC_CAL)   the positive control
    8 -> 10   LPF only, per 1 dB              the same-session floor

Decision rule (pre-registered in experiment_readme.md):

    |dH(5->6)| >= 5x floor AND >= 1.5 deg, sem < 0.35  -> sensitivity demonstrated
    2x .. 5x floor                                     -> partial
    < 2x floor                                         -> FALSIFYING: both E-CAL1
                                                          nulls are uninformative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARM1_ANALYZE = (
    REPO / "spf/calibrations/dual_rx_gain_frequency/reports"
    "/e_cal1_rfdc_20260807_v1/analyze.py"
)


def _load_arm1():
    """Load E-CAL1 arm 1's estimator by path.

    This file is also called `analyze.py`, so a plain `import analyze` resolves
    to whichever copy sys.path happens to reach first -- and when this module is
    imported (rather than run as a script) that is *this* file, silently
    self-importing. Loading by explicit path removes the ambiguity.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "e_cal1_arm1_analyze", ARM1_ANALYZE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A1 = _load_arm1()  # E-CAL1 arm 1 estimator, unmodified

# (low, high, per_db_divisor)
STEPS = {
    "mixer_5_to_6": (5, 6, 1),
    "lpf_only_8_to_10": (8, 10, 2),
}
CAMPAIGN_MIXER_MEDIAN_DEG = 2.664
GATE_RATIO = 5.0
GATE_ABS_DEG = 1.5
GATE_SEM_DEG = 0.35


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    datasets = sorted(args.root.glob("*/calibration.v7.zarr"))
    if not datasets:
        raise SystemExit(f"no datasets under {args.root}")

    pooled: dict[str, list] = {name: [] for name in STEPS}
    clusters: dict[str, dict] = {name: {} for name in STEPS}
    per_radio = {}

    for path in datasets:
        serial = path.parent.name
        curves = A1.load_epoch_h(path)
        radio = {"n_epoch_curves": len(curves), "steps": {}, "per_lo": {}}
        for name, (low, high, per_db) in STEPS.items():
            series = A1.step_series(curves, low, high, per_db)
            values = np.asarray(list(series.values()), dtype=np.float64)
            radio["steps"][name] = A1.summarize(values)
            pooled[name].extend(values.tolist())
            for (lo_hz, _epoch), value in series.items():
                clusters[name].setdefault((serial, lo_hz), []).append(value)
                radio["per_lo"].setdefault(str(lo_hz), {}).setdefault(
                    name, []
                ).append(value)
        radio["per_lo"] = {
            lo: {
                n: A1.summarize(np.asarray(v, dtype=np.float64))
                for n, v in by_step.items()
            }
            for lo, by_step in radio["per_lo"].items()
        }
        per_radio[serial] = radio

    summary = {
        name: A1.summarize(np.asarray(vals, dtype=np.float64))
        for name, vals in pooled.items()
    }
    cluster_ci = {
        name: A1.cluster_bootstrap_ci(
            {k: np.asarray(v, dtype=np.float64) for k, v in by_cluster.items()}
        )
        for name, by_cluster in clusters.items()
    }

    mixer = summary["mixer_5_to_6"]
    floor = summary["lpf_only_8_to_10"]
    mixer_mag = mixer["mean_abs_deg"]
    floor_mag = floor["mean_abs_deg"]
    ratio = mixer_mag / floor_mag if floor_mag else float("inf")
    sem = mixer["sem_of_abs_deg"]

    if ratio >= GATE_RATIO and mixer_mag >= GATE_ABS_DEG and sem < GATE_SEM_DEG:
        verdict = "sensitivity_demonstrated"
    elif ratio < 2.0:
        verdict = "FALSIFYING_chain_cannot_see_a_known_step"
    else:
        verdict = "partial_sensitivity"

    result = {
        "experiment": "E-CAL5 positive control",
        "steps_measured": {
            n: {"from_db": lo, "to_db": hi, "per_db_divisor": d}
            for n, (lo, hi, d) in STEPS.items()
        },
        "per_radio": per_radio,
        "pooled": summary,
        "cluster_ci": cluster_ci,
        "mixer_step_deg": mixer_mag,
        "lpf_floor_deg": floor_mag,
        "ratio_mixer_over_floor": ratio,
        "campaign_reference_deg": CAMPAIGN_MIXER_MEDIAN_DEG,
        "decision": {
            "gate_ratio": GATE_RATIO,
            "gate_abs_deg": GATE_ABS_DEG,
            "gate_sem_deg": GATE_SEM_DEG,
            "verdict": verdict,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"mixer 5->6      : {mixer_mag:.3f} deg  (median |.| "
          f"{mixer['median_abs_deg']:.3f}, sem {sem:.3f})")
    print(f"LPF floor /1 dB : {floor_mag:.3f} deg  (median |.| "
          f"{floor['median_abs_deg']:.3f})")
    print(f"ratio           : {ratio:.2f}x   campaign reference "
          f"{CAMPAIGN_MIXER_MEDIAN_DEG} deg")
    print(f"verdict         : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
