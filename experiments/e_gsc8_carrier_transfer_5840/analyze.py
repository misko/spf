#!/usr/bin/env python3
"""Grade E-GSC8's preregistered carrier-transfer hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SERIALS = {
    "R17": "104000bac4950008230026001b440a003a",
    "R18": "1040007c4a94000211000b009186843ef2",
}
REFERENCE_LO_HZ = 5_766_000_000
TARGET_LO_HZ = 5_840_000_000
H1_MAXIMUM_RMS_DEG = 3.0
PRIOR_5300_RMS_DEG = {"R17": 79.8416680293495, "R18": 9.05925981708565}
PRIOR_5766_SUM_DEG = {
    "R17": {"usb": 7.026463251827139, "ip": 8.004107976364072},
    "R18": {"usb": 5.919374462958696, "ip": 5.405405489286933},
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _frequency(fit: dict, frequency_hz: int) -> dict:
    return next(
        row for row in fit["frequency_results"] if row["frequency_hz"] == frequency_hz
    )


def _curve_difference_deg(reference: dict, target: dict) -> np.ndarray:
    reference_curve = np.asarray(reference["shared_gain_effect_rad"], dtype=float)[1:]
    target_curve = np.asarray(target["shared_gain_effect_rad"], dtype=float)[1:]
    return np.rad2deg(
        np.angle(np.exp(1j * (target_curve - reference_curve)))
    )


def _bootstrap_rms_ci(
    paired_differences_deg: np.ndarray,
    *,
    iterations: int = 20_000,
    seed: int = 20260813,
) -> list[float]:
    """Paired-state nonparametric 95% CI for the RMS transfer error."""

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        paired_differences_deg,
        size=(iterations, len(paired_differences_deg)),
        replace=True,
    )
    rms = np.sqrt(np.mean(np.square(samples), axis=1))
    return [float(value) for value in np.quantile(rms, [0.025, 0.975])]


def _step_sum(row: dict) -> float:
    steps = [
        item["shared_effect_step_deg"]
        for item in row["adjacent_shared_gain_steps"]
        if item["gain_from_db"] >= 52
    ]
    if len(steps) != 10:
        raise ValueError(f"expected ten 52-to-62 steps, found {len(steps)}")
    return float(np.sum(steps))


def analyze(run_root: Path) -> dict:
    rows = []
    capture_gates = {}
    for radio, serial in SERIALS.items():
        fit = _load(run_root / serial / "additive_cross/analysis.json")
        validation = _load(run_root / serial / "validation.json")
        reference = _frequency(fit, REFERENCE_LO_HZ)
        target = _frequency(fit, TARGET_LO_HZ)
        paired = _curve_difference_deg(reference, target)
        transfer_rms = float(np.sqrt(np.mean(np.square(paired))))
        transfer_rms_ci = _bootstrap_rms_ci(paired)
        repeat_sum = _step_sum(reference)
        prior_values = list(PRIOR_5766_SUM_DEG[radio].values())
        prior_midpoint = float(np.mean(prior_values))
        # GSC7's observed transport repeatability is the full USB/IP
        # difference. Apply that tolerance about their midpoint; using only the
        # min/max span would silently halve the preregistered tolerance.
        transport_repeatability = float(abs(prior_values[0] - prior_values[1]))
        repeat_band = [
            prior_midpoint - transport_repeatability,
            prior_midpoint + transport_repeatability,
        ]
        h3_pass = repeat_band[0] <= repeat_sum <= repeat_band[1]
        rows.append(
            {
                "radio": radio,
                "serial": serial,
                # H3 is intentionally first because H1/H2 are uninterpretable
                # when the same-LO control has drifted outside GSC7's USB/IP band.
                "H3": {
                    "status": "pass" if h3_pass else "fail_invalidates_transfer",
                    "new_5766_step_sum_deg": repeat_sum,
                    "prior_usb_ip_values_deg": PRIOR_5766_SUM_DEG[radio],
                    "prior_midpoint_deg": prior_midpoint,
                    "transport_repeatability_tolerance_deg": transport_repeatability,
                    "acceptance_band_deg": repeat_band,
                    "distance_to_band_deg": float(
                        max(repeat_band[0] - repeat_sum, repeat_sum - repeat_band[1], 0)
                    ),
                },
                "H1": {
                    "status": (
                        "pass"
                        if h3_pass and transfer_rms_ci[1] <= H1_MAXIMUM_RMS_DEG
                        else "fail" if h3_pass
                        else "not_interpretable"
                    ),
                    "paired_gain_states": len(paired),
                    "paired_curve_difference_deg": paired.tolist(),
                    "rms_deg": transfer_rms,
                    "paired_state_bootstrap_95pct_ci_deg": transfer_rms_ci,
                    "threshold_deg": H1_MAXIMUM_RMS_DEG,
                },
                "H2": {
                    "status": (
                        "pass"
                        if h3_pass and transfer_rms < PRIOR_5300_RMS_DEG[radio]
                        else "fail" if h3_pass
                        else "not_interpretable"
                    ),
                    "new_5766_to_5840_rms_deg": transfer_rms,
                    "prior_usb_5766_to_5300_rms_deg": PRIOR_5300_RMS_DEG[radio],
                },
                "diagnostic_cross_lo_rms_deg": {
                    str(row["frequency_hz"]): float(
                        np.sqrt(
                            np.mean(
                                np.square(_curve_difference_deg(reference, row))
                            )
                        )
                    )
                    for row in fit["frequency_results"]
                    if row["frequency_hz"] != REFERENCE_LO_HZ
                },
            }
        )
        capture_gates[radio] = {
            key: validation[key]
            for key in (
                "status",
                "quality_valid_frames",
                "expected_frames",
                "passing_cells",
                "expected_cells",
            )
        }

    return {
        "schema": "spf.experiment.e_gsc8.result",
        "schema_version": 1,
        "artifact": str(run_root),
        "capture_gates": capture_gates,
        "grading_order": ["H3", "H1", "H2"],
        "confidence_interval_method": (
            "paired gain-state nonparametric bootstrap, 20000 resamples; "
            "the 11 gain states remain paired between 5766 and 5840 MHz"
        ),
        "results": rows,
        "hypotheses": {
            radio: {key: row[key]["status"] for key in ("H3", "H1", "H2")}
            for radio, row in zip(SERIALS, rows, strict=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["hypotheses"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
