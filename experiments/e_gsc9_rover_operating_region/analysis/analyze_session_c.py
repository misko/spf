#!/usr/bin/env python3
"""Analyze the E-GSC9 no-pad/pads/pads-removed A/B/A-prime sequence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import wrap_phase
from spf.calibrations.dual_rx_gain_frequency.comparative_analysis import _load_radio
from spf.calibrations.dual_rx_gain_frequency.model_matrix import _metrics
from spf.calibrations.dual_rx_gain_frequency.runner import load_calibration_document


SCHEMA = "spf.experiment.e_gsc9.pad_discriminator"
SCHEMA_VERSION = 1
HCP1_WORST_CASE_PHASE_BOUND_DEG = 8.9
RESOLVED_MARGIN_RATIO = 3.0
LEG_STATE = {
    "a": "no_pads",
    "b": "pads_installed",
    "aprime": "pads_removed",
}
LEG_TX_GAIN_DB = {"a": -35, "b": -25, "aprime": -35}
SERIALS = (
    "104000bac4950008230026001b440a003a",
    "1040007c4a94000211000b009186843ef2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _circular_mean(values: list[float]) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if not vector.size:
        raise ValueError("cannot average an empty circular vector")
    return float(np.angle(np.mean(np.exp(1j * vector))))


def _cell_means(radio: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    selected = radio["quality_mask"] & np.isfinite(radio["phase"])
    for frequency, gain1, gain2, phase in zip(
        radio["frequency_hz"][selected],
        radio["gain1"][selected],
        radio["gain2"][selected],
        radio["phase"][selected],
    ):
        groups[(int(frequency), int(gain1), int(gain2))].append(float(phase))
    return {coordinate: _circular_mean(values) for coordinate, values in groups.items()}


def _coordinates(rows: set[tuple[int, int, int]]) -> list[list[int]]:
    return [list(row) for row in sorted(rows)]


def _summarize(
    residuals: np.ndarray, *, expected: int
) -> dict[str, float | int | None]:
    result = _metrics(residuals, expected=expected)
    absolute_deg = np.abs(np.degrees(residuals))
    result["circular_median_absolute_deg"] = (
        float(np.median(absolute_deg)) if absolute_deg.size else None
    )
    return result


def _validation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing strict validation: {path}")
    document = json.loads(path.read_text())
    if document.get("status") != "pass":
        raise ValueError(f"strict validation is not pass: {path}")
    return document


def _load_state(root: Path, *, leg: str) -> dict[str, Any]:
    path = root / "physical_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing physical-state record: {path}")
    state = json.loads(path.read_text())
    if state.get("leg") != leg:
        raise ValueError(f"{path}: expected leg {leg!r}, got {state.get('leg')!r}")
    expected_state = LEG_STATE[leg]
    if state.get("physical_state") != expected_state:
        raise ValueError(
            f"{path}: expected physical state {expected_state!r}, "
            f"got {state.get('physical_state')!r}"
        )
    if not str(state.get("operator_note", "")).strip():
        raise ValueError(f"{path}: operator note is empty")
    return state


def _schedule_signature(config: Any) -> tuple[Any, ...]:
    return (
        tuple(config.frequencies_hz),
        tuple(config.gains_db),
        tuple(tuple(pair) for pair in config.gain_pairs),
        config.repetitions,
        config.sample_rate_hz,
        config.bandwidth_hz,
        config.buffer_size,
        config.tone_offset_hz,
        config.tx_digital_amplitude,
    )


def _anchor_drift_by_frequency(
    radio: dict[str, Any], frequencies: tuple[int, ...]
) -> dict[int, dict[str, float | int | None]]:
    selected = radio["quality_mask"] & np.isfinite(radio["phase"])
    groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for frequency, gain1, gain2, epoch, phase in zip(
        radio["frequency_hz"][selected],
        radio["gain1"][selected],
        radio["gain2"][selected],
        radio["epoch"][selected],
        radio["phase"][selected],
    ):
        if int(gain1) == int(gain2):
            groups[(int(frequency), int(gain1), int(epoch))].append(float(phase))

    result = {}
    for frequency in frequencies:
        by_gain: dict[int, list[float]] = defaultdict(list)
        for (row_frequency, gain, _epoch), values in groups.items():
            if row_frequency == frequency:
                by_gain[gain].append(
                    float(np.angle(np.mean(np.exp(1j * np.asarray(values)))))
                )
        pairwise: list[float] = []
        for values in by_gain.values():
            for first_index, first in enumerate(values):
                for second in values[first_index + 1 :]:
                    pairwise.append(abs(float(np.degrees(wrap_phase(second - first)))))
        result[frequency] = {
            "equal_gain_cells": len(by_gain),
            "pairwise_epoch_comparisons": len(pairwise),
            "median_pairwise_drift_deg": (
                float(np.median(pairwise)) if pairwise else None
            ),
            "maximum_pairwise_drift_deg": max(pairwise) if pairwise else None,
        }
    return result


def _comparison_metrics(
    first: dict[tuple[int, int, int], float],
    second: dict[tuple[int, int, int], float],
    common: set[tuple[int, int, int]],
    *,
    expected: int,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    residuals = np.asarray(
        [wrap_phase(second[cell] - first[cell]) for cell in sorted(common)],
        dtype=np.float64,
    )
    return residuals, _summarize(residuals, expected=expected)


def _compare_radio(
    *,
    serial: str,
    radios: dict[str, dict[str, Any]],
    expected_by_frequency: dict[int, set[tuple[int, int, int]]],
) -> dict[str, Any]:
    if any(radio["serial"] != serial for radio in radios.values()):
        raise ValueError(f"dataset serial mismatch for {serial}")
    means = {leg: _cell_means(radio) for leg, radio in radios.items()}
    frequencies = tuple(sorted(expected_by_frequency))
    anchor_drift = {
        leg: _anchor_drift_by_frequency(radio, frequencies)
        for leg, radio in radios.items()
    }
    frequency_rows = []
    for frequency, expected in sorted(expected_by_frequency.items()):
        present = {leg: set(values) & expected for leg, values in means.items()}
        common = set.intersection(*present.values())
        ab_values, treatment = _comparison_metrics(
            means["a"], means["b"], common, expected=len(expected)
        )
        _, restoration = _comparison_metrics(
            means["a"], means["aprime"], common, expected=len(expected)
        )
        _, retained = _comparison_metrics(
            means["b"], means["aprime"], common, expected=len(expected)
        )
        absolute_restoration = np.abs(
            np.degrees(
                np.asarray(
                    [
                        wrap_phase(means["aprime"][cell] - means["a"][cell])
                        for cell in sorted(common)
                    ]
                )
            )
        )
        absolute_retained = np.abs(
            np.degrees(
                np.asarray(
                    [
                        wrap_phase(means["aprime"][cell] - means["b"][cell])
                        for cell in sorted(common)
                    ]
                )
            )
        )
        closer_fraction = (
            float(np.mean(absolute_restoration < absolute_retained)) if common else None
        )
        median_anchor_drifts = [
            anchor_drift[leg][frequency]["median_pairwise_drift_deg"]
            for leg in LEG_STATE
        ]
        available_anchor_drifts = [
            float(value) for value in median_anchor_drifts if value is not None
        ]
        noise_floor = max(available_anchor_drifts) if available_anchor_drifts else None
        treatment_mae = treatment["circular_mae_deg"]
        resolved_ratio = (
            float(treatment_mae / noise_floor)
            if treatment_mae is not None and noise_floor is not None and noise_floor > 0
            else None
        )
        full_coverage = len(common) == len(expected)
        coupling_bound_pass = bool(
            full_coverage
            and treatment_mae is not None
            and treatment_mae < HCP1_WORST_CASE_PHASE_BOUND_DEG
        )
        reversal_pass = bool(
            full_coverage
            and restoration["circular_mae_deg"] is not None
            and retained["circular_mae_deg"] is not None
            and restoration["circular_mae_deg"] < retained["circular_mae_deg"]
        )
        frequency_rows.append(
            {
                "frequency_hz": frequency,
                "expected_cells": len(expected),
                "common_quality_valid_cells": len(common),
                "missing_cells": {
                    leg: _coordinates(expected - cells)
                    for leg, cells in present.items()
                },
                "treatment_b_minus_a": treatment,
                "restoration_aprime_minus_a": restoration,
                "retained_treatment_aprime_minus_b": retained,
                "reversal_cell_fraction_closer_to_a_than_b": closer_fraction,
                "within_leg_anchor_drift": {
                    leg: anchor_drift[leg][frequency] for leg in LEG_STATE
                },
                "conservative_same_leg_noise_floor_deg": noise_floor,
                "treatment_to_noise_ratio": resolved_ratio,
                "treatment_reporting": (
                    "resolved_effect"
                    if resolved_ratio is not None
                    and resolved_ratio >= RESOLVED_MARGIN_RATIO
                    else "upper_bound_only"
                ),
                "coupling_bound_pass": coupling_bound_pass,
                "reversal_pass": reversal_pass,
                "h7_pass": coupling_bound_pass and reversal_pass,
                "signed_treatment_cell_mean_bias_deg": (
                    float(np.degrees(np.angle(np.mean(np.exp(1j * ab_values)))))
                    if ab_values.size
                    else None
                ),
            }
        )
    return {
        "serial": serial,
        "analysis_input_sha256": {
            leg: radio["analysis_input_sha256"] for leg, radio in radios.items()
        },
        "per_frequency": frequency_rows,
        "h7_pass": all(row["h7_pass"] for row in frequency_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg-a-root", type=Path, required=True)
    parser.add_argument("--leg-b-root", type=Path, required=True)
    parser.add_argument("--leg-aprime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "a": args.leg_a_root,
        "b": args.leg_b_root,
        "aprime": args.leg_aprime_root,
    }

    states = {leg: _load_state(root, leg=leg) for leg, root in roots.items()}
    config_paths = {leg: root / "capture_config.yaml" for leg, root in roots.items()}
    configs = {
        leg: load_calibration_document(path)[1] for leg, path in config_paths.items()
    }
    signatures = {_schedule_signature(config) for config in configs.values()}
    if len(signatures) != 1:
        raise ValueError("session-C leg schedules differ")
    for leg, config in configs.items():
        expected_tx_gain = LEG_TX_GAIN_DB[leg]
        if config.tx_gain_db != expected_tx_gain:
            raise ValueError(
                f"session-C leg {leg} TX gain is {config.tx_gain_db}, "
                f"expected {expected_tx_gain} dB"
            )
        if states[leg].get("tx_gain_db") != expected_tx_gain:
            raise ValueError(
                f"session-C leg {leg} physical-state record has unexpected TX gain"
            )
    config = configs["a"]
    expected_by_frequency = {
        int(frequency): {
            (int(frequency), int(gain1), int(gain2))
            for gain1, gain2 in config.gain_pairs
        }
        for frequency in config.frequencies_hz
    }

    validations = []
    rows = []
    for serial in SERIALS:
        radios = {}
        for leg, root in roots.items():
            validation_path = root / serial / "validation.json"
            validations.append(
                {
                    "leg": leg,
                    "serial": serial,
                    "path": str(validation_path.resolve()),
                    "status": _validation(validation_path)["status"],
                }
            )
            radios[leg] = _load_radio(
                root / serial / "calibration.v7.zarr", config=configs[leg]
            )
        rows.append(
            _compare_radio(
                serial=serial,
                radios=radios,
                expected_by_frequency=expected_by_frequency,
            )
        )

    repo = Path(__file__).resolve().parents[3]
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix_ns": time.time_ns(),
        "spf_git_sha": git_sha,
        "decision_rule": {
            "coupling_metric": "circular MAE of B-minus-A matching cell means",
            "coupling_bound_deg_exclusive": HCP1_WORST_CASE_PHASE_BOUND_DEG,
            "coupling_bound_source": (
                "experiments/e_hcp1_cross_arm_coupling/RESULTS.md worst-case "
                "phase upper bound"
            ),
            "reversal_metric": (
                "A-prime-minus-A circular MAE is less than "
                "A-prime-minus-B circular MAE"
            ),
            "strata": "each physical radio and each LO independently",
            "coverage": "all scheduled cells must be quality-valid in all three legs",
            "matched_level_tx_gain_db": LEG_TX_GAIN_DB,
            "resolved_effect_ratio": RESOLVED_MARGIN_RATIO,
            "overall_pass": "every radio-by-LO stratum passes bound and reversal",
        },
        "legs": {
            leg: {
                "root": str(root.resolve()),
                "config": str(config_paths[leg].resolve()),
                "config_sha256": _sha256(config_paths[leg]),
                "physical_state": states[leg],
            }
            for leg, root in roots.items()
        },
        "validations": validations,
        "per_radio": rows,
        "h7_pass": all(row["h7_pass"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
