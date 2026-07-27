"""Circular additive phase model for validated gain/frequency calibration data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import circular_stats, wrap_phase
from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


def _circular_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * values))))


def fit_additive_surface(
    phase: np.ndarray,
    gain1_index: np.ndarray,
    gain2_index: np.ndarray,
    gain_count: int,
    *,
    iterations: int = 50,
) -> dict[str, Any]:
    """Fit phase ~= intercept + RX1(g1) + RX2(g2) on the unit circle."""

    phase = np.asarray(phase, dtype=np.float64)
    gain1_index = np.asarray(gain1_index, dtype=np.int64)
    gain2_index = np.asarray(gain2_index, dtype=np.int64)
    if not (
        phase.ndim == gain1_index.ndim == gain2_index.ndim == 1
        and phase.size == gain1_index.size == gain2_index.size
    ):
        raise ValueError("phase and gain indices must be equal-length vectors")
    if phase.size == 0:
        raise ValueError("cannot fit an empty phase surface")
    intercept = _circular_mean(phase)
    rx1 = np.zeros(gain_count, dtype=np.float64)
    rx2 = np.zeros(gain_count, dtype=np.float64)
    for _ in range(iterations):
        for index in range(gain_count):
            selected = gain1_index == index
            if np.any(selected):
                rx1[index] = _circular_mean(
                    wrap_phase(phase[selected] - intercept - rx2[gain2_index[selected]])
                )
        for index in range(gain_count):
            selected = gain2_index == index
            if np.any(selected):
                rx2[index] = _circular_mean(
                    wrap_phase(phase[selected] - intercept - rx1[gain1_index[selected]])
                )
        residual_without_intercept = wrap_phase(
            phase - rx1[gain1_index] - rx2[gain2_index]
        )
        intercept = _circular_mean(residual_without_intercept)

    common_indices = np.intersect1d(np.unique(gain1_index), np.unique(gain2_index))
    if common_indices.size == 0:
        raise ValueError(
            "additive surface is not identifiable: no gain appears in both roles"
        )
    reference = int(common_indices[np.argmin(np.abs(common_indices - gain_count // 2))])
    intercept = float(wrap_phase(intercept + rx1[reference] + rx2[reference]))
    rx1 = wrap_phase(rx1 - rx1[reference])
    rx2 = wrap_phase(rx2 - rx2[reference])
    prediction = wrap_phase(intercept + rx1[gain1_index] + rx2[gain2_index])
    residual = wrap_phase(phase - prediction)
    return {
        "intercept_rad": intercept,
        "reference_gain_index": reference,
        "rx1_effect_rad": rx1,
        "rx2_effect_rad": rx2,
        "prediction_rad": prediction,
        "residual_rad": residual,
    }


def fit_grouped_additive_surface(
    phase: np.ndarray,
    gain1_index: np.ndarray,
    gain2_index: np.ndarray,
    group_index: np.ndarray,
    gain_count: int,
    group_count: int,
    *,
    iterations: int = 50,
) -> dict[str, Any]:
    """Fit phase ~= intercept[group] + RX1(g1) + RX2(g2).

    This is the frequency-shared counterpart to :func:`fit_additive_surface`.
    Each frequency keeps its own intercept while both receivers share gain
    curves. Missing groups are represented by a NaN intercept and must not be
    used for prediction.
    """

    phase = np.asarray(phase, dtype=np.float64)
    gain1_index = np.asarray(gain1_index, dtype=np.int64)
    gain2_index = np.asarray(gain2_index, dtype=np.int64)
    group_index = np.asarray(group_index, dtype=np.int64)
    if not (
        phase.ndim == gain1_index.ndim == gain2_index.ndim == group_index.ndim == 1
        and phase.size == gain1_index.size == gain2_index.size == group_index.size
    ):
        raise ValueError("phase and model indices must be equal-length vectors")
    if phase.size == 0:
        raise ValueError("cannot fit an empty grouped phase surface")
    if (
        np.any(gain1_index < 0)
        or np.any(gain1_index >= gain_count)
        or np.any(gain2_index < 0)
        or np.any(gain2_index >= gain_count)
        or np.any(group_index < 0)
        or np.any(group_index >= group_count)
    ):
        raise ValueError("model index outside configured range")

    intercept = np.full(group_count, np.nan, dtype=np.float64)
    for index in range(group_count):
        selected = group_index == index
        if np.any(selected):
            intercept[index] = _circular_mean(phase[selected])
    rx1 = np.zeros(gain_count, dtype=np.float64)
    rx2 = np.zeros(gain_count, dtype=np.float64)
    for _ in range(iterations):
        for index in range(gain_count):
            selected = gain1_index == index
            if np.any(selected):
                rx1[index] = _circular_mean(
                    wrap_phase(
                        phase[selected]
                        - intercept[group_index[selected]]
                        - rx2[gain2_index[selected]]
                    )
                )
        for index in range(gain_count):
            selected = gain2_index == index
            if np.any(selected):
                rx2[index] = _circular_mean(
                    wrap_phase(
                        phase[selected]
                        - intercept[group_index[selected]]
                        - rx1[gain1_index[selected]]
                    )
                )
        for index in range(group_count):
            selected = group_index == index
            if np.any(selected):
                intercept[index] = _circular_mean(
                    wrap_phase(
                        phase[selected]
                        - rx1[gain1_index[selected]]
                        - rx2[gain2_index[selected]]
                    )
                )

    common_indices = np.intersect1d(np.unique(gain1_index), np.unique(gain2_index))
    if common_indices.size == 0:
        raise ValueError(
            "grouped additive surface is not identifiable: "
            "no gain appears in both roles"
        )
    reference = int(common_indices[np.argmin(np.abs(common_indices - gain_count // 2))])
    intercept = wrap_phase(intercept + rx1[reference] + rx2[reference])
    rx1 = wrap_phase(rx1 - rx1[reference])
    rx2 = wrap_phase(rx2 - rx2[reference])
    prediction = wrap_phase(
        intercept[group_index] + rx1[gain1_index] + rx2[gain2_index]
    )
    residual = wrap_phase(phase - prediction)
    return {
        "intercept_rad_by_group": intercept,
        "reference_gain_index": reference,
        "rx1_effect_rad": rx1,
        "rx2_effect_rad": rx2,
        "prediction_rad": prediction,
        "residual_rad": residual,
    }


def _residual_metrics(residual: np.ndarray) -> dict[str, float]:
    degrees = np.abs(np.degrees(np.asarray(residual)))
    return {
        "circular_mae_deg": float(np.mean(degrees)),
        "circular_rmse_deg": float(np.sqrt(np.mean(degrees**2))),
        "circular_p95_deg": float(np.percentile(degrees, 95)),
        "circular_max_deg": float(np.max(degrees)),
    }


def fit_frequency_delay(
    frequency_hz: np.ndarray,
    phase_rad: np.ndarray,
) -> dict[str, Any]:
    """Fit phase(f) = phase(reference) - 2*pi*f_offset*delay.

    This estimates an *effective differential delay*. It is a useful
    description of a linear phase slope, but it does not identify which
    combination of cables, PCB paths, analogue filters, and retune state
    produced that slope.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    phase_rad = np.asarray(phase_rad, dtype=np.float64)
    if (
        frequency_hz.ndim != 1
        or phase_rad.ndim != 1
        or frequency_hz.size != phase_rad.size
        or frequency_hz.size < 2
        or not np.all(np.isfinite(frequency_hz))
        or not np.all(np.isfinite(phase_rad))
        or np.unique(frequency_hz).size != frequency_hz.size
    ):
        raise ValueError(
            "delay fit requires equal-length finite vectors at two or more "
            "distinct frequencies"
        )

    order = np.argsort(frequency_hz)
    frequency_sorted = frequency_hz[order]
    phase_sorted = phase_rad[order]
    unwrapped = np.unwrap(phase_sorted)
    reference_frequency_hz = float(np.mean(frequency_sorted))
    offset_hz = frequency_sorted - reference_frequency_hz
    slope, phase_at_reference = np.polyfit(offset_hz, unwrapped, 1)
    fitted_unwrapped = phase_at_reference + slope * offset_hz
    residual = wrap_phase(unwrapped - fitted_unwrapped)
    delay_seconds = -float(slope) / (2 * np.pi)
    points = [
        {
            "frequency_hz": int(frequency),
            "phase_rad_unwrapped": float(observed),
            "fitted_phase_rad_unwrapped": float(fitted),
            "residual_rad": float(error),
        }
        for frequency, observed, fitted, error in zip(
            frequency_sorted,
            unwrapped,
            fitted_unwrapped,
            residual,
        )
    ]
    return {
        "reference_frequency_hz": reference_frequency_hz,
        "phase_at_reference_rad_unwrapped": float(phase_at_reference),
        "slope_rad_per_hz": float(slope),
        "descriptive_delay_seconds": delay_seconds,
        "equivalent_free_space_path_m": delay_seconds * 299_792_458.0,
        "fit_residual_metrics": _residual_metrics(residual),
        "frequency_points": points,
        "phase_unwrap_assumption": (
            "Adjacent measured frequencies differ by less than 180 degrees "
            "of true unwrapped phase."
        ),
    }


def _comparison(
    first_name: str,
    first_residuals: list[float],
    second_name: str,
    second_residuals: list[float],
    *,
    preferred_if_equivalent: str | None = None,
    mae_equivalence_margin_deg: float = 0.1,
) -> dict[str, Any] | None:
    """Summarize two models evaluated on exactly the same observations."""

    if not first_residuals:
        return None
    first = np.asarray(first_residuals, dtype=np.float64)
    second = np.asarray(second_residuals, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("paired model residuals have different shapes")
    first_metrics = _residual_metrics(first)
    second_metrics = _residual_metrics(second)
    first_mae = first_metrics["circular_mae_deg"]
    second_mae = second_metrics["circular_mae_deg"]
    if math.isclose(first_mae, second_mae, abs_tol=1e-12):
        winner = "tie"
    else:
        winner = first_name if first_mae < second_mae else second_name
    delta_mae = float(second_mae - first_mae)
    preferred_if_equivalent = preferred_if_equivalent or first_name
    if abs(delta_mae) <= mae_equivalence_margin_deg:
        recommended = preferred_if_equivalent
        recommendation_reason = (
            "MAE difference is inside the predeclared practical-equivalence margin"
        )
    else:
        recommended = winner
        recommendation_reason = "lower held-out MAE outside equivalence margin"
    return {
        "evaluation": "paired leave-one-epoch-out",
        "n_observations": int(first.size),
        first_name: first_metrics,
        second_name: second_metrics,
        "delta_mae_deg_second_minus_first": delta_mae,
        "winner_by_mae": winner,
        "mae_equivalence_margin_deg": mae_equivalence_margin_deg,
        "recommended_model": recommended,
        "recommendation_reason": recommendation_reason,
    }


def _fit_gain_difference(
    phase: np.ndarray, gain_difference_db: np.ndarray
) -> dict[int, float]:
    """Fit a deliberately simple gain-difference-only baseline."""

    return {
        int(difference): _circular_mean(phase[gain_difference_db == difference])
        for difference in np.unique(gain_difference_db)
    }


def _evaluate_independent_frequency_cv(
    *,
    eligible: np.ndarray,
    phases: np.ndarray,
    epochs: np.ndarray,
    frequency_indices: np.ndarray,
    requested: np.ndarray,
    gain1_indices: np.ndarray,
    gain2_indices: np.ndarray,
    config: CalibrationConfig,
    include_diagnostics: bool,
) -> dict[str, Any]:
    """Evaluate independent per-frequency models on held-out epochs."""

    gain_count = len(config.gains_db)
    additive_all: list[float] = []
    additive_interaction: list[float] = []
    interaction: list[float] = []
    additive_difference: list[float] = []
    difference_only: list[float] = []
    additive_anchor: list[float] = []
    anchored: list[float] = []
    anchor_gain_db = min(
        config.gains_db,
        key=lambda gain: abs(gain - config.tx_reference_rx_gain_db),
    )

    for frequency_index, _ in enumerate(config.frequencies_hz):
        selected = eligible & (frequency_indices == frequency_index)
        for held_epoch in range(config.repetitions):
            train = selected & (epochs != held_epoch)
            test = selected & (epochs == held_epoch)
            if not np.any(train) or not np.any(test):
                continue
            train_rx1 = np.bincount(gain1_indices[train], minlength=gain_count)
            train_rx2 = np.bincount(gain2_indices[train], minlength=gain_count)
            supported = (
                test & (train_rx1[gain1_indices] > 0) & (train_rx2[gain2_indices] > 0)
            )
            if not np.any(supported):
                continue
            fitted = fit_additive_surface(
                phases[train],
                gain1_indices[train],
                gain2_indices[train],
                gain_count,
            )
            additive_prediction = wrap_phase(
                fitted["intercept_rad"]
                + fitted["rx1_effect_rad"][gain1_indices]
                + fitted["rx2_effect_rad"][gain2_indices]
            )
            additive_residual = wrap_phase(phases - additive_prediction)
            additive_all.extend(additive_residual[supported])
            if not include_diagnostics:
                continue

            # A pair-specific interaction is estimated only from training
            # residuals. The additive comparator uses the exact same test mask.
            train_indices = np.flatnonzero(train)
            cell_residuals: dict[tuple[int, int], list[float]] = {}
            for index in train_indices:
                pair = (int(gain1_indices[index]), int(gain2_indices[index]))
                cell_residuals.setdefault(pair, []).append(
                    float(additive_residual[index])
                )
            interaction_by_pair = {
                pair: _circular_mean(np.asarray(values))
                for pair, values in cell_residuals.items()
            }
            interaction_indices = np.asarray(
                [
                    index
                    for index in np.flatnonzero(supported)
                    if (
                        int(gain1_indices[index]),
                        int(gain2_indices[index]),
                    )
                    in interaction_by_pair
                ],
                dtype=np.int64,
            )
            if interaction_indices.size:
                interaction_effect = np.asarray(
                    [
                        interaction_by_pair[
                            (
                                int(gain1_indices[index]),
                                int(gain2_indices[index]),
                            )
                        ]
                        for index in interaction_indices
                    ]
                )
                additive_interaction.extend(additive_residual[interaction_indices])
                interaction.extend(
                    wrap_phase(
                        additive_residual[interaction_indices] - interaction_effect
                    )
                )

            gain_difference = requested[:, 0] - requested[:, 1]
            difference_model = _fit_gain_difference(
                phases[train], gain_difference[train]
            )
            difference_indices = np.asarray(
                [
                    index
                    for index in np.flatnonzero(supported)
                    if int(gain_difference[index]) in difference_model
                ],
                dtype=np.int64,
            )
            if difference_indices.size:
                difference_prediction = np.asarray(
                    [
                        difference_model[int(gain_difference[index])]
                        for index in difference_indices
                    ]
                )
                additive_difference.extend(additive_residual[difference_indices])
                difference_only.extend(
                    wrap_phase(phases[difference_indices] - difference_prediction)
                )

            # A practical anchored strategy spends one held-out reference frame
            # per frequency/epoch to estimate a session intercept. The anchor
            # itself is excluded from scoring to avoid leakage.
            anchor = (
                test
                & (requested[:, 0] == anchor_gain_db)
                & (requested[:, 1] == anchor_gain_db)
            )
            anchor_indices = np.flatnonzero(anchor)
            if anchor_indices.size == 1:
                anchor_index = int(anchor_indices[0])
                anchor_delta = float(additive_residual[anchor_index])
                anchored_test = supported.copy()
                anchored_test[anchor_index] = False
                if np.any(anchored_test):
                    additive_anchor.extend(additive_residual[anchored_test])
                    anchored.extend(
                        wrap_phase(additive_residual[anchored_test] - anchor_delta)
                    )

    result: dict[str, Any] = {
        "eligible_observations": int(np.count_nonzero(eligible)),
        "additive": (
            {
                "n_observations": len(additive_all),
                **_residual_metrics(np.asarray(additive_all)),
            }
            if additive_all
            else None
        ),
    }
    if include_diagnostics:
        result.update(
            {
                "additive_vs_cell_interaction": _comparison(
                    "additive",
                    additive_interaction,
                    "additive_plus_cell_interaction",
                    interaction,
                ),
                "additive_vs_gain_difference_only": _comparison(
                    "additive",
                    additive_difference,
                    "gain_difference_only",
                    difference_only,
                ),
                "unanchored_vs_one_frame_anchor": _comparison(
                    "unanchored",
                    additive_anchor,
                    "one_frame_anchored",
                    anchored,
                ),
                "anchor_gain_db": anchor_gain_db,
                "anchor_policy": (
                    "One held-out equal-gain frame per frequency/epoch; "
                    "anchor frame excluded from scoring."
                ),
            }
        )
    return result


def _evaluate_shared_gain_curve_cv(
    *,
    eligible: np.ndarray,
    phases: np.ndarray,
    epochs: np.ndarray,
    frequency_indices: np.ndarray,
    gain1_indices: np.ndarray,
    gain2_indices: np.ndarray,
    config: CalibrationConfig,
) -> dict[str, Any] | None:
    """Compare frequency-specific and frequency-shared gain curves fairly."""

    gain_count = len(config.gains_db)
    independent_residuals: list[float] = []
    shared_residuals: list[float] = []
    for held_epoch in range(config.repetitions):
        train_all = eligible & (epochs != held_epoch)
        test_all = eligible & (epochs == held_epoch)
        if not np.any(train_all) or not np.any(test_all):
            continue
        shared = fit_grouped_additive_surface(
            phases[train_all],
            gain1_indices[train_all],
            gain2_indices[train_all],
            frequency_indices[train_all],
            gain_count,
            len(config.frequencies_hz),
        )
        shared_rx1_count = np.bincount(gain1_indices[train_all], minlength=gain_count)
        shared_rx2_count = np.bincount(gain2_indices[train_all], minlength=gain_count)
        for frequency_index, _ in enumerate(config.frequencies_hz):
            train = train_all & (frequency_indices == frequency_index)
            test = test_all & (frequency_indices == frequency_index)
            if not np.any(train) or not np.any(test):
                continue
            independent = fit_additive_surface(
                phases[train],
                gain1_indices[train],
                gain2_indices[train],
                gain_count,
            )
            independent_rx1_count = np.bincount(
                gain1_indices[train], minlength=gain_count
            )
            independent_rx2_count = np.bincount(
                gain2_indices[train], minlength=gain_count
            )
            paired = (
                test
                & (shared_rx1_count[gain1_indices] > 0)
                & (shared_rx2_count[gain2_indices] > 0)
                & (independent_rx1_count[gain1_indices] > 0)
                & (independent_rx2_count[gain2_indices] > 0)
            )
            if not np.any(paired):
                continue
            independent_prediction = wrap_phase(
                independent["intercept_rad"]
                + independent["rx1_effect_rad"][gain1_indices[paired]]
                + independent["rx2_effect_rad"][gain2_indices[paired]]
            )
            shared_prediction = wrap_phase(
                shared["intercept_rad_by_group"][frequency_index]
                + shared["rx1_effect_rad"][gain1_indices[paired]]
                + shared["rx2_effect_rad"][gain2_indices[paired]]
            )
            independent_residuals.extend(
                wrap_phase(phases[paired] - independent_prediction)
            )
            shared_residuals.extend(wrap_phase(phases[paired] - shared_prediction))
    return _comparison(
        "frequency_specific_gain_curves",
        independent_residuals,
        "frequency_shared_gain_curves",
        shared_residuals,
        preferred_if_equivalent="frequency_shared_gain_curves",
    )


def _evaluate_linear_delay_intercept_cv(
    *,
    eligible: np.ndarray,
    phases: np.ndarray,
    epochs: np.ndarray,
    frequency_indices: np.ndarray,
    gain1_indices: np.ndarray,
    gain2_indices: np.ndarray,
    config: CalibrationConfig,
) -> dict[str, Any] | None:
    """Compare independent intercepts with a constant-plus-delay intercept.

    Gain effects remain independently fitted at every frequency. Only the
    per-frequency baseline at one common reference gain is constrained to a
    straight line in frequency, so both alternatives are evaluated on the
    exact same held-out observations.
    """

    gain_count = len(config.gains_db)
    independent_residuals: list[float] = []
    delay_residuals: list[float] = []
    for held_epoch in range(config.repetitions):
        fits: dict[int, dict[str, Any]] = {}
        counts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for frequency_index, _ in enumerate(config.frequencies_hz):
            train = (
                eligible
                & (epochs != held_epoch)
                & (frequency_indices == frequency_index)
            )
            test = (
                eligible
                & (epochs == held_epoch)
                & (frequency_indices == frequency_index)
            )
            if not np.any(train) or not np.any(test):
                continue
            fits[frequency_index] = fit_additive_surface(
                phases[train],
                gain1_indices[train],
                gain2_indices[train],
                gain_count,
            )
            counts[frequency_index] = (
                np.bincount(gain1_indices[train], minlength=gain_count),
                np.bincount(gain2_indices[train], minlength=gain_count),
            )
        if len(fits) < 2:
            continue

        common_reference = [
            index
            for index in range(gain_count)
            if all(
                counts[frequency_index][0][index] > 0
                and counts[frequency_index][1][index] > 0
                for frequency_index in fits
            )
        ]
        if not common_reference:
            continue
        preferred_gain = config.tx_reference_rx_gain_db
        reference_index = min(
            common_reference,
            key=lambda index: abs(config.gains_db[index] - preferred_gain),
        )
        fit_frequency_indices = sorted(fits)
        baseline = np.asarray(
            [
                wrap_phase(
                    fits[index]["intercept_rad"]
                    + fits[index]["rx1_effect_rad"][reference_index]
                    + fits[index]["rx2_effect_rad"][reference_index]
                )
                for index in fit_frequency_indices
            ]
        )
        delay_fit = fit_frequency_delay(
            np.asarray(
                [config.frequencies_hz[index] for index in fit_frequency_indices]
            ),
            baseline,
        )

        for frequency_index in fit_frequency_indices:
            test = (
                eligible
                & (epochs == held_epoch)
                & (frequency_indices == frequency_index)
            )
            rx1_count, rx2_count = counts[frequency_index]
            paired = (
                test & (rx1_count[gain1_indices] > 0) & (rx2_count[gain2_indices] > 0)
            )
            if not np.any(paired):
                continue
            fitted = fits[frequency_index]
            independent_prediction = wrap_phase(
                fitted["intercept_rad"]
                + fitted["rx1_effect_rad"][gain1_indices[paired]]
                + fitted["rx2_effect_rad"][gain2_indices[paired]]
            )
            frequency_offset = (
                config.frequencies_hz[frequency_index]
                - delay_fit["reference_frequency_hz"]
            )
            delay_baseline = (
                delay_fit["phase_at_reference_rad_unwrapped"]
                + delay_fit["slope_rad_per_hz"] * frequency_offset
            )
            delay_prediction = wrap_phase(
                delay_baseline
                + fitted["rx1_effect_rad"][gain1_indices[paired]]
                - fitted["rx1_effect_rad"][reference_index]
                + fitted["rx2_effect_rad"][gain2_indices[paired]]
                - fitted["rx2_effect_rad"][reference_index]
            )
            independent_residuals.extend(
                wrap_phase(phases[paired] - independent_prediction)
            )
            delay_residuals.extend(wrap_phase(phases[paired] - delay_prediction))

    return _comparison(
        "frequency_specific_intercepts",
        independent_residuals,
        "linear_delay_intercept",
        delay_residuals,
        preferred_if_equivalent="linear_delay_intercept",
    )


def fit_dataset(path: Path, *, config: CalibrationConfig) -> dict[str, Any]:
    """Fit one additive gain model per frequency and leave-one-epoch-out CV."""

    zarr = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        receiver = zarr["receivers/r0"]
        completed = np.asarray(receiver.sweep_completed[:], dtype=bool)
        valid = completed & np.asarray(receiver.sweep_quality_valid[:], dtype=bool)
        phases = np.asarray(receiver.phase_difference_rad[:], dtype=np.float64)
        epochs = np.asarray(receiver.sweep_epoch[:], dtype=np.int64)
        frequency_indices = np.asarray(
            receiver.sweep_frequency_index[:], dtype=np.int64
        )
        requested = np.asarray(receiver.sweep_requested_gain_db[:], dtype=np.int64)
        gain_lookup = {gain: index for index, gain in enumerate(config.gains_db)}
        gain1_indices = np.asarray(
            [gain_lookup.get(int(value), -1) for value in requested[:, 0]]
        )
        gain2_indices = np.asarray(
            [gain_lookup.get(int(value), -1) for value in requested[:, 1]]
        )
        if np.any(valid & ((gain1_indices < 0) | (gain2_indices < 0))):
            raise ValueError("dataset contains gains outside the configured set")

        frequency_models = []
        cv_residuals = []
        for frequency_index, frequency_hz in enumerate(config.frequencies_hz):
            selected = valid & (frequency_indices == frequency_index)
            if not np.any(selected):
                frequency_models.append(
                    {
                        "frequency_hz": frequency_hz,
                        "status": "no_quality_valid_data",
                    }
                )
                continue
            fitted = fit_additive_surface(
                phases[selected],
                gain1_indices[selected],
                gain2_indices[selected],
                len(config.gains_db),
            )
            rx1_counts = np.bincount(
                gain1_indices[selected], minlength=len(config.gains_db)
            )
            rx2_counts = np.bincount(
                gain2_indices[selected], minlength=len(config.gains_db)
            )
            rx1_effect = [
                float(value) if rx1_counts[index] else None
                for index, value in enumerate(fitted["rx1_effect_rad"])
            ]
            rx2_effect = [
                float(value) if rx2_counts[index] else None
                for index, value in enumerate(fitted["rx2_effect_rad"])
            ]
            cell_residuals: dict[tuple[int, int], list[float]] = {}
            cell_phases: dict[tuple[int, int], list[float]] = {}
            selected_indices = np.flatnonzero(selected)
            for source_index, residual in zip(selected_indices, fitted["residual_rad"]):
                pair = (
                    int(requested[source_index, 0]),
                    int(requested[source_index, 1]),
                )
                cell_residuals.setdefault(pair, []).append(float(residual))
                cell_phases.setdefault(pair, []).append(float(phases[source_index]))
            interaction = np.full((len(config.gains_db), len(config.gains_db)), np.nan)
            cell_counts = np.zeros(
                (len(config.gains_db), len(config.gains_db)), dtype=np.uint8
            )
            supported_pairs = np.zeros(
                (len(config.gains_db), len(config.gains_db)), dtype=bool
            )
            for (gain1, gain2), values in cell_residuals.items():
                gain1_index = gain_lookup[gain1]
                gain2_index = gain_lookup[gain2]
                interaction[gain1_index, gain2_index] = circular_stats(values)[
                    "mean_rad"
                ]
                phases_for_pair = cell_phases[(gain1, gain2)]
                phase_stats = circular_stats(phases_for_pair)
                circular_std_rad = phase_stats["circular_std_rad"]
                cell_counts[gain1_index, gain2_index] = len(phases_for_pair)
                supported_pairs[gain1_index, gain2_index] = bool(
                    len(phases_for_pair) >= config.min_quality_valid_per_cell
                    and circular_std_rad is not None
                    and math.degrees(circular_std_rad)
                    <= config.max_across_repeat_phase_std_deg
                )
            frequency_models.append(
                {
                    "frequency_hz": frequency_hz,
                    "status": "fit",
                    "n_observations": int(np.count_nonzero(selected)),
                    "reference_gain_db": config.gains_db[
                        fitted["reference_gain_index"]
                    ],
                    "rx1_observation_count_by_gain": rx1_counts.tolist(),
                    "rx2_observation_count_by_gain": rx2_counts.tolist(),
                    "intercept_rad": fitted["intercept_rad"],
                    "rx1_effect_rad": rx1_effect,
                    "rx2_effect_rad": rx2_effect,
                    "quality_valid_observation_count_by_gain_pair": (
                        cell_counts.tolist()
                    ),
                    "supported_gain_pair": supported_pairs.tolist(),
                    "interaction_residual_rad": interaction.tolist(),
                    "training_metrics": _residual_metrics(fitted["residual_rad"]),
                }
            )

            frequency_cv_residuals = []
            for held_epoch in range(config.repetitions):
                train = selected & (epochs != held_epoch)
                test = selected & (epochs == held_epoch)
                if not np.any(train) or not np.any(test):
                    continue
                train_rx1 = np.bincount(
                    gain1_indices[train], minlength=len(config.gains_db)
                )
                train_rx2 = np.bincount(
                    gain2_indices[train], minlength=len(config.gains_db)
                )
                test &= (train_rx1[gain1_indices] > 0) & (train_rx2[gain2_indices] > 0)
                if not np.any(test):
                    continue
                fold = fit_additive_surface(
                    phases[train],
                    gain1_indices[train],
                    gain2_indices[train],
                    len(config.gains_db),
                )
                prediction = wrap_phase(
                    fold["intercept_rad"]
                    + fold["rx1_effect_rad"][gain1_indices[test]]
                    + fold["rx2_effect_rad"][gain2_indices[test]]
                )
                fold_residuals = wrap_phase(phases[test] - prediction)
                cv_residuals.extend(fold_residuals)
                frequency_cv_residuals.extend(fold_residuals)
            frequency_models[-1]["cross_validation_metrics"] = (
                _residual_metrics(np.asarray(frequency_cv_residuals))
                if frequency_cv_residuals
                else None
            )

        fitted_frequency_models = [
            model for model in frequency_models if model.get("status") == "fit"
        ]
        delay_model = None
        if len(fitted_frequency_models) >= 2:
            common_reference = [
                index
                for index in range(len(config.gains_db))
                if all(
                    model["rx1_effect_rad"][index] is not None
                    and model["rx2_effect_rad"][index] is not None
                    for model in fitted_frequency_models
                )
            ]
            if common_reference:
                reference_index = min(
                    common_reference,
                    key=lambda index: abs(
                        config.gains_db[index] - config.tx_reference_rx_gain_db
                    ),
                )
                baseline = np.asarray(
                    [
                        wrap_phase(
                            model["intercept_rad"]
                            + model["rx1_effect_rad"][reference_index]
                            + model["rx2_effect_rad"][reference_index]
                        )
                        for model in fitted_frequency_models
                    ]
                )
                delay_model = {
                    **fit_frequency_delay(
                        np.asarray(
                            [model["frequency_hz"] for model in fitted_frequency_models]
                        ),
                        baseline,
                    ),
                    "reference_gain_db": config.gains_db[reference_index],
                    "interpretation": (
                        "Effective RX1-minus-RX2 electrical group delay at the "
                        "common reference gain."
                    ),
                    "warning": (
                        "Descriptive only: cables, PCB and analogue paths, LO "
                        "retunes, and calibration state can all contribute. It "
                        "is not asserted to be literal cable length."
                    ),
                }
        model_comparisons = _evaluate_independent_frequency_cv(
            eligible=valid,
            phases=phases,
            epochs=epochs,
            frequency_indices=frequency_indices,
            requested=requested,
            gain1_indices=gain1_indices,
            gain2_indices=gain2_indices,
            config=config,
            include_diagnostics=True,
        )
        model_comparisons[
            "frequency_specific_vs_shared_gain_curves"
        ] = _evaluate_shared_gain_curve_cv(
            eligible=valid,
            phases=phases,
            epochs=epochs,
            frequency_indices=frequency_indices,
            gain1_indices=gain1_indices,
            gain2_indices=gain2_indices,
            config=config,
        )
        model_comparisons[
            "frequency_specific_vs_linear_delay_intercept"
        ] = _evaluate_linear_delay_intercept_cv(
            eligible=valid,
            phases=phases,
            epochs=epochs,
            frequency_indices=frequency_indices,
            gain1_indices=gain1_indices,
            gain2_indices=gain2_indices,
            config=config,
        )
        tone_snr_db = np.asarray(receiver.tone_snr_db[:], dtype=np.float64)
        model_comparisons["confidence_tiers"] = [
            {
                "minimum_both_channel_tone_snr_db": threshold,
                **_evaluate_independent_frequency_cv(
                    eligible=valid & np.all(tone_snr_db >= threshold, axis=1),
                    phases=phases,
                    epochs=epochs,
                    frequency_indices=frequency_indices,
                    requested=requested,
                    gain1_indices=gain1_indices,
                    gain2_indices=gain2_indices,
                    config=config,
                    include_diagnostics=False,
                ),
            }
            for threshold in (-10.0, 0.0, 10.0)
        ]
        return {
            "schema": "spf.calibration.dual_rx_gain_frequency.model",
            "schema_version": 2,
            "serial": receiver.attrs.get("sdr_serial"),
            "phase_convention": zarr.attrs.get("phase_convention"),
            "gain_values_db": list(config.gains_db),
            "prediction_support_policy": {
                "minimum_quality_valid_epochs": config.min_quality_valid_per_cell,
                "maximum_repeat_circular_std_deg": (
                    config.max_across_repeat_phase_std_deg
                ),
                "interpolation_or_extrapolation": False,
            },
            "quality_valid_observations": int(np.count_nonzero(valid)),
            "frequency_models": frequency_models,
            "cross_validation_metrics": (
                _residual_metrics(np.asarray(cv_residuals)) if cv_residuals else None
            ),
            "model_comparisons": model_comparisons,
            "frequency_intercept_delay_model": delay_model,
        }
    finally:
        zarr.store.close()


def predict_phase_offset(
    model: dict[str, Any],
    *,
    frequency_hz: int,
    gain_rx1_db: int,
    gain_rx2_db: int,
) -> float:
    """Return an exact-grid additive correction or fail closed.

    Frequency and both manual gains must be present in the fitted artifact.
    This intentionally performs no interpolation or extrapolation.
    """

    gains = [int(value) for value in model.get("gain_values_db", [])]
    gain_lookup = {gain: index for index, gain in enumerate(gains)}
    if gain_rx1_db not in gain_lookup:
        raise ValueError(f"unsupported RX1 gain: {gain_rx1_db}")
    if gain_rx2_db not in gain_lookup:
        raise ValueError(f"unsupported RX2 gain: {gain_rx2_db}")
    matching = [
        row
        for row in model.get("frequency_models", [])
        if int(row.get("frequency_hz", -1)) == int(frequency_hz)
    ]
    if len(matching) != 1 or matching[0].get("status") != "fit":
        raise ValueError(f"unsupported RF frequency: {frequency_hz}")
    frequency_model = matching[0]
    supported_pairs = frequency_model.get("supported_gain_pair")
    if supported_pairs is None or not bool(
        supported_pairs[gain_lookup[gain_rx1_db]][gain_lookup[gain_rx2_db]]
    ):
        raise ValueError(
            "phase correction unavailable for an unvalidated ordered gain pair"
        )
    rx1_effect = frequency_model["rx1_effect_rad"][gain_lookup[gain_rx1_db]]
    rx2_effect = frequency_model["rx2_effect_rad"][gain_lookup[gain_rx2_db]]
    if rx1_effect is None or rx2_effect is None:
        raise ValueError("phase correction unavailable for an unobserved receiver gain")
    return float(
        wrap_phase(
            float(frequency_model["intercept_rad"])
            + float(rx1_effect)
            + float(rx2_effect)
        )
    )


def write_model(path: Path, model: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
