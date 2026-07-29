"""Compare radio-specific, universal, lookup, and physical delay phase models.

The primary evaluation is leave-one-epoch-out: every requested
frequency/gain cell remains represented in training by the other repeats.
Models that can predict an unseen frequency are also evaluated with
leave-one-frequency-out. Universal models are additionally evaluated by
training on one Pluto and predicting the other.

Both Cartesian and additive-cross schedules are supported. For an
additive-cross schedule, leave-one-epoch-out evaluates repeatability on the
requested cells; the separate additive-cross analysis remains the decisive
test on off-axis cells excluded from fitting.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

from spf.bench.dual_rx_phase import wrap_phase
from spf.calibrations.dual_rx_gain_frequency.comparative_analysis import _load_radio
from spf.calibrations.dual_rx_gain_frequency.runner import load_calibration_document

SCHEMA = "spf.calibration.dual_rx_gain_frequency.model_matrix"
SCHEMA_VERSION = 1
LIGHT_SPEED_M_PER_S = 299_792_458.0
INDEPENDENT_GAIN_MODEL = "frequency_specific_additive_gain_per_radio"
SYMMETRIC_GAIN_MODEL = "frequency_specific_antisymmetric_gain_per_radio"
FREQUENCY_SCALING_MODELS = (
    "frequency_lut_symmetric_gain_per_radio",
    "frequency_lut_frequency_scaled_symmetric_gain_per_radio",
    "frequency_lut_gain_table_symmetric_gain_per_radio",
    "frequency_lut_gain_table_frequency_scaled_symmetric_gain_per_radio",
    SYMMETRIC_GAIN_MODEL,
    INDEPENDENT_GAIN_MODEL,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    scope: str
    kind: str
    formula: str
    unseen_frequency: bool = False


MODEL_SPECS = (
    ModelSpec(
        "constant_per_radio",
        "Constant per radio",
        "per_radio",
        "constant",
        "phase = radio_intercept",
        True,
    ),
    ModelSpec(
        "gain_linear_per_radio",
        "Linear gains per radio",
        "per_radio",
        "gain_linear",
        "phase = intercept + beta1*g1 + beta2*g2",
        True,
    ),
    ModelSpec(
        "gain_additive_lut_per_radio",
        "Additive gain LUT per radio",
        "per_radio",
        "gain_additive",
        "phase = intercept + RX1[g1] + RX2[g2]",
        True,
    ),
    ModelSpec(
        "frequency_lut_linear_gain_per_radio",
        "Frequency LUT + linear gains per radio",
        "per_radio",
        "frequency_lut_gain_linear",
        "phase = frequency_intercept[f] + beta1*g1 + beta2*g2",
    ),
    ModelSpec(
        "frequency_lut_additive_gain_per_radio",
        "Frequency LUT + additive gain LUT per radio",
        "per_radio",
        "frequency_lut_gain_additive",
        "phase = frequency_intercept[f] + RX1[g1] + RX2[g2]",
    ),
    ModelSpec(
        "frequency_lut_symmetric_gain_per_radio",
        "Frequency LUT + symmetric gain LUT per radio",
        "per_radio",
        "frequency_lut_gain_symmetric",
        "phase = intercept[f] + H[g1] - H[g2]",
    ),
    ModelSpec(
        "frequency_lut_frequency_scaled_symmetric_gain_per_radio",
        "Frequency LUT + f-scaled symmetric gain LUT per radio",
        "per_radio",
        "frequency_lut_frequency_scaled_gain_symmetric",
        "phase = intercept[f] + (f/GHz) * (G[g1] - G[g2])",
    ),
    ModelSpec(
        "frequency_lut_gain_table_symmetric_gain_per_radio",
        "Frequency LUT + gain-table-specific symmetric gain LUT per radio",
        "per_radio",
        "frequency_lut_gain_table_gain_symmetric",
        "phase = intercept[f] + H[table(f),g1] - H[table(f),g2]",
    ),
    ModelSpec(
        "frequency_lut_gain_table_frequency_scaled_symmetric_gain_per_radio",
        "Frequency LUT + gain-table-specific f-scaled symmetric gain LUT per radio",
        "per_radio",
        "frequency_lut_gain_table_frequency_scaled_gain_symmetric",
        "phase = intercept[f] + (f/GHz) * (G[table(f),g1] - G[table(f),g2])",
    ),
    ModelSpec(
        "frequency_specific_additive_gain_per_radio",
        "Per-frequency additive gain LUT per radio",
        "per_radio",
        "frequency_specific_gain_additive",
        "phase = intercept[f] + RX1[f,g1] + RX2[f,g2]",
    ),
    ModelSpec(
        "frequency_specific_antisymmetric_gain_per_radio",
        "Per-frequency antisymmetric gain LUT per radio",
        "per_radio",
        "frequency_specific_gain_antisymmetric",
        "phase = intercept[f] + H[f,g1] - H[f,g2]",
    ),
    ModelSpec(
        "full_cell_lut_per_radio",
        "Full frequency/gain-pair LUT per radio",
        "per_radio",
        "full_cell",
        "phase = cell[f,g1,g2]",
    ),
    ModelSpec(
        "delay_additive_gain_per_radio",
        "One delay + additive gain LUT per radio",
        "per_radio",
        "delay_gain_additive",
        "phase = intercept - 2*pi*(f-f_ref)*delay + RX1[g1] + RX2[g2]",
        True,
    ),
    ModelSpec(
        "branch_gain_delay_lut_per_radio",
        "Gain-dependent branch-delay LUT per radio",
        "per_radio",
        "branch_gain_delay",
        (
            "phase = intercept + RX1_phase[g1] + RX2_phase[g2] "
            "- 2*pi*(f-f_ref)*(delay0 + RX1_delay[g1] + RX2_delay[g2])"
        ),
        True,
    ),
    ModelSpec(
        "frequency_lut_additive_gain_universal",
        "Strict universal frequency + gain LUT",
        "universal",
        "frequency_lut_gain_additive",
        "phase = frequency_intercept[f] + RX1[g1] + RX2[g2]",
    ),
    ModelSpec(
        "frequency_specific_additive_gain_universal",
        "Strict universal per-frequency additive LUT",
        "universal",
        "frequency_specific_gain_additive",
        "phase = intercept[f] + RX1[f,g1] + RX2[f,g2]",
    ),
    ModelSpec(
        "full_cell_lut_universal",
        "Strict universal full cell LUT",
        "universal",
        "full_cell",
        "phase = cell[f,g1,g2]",
    ),
    ModelSpec(
        "branch_gain_delay_lut_universal",
        "Strict universal gain-dependent delay LUT",
        "universal",
        "branch_gain_delay",
        (
            "phase = intercept + RX1_phase[g1] + RX2_phase[g2] "
            "- 2*pi*(f-f_ref)*(delay0 + RX1_delay[g1] + RX2_delay[g2])"
        ),
        True,
    ),
)


@dataclass
class Fit:
    spec: ModelSpec
    beta: np.ndarray
    feature_names: list[str]
    reference_frequency_hz: float
    reference_gain_db: int
    seen_frequencies: set[int]
    seen_gain1: set[int]
    seen_gain2: set[int]
    seen_cells: set[tuple[int, int, int]]


def _metrics(residual_rad: np.ndarray, *, expected: int) -> dict[str, Any]:
    residual = np.asarray(residual_rad, dtype=np.float64)
    absolute_deg = np.abs(np.degrees(residual))
    if not residual.size:
        return {
            "evaluated_observations": 0,
            "expected_observations": int(expected),
            "coverage_fraction": 0.0,
            "circular_mae_deg": None,
            "circular_rmse_deg": None,
            "circular_p95_deg": None,
            "circular_max_deg": None,
            "circular_bias_deg": None,
        }
    return {
        "evaluated_observations": int(residual.size),
        "expected_observations": int(expected),
        "coverage_fraction": float(residual.size / expected) if expected else 0.0,
        "circular_mae_deg": float(np.mean(absolute_deg)),
        "circular_rmse_deg": float(np.sqrt(np.mean(absolute_deg**2))),
        "circular_p95_deg": float(np.percentile(absolute_deg, 95)),
        "circular_max_deg": float(np.max(absolute_deg)),
        "circular_bias_deg": float(
            np.degrees(np.angle(np.mean(np.exp(1j * residual))))
        ),
    }


def _radio_arrays(
    *,
    config_path: Path,
    artifact_root: Path,
) -> tuple[Any, dict[str, np.ndarray], list[dict[str, Any]]]:
    paths = sorted(artifact_root.glob("*/calibration.v7.zarr"))
    if not paths:
        raise FileNotFoundError(f"no V7 calibration datasets below {artifact_root}")
    return _radio_arrays_from_paths(config_path=config_path, dataset_paths=paths)


def _radio_arrays_from_paths(
    *,
    config_path: Path,
    dataset_paths: Iterable[Path],
) -> tuple[Any, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Load one complete calibration dataset per physical radio."""

    _, config = load_calibration_document(config_path)
    paths = [Path(path) for path in dataset_paths]
    if not paths:
        raise ValueError("at least one dataset path is required")
    if len(paths) != len(set(paths)):
        raise ValueError("dataset paths must be unique")
    radios = [_load_radio(path, config=config) for path in paths]
    serials = [radio["serial"] for radio in radios]
    if len(serials) != len(set(serials)):
        raise ValueError(
            "each dataset must represent a different physical radio; "
            f"duplicate serials: {serials}"
        )
    gain_lookup = {int(value): index for index, value in enumerate(config.gains_db)}
    frequency_lookup = {
        int(value): index for index, value in enumerate(config.frequencies_hz)
    }
    pieces: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "radio",
            "epoch",
            "frequency_hz",
            "frequency",
            "gain1_db",
            "gain2_db",
            "gain1",
            "gain2",
            "phase",
        )
    }
    provenance = []
    for radio_index, (radio, dataset_path) in enumerate(zip(radios, paths)):
        selected = radio["quality_mask"] & np.isfinite(radio["phase"])
        gain1_db = radio["gain1"][selected].astype(np.int64)
        gain2_db = radio["gain2"][selected].astype(np.int64)
        frequency_hz = radio["frequency_hz"][selected].astype(np.int64)
        count = int(np.count_nonzero(selected))
        pieces["radio"].append(np.full(count, radio_index, dtype=np.int64))
        pieces["epoch"].append(radio["epoch"][selected].astype(np.int64))
        pieces["frequency_hz"].append(frequency_hz)
        pieces["frequency"].append(
            np.asarray(
                [frequency_lookup[int(value)] for value in frequency_hz],
                dtype=np.int64,
            )
        )
        pieces["gain1_db"].append(gain1_db)
        pieces["gain2_db"].append(gain2_db)
        pieces["gain1"].append(
            np.asarray(
                [gain_lookup[int(value)] for value in gain1_db],
                dtype=np.int64,
            )
        )
        pieces["gain2"].append(
            np.asarray(
                [gain_lookup[int(value)] for value in gain2_db],
                dtype=np.int64,
            )
        )
        pieces["phase"].append(radio["phase"][selected].astype(np.float64))
        validation_path = dataset_path.parent / "validation.json"
        validation = (
            json.loads(validation_path.read_text()) if validation_path.exists() else {}
        )
        provenance.append(
            {
                "radio_index": radio_index,
                "serial": radio["serial"],
                "dataset_path": radio["dataset_path"],
                "analysis_input_sha256": radio["analysis_input_sha256"],
                "completed_frames": radio["completed_frames"],
                "quality_valid_observations": count,
                "validation_status": validation.get("status"),
                "quality_reason_counts": validation.get("quality_reason_counts", {}),
                "attrs": radio["attrs"],
            }
        )
    return (
        config,
        {key: np.concatenate(value) for key, value in pieces.items()},
        provenance,
    )


def _one_hot(index: np.ndarray, columns: int) -> csr_matrix:
    rows = np.arange(index.size)
    return csr_matrix(
        (np.ones(index.size), (rows, index)),
        shape=(index.size, columns),
        dtype=np.float64,
    )


def _categorical(
    index: np.ndarray,
    count: int,
    reference: int,
    *,
    prefix: str,
) -> tuple[csr_matrix, list[str]]:
    keep = [value for value in range(count) if value != reference]
    mapping = np.full(count, -1, dtype=np.int64)
    mapping[keep] = np.arange(len(keep))
    selected = mapping[index]
    rows = np.flatnonzero(selected >= 0)
    matrix = csr_matrix(
        (np.ones(rows.size), (rows, selected[rows])),
        shape=(index.size, len(keep)),
    )
    return matrix, [f"{prefix}[{value}]" for value in keep]


def _design(
    kind: str,
    data: dict[str, np.ndarray],
    *,
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
    reference_frequency_hz: float,
) -> tuple[csr_matrix, list[str]]:
    count = data["phase"].size
    ones = csr_matrix(np.ones((count, 1)))
    frequency_ghz = (
        data["frequency_hz"].astype(np.float64) - reference_frequency_hz
    ) / 1e9
    gain1_linear = csr_matrix(data["gain1_db"][:, None] / 20.0)
    gain2_linear = csr_matrix(data["gain2_db"][:, None] / 20.0)
    gain1_lut, gain1_names = _categorical(
        data["gain1"], gain_count, reference_gain, prefix="rx1_phase"
    )
    gain2_lut, gain2_names = _categorical(
        data["gain2"], gain_count, reference_gain, prefix="rx2_phase"
    )
    if kind == "constant":
        return ones, ["intercept"]
    if kind == "gain_linear":
        return hstack((ones, gain1_linear, gain2_linear), format="csr"), [
            "intercept",
            "rx1_rad_per_20db",
            "rx2_rad_per_20db",
        ]
    if kind == "gain_additive":
        return hstack((ones, gain1_lut, gain2_lut), format="csr"), [
            "intercept",
            *gain1_names,
            *gain2_names,
        ]
    if kind == "frequency_lut_gain_linear":
        frequencies = _one_hot(data["frequency"], frequency_count)
        return hstack((frequencies, gain1_linear, gain2_linear), format="csr"), [
            *[f"frequency_intercept[{index}]" for index in range(frequency_count)],
            "rx1_rad_per_20db",
            "rx2_rad_per_20db",
        ]
    if kind == "frequency_lut_gain_additive":
        frequencies = _one_hot(data["frequency"], frequency_count)
        return hstack((frequencies, gain1_lut, gain2_lut), format="csr"), [
            *[f"frequency_intercept[{index}]" for index in range(frequency_count)],
            *gain1_names,
            *gain2_names,
        ]
    if kind in (
        "frequency_lut_gain_symmetric",
        "frequency_lut_frequency_scaled_gain_symmetric",
        "frequency_lut_gain_table_gain_symmetric",
        "frequency_lut_gain_table_frequency_scaled_gain_symmetric",
    ):
        # A differential delay produces phase proportional to absolute RF
        # frequency. The scalar k and a learned G(g) are not separately
        # identifiable, so G stores their product in radians/GHz.
        frequencies = _one_hot(data["frequency"], frequency_count)
        frequency_scaled = "frequency_scaled" in kind
        gain_table_specific = "gain_table" in kind
        absolute_frequency_ghz = data["frequency_hz"].astype(np.float64) / 1e9
        symmetric_gain_lut = gain1_lut - gain2_lut
        frequency_names = [
            f"frequency_intercept[{index}]" for index in range(frequency_count)
        ]
        non_reference = [
            value for value in range(gain_count) if value != reference_gain
        ]
        if not gain_table_specific:
            scale = absolute_frequency_ghz if frequency_scaled else np.ones(count)
            scaled_gain = symmetric_gain_lut.multiply(scale[:, None])
            return hstack((frequencies, scaled_gain), format="csr"), [
                *frequency_names,
                *[
                    (
                        f"symmetric_gain_rad_per_ghz[{value}]"
                        if frequency_scaled
                        else f"symmetric_gain_phase[{value}]"
                    )
                    for value in non_reference
                ],
            ]

        # The AD9361/AD9363 driver selects three distinct full gain tables:
        # low for f <= 1.3 GHz, middle for 1.3 < f <= 4.0 GHz, and high
        # above 4.0 GHz. Allow one delay-like gain curve per physical table.
        table_index = np.where(
            data["frequency_hz"] <= 1_300_000_000,
            0,
            np.where(data["frequency_hz"] <= 4_000_000_000, 1, 2),
        )
        blocks = [frequencies]
        names = list(frequency_names)
        for table in range(3):
            scale = (
                absolute_frequency_ghz if frequency_scaled else np.ones(count)
            ) * (table_index == table)
            blocks.append(symmetric_gain_lut.multiply(scale[:, None]))
            names.extend(
                (
                    f"gain_table[{table}].symmetric_gain_rad_per_ghz[{value}]"
                    if frequency_scaled
                    else f"gain_table[{table}].symmetric_gain_phase[{value}]"
                )
                for value in non_reference
            )
        return hstack(blocks, format="csr"), names
    if kind == "frequency_specific_gain_additive":
        width = 1 + 2 * (gain_count - 1)
        gain1_local = np.full(gain_count, -1, dtype=np.int64)
        gain2_local = np.full(gain_count, -1, dtype=np.int64)
        non_reference = [
            value for value in range(gain_count) if value != reference_gain
        ]
        gain1_local[non_reference] = 1 + np.arange(gain_count - 1)
        gain2_local[non_reference] = gain_count + np.arange(gain_count - 1)
        columns = data["frequency"] * width
        rows = list(np.arange(count))
        cols = list(columns)
        values = [1.0] * count
        for source, local in (
            (data["gain1"], gain1_local),
            (data["gain2"], gain2_local),
        ):
            present = local[source] >= 0
            rows.extend(np.flatnonzero(present))
            cols.extend(columns[present] + local[source[present]])
            values.extend([1.0] * int(np.count_nonzero(present)))
        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(count, frequency_count * width),
        )
        names = []
        for frequency in range(frequency_count):
            names.append(f"frequency[{frequency}].intercept")
            names.extend(
                f"frequency[{frequency}].rx1_phase[{value}]" for value in non_reference
            )
            names.extend(
                f"frequency[{frequency}].rx2_phase[{value}]" for value in non_reference
            )
        return matrix, names
    if kind == "frequency_specific_gain_antisymmetric":
        # The measured phase convention is RX1 minus RX2, so the same
        # gain-state phase response enters with opposite signs:
        #
        #   phase = intercept[f] + H[f,g1] - H[f,g2]
        #
        # H at the reference gain is fixed to zero for identifiability.
        width = gain_count
        gain_local = np.full(gain_count, -1, dtype=np.int64)
        non_reference = [
            value for value in range(gain_count) if value != reference_gain
        ]
        gain_local[non_reference] = 1 + np.arange(gain_count - 1)
        columns = data["frequency"] * width
        rows = list(np.arange(count))
        cols = list(columns)
        values = [1.0] * count
        for source, sign in ((data["gain1"], 1.0), (data["gain2"], -1.0)):
            present = gain_local[source] >= 0
            rows.extend(np.flatnonzero(present))
            cols.extend(columns[present] + gain_local[source[present]])
            values.extend([sign] * int(np.count_nonzero(present)))
        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(count, frequency_count * width),
        )
        names = []
        for frequency in range(frequency_count):
            names.append(f"frequency[{frequency}].intercept")
            names.extend(
                f"frequency[{frequency}].gain_phase[{value}]" for value in non_reference
            )
        return matrix, names
    if kind == "full_cell":
        cell = (data["frequency"] * gain_count + data["gain1"]) * gain_count + data[
            "gain2"
        ]
        return _one_hot(cell, frequency_count * gain_count * gain_count), [
            f"cell[{frequency},{gain1},{gain2}]"
            for frequency in range(frequency_count)
            for gain1 in range(gain_count)
            for gain2 in range(gain_count)
        ]
    if kind == "delay_gain_additive":
        return hstack(
            (
                ones,
                csr_matrix(frequency_ghz[:, None]),
                gain1_lut,
                gain2_lut,
            ),
            format="csr",
        ), ["intercept", "frequency_rad_per_ghz", *gain1_names, *gain2_names]
    if kind == "branch_gain_delay":
        frequency_column = csr_matrix(frequency_ghz[:, None])
        return hstack(
            (
                ones,
                frequency_column,
                gain1_lut,
                gain2_lut,
                gain1_lut.multiply(frequency_ghz[:, None]),
                gain2_lut.multiply(frequency_ghz[:, None]),
            ),
            format="csr",
        ), [
            "intercept",
            "frequency_rad_per_ghz",
            *gain1_names,
            *gain2_names,
            *[name.replace("_phase", "_delay_slope") for name in gain1_names],
            *[name.replace("_phase", "_delay_slope") for name in gain2_names],
        ]
    raise ValueError(f"unknown model kind: {kind}")


def _subset(data: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[selected] for key, value in data.items()}


def _fit(
    spec: ModelSpec,
    data: dict[str, np.ndarray],
    *,
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
    reference_frequency_hz: float,
) -> Fit:
    matrix, names = _design(
        spec.kind,
        data,
        gain_count=gain_count,
        frequency_count=frequency_count,
        reference_gain=reference_gain,
        reference_frequency_hz=reference_frequency_hz,
    )
    target = data["phase"].astype(np.float64)
    beta = lsqr(matrix, target, atol=1e-11, btol=1e-11, iter_lim=2000)[0]
    # Iteratively choose the nearest 2*pi branch and refit. This preserves the
    # sparse linear solve while making residuals circular.
    for _ in range(8):
        prediction = matrix @ beta
        adjusted = prediction + wrap_phase(target - prediction)
        updated = lsqr(matrix, adjusted, atol=1e-11, btol=1e-11, iter_lim=2000)[0]
        if np.max(np.abs(updated - beta)) < 1e-10:
            beta = updated
            break
        beta = updated
    return Fit(
        spec=spec,
        beta=np.asarray(beta),
        feature_names=names,
        reference_frequency_hz=reference_frequency_hz,
        reference_gain_db=reference_gain,
        seen_frequencies=set(int(value) for value in data["frequency_hz"]),
        seen_gain1=set(int(value) for value in data["gain1_db"]),
        seen_gain2=set(int(value) for value in data["gain2_db"]),
        seen_cells=set(
            zip(
                (int(value) for value in data["frequency_hz"]),
                (int(value) for value in data["gain1_db"]),
                (int(value) for value in data["gain2_db"]),
            )
        ),
    )


def _support(fit: Fit, data: dict[str, np.ndarray]) -> np.ndarray:
    gain_support = np.isin(data["gain1_db"], list(fit.seen_gain1)) & np.isin(
        data["gain2_db"], list(fit.seen_gain2)
    )
    if fit.spec.kind == "full_cell":
        return np.asarray(
            [
                (int(frequency), int(gain1), int(gain2)) in fit.seen_cells
                for frequency, gain1, gain2 in zip(
                    data["frequency_hz"], data["gain1_db"], data["gain2_db"]
                )
            ]
        )
    if fit.spec.unseen_frequency:
        return gain_support
    return gain_support & np.isin(data["frequency_hz"], list(fit.seen_frequencies))


def _predict(
    fit: Fit,
    data: dict[str, np.ndarray],
    *,
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
) -> tuple[np.ndarray, np.ndarray]:
    supported = _support(fit, data)
    if not np.any(supported):
        return np.asarray([], dtype=np.float64), supported
    matrix, _ = _design(
        fit.spec.kind,
        _subset(data, supported),
        gain_count=gain_count,
        frequency_count=frequency_count,
        reference_gain=reference_gain,
        reference_frequency_hz=fit.reference_frequency_hz,
    )
    return wrap_phase(matrix @ fit.beta), supported


def _fit_predict(
    spec: ModelSpec,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    radio_count: int,
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
    reference_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(test["phase"].shape, np.nan, dtype=np.float64)
    supported = np.zeros(test["phase"].shape, dtype=bool)
    groups: Iterable[int | None] = (
        range(radio_count) if spec.scope == "per_radio" else (None,)
    )
    for radio in groups:
        train_selected = (
            np.ones(train["phase"].shape, dtype=bool)
            if radio is None
            else train["radio"] == radio
        )
        test_selected = (
            np.ones(test["phase"].shape, dtype=bool)
            if radio is None
            else test["radio"] == radio
        )
        if not np.any(train_selected) or not np.any(test_selected):
            continue
        fit = _fit(
            spec,
            _subset(train, train_selected),
            gain_count=gain_count,
            frequency_count=frequency_count,
            reference_gain=reference_gain,
            reference_frequency_hz=reference_frequency_hz,
        )
        prediction, local_support = _predict(
            fit,
            _subset(test, test_selected),
            gain_count=gain_count,
            frequency_count=frequency_count,
            reference_gain=reference_gain,
        )
        target_indices = np.flatnonzero(test_selected)
        supported_indices = target_indices[local_support]
        predictions[supported_indices] = prediction
        supported[supported_indices] = True
    return predictions, supported


def _cross_validate(
    specs: Iterable[ModelSpec],
    data: dict[str, np.ndarray],
    *,
    split_values: Iterable[int],
    split_field: str,
    train_on_equal: bool = False,
    radio_count: int,
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
    reference_frequency_hz: float,
) -> dict[str, dict[str, Any]]:
    residuals = {spec.name: [] for spec in specs}
    expected = {spec.name: 0 for spec in specs}
    residuals_by_radio = {
        spec.name: {radio: [] for radio in range(radio_count)} for spec in specs
    }
    expected_by_radio = {
        spec.name: {radio: 0 for radio in range(radio_count)} for spec in specs
    }
    per_split: dict[str, dict[str, Any]] = {}
    for held in split_values:
        equal = data[split_field] == held
        train_selected = equal if train_on_equal else ~equal
        test_selected = ~equal if train_on_equal else equal
        train = _subset(data, train_selected)
        test = _subset(data, test_selected)
        split_result = {}
        for spec in specs:
            prediction, supported = _fit_predict(
                spec,
                train,
                test,
                radio_count=radio_count,
                gain_count=gain_count,
                frequency_count=frequency_count,
                reference_gain=reference_gain,
                reference_frequency_hz=reference_frequency_hz,
            )
            residual = wrap_phase(test["phase"][supported] - prediction[supported])
            residuals[spec.name].extend(residual.tolist())
            expected[spec.name] += int(test["phase"].size)
            supported_radios = test["radio"][supported]
            for radio in range(radio_count):
                residuals_by_radio[spec.name][radio].extend(
                    residual[supported_radios == radio].tolist()
                )
                expected_by_radio[spec.name][radio] += int(
                    np.count_nonzero(test["radio"] == radio)
                )
            split_result[spec.name] = _metrics(
                residual, expected=int(test["phase"].size)
            )
        per_split[str(held)] = split_result
    return {
        spec.name: {
            **_metrics(
                np.asarray(residuals[spec.name]),
                expected=expected[spec.name],
            ),
            "per_split": {key: value[spec.name] for key, value in per_split.items()},
            "per_radio": {
                str(radio): _metrics(
                    np.asarray(residuals_by_radio[spec.name][radio]),
                    expected=expected_by_radio[spec.name][radio],
                )
                for radio in range(radio_count)
            },
        }
        for spec in specs
    }


def _fit_summary(
    spec: ModelSpec,
    data: dict[str, np.ndarray],
    *,
    radio_count: int,
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
    reference_frequency_hz: float,
) -> dict[str, Any]:
    groups: Iterable[int | None] = (
        range(radio_count) if spec.scope == "per_radio" else (None,)
    )
    fits = []
    residuals = []
    for radio in groups:
        selected = (
            np.ones(data["phase"].shape, dtype=bool)
            if radio is None
            else data["radio"] == radio
        )
        local = _subset(data, selected)
        fit = _fit(
            spec,
            local,
            gain_count=gain_count,
            frequency_count=frequency_count,
            reference_gain=reference_gain,
            reference_frequency_hz=reference_frequency_hz,
        )
        prediction, support = _predict(
            fit,
            local,
            gain_count=gain_count,
            frequency_count=frequency_count,
            reference_gain=reference_gain,
        )
        residuals.extend(wrap_phase(local["phase"][support] - prediction).tolist())
        coefficient_indices = np.arange(fit.beta.size, dtype=np.int64)
        if spec.kind == "full_cell":
            # Additive-cross schedules intentionally observe only a subset of
            # the Cartesian gain-pair cube. Do not report the all-zero,
            # structurally unidentifiable columns as trained parameters.
            coefficient_indices = np.unique(
                (local["frequency"] * gain_count + local["gain1"]) * gain_count
                + local["gain2"]
            )
        coefficients = {
            fit.feature_names[int(index)]: float(fit.beta[int(index)])
            for index in coefficient_indices
        }
        delay = {}
        if "frequency_rad_per_ghz" in coefficients:
            delay["base_differential_delay_ps"] = float(
                -coefficients["frequency_rad_per_ghz"] / (2 * np.pi * 1e9) * 1e12
            )
            delay["base_equivalent_path_mm"] = float(
                delay["base_differential_delay_ps"] * 1e-12 * LIGHT_SPEED_M_PER_S * 1e3
            )
        branch_delays = {
            name: float(-value / (2 * np.pi * 1e9) * 1e12)
            for name, value in coefficients.items()
            if "_delay_slope[" in name
        }
        if branch_delays:
            delay["relative_branch_delay_ps_by_gain_index"] = branch_delays
        fits.append(
            {
                "radio_index": radio,
                "parameter_count": len(coefficients),
                "coefficients_rad": coefficients,
                "delay_interpretation": delay or None,
            }
        )
    return {
        "training_metrics": _metrics(
            np.asarray(residuals), expected=int(data["phase"].size)
        ),
        "total_parameter_count": int(sum(fit["parameter_count"] for fit in fits)),
        "fits": fits,
    }


def _metric_difference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float | None]:
    """Return candidate-minus-reference prediction metric differences."""

    keys = (
        "circular_mae_deg",
        "circular_rmse_deg",
        "circular_p95_deg",
        "circular_max_deg",
    )
    return {
        key: (
            None
            if candidate.get(key) is None or reference.get(key) is None
            else float(candidate[key] - reference[key])
        )
        for key in keys
    }


def _symmetric_model_comparison(
    models: dict[str, dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the parsimonious symmetric baseline with independent A/B."""

    symmetric = models[SYMMETRIC_GAIN_MODEL]
    independent = models[INDEPENDENT_GAIN_MODEL]
    symmetric_metric = symmetric["leave_one_epoch_out"]
    independent_metric = independent["leave_one_epoch_out"]
    parameter_reduction = (
        independent["total_parameter_count"] - symmetric["total_parameter_count"]
    )
    per_radio = []
    for radio in provenance:
        radio_index = str(radio["radio_index"])
        symmetric_radio = symmetric_metric["per_radio"][radio_index]
        independent_radio = independent_metric["per_radio"][radio_index]
        per_radio.append(
            {
                "radio_index": radio["radio_index"],
                "serial": radio["serial"],
                "symmetric": {
                    key: symmetric_radio[key]
                    for key in (
                        "circular_mae_deg",
                        "circular_rmse_deg",
                        "circular_p95_deg",
                        "circular_max_deg",
                    )
                },
                "independent": {
                    key: independent_radio[key]
                    for key in (
                        "circular_mae_deg",
                        "circular_rmse_deg",
                        "circular_p95_deg",
                        "circular_max_deg",
                    )
                },
                "symmetric_minus_independent": _metric_difference(
                    symmetric_radio, independent_radio
                ),
            }
        )
    return {
        "default_parsimonious_model": SYMMETRIC_GAIN_MODEL,
        "accuracy_reference_model": INDEPENDENT_GAIN_MODEL,
        "interpretation": (
            "H is one physical gain-response LUT whose contribution is "
            "antisymmetric under the RX1-minus-RX2 phase convention. Always "
            "fit both H and independent A/B, and report H-minus-A/B error."
        ),
        "parameter_count": {
            "symmetric": symmetric["total_parameter_count"],
            "independent": independent["total_parameter_count"],
            "reduction": parameter_reduction,
            "reduction_fraction": float(
                parameter_reduction / independent["total_parameter_count"]
            ),
        },
        "leave_one_epoch_out": {
            "symmetric": {
                key: symmetric_metric[key]
                for key in (
                    "circular_mae_deg",
                    "circular_rmse_deg",
                    "circular_p95_deg",
                    "circular_max_deg",
                )
            },
            "independent": {
                key: independent_metric[key]
                for key in (
                    "circular_mae_deg",
                    "circular_rmse_deg",
                    "circular_p95_deg",
                    "circular_max_deg",
                )
            },
            "symmetric_minus_independent": _metric_difference(
                symmetric_metric, independent_metric
            ),
        },
        "per_radio": per_radio,
    }


def _symmetric_curve_rad(
    fit: dict[str, Any],
    *,
    frequency_index: int,
    gain_count: int,
    reference_gain: int,
) -> np.ndarray:
    """Recover the reference-zero H(f,g) curve from a fitted coefficient map."""

    coefficients = fit["coefficients_rad"]
    curve = np.zeros(gain_count, dtype=np.float64)
    for gain_index in range(gain_count):
        if gain_index == reference_gain:
            continue
        curve[gain_index] = coefficients[
            f"frequency[{frequency_index}].gain_phase[{gain_index}]"
        ]
    curve = np.unwrap(curve)
    return curve - curve[reference_gain]


def _representative_frequencies(frequencies_hz: list[int]) -> list[int]:
    preferred = (
        915_000_000,
        2_412_000_000,
        2_467_100_000,
        4_000_000_000,
        4_001_000_000,
        5_804_000_000,
    )
    selected = [frequency for frequency in preferred if frequency in frequencies_hz]
    if len(selected) >= 6:
        return selected[:6]
    for index in np.linspace(0, len(frequencies_hz) - 1, 6, dtype=int):
        frequency = frequencies_hz[int(index)]
        if frequency not in selected:
            selected.append(frequency)
        if len(selected) == 6:
            break
    return selected


def _symmetric_gain_structure(
    models: dict[str, dict[str, Any]],
    provenance: list[dict[str, Any]],
    frequencies_hz: list[int],
    gains_db: list[int],
    reference_gain_db: int,
) -> dict[str, Any]:
    """Summarize how the fitted H(r,f,g) LUT changes across gain."""

    model = models[SYMMETRIC_GAIN_MODEL]
    reference_gain = gains_db.index(reference_gain_db)
    representative = _representative_frequencies(frequencies_hz)
    frequency_lookup = {
        frequency_hz: index for index, frequency_hz in enumerate(frequencies_hz)
    }
    serial_by_index = {radio["radio_index"]: radio["serial"] for radio in provenance}
    all_adjacent_steps = []
    curves = []
    largest_steps = []
    for fit in model["fits"]:
        serial = serial_by_index[fit["radio_index"]]
        for frequency_index, frequency_hz in enumerate(frequencies_hz):
            curve_deg = np.degrees(
                _symmetric_curve_rad(
                    fit,
                    frequency_index=frequency_index,
                    gain_count=len(gains_db),
                    reference_gain=reference_gain,
                )
            )
            adjacent = np.diff(curve_deg)
            all_adjacent_steps.extend(np.abs(adjacent).tolist())
            largest_index = int(np.argmax(np.abs(adjacent)))
            largest_steps.append(
                {
                    "serial": serial,
                    "frequency_hz": frequency_hz,
                    "gain_from_db": gains_db[largest_index],
                    "gain_to_db": gains_db[largest_index + 1],
                    "step_deg": float(adjacent[largest_index]),
                    "absolute_step_deg": float(abs(adjacent[largest_index])),
                }
            )
        for frequency_hz in representative:
            frequency_index = frequency_lookup[frequency_hz]
            curve_deg = np.degrees(
                _symmetric_curve_rad(
                    fit,
                    frequency_index=frequency_index,
                    gain_count=len(gains_db),
                    reference_gain=reference_gain,
                )
            )
            adjacent = np.diff(curve_deg)
            largest_index = int(np.argmax(np.abs(adjacent)))
            curves.append(
                {
                    "radio_index": fit["radio_index"],
                    "serial": serial,
                    "frequency_hz": frequency_hz,
                    "minimum_deg": float(np.min(curve_deg)),
                    "maximum_deg": float(np.max(curve_deg)),
                    "span_deg": float(np.ptp(curve_deg)),
                    "largest_step_gain_from_db": gains_db[largest_index],
                    "largest_step_gain_to_db": gains_db[largest_index + 1],
                    "largest_step_deg": float(adjacent[largest_index]),
                }
            )
    absolute_steps = np.asarray(all_adjacent_steps, dtype=np.float64)
    return {
        "formula": "H(r,f,g1) - H(r,f,g2)",
        "units": "degrees of RX1-minus-RX2 phase contribution",
        "reference_gain_db": reference_gain_db,
        "representative_frequencies_hz": representative,
        "adjacent_one_db_step_distribution": {
            "n_steps": int(absolute_steps.size),
            "median_absolute_deg": float(np.median(absolute_steps)),
            "p90_absolute_deg": float(np.percentile(absolute_steps, 90)),
            "p95_absolute_deg": float(np.percentile(absolute_steps, 95)),
            "p99_absolute_deg": float(np.percentile(absolute_steps, 99)),
            "maximum_absolute_deg": float(np.max(absolute_steps)),
        },
        "representative_curves": curves,
        "largest_steps": sorted(
            largest_steps,
            key=lambda row: row["absolute_step_deg"],
            reverse=True,
        )[:20],
    }


def _gain_table_band_index(frequency_hz: np.ndarray) -> np.ndarray:
    """Map RF frequency to the three AD936x full gain-table bands."""

    return np.where(
        frequency_hz <= 1_300_000_000,
        0,
        np.where(frequency_hz <= 4_000_000_000, 1, 2),
    )


def _held_out_gain_pair_comparison(
    *,
    data: dict[str, np.ndarray],
    held_out_gain_pairs: Iterable[tuple[int, int]],
    models: dict[str, dict[str, Any]],
    radio_count: int,
    gain_count: int,
    frequency_count: int,
    reference_gain: int,
    reference_frequency_hz: float,
) -> dict[str, Any]:
    """Score compact gain models on pairs explicitly excluded from fitting."""

    held_out = {tuple(map(int, pair)) for pair in held_out_gain_pairs}
    if not held_out:
        return {
            "available": False,
            "reason": "configuration contains no held-out gain pairs",
            "models": {},
        }
    selected = np.asarray(
        [
            (int(gain1), int(gain2)) in held_out
            for gain1, gain2 in zip(data["gain1_db"], data["gain2_db"])
        ],
        dtype=bool,
    )
    if not np.any(selected):
        return {
            "available": False,
            "reason": "no quality-valid held-out gain-pair observations",
            "models": {},
        }
    train = _subset(data, ~selected)
    test = _subset(data, selected)
    bands = _gain_table_band_index(test["frequency_hz"])
    spec_by_name = {spec.name: spec for spec in MODEL_SPECS}
    common = {
        "radio_count": radio_count,
        "gain_count": gain_count,
        "frequency_count": frequency_count,
        "reference_gain": reference_gain,
        "reference_frequency_hz": reference_frequency_hz,
    }
    results = {}
    for name in FREQUENCY_SCALING_MODELS:
        prediction, supported = _fit_predict(
            spec_by_name[name],
            train,
            test,
            **common,
        )
        residual = wrap_phase(test["phase"][supported] - prediction[supported])
        supported_radios = test["radio"][supported]
        supported_bands = bands[supported]
        results[name] = {
            "label": models[name]["label"],
            "parameter_count": models[name]["total_parameter_count"],
            **_metrics(residual, expected=int(test["phase"].size)),
            "per_radio": {
                str(radio): _metrics(
                    residual[supported_radios == radio],
                    expected=int(np.count_nonzero(test["radio"] == radio)),
                )
                for radio in range(radio_count)
            },
            "per_gain_table_band": {
                str(band): _metrics(
                    residual[supported_bands == band],
                    expected=int(np.count_nonzero(bands == band)),
                )
                for band in range(3)
            },
        }
    return {
        "available": True,
        "held_out_gain_pair_count": len(held_out),
        "quality_valid_observations": int(test["phase"].size),
        "training_quality_valid_observations": int(train["phase"].size),
        "evaluation": (
            "Fit on the additive-cross axes and predict only RX1/RX2 gain "
            "pairs explicitly excluded from fitting."
        ),
        "models": results,
    }


def _frequency_scaling_structure(
    *,
    models: dict[str, dict[str, Any]],
    provenance: list[dict[str, Any]],
    frequencies_hz: list[int],
    gains_db: list[int],
    reference_gain_db: int,
) -> dict[str, Any]:
    """Measure whether exact-frequency gain LUTs are scaled copies."""

    frequencies_ghz = np.asarray(frequencies_hz, dtype=np.float64) / 1e9
    reference_gain = gains_db.index(reference_gain_db)
    masks = {
        "all": np.ones(frequencies_ghz.size, dtype=bool),
        "low": frequencies_ghz <= 1.3,
        "middle": (frequencies_ghz > 1.3) & (frequencies_ghz <= 4.0),
        "high": frequencies_ghz > 4.0,
    }
    serial_by_index = {radio["radio_index"]: radio["serial"] for radio in provenance}
    rows = []
    for fit in models[SYMMETRIC_GAIN_MODEL]["fits"]:
        exact = np.stack(
            [
                _symmetric_curve_rad(
                    fit,
                    frequency_index=frequency_index,
                    gain_count=len(gains_db),
                    reference_gain=reference_gain,
                )
                for frequency_index in range(len(frequencies_hz))
            ]
        )
        for scope, mask in masks.items():
            if not np.any(mask):
                continue
            curves = exact[mask]
            _, singular_values, _ = np.linalg.svd(curves, full_matrices=False)
            energy = singular_values**2
            cumulative = np.cumsum(energy) / np.sum(energy)
            frequencies = frequencies_ghz[mask]
            forced_curve = np.sum(frequencies[:, None] * curves, axis=0) / np.sum(
                frequencies**2
            )
            forced_prediction = frequencies[:, None] * forced_curve
            forced_residual = curves - forced_prediction
            forced_energy_fraction = 1.0 - float(
                np.sum(forced_residual**2) / np.sum(curves**2)
            )
            rows.append(
                {
                    "radio_index": fit["radio_index"],
                    "serial": serial_by_index[fit["radio_index"]],
                    "scope": scope,
                    "frequency_count": int(np.count_nonzero(mask)),
                    "forced_frequency_energy_fraction": forced_energy_fraction,
                    "forced_frequency_rms_deg": float(
                        np.degrees(np.sqrt(np.mean(forced_residual**2)))
                    ),
                    "best_rank1_energy_fraction": float(cumulative[0]),
                    "best_rank2_energy_fraction": float(
                        cumulative[min(1, cumulative.size - 1)]
                    ),
                    "best_rank3_energy_fraction": float(
                        cumulative[min(2, cumulative.size - 1)]
                    ),
                }
            )
    aggregate = {}
    for scope in masks:
        local = [row for row in rows if row["scope"] == scope]
        aggregate[scope] = {
            "frequency_count": int(local[0]["frequency_count"]) if local else 0,
            **{
                key: (
                    float(np.mean([row[key] for row in local])) if local else None
                )
                for key in (
                    "forced_frequency_energy_fraction",
                    "forced_frequency_rms_deg",
                    "best_rank1_energy_fraction",
                    "best_rank2_energy_fraction",
                    "best_rank3_energy_fraction",
                )
            },
        }
    return {
        "interpretation": (
            "The best rank-k values are optimistic low-rank summaries of the "
            "exact-frequency symmetric LUTs. Forced-frequency scaling uses "
            "H(f,g)=(f/GHz)*G(g). Explained LUT energy is descriptive and is "
            "not a substitute for held-out phase-prediction error."
        ),
        "per_radio": rows,
        "mean_across_radios": aggregate,
    }


def analyze_model_matrix(
    *,
    config_path: Path,
    artifact_root: Path | None = None,
    dataset_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    explicit_paths = tuple(dataset_paths or ())
    if (artifact_root is None) == (not explicit_paths):
        raise ValueError("provide exactly one of artifact_root or dataset_paths")
    if explicit_paths:
        config, data, provenance = _radio_arrays_from_paths(
            config_path=config_path,
            dataset_paths=explicit_paths,
        )
    else:
        config, data, provenance = _radio_arrays(
            config_path=config_path,
            artifact_root=artifact_root,
        )
    radio_count = len(provenance)
    gain_count = len(config.gains_db)
    frequency_count = len(config.frequencies_hz)
    reference_gain_db = min(
        config.gains_db,
        key=lambda value: abs(value - config.tx_reference_rx_gain_db),
    )
    reference_gain = config.gains_db.index(reference_gain_db)
    reference_frequency_hz = float(np.mean(config.frequencies_hz))
    common = {
        "radio_count": radio_count,
        "gain_count": gain_count,
        "frequency_count": frequency_count,
        "reference_gain": reference_gain,
        "reference_frequency_hz": reference_frequency_hz,
    }

    leave_epoch_out = _cross_validate(
        MODEL_SPECS,
        data,
        split_values=range(config.repetitions),
        split_field="epoch",
        **common,
    )
    frequency_specs = tuple(spec for spec in MODEL_SPECS if spec.unseen_frequency)
    leave_frequency_out = _cross_validate(
        frequency_specs,
        data,
        split_values=range(frequency_count),
        split_field="frequency",
        **common,
    )
    universal_specs = tuple(spec for spec in MODEL_SPECS if spec.scope == "universal")
    leave_radio_out = _cross_validate(
        universal_specs,
        data,
        split_values=range(radio_count),
        split_field="radio",
        **common,
    )
    models = {}
    for spec in MODEL_SPECS:
        models[spec.name] = {
            "label": spec.label,
            "scope": spec.scope,
            "kind": spec.kind,
            "formula": spec.formula,
            "can_predict_unseen_frequency": spec.unseen_frequency,
            **_fit_summary(spec, data, **common),
            "leave_one_epoch_out": leave_epoch_out[spec.name],
            "leave_one_frequency_out": leave_frequency_out.get(spec.name),
            "leave_one_radio_out": leave_radio_out.get(spec.name),
        }
    symmetric_comparison = _symmetric_model_comparison(models, provenance)
    symmetric_gain_structure = _symmetric_gain_structure(
        models,
        provenance,
        list(config.frequencies_hz),
        list(config.gains_db),
        int(reference_gain_db),
    )
    held_out_gain_pair_comparison = _held_out_gain_pair_comparison(
        data=data,
        held_out_gain_pairs=config.held_out_gain_pairs,
        models=models,
        **common,
    )
    frequency_scaling_structure = _frequency_scaling_structure(
        models=models,
        provenance=provenance,
        frequencies_hz=list(config.frequencies_hz),
        gains_db=list(config.gains_db),
        reference_gain_db=int(reference_gain_db),
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "dataset_paths": [radio["dataset_path"] for radio in provenance],
        "phase_convention": "RX1 minus RX2",
        "evaluation": {
            "leave_one_epoch_out": (
                "Train on two complete sweep repeats and test the third. This "
                "measures correction accuracy for known frequency/gain cells."
            ),
            "leave_one_frequency_out": (
                "Train without one frequency and test that frequency. Only "
                "models without a frequency lookup table are eligible."
            ),
            "leave_one_radio_out": (
                "Fit a strict universal model on one radio and predict the "
                "other with no radio-specific baseline adjustment."
            ),
        },
        "identifiability": (
            "Differential phase identifies only RX1 delay minus RX2 delay. "
            "Reported branch-delay LUT terms are relative contributions under "
            "the stated reference constraints, not absolute physical delays."
        ),
        "reference_frequency_hz": reference_frequency_hz,
        "reference_gain_db": int(reference_gain_db),
        "frequencies_hz": list(config.frequencies_hz),
        "gains_db": list(config.gains_db),
        "provenance": provenance,
        "models": models,
        "default_model_policy": {
            "parsimonious_default": SYMMETRIC_GAIN_MODEL,
            "accuracy_reference": INDEPENDENT_GAIN_MODEL,
            "required_comparison": (
                "Always report symmetric-minus-independent prediction error "
                "and parameter reduction."
            ),
        },
        "symmetric_vs_independent": symmetric_comparison,
        "symmetric_gain_structure": symmetric_gain_structure,
        "held_out_gain_pair_comparison": held_out_gain_pair_comparison,
        "frequency_scaling_structure": frequency_scaling_structure,
    }


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_model_matrix_report(result: dict[str, Any]) -> str:
    lines = [
        "# Dual-RX phase model matrix",
        "",
        "## Scope and evaluation",
        "",
        f"- Phase convention: `{result['phase_convention']}`.",
        f"- Reference gain: {result['reference_gain_db']} dB.",
        (
            f"- Reference frequency for delay fits: "
            f"{result['reference_frequency_hz'] / 1e9:.6f} GHz."
        ),
        f"- {result['identifiability']}",
        "",
        "## Input checkpoint",
        "",
        "| Pluto serial | Completed | Quality-valid | Validation | Scalar-input SHA-256 |",
        "|---|---:|---:|---|---|",
    ]
    for radio in result["provenance"]:
        lines.append(
            f"| `{radio['serial']}` | {radio['completed_frames']} | "
            f"{radio['quality_valid_observations']} | "
            f"`{radio['validation_status']}` | "
            f"`{radio['analysis_input_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "All datasets are structurally complete. `fail_quality` means "
            "quality-rejected frames remain explicit in the V7 dataset; only "
            "quality-valid observations enter these fits.",
            "",
            "## Evaluation design",
            "",
            "The main table uses leave-one-epoch-out prediction: two repeats train "
            "the model and the third is unseen. This is the fair operational test "
            "for lookup tables whose deployment support is the measured grid.",
            "",
            "## Known-cell prediction",
            "",
            "| Model | Scope | Parameters | MAE ° | RMSE ° | P95 ° | Max ° | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered = sorted(
        result["models"].values(),
        key=lambda model: model["leave_one_epoch_out"]["circular_mae_deg"],
    )
    for model in ordered:
        metric = model["leave_one_epoch_out"]
        lines.append(
            f"| {model['label']} | {model['scope']} | "
            f"{model['total_parameter_count']} | "
            f"{_format_metric(metric['circular_mae_deg'])} | "
            f"{_format_metric(metric['circular_rmse_deg'])} | "
            f"{_format_metric(metric['circular_p95_deg'])} | "
            f"{_format_metric(metric['circular_max_deg'])} | "
            f"{100 * metric['coverage_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "![Known-cell model comparison](known_cell_model_comparison.png)",
        ]
    )
    lines.extend(
        [
            "",
            "### Best model by radio",
            "",
            "| Pluto serial | MAE ° | RMSE ° | P95 ° | Max ° |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    best = result["models"]["frequency_specific_additive_gain_per_radio"][
        "leave_one_epoch_out"
    ]
    for radio in result["provenance"]:
        metric = best["per_radio"][str(radio["radio_index"])]
        lines.append(
            f"| `{radio['serial']}` | "
            f"{_format_metric(metric['circular_mae_deg'])} | "
            f"{_format_metric(metric['circular_rmse_deg'])} | "
            f"{_format_metric(metric['circular_p95_deg'])} | "
            f"{_format_metric(metric['circular_max_deg'])} |"
        )
    independent_model = result["models"][INDEPENDENT_GAIN_MODEL]
    antisymmetric_model = result["models"][SYMMETRIC_GAIN_MODEL]
    independent_metric = independent_model["leave_one_epoch_out"]
    antisymmetric_metric = antisymmetric_model["leave_one_epoch_out"]
    comparison = result["symmetric_vs_independent"]
    gap = comparison["leave_one_epoch_out"]["symmetric_minus_independent"]
    radio_count = len(result["provenance"])
    frequency_count = len(result["frequencies_hz"])
    gain_count = len(result["gains_db"])
    lines.extend(
        [
            "",
            "### Symmetric default versus independent accuracy reference",
            "",
            "The parsimonious default assumes RX1 and RX2 share one physical "
            "gain-state response. Under the RX1-minus-RX2 phase convention, "
            "that response enters with opposite signs:",
            "",
            "```text",
            "independent:   phase = C[f] + A[f,g1] + B[f,g2]",
            "antisymmetric: phase = C[f] + H[f,g1] - H[f,g2]",
            "```",
            "",
            "| Model | Parameters per radio/frequency | Per radio | Fleet total | "
            "LOEO MAE | LOEO p95 |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| Independent A/B | 1 + {gain_count - 1} + "
                f"{gain_count - 1} = {1 + 2 * (gain_count - 1)} | "
                f"{frequency_count * (1 + 2 * (gain_count - 1))} | "
                f"{independent_model['total_parameter_count']} | "
                f"{_format_metric(independent_metric['circular_mae_deg'])}° | "
                f"{_format_metric(independent_metric['circular_p95_deg'])}° |"
            ),
            (
                f"| Shared antisymmetric H | 1 + {gain_count - 1} = "
                f"{gain_count} | {frequency_count * gain_count} | "
                f"{antisymmetric_model['total_parameter_count']} | "
                f"{_format_metric(antisymmetric_metric['circular_mae_deg'])}° | "
                f"{_format_metric(antisymmetric_metric['circular_p95_deg'])}° |"
            ),
            "",
            f"The fleet totals above cover {radio_count} radios. The reference "
            f"gain ({result['reference_gain_db']} dB) is fixed to zero in each "
            "gain LUT, so it is not an additional fitted coefficient.",
            "",
            f"Relative to independent A/B, symmetric H changes LOEO MAE by "
            f"{gap['circular_mae_deg']:+.3f}°, RMSE by "
            f"{gap['circular_rmse_deg']:+.3f}°, p95 by "
            f"{gap['circular_p95_deg']:+.3f}°, and maximum error by "
            f"{gap['circular_max_deg']:+.3f}°. It removes "
            f"{comparison['parameter_count']['reduction']} parameters "
            f"({100 * comparison['parameter_count']['reduction_fraction']:.1f}%).",
            "",
            "Both models are always fitted. Symmetric H is the default "
            "structural model; independent A/B remains the accuracy reference, "
            "and the H-minus-A/B gap is a required output.",
            "",
            "#### Gap by radio",
            "",
            "| Pluto serial | Symmetric MAE | Independent MAE | MAE gap | "
            "Symmetric p95 | Independent p95 | P95 gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["per_radio"]:
        symmetric = row["symmetric"]
        independent = row["independent"]
        delta = row["symmetric_minus_independent"]
        lines.append(
            f"| `{row['serial']}` | "
            f"{symmetric['circular_mae_deg']:.3f}° | "
            f"{independent['circular_mae_deg']:.3f}° | "
            f"{delta['circular_mae_deg']:+.3f}° | "
            f"{symmetric['circular_p95_deg']:.3f}° | "
            f"{independent['circular_p95_deg']:.3f}° | "
            f"{delta['circular_p95_deg']:+.3f}° |"
        )
    held_out = result["held_out_gain_pair_comparison"]
    if held_out["available"]:
        held_models = held_out["models"]
        global_unscaled = held_models[
            "frequency_lut_symmetric_gain_per_radio"
        ]
        global_scaled = held_models[
            "frequency_lut_frequency_scaled_symmetric_gain_per_radio"
        ]
        table_unscaled = held_models[
            "frequency_lut_gain_table_symmetric_gain_per_radio"
        ]
        table_scaled = held_models[
            "frequency_lut_gain_table_frequency_scaled_symmetric_gain_per_radio"
        ]
        lines.extend(
            [
                "",
                "### Can one gain LUT be scaled across frequency?",
                "",
                "The direct test is deliberately stricter than the epoch "
                "holdout above. It fits only the additive-cross axes and "
                f"predicts {held_out['quality_valid_observations']:,} "
                "quality-valid observations from "
                f"{held_out['held_out_gain_pair_count']} RX1/RX2 gain pairs "
                "that were never used for fitting.",
                "",
                "```text",
                "frequency-scaled: phase = C(r,f) + (f/GHz) * "
                "[G(r,g1) - G(r,g2)]",
                "```",
                "",
                "`C(r,f)` is still an exact-frequency equal-gain anchor. A "
                "separate scalar `k` and learned `G` are not identifiable, so "
                "`k` is absorbed into the LUT coefficients.",
                "",
                "| Gain model | Parameters | Held-out MAE | Held-out p95 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in FREQUENCY_SCALING_MODELS:
            metric = held_models[name]
            lines.append(
                f"| {metric['label']} | {metric['parameter_count']} | "
                f"{metric['circular_mae_deg']:.3f}° | "
                f"{metric['circular_p95_deg']:.3f}° |"
            )
        lines.extend(
            [
                "",
                "Forcing one global LUT to scale with frequency improves MAE "
                f"from {global_unscaled['circular_mae_deg']:.3f}° to "
                f"{global_scaled['circular_mae_deg']:.3f}°. Separating the "
                "three AD936x gain-table bands matters much more; adding the "
                "frequency factor to those three LUTs improves MAE from "
                f"{table_unscaled['circular_mae_deg']:.3f}° to "
                f"{table_scaled['circular_mae_deg']:.3f}°.",
                "",
                "![Frequency-scaled model held-out comparison]"
                "(frequency_scaled_gain_model_comparison.png)",
                "",
                "The exact fitted `H(r,f,g)` curves provide a second, "
                "descriptive low-rank test. A best rank-1 approximation may "
                "learn any scale at each frequency; the forced-frequency "
                "column specifically requires the scale to be proportional "
                "to `f`.",
                "",
                "| Frequency scope | Forced f scaling | Best rank 1 | "
                "Best rank 2 |",
                "|---|---:|---:|---:|",
            ]
        )
        structure_scaling = result["frequency_scaling_structure"][
            "mean_across_radios"
        ]
        scope_labels = {
            "all": "All frequencies",
            "low": "Low table, ≤1.3 GHz",
            "middle": "Middle table, 1.3–4.0 GHz",
            "high": "High table, >4.0 GHz",
        }
        for scope in ("all", "low", "middle", "high"):
            metric = structure_scaling[scope]
            percentages = [
                (
                    "n/a"
                    if metric[key] is None
                    else f"{100 * metric[key]:.1f}%"
                )
                for key in (
                    "forced_frequency_energy_fraction",
                    "best_rank1_energy_fraction",
                    "best_rank2_energy_fraction",
                )
            ]
            lines.append(
                f"| {scope_labels[scope]} | {percentages[0]} | "
                f"{percentages[1]} | {percentages[2]} |"
            )
        lines.extend(
            [
                "",
                "![Gain LUT low-rank structure]"
                "(gain_lut_low_rank_structure.png)",
                "",
                "Conclusion: proportional-to-frequency scaling is real but "
                "only a coarse compression. It cannot move gain-stage "
                "discontinuities or represent frequency-dependent analogue "
                "dispersion. One scaled LUT per hardware gain-table band is a "
                "useful fallback; exact-frequency H remains the parsimonious "
                "precision default, with independent A/B as the accuracy "
                "reference. Rank two is a promising future compressed model, "
                "but it is not recommended until its phase prediction is "
                "evaluated on held-out data.",
            ]
        )
    structure = result["symmetric_gain_structure"]
    distribution = structure["adjacent_one_db_step_distribution"]
    lines.extend(
        [
            "",
            "### What H(r,f,g) looks like across gain",
            "",
            "`H` is a circular phase correction LUT in degrees, not a gain "
            "value. At fixed radio and frequency it is anchored at "
            f"`H({result['reference_gain_db']} dB) = 0`. Across all fitted "
            "curves, the median absolute 1 dB step is "
            f"{distribution['median_absolute_deg']:.3f}°, p95 is "
            f"{distribution['p95_absolute_deg']:.3f}°, and p99 is "
            f"{distribution['p99_absolute_deg']:.3f}°. The low median and "
            "large tail describe a mostly flat or gently varying staircase "
            "with a few hardware gain-stage jumps.",
            "",
            "| Radio | Frequency | H range | Span | Largest adjacent step |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in structure["representative_curves"]:
        lines.append(
            f"| `{row['serial']}` | {row['frequency_hz'] / 1e6:.3f} MHz | "
            f"{row['minimum_deg']:.2f}° to {row['maximum_deg']:.2f}° | "
            f"{row['span_deg']:.2f}° | "
            f"{row['largest_step_gain_from_db']}→"
            f"{row['largest_step_gain_to_db']} dB: "
            f"{row['largest_step_deg']:+.2f}° |"
        )
    lines.extend(
        [
            "",
            "![Symmetric gain-response LUT slices]" "(symmetric_gain_lut_slices.png)",
        ]
    )
    lines.extend(
        [
            "",
            "## Unseen-frequency test",
            "",
            "| Model | Scope | MAE ° | RMSE ° | P95 ° | Coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    frequency_models = [
        model
        for model in result["models"].values()
        if model["leave_one_frequency_out"] is not None
    ]
    frequency_models.sort(
        key=lambda model: model["leave_one_frequency_out"]["circular_mae_deg"]
    )
    for model in frequency_models:
        metric = model["leave_one_frequency_out"]
        lines.append(
            f"| {model['label']} | {model['scope']} | "
            f"{_format_metric(metric['circular_mae_deg'])} | "
            f"{_format_metric(metric['circular_rmse_deg'])} | "
            f"{_format_metric(metric['circular_p95_deg'])} | "
            f"{100 * metric['coverage_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "![Unseen-frequency model comparison](unseen_frequency_model_comparison.png)",
        ]
    )
    lines.extend(
        [
            "",
            "### Fitted effective differential delays",
            "",
            "| Model | Pluto serial | Base delay ps | Free-space equivalent mm |",
            "|---|---|---:|---:|",
        ]
    )
    serial_by_index = {
        radio["radio_index"]: radio["serial"] for radio in result["provenance"]
    }
    for name in (
        "delay_additive_gain_per_radio",
        "branch_gain_delay_lut_per_radio",
    ):
        model = result["models"][name]
        for fit in model["fits"]:
            delay = fit["delay_interpretation"]
            lines.append(
                f"| {model['label']} | "
                f"`{serial_by_index[fit['radio_index']]}` | "
                f"{delay['base_differential_delay_ps']:.3f} | "
                f"{delay['base_equivalent_path_mm']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Strict universal transfer to an unseen radio",
            "",
            "| Model | MAE ° | RMSE ° | P95 ° | Max ° | Coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    universal_models = [
        model
        for model in result["models"].values()
        if model["leave_one_radio_out"] is not None
    ]
    universal_models.sort(
        key=lambda model: model["leave_one_radio_out"]["circular_mae_deg"]
    )
    for model in universal_models:
        metric = model["leave_one_radio_out"]
        lines.append(
            f"| {model['label']} | "
            f"{_format_metric(metric['circular_mae_deg'])} | "
            f"{_format_metric(metric['circular_rmse_deg'])} | "
            f"{_format_metric(metric['circular_p95_deg'])} | "
            f"{_format_metric(metric['circular_max_deg'])} | "
            f"{100 * metric['coverage_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Physical interpretation",
            "",
            "A delay-only explanation is supported only if the delay models are "
            "competitive on the unseen-frequency test. A good fit on the "
            "measured frequencies alone is insufficient: a frequency lookup "
            "can absorb arbitrary retune- or band-dependent phase offsets.",
            "",
            "The gain-dependent branch-delay model assigns relative delay "
            "terms to RX1 and RX2 gain states. Because only RX1−RX2 phase is "
            "observed, adding the same absolute delay to both paths is "
            "unobservable; the individual physical path lengths cannot be "
            "recovered from this experiment.",
            "",
            "## Interpretation and recommendation",
            "",
            "The recommended correction on this measured grid is the per-radio, "
            "per-frequency additive lookup. The full cell lookup has many more "
            "parameters and no held-out benefit, so an RX1-by-RX2 interaction "
            "table is not justified by these data.",
            "",
            "Do not use the strict universal model as a precision correction "
            "for an unseen radio. Its leave-one-radio-out error measures the "
            "cost of omitting radio-specific calibration.",
            "",
            "The delay models are useful descriptions of a broad phase slope, "
            "but their unseen-frequency errors reject path imbalance as the "
            "only mechanism. Retune state, frequency-band analogue response, "
            "and frequency-dependent gain-table effects still require explicit "
            "frequency calibration or denser within-band characterization.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m spf.calibrations.dual_rx_gain_frequency model-matrix \\",
            f"  --config {result['config_path']} \\",
        ]
    )
    if result["artifact_root"] is not None:
        lines.extend(
            [
                f"  --artifact-root {result['artifact_root']} \\",
            ]
        )
    else:
        for dataset_path in result["dataset_paths"]:
            lines.append(f"  --dataset {dataset_path} \\")
    lines.extend(
        [
            "  --output-dir artifacts/dual_rx_gain_frequency/model_matrix",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _write_plots(result: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["models"])
    labels = [result["models"][name]["label"] for name in names]
    epoch_mae = [
        result["models"][name]["leave_one_epoch_out"]["circular_mae_deg"]
        for name in names
    ]
    order = np.argsort(epoch_mae)
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        np.arange(len(names)),
        np.asarray(epoch_mae)[order],
        color=[
            "#4472c4"
            if result["models"][names[index]]["scope"] == "per_radio"
            else "#ed7d31"
            for index in order
        ],
    )
    axis.set_yticks(np.arange(len(names)), [labels[index] for index in order])
    axis.invert_yaxis()
    axis.set_xlabel("Leave-one-epoch-out circular MAE (degrees)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    epoch_path = output_dir / "known_cell_model_comparison.png"
    fig.savefig(epoch_path, dpi=160)
    plt.close(fig)

    frequency = [
        (model["label"], model["leave_one_frequency_out"]["circular_mae_deg"])
        for model in result["models"].values()
        if model["leave_one_frequency_out"] is not None
    ]
    frequency.sort(key=lambda item: item[1])
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(
        np.arange(len(frequency)),
        [value for _, value in frequency],
        color="#70ad47",
    )
    axis.set_yticks(np.arange(len(frequency)), [label for label, _ in frequency])
    axis.invert_yaxis()
    axis.set_xlabel("Leave-one-frequency-out circular MAE (degrees)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    frequency_path = output_dir / "unseen_frequency_model_comparison.png"
    fig.savefig(frequency_path, dpi=160)
    plt.close(fig)

    structure = result["symmetric_gain_structure"]
    representative = structure["representative_frequencies_hz"]
    frequency_lookup = {
        frequency_hz: index
        for index, frequency_hz in enumerate(result["frequencies_hz"])
    }
    serial_by_index = {
        radio["radio_index"]: radio["serial"] for radio in result["provenance"]
    }
    reference_gain = result["gains_db"].index(result["reference_gain_db"])
    model = result["models"][SYMMETRIC_GAIN_MODEL]
    column_count = 3
    row_count = int(np.ceil(len(representative) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, 4.5 * row_count),
        sharex=True,
        squeeze=False,
    )
    for axis, frequency_hz in zip(axes.flat, representative):
        frequency_index = frequency_lookup[frequency_hz]
        for fit in model["fits"]:
            curve_deg = np.degrees(
                _symmetric_curve_rad(
                    fit,
                    frequency_index=frequency_index,
                    gain_count=len(result["gains_db"]),
                    reference_gain=reference_gain,
                )
            )
            axis.plot(
                result["gains_db"],
                curve_deg,
                linewidth=1.8,
                label=serial_by_index[fit["radio_index"]][-6:],
            )
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        axis.axvline(
            result["reference_gain_db"],
            color="black",
            linewidth=0.7,
            linestyle="--",
            alpha=0.4,
        )
        axis.set_title(f"{frequency_hz / 1e6:.3f} MHz")
        axis.set_xlabel("Gain (dB)")
        axis.set_ylabel("H(r,f,g) phase contribution (degrees)")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(representative) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Pluto serial suffix",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(1, len(labels)),
    )
    fig.suptitle(
        "Symmetric gain-response LUT H(r,f,g), reference H(26 dB) = 0",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    symmetric_path = output_dir / "symmetric_gain_lut_slices.png"
    fig.savefig(symmetric_path, dpi=160)
    plt.close(fig)
    plot_paths = [epoch_path.name, frequency_path.name, symmetric_path.name]

    held_out = result["held_out_gain_pair_comparison"]
    if held_out["available"]:
        short_labels = {
            "frequency_lut_symmetric_gain_per_radio": "One H(g)",
            "frequency_lut_frequency_scaled_symmetric_gain_per_radio": "f · G(g)",
            "frequency_lut_gain_table_symmetric_gain_per_radio": "One Hₜ(g) / table",
            "frequency_lut_gain_table_frequency_scaled_symmetric_gain_per_radio": (
                "f · Gₜ(g) / table"
            ),
            SYMMETRIC_GAIN_MODEL: "Exact H(f,g)",
            INDEPENDENT_GAIN_MODEL: "Independent A/B(f,g)",
        }
        held_models = held_out["models"]
        y = np.arange(len(FREQUENCY_SCALING_MODELS))
        mae = [
            held_models[name]["circular_mae_deg"]
            for name in FREQUENCY_SCALING_MODELS
        ]
        p95 = [
            held_models[name]["circular_p95_deg"]
            for name in FREQUENCY_SCALING_MODELS
        ]
        labels = [
            f"{short_labels[name]}  "
            f"({held_models[name]['parameter_count']:,} parameters)"
            for name in FREQUENCY_SCALING_MODELS
        ]
        fig, axis = plt.subplots(figsize=(12, 6.5))
        axis.barh(y - 0.18, mae, height=0.34, label="MAE", color="#4472c4")
        axis.barh(y + 0.18, p95, height=0.34, label="P95", color="#ed7d31")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Circular phase error on held-out gain pairs (degrees)")
        axis.set_title(
            "Frequency-scaled gain LUTs: compression versus prediction error"
        )
        axis.grid(axis="x", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        scaling_path = output_dir / "frequency_scaled_gain_model_comparison.png"
        fig.savefig(scaling_path, dpi=160)
        plt.close(fig)
        plot_paths.append(scaling_path.name)

        aggregate = result["frequency_scaling_structure"]["mean_across_radios"]
        scopes = ("all", "low", "middle", "high")
        scope_labels = (
            "All",
            "Low\n≤1.3 GHz",
            "Middle\n1.3–4.0 GHz",
            "High\n>4.0 GHz",
        )
        x = np.arange(len(scopes))
        width = 0.25
        fig, axis = plt.subplots(figsize=(10.5, 5.8))
        for offset, key, label, color in (
            (
                -width,
                "forced_frequency_energy_fraction",
                "Forced f scaling",
                "#ed7d31",
            ),
            (0.0, "best_rank1_energy_fraction", "Best rank 1", "#4472c4"),
            (width, "best_rank2_energy_fraction", "Best rank 2", "#70ad47"),
        ):
            axis.bar(
                x + offset,
                [
                    (
                        0.0
                        if aggregate[scope][key] is None
                        else 100 * aggregate[scope][key]
                    )
                    for scope in scopes
                ],
                width=width,
                label=label,
                color=color,
            )
        axis.set_xticks(x, scope_labels)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Exact gain-LUT energy captured (%)")
        axis.set_title(
            "Can exact-frequency gain LUTs be represented by scaled templates?"
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right")
        fig.tight_layout()
        rank_path = output_dir / "gain_lut_low_rank_structure.png"
        fig.savefig(rank_path, dpi=160)
        plt.close(fig)
        plot_paths.append(rank_path.name)

    return plot_paths


def write_model_matrix_bundle(
    *,
    config_path: Path,
    artifact_root: Path | None = None,
    dataset_paths: Iterable[Path] | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_model_matrix(
        config_path=config_path,
        artifact_root=artifact_root,
        dataset_paths=dataset_paths,
    )
    result["plots"] = _write_plots(result, output_dir)
    (output_dir / "model_matrix.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "MODEL_MATRIX_REPORT.md").write_text(
        render_model_matrix_report(result)
    )
    with (output_dir / "model_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "model",
                "label",
                "scope",
                "parameter_count",
                "loeo_mae_deg",
                "loeo_rmse_deg",
                "loeo_p95_deg",
                "lofo_mae_deg",
                "loro_mae_deg",
            )
        )
        for name, model in result["models"].items():
            writer.writerow(
                (
                    name,
                    model["label"],
                    model["scope"],
                    model["total_parameter_count"],
                    model["leave_one_epoch_out"]["circular_mae_deg"],
                    model["leave_one_epoch_out"]["circular_rmse_deg"],
                    model["leave_one_epoch_out"]["circular_p95_deg"],
                    (
                        model["leave_one_frequency_out"]["circular_mae_deg"]
                        if model["leave_one_frequency_out"]
                        else ""
                    ),
                    (
                        model["leave_one_radio_out"]["circular_mae_deg"]
                        if model["leave_one_radio_out"]
                        else ""
                    ),
                )
            )
    return result
