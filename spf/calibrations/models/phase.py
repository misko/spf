"""Fail-closed runtime prediction for exported phase-offset models."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spf.bench.dual_rx_phase import wrap_phase


MODEL_SCHEMA = "spf.calibration.phase_offset_model"
MODEL_SCHEMA_VERSION = 1
SUPPORT_SCHEMA = "spf.calibration.phase_offset_support"
SUPPORT_SCHEMA_VERSION = 1


class UnsupportedPhaseModelInput(ValueError):
    """The requested radio state is outside the model's declared support."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"phase model file does not exist: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"phase model document must be an object: {path}")
    return value


@dataclass(frozen=True)
class PhaseOffsetModel:
    """One model family fitted to one immutable Pluto serial."""

    path: Path
    document: dict[str, Any]
    supported_cells: frozenset[tuple[int, int, int]]

    @property
    def model_name(self) -> str:
        return str(self.document["model_name"])

    @property
    def serial(self) -> str:
        return str(self.document["radio_serial"])

    @property
    def kind(self) -> str:
        return str(self.document["kind"])

    @property
    def frequencies_hz(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.document["frequencies_hz"])

    @property
    def gains_db(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.document["gains_db"])

    @property
    def reference_gain_db(self) -> int:
        return int(self.document["reference_gain_db"])

    @property
    def reference_frequency_hz(self) -> float:
        return float(self.document["reference_frequency_hz"])

    def _coefficient(self, name: str, *, reference_zero: bool = False) -> float:
        coefficients = self.document["coefficients_rad"]
        if name not in coefficients:
            if reference_zero:
                return 0.0
            raise ValueError(f"{self.path}: missing coefficient {name}")
        value = float(coefficients[name])
        if not math.isfinite(value):
            raise ValueError(f"{self.path}: non-finite coefficient {name}")
        return value

    def _gain_effect(self, receiver: str, gain_index: int) -> float:
        reference_index = self.gains_db.index(self.reference_gain_db)
        return self._coefficient(
            f"{receiver}_phase[{gain_index}]",
            reference_zero=gain_index == reference_index,
        )

    def _frequency_gain_effect(
        self, frequency_index: int, receiver: str, gain_index: int
    ) -> float:
        reference_index = self.gains_db.index(self.reference_gain_db)
        return self._coefficient(
            f"frequency[{frequency_index}].{receiver}_phase[{gain_index}]",
            reference_zero=gain_index == reference_index,
        )

    def _delay_slope(self, receiver: str, gain_index: int) -> float:
        reference_index = self.gains_db.index(self.reference_gain_db)
        return self._coefficient(
            f"{receiver}_delay_slope[{gain_index}]",
            reference_zero=gain_index == reference_index,
        )

    def canonicalize_frequency_hz(
        self,
        frequency_hz: int,
        *,
        allow_float32_alias: bool = False,
    ) -> int:
        """Resolve an exact model LO, optionally recovering a float32 alias.

        Some historical datasets stored integer-Hz LOs in float32 fields.
        For example, 2,467,100,000 Hz becomes 2,467,099,904 when represented
        as float32. Alias recovery only accepts that exact representation of a
        fitted frequency; it is not a frequency tolerance or interpolation.
        """

        frequency_hz = int(frequency_hz)
        if frequency_hz in self.frequencies_hz or not allow_float32_alias:
            return frequency_hz
        aliases = [
            candidate
            for candidate in self.frequencies_hz
            if int(np.float32(candidate)) == frequency_hz
        ]
        if len(aliases) == 1:
            return aliases[0]
        if len(aliases) > 1:
            raise UnsupportedPhaseModelInput(
                f"float32 frequency alias {frequency_hz} is ambiguous: {aliases}"
            )
        return frequency_hz

    def predict_phase_offset(
        self,
        *,
        frequency_hz: int,
        gain_rx1_db: int,
        gain_rx2_db: int,
        strict: bool = True,
        allow_float32_frequency_alias: bool = False,
    ) -> float:
        """Predict wrapped RX1-minus-RX2 phase offset in radians.

        Strict mode is the operational default: the exact
        ``(frequency, RX1 gain, RX2 gain)`` cell must have passed the source
        dataset's three-epoch quality gate. Setting ``strict=False`` exposes
        the mathematical model support for diagnostics.
        """

        frequency_hz = self.canonicalize_frequency_hz(
            frequency_hz,
            allow_float32_alias=allow_float32_frequency_alias,
        )
        gain_rx1_db = int(gain_rx1_db)
        gain_rx2_db = int(gain_rx2_db)
        if frequency_hz <= 0:
            raise UnsupportedPhaseModelInput("frequency must be positive")
        try:
            gain1_index = self.gains_db.index(gain_rx1_db)
            gain2_index = self.gains_db.index(gain_rx2_db)
        except ValueError as error:
            raise UnsupportedPhaseModelInput(
                f"unsupported ordered gains ({gain_rx1_db}, {gain_rx2_db}); "
                f"available gains are {list(self.gains_db)}"
            ) from error

        frequency_index = (
            self.frequencies_hz.index(frequency_hz)
            if frequency_hz in self.frequencies_hz
            else None
        )
        coordinate = (frequency_hz, gain_rx1_db, gain_rx2_db)
        if strict and coordinate not in self.supported_cells:
            raise UnsupportedPhaseModelInput(
                f"{self.serial}/{self.model_name} has no validated support for "
                f"{coordinate}"
            )
        if (
            frequency_index is None
            and not self.document["can_predict_unseen_frequency"]
        ):
            raise UnsupportedPhaseModelInput(
                f"model requires one of the fitted frequencies "
                f"{list(self.frequencies_hz)}"
            )

        coefficients = self.document["coefficients_rad"]
        kind = self.kind
        frequency_offset_ghz = (frequency_hz - self.reference_frequency_hz) / 1e9
        if kind == "constant":
            prediction = self._coefficient("intercept")
        elif kind == "gain_linear":
            prediction = (
                self._coefficient("intercept")
                + self._coefficient("rx1_rad_per_20db") * gain_rx1_db / 20.0
                + self._coefficient("rx2_rad_per_20db") * gain_rx2_db / 20.0
            )
        elif kind == "gain_additive":
            prediction = (
                self._coefficient("intercept")
                + self._gain_effect("rx1", gain1_index)
                + self._gain_effect("rx2", gain2_index)
            )
        elif kind == "frequency_lut_gain_linear":
            assert frequency_index is not None
            prediction = (
                self._coefficient(f"frequency_intercept[{frequency_index}]")
                + self._coefficient("rx1_rad_per_20db") * gain_rx1_db / 20.0
                + self._coefficient("rx2_rad_per_20db") * gain_rx2_db / 20.0
            )
        elif kind == "frequency_lut_gain_additive":
            assert frequency_index is not None
            prediction = (
                self._coefficient(f"frequency_intercept[{frequency_index}]")
                + self._gain_effect("rx1", gain1_index)
                + self._gain_effect("rx2", gain2_index)
            )
        elif kind == "frequency_specific_gain_additive":
            assert frequency_index is not None
            prediction = (
                self._coefficient(f"frequency[{frequency_index}].intercept")
                + self._frequency_gain_effect(frequency_index, "rx1", gain1_index)
                + self._frequency_gain_effect(frequency_index, "rx2", gain2_index)
            )
        elif kind == "full_cell":
            assert frequency_index is not None
            prediction = self._coefficient(
                f"cell[{frequency_index},{gain1_index},{gain2_index}]"
            )
        elif kind == "delay_gain_additive":
            prediction = (
                self._coefficient("intercept")
                + self._coefficient("frequency_rad_per_ghz") * frequency_offset_ghz
                + self._gain_effect("rx1", gain1_index)
                + self._gain_effect("rx2", gain2_index)
            )
        elif kind == "branch_gain_delay":
            prediction = (
                self._coefficient("intercept")
                + self._coefficient("frequency_rad_per_ghz") * frequency_offset_ghz
                + self._gain_effect("rx1", gain1_index)
                + self._gain_effect("rx2", gain2_index)
                + self._delay_slope("rx1", gain1_index) * frequency_offset_ghz
                + self._delay_slope("rx2", gain2_index) * frequency_offset_ghz
            )
        else:
            raise ValueError(f"{self.path}: unsupported model kind {kind!r}")

        if not coefficients:
            raise ValueError(f"{self.path}: model has no coefficients")
        return float(wrap_phase(prediction))

    def correct_measured_phase(
        self,
        measured_phase_rad: float,
        *,
        frequency_hz: int,
        gain_rx1_db: int,
        gain_rx2_db: int,
        strict: bool = True,
        allow_float32_frequency_alias: bool = False,
    ) -> float:
        """Subtract the predicted system offset from a measured phase."""

        offset = self.predict_phase_offset(
            frequency_hz=frequency_hz,
            gain_rx1_db=gain_rx1_db,
            gain_rx2_db=gain_rx2_db,
            strict=strict,
            allow_float32_frequency_alias=allow_float32_frequency_alias,
        )
        return float(wrap_phase(float(measured_phase_rad) - offset))


def load_phase_model(path: Path | str) -> PhaseOffsetModel:
    """Load and validate one exported serial-specific JSON model."""

    path = Path(path).resolve()
    document = _read_document(path)
    if document.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"{path}: unsupported phase-model schema")
    if document.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported phase-model schema version")
    if document.get("scope") != "per_radio":
        raise ValueError(f"{path}: runtime registry requires a per-radio model")
    serial = str(document.get("radio_serial", ""))
    if not serial:
        raise ValueError(f"{path}: missing radio serial")
    coefficients = document.get("coefficients_rad")
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError(f"{path}: coefficients_rad must be a non-empty object")

    support_reference = document.get("support_profile")
    if not isinstance(support_reference, dict):
        raise ValueError(f"{path}: missing support profile")
    support_path = (path.parent / support_reference["path"]).resolve()
    expected_sha = str(support_reference.get("sha256", ""))
    if _sha256(support_path) != expected_sha:
        raise ValueError(f"{path}: support profile hash mismatch")
    support = _read_document(support_path)
    if (
        support.get("schema") != SUPPORT_SCHEMA
        or support.get("schema_version") != SUPPORT_SCHEMA_VERSION
    ):
        raise ValueError(f"{path}: unsupported support-profile schema")
    if support.get("radio_serial") != serial:
        raise ValueError(f"{path}: support profile belongs to another radio")
    if tuple(support.get("frequencies_hz", ())) != tuple(
        document.get("frequencies_hz", ())
    ):
        raise ValueError(f"{path}: support frequencies do not match model")
    if tuple(support.get("gains_db", ())) != tuple(document.get("gains_db", ())):
        raise ValueError(f"{path}: support gains do not match model")
    support_kind = support.get("support_kind", "explicit_cells")
    if support_kind == "explicit_cells":
        supported_rows = support.get("supported_cells", ())
        supported_cells = frozenset(
            (int(row[0]), int(row[1]), int(row[2])) for row in supported_rows
        )
        if len(supported_cells) != len(supported_rows):
            raise ValueError(f"{path}: duplicate supported cells")
    elif support_kind == "cartesian_product":
        supported_cells = frozenset(
            itertools.product(
                (int(value) for value in support["frequencies_hz"]),
                (int(value) for value in support["gains_db"]),
                (int(value) for value in support["gains_db"]),
            )
        )
        if int(support.get("supported_cell_count", -1)) != len(supported_cells):
            raise ValueError(f"{path}: cartesian support count mismatch")
    else:
        raise ValueError(f"{path}: unsupported support kind {support_kind!r}")
    model = PhaseOffsetModel(
        path=path,
        document=document,
        supported_cells=supported_cells,
    )
    if model.reference_gain_db not in model.gains_db:
        raise ValueError(f"{path}: reference gain is not in configured gains")
    return model


def load_model(
    model_name: str,
    serial: str,
    *,
    registry_root: Path | str | None = None,
) -> PhaseOffsetModel:
    """Load ``models/<model_name>/<serial>.json`` from the registry."""

    root = (
        Path(__file__).resolve().parent
        if registry_root is None
        else Path(registry_root).resolve()
    )
    return load_phase_model(root / str(model_name) / f"{serial}.json")
