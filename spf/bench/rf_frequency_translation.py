"""Pure scheduling and spectral checks for closed-loop RF translation tests."""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Iterable

import numpy as np

DEFAULT_EMITTED_FREQUENCIES_HZ = (
    433_920_000,
    1_200_000_000,
    2_450_000_000,
    5_800_000_000,
)
DEFAULT_RX_LO_OFFSETS_HZ = (-900_000, -350_000, 225_000, 800_000)
DEFAULT_DDS_OFFSET_HZ = 100_000


@dataclasses.dataclass(frozen=True, slots=True)
class TranslationCell:
    emitted_frequency_hz: int
    tx_lo_hz: int
    dds_offset_hz: int
    rx_lo_offset_hz: int
    rx_lo_hz: int
    expected_if_hz: int


def parse_hz_list(value: str, *, signed: bool = False) -> tuple[int, ...]:
    """Parse a unique comma-separated list with optional K/M/G suffixes."""

    values: list[int] = []
    scale = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    for raw_token in value.split(","):
        token = raw_token.strip().lower()
        if not token:
            raise ValueError("frequency list contains an empty value")
        multiplier = scale.get(token[-1], 1)
        number = token[:-1] if multiplier != 1 else token
        try:
            parsed = round(float(number) * multiplier)
        except ValueError as error:
            raise ValueError(f"invalid frequency: {raw_token!r}") from error
        if (not signed and parsed <= 0) or (signed and parsed == 0):
            qualifier = "non-zero" if signed else "positive"
            raise ValueError(f"frequencies must be {qualifier}")
        values.append(parsed)
    if not values:
        raise ValueError("at least one frequency is required")
    if len(set(values)) != len(values):
        raise ValueError("frequencies must be unique")
    return tuple(values)


def build_translation_cells(
    emitted_frequencies_hz: Iterable[int] = DEFAULT_EMITTED_FREQUENCIES_HZ,
    rx_lo_offsets_hz: Iterable[int] = DEFAULT_RX_LO_OFFSETS_HZ,
    *,
    dds_offset_hz: int = DEFAULT_DDS_OFFSET_HZ,
    sample_rate_hz: int = 3_000_000,
    search_width_hz: int = 25_000,
    shuffle_seed: int | None = None,
) -> tuple[TranslationCell, ...]:
    """Build cells where RF stays fixed while the receive LO moves.

    ``rx_lo_offset_hz`` is relative to the emitted RF frequency, therefore the
    signed complex baseband expectation is its negation.
    """

    emitted = tuple(int(value) for value in emitted_frequencies_hz)
    offsets = tuple(int(value) for value in rx_lo_offsets_hz)
    if not emitted or len(set(emitted)) != len(emitted):
        raise ValueError("emitted frequencies must be non-empty and unique")
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("RX LO offsets must be non-empty and unique")
    if not any(value < 0 for value in offsets) or not any(
        value > 0 for value in offsets
    ):
        raise ValueError("RX LO offsets must exercise both sides of the emitter")
    if dds_offset_hz == 0:
        raise ValueError("DDS offset must be non-zero to avoid LO leakage")
    if sample_rate_hz <= 0 or search_width_hz <= 0:
        raise ValueError("sample rate and search width must be positive")
    nyquist_hz = sample_rate_hz / 2
    if any(abs(value) + search_width_hz >= nyquist_hz for value in offsets):
        raise ValueError("every expected IF search window must fit inside Nyquist")

    carrier_order = list(emitted)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(carrier_order)
    cells: list[TranslationCell] = []
    for carrier_index, emitted_frequency_hz in enumerate(carrier_order):
        if emitted_frequency_hz <= 0:
            raise ValueError("emitted frequencies must be positive")
        offset_order = list(offsets)
        if shuffle_seed is not None:
            random.Random(shuffle_seed + carrier_index + 1).shuffle(offset_order)
        tx_lo_hz = emitted_frequency_hz - dds_offset_hz
        for rx_lo_offset_hz in offset_order:
            rx_lo_hz = emitted_frequency_hz + rx_lo_offset_hz
            if tx_lo_hz <= 0 or rx_lo_hz <= 0:
                raise ValueError("planned TX and RX LOs must be positive")
            cells.append(
                TranslationCell(
                    emitted_frequency_hz=emitted_frequency_hz,
                    tx_lo_hz=tx_lo_hz,
                    dds_offset_hz=dds_offset_hz,
                    rx_lo_offset_hz=rx_lo_offset_hz,
                    rx_lo_hz=rx_lo_hz,
                    expected_if_hz=-rx_lo_offset_hz,
                )
            )
    return tuple(cells)


def spectral_dominance(
    signal_matrix: np.ndarray,
    *,
    sample_rate_hz: int,
    expected_if_hz: float,
    search_width_hz: float = 25_000,
    dc_exclusion_hz: float = 25_000,
) -> dict[str, float]:
    """Compare expected signed IF energy with images and unrelated peaks."""

    signal = np.asarray(signal_matrix)
    if signal.ndim != 2 or signal.shape[0] != 2 or signal.shape[1] < 256:
        raise ValueError("signal_matrix must have shape (2, samples>=256)")
    if not np.isfinite(signal).all():
        raise ValueError("signal_matrix contains non-finite values")
    if abs(expected_if_hz) + search_width_hz >= sample_rate_hz / 2:
        raise ValueError("expected IF search window must fit inside Nyquist")

    signal = signal.astype(np.complex128, copy=False)
    signal = signal - np.mean(signal, axis=1, keepdims=True)
    window = np.hanning(signal.shape[1])
    spectrum = np.fft.fft(signal * window[None, :], axis=1)
    frequencies = np.fft.fftfreq(signal.shape[1], d=1 / sample_rate_hz)
    power = np.sum(np.abs(spectrum) ** 2, axis=0)
    expected_mask = np.abs(frequencies - expected_if_hz) <= search_width_hz
    mirror_mask = np.abs(frequencies + expected_if_hz) <= search_width_hz
    other_mask = ~expected_mask & (np.abs(frequencies) > dc_exclusion_hz)
    if not expected_mask.any() or not mirror_mask.any() or not other_mask.any():
        raise ValueError("spectral comparison masks are empty")

    expected_index = int(np.flatnonzero(expected_mask)[np.argmax(power[expected_mask])])
    mirror_index = int(np.flatnonzero(mirror_mask)[np.argmax(power[mirror_mask])])
    other_index = int(np.flatnonzero(other_mask)[np.argmax(power[other_mask])])
    floor = np.finfo(np.float64).tiny

    def ratio_db(numerator: float, denominator: float) -> float:
        return 10 * math.log10(max(numerator, floor) / max(denominator, floor))

    expected_power = float(power[expected_index])
    return {
        "expected_peak_hz": float(frequencies[expected_index]),
        "mirror_peak_hz": float(frequencies[mirror_index]),
        "strongest_other_peak_hz": float(frequencies[other_index]),
        "mirror_rejection_db": ratio_db(expected_power, float(power[mirror_index])),
        "global_dominance_db": ratio_db(expected_power, float(power[other_index])),
    }
