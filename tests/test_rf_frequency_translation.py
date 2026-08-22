import numpy as np
import pytest

from spf.bench.rf_frequency_translation import (
    DEFAULT_EMITTED_FREQUENCIES_HZ,
    DEFAULT_RX_LO_OFFSETS_HZ,
    build_translation_cells,
    parse_hz_list,
    spectral_dominance,
)


def test_translation_matrix_keeps_emitter_fixed_and_preserves_signed_if() -> None:
    cells = build_translation_cells()
    assert len(cells) == 16
    assert {cell.emitted_frequency_hz for cell in cells} == set(
        DEFAULT_EMITTED_FREQUENCIES_HZ
    )
    assert {cell.rx_lo_offset_hz for cell in cells} == set(DEFAULT_RX_LO_OFFSETS_HZ)
    for cell in cells:
        assert cell.tx_lo_hz + cell.dds_offset_hz == cell.emitted_frequency_hz
        assert cell.emitted_frequency_hz - cell.rx_lo_hz == cell.expected_if_hz


def test_translation_schedule_shuffle_is_repeatable_and_complete() -> None:
    first = build_translation_cells(shuffle_seed=7)
    second = build_translation_cells(shuffle_seed=7)
    third = build_translation_cells(shuffle_seed=8)
    assert first == second
    assert first != third
    assert set(first) == set(third)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("433.92M,1.2G", (433_920_000, 1_200_000_000)),
        ("-900K,225K", (-900_000, 225_000)),
    ],
)
def test_parse_hz_list(value: str, expected: tuple[int, ...]) -> None:
    assert parse_hz_list(value, signed=value.startswith("-")) == expected


def test_translation_matrix_rejects_one_sided_or_out_of_band_offsets() -> None:
    with pytest.raises(ValueError, match="both sides"):
        build_translation_cells(rx_lo_offsets_hz=(-900_000, -350_000))
    with pytest.raises(ValueError, match="Nyquist"):
        build_translation_cells(rx_lo_offsets_hz=(-1_490_000, 225_000))


def test_spectral_dominance_distinguishes_signed_tone_from_image() -> None:
    sample_rate_hz = 3_000_000
    sample_index = np.arange(65_536)
    expected_hz = -225_000
    wanted = np.exp(2j * np.pi * expected_hz * sample_index / sample_rate_hz)
    image = 0.1 * np.exp(-2j * np.pi * expected_hz * sample_index / sample_rate_hz)
    signal = np.vstack((wanted + image, 0.8j * (wanted + image)))
    result = spectral_dominance(
        signal,
        sample_rate_hz=sample_rate_hz,
        expected_if_hz=expected_hz,
    )
    assert result["expected_peak_hz"] == pytest.approx(expected_hz, abs=50)
    assert result["mirror_rejection_db"] == pytest.approx(20, abs=0.2)
    assert result["global_dominance_db"] > 19
