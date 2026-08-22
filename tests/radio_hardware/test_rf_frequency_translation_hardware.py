"""Opt-in fixed-emitter, swept-RX-LO signed-IF hardware qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from spf.bench.rf_frequency_translation import parse_hz_list
from spf.scripts.rf_frequency_translation_burn import UsbRadio, run_campaign

pytestmark = [
    pytest.mark.radio_hardware,
    pytest.mark.radio_tandem_agc,
    pytest.mark.radio_tx_loopback,
    pytest.mark.radio_frequency_translation,
]


def _selected_radios(attached_plutos) -> tuple[UsbRadio, ...]:
    return tuple(
        UsbRadio(
            serial=radio.serial,
            bus=radio.bus,
            address=radio.address,
            port_path=radio.port_path,
        )
        for radio in attached_plutos
    )


def _options(pytestconfig) -> dict:
    return {
        "emitted_frequencies_hz": parse_hz_list(
            pytestconfig.getoption("--radio-frequency-translation-carriers")
        ),
        "rx_lo_offsets_hz": parse_hz_list(
            pytestconfig.getoption("--radio-frequency-translation-offsets"),
            signed=True,
        ),
        "tx_gain_db": pytestconfig.getoption("--radio-tx-gain-db"),
        "physical_attenuation_db": pytestconfig.getoption(
            "--radio-tx-loopback-attenuation-db"
        ),
    }


def test_fixed_emitter_swept_rx_lo_signed_if_matrix(
    attached_plutos, pytestconfig, radio_report_dir: Path
) -> None:
    report = run_campaign(
        _selected_radios(attached_plutos),
        report_path=radio_report_dir / "rf-frequency-translation-matrix.json",
        duration_seconds=0,
        **_options(pytestconfig),
    )
    assert report["outcome"] == "pass"
    assert report["epochs_completed"] == 1


@pytest.mark.radio_soak
def test_fixed_emitter_swept_rx_lo_signed_if_burn(
    attached_plutos, pytestconfig, radio_report_dir: Path
) -> None:
    report = run_campaign(
        _selected_radios(attached_plutos),
        report_path=radio_report_dir / "rf-frequency-translation-burn.json",
        duration_seconds=pytestconfig.getoption(
            "--radio-frequency-translation-duration-seconds"
        ),
        **_options(pytestconfig),
    )
    assert report["outcome"] == "pass"
    assert report["elapsed_seconds"] >= pytestconfig.getoption(
        "--radio-frequency-translation-duration-seconds"
    )
