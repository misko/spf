"""Explicit TX2-to-RX1/RX2 protocol-v3 release-candidate gate.

This module transmits. It remains skipped unless both ``--radio-hardware`` and
``--radio-tx-loopback`` are present, and it additionally requires the operator
to state the physical attenuation on the command line.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from spf.bench.dual_rx_phase import ToneQualityThresholds, analyze_common_tone
from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio
from spf.scripts.mute_pluto_tx import mute_attached_plutos
from spf.sdrpluto.direct_usb_protocol import (
    GainObservationFlags,
    MetadataFlags,
    RadioMetadataV3,
)
from spf.sdrpluto.direct_usb_receiver import iq_payload_to_complex64


pytestmark = [
    pytest.mark.radio_hardware,
    pytest.mark.radio_gain_series_v3,
    pytest.mark.radio_tx_loopback,
]

MINIMUM_LOOPBACK_ATTENUATION_DB = 30.0
MANUAL_GAINS_DB = (20, 35)
TX_CORE_REGISTERS = {
    "version": 0x00,
    "control": 0x40,
    "configuration": 0x44,
    "status": 0x5C,
    "underflow": 0x88,
    "timestamp_discard_count": 0xB8,
    "timestamp_interval_control": 0xBC,
}


@pytest.fixture(autouse=True)
def tx_safety_guard(attached_plutos):
    serials = [radio.serial for radio in attached_plutos]
    mute_attached_plutos(serials=serials, expected_count=len(serials))
    try:
        yield
    finally:
        mute_attached_plutos(serials=serials, expected_count=len(serials))


def _capture_frames(radio: DirectUsbLoopbackRadio, samples: int, count: int):
    frames = list(
        radio.direct.stream_frames(
            samples_per_channel=samples,
            frame_count=count,
            queue_depth=1,
        )
    )
    assert len(frames) == count
    assert [frame.metadata.buffer_sequence for frame in frames] == list(range(count))
    starts = [frame.metadata.first_sample_sequence for frame in frames]
    assert all(right - left == samples for left, right in zip(starts, starts[1:]))
    return frames


def _validate_metadata(frame, samples: int, interval: int) -> RadioMetadataV3:
    metadata = frame.metadata
    assert isinstance(metadata, RadioMetadataV3)
    assert metadata.samples_per_channel == samples
    assert metadata.iq_payload_bytes == samples * 8
    assert metadata.gain_observation_interval_samples == interval
    assert metadata.flags & MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
    assert metadata.flags & MetadataFlags.GAIN_OBSERVATIONS_VALID
    assert not metadata.flags & MetadataFlags.GAIN_OBSERVATION_OVERFLOW
    assert not metadata.flags & MetadataFlags.DEVICE_IIO_OVERFLOW
    assert metadata.gain_observation_overflow_count == 0
    assert metadata.gain_observations
    for observation in metadata.gain_observations:
        required = (
            GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID
        )
        assert observation.flags & required == required
        assert 0 <= observation.rx1_gain_index <= 0x7F
        assert 0 <= observation.rx2_gain_index <= 0x7F
        assert observation.read_duration_ns > 0
    return metadata


def _signal(frame, samples: int) -> np.ndarray:
    signal = iq_payload_to_complex64(frame.iq_payload, samples)
    assert signal.shape == (2, samples)
    assert np.isfinite(signal).all()
    return signal


def _matched_tone_dbfs(
    signal: np.ndarray, *, sample_rate_hz: int, tone_hz: int
) -> np.ndarray:
    raw = signal.astype(np.complex128, copy=False)
    raw = raw - np.mean(raw, axis=1, keepdims=True)
    sample_index = np.arange(raw.shape[1], dtype=np.float64)
    oscillator = np.exp(-2j * np.pi * tone_hz * sample_index / sample_rate_hz)
    amplitude = np.abs(np.mean(raw * oscillator[None, :], axis=1))
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(amplitude, 1e-15) / 2048.0)


def _assert_manual_gain_metadata(metadata: RadioMetadataV3) -> None:
    observed = np.asarray(
        [[item.rx1_gain_db, item.rx2_gain_db] for item in metadata.gain_observations],
        dtype=np.float64,
    )
    expected = np.asarray(MANUAL_GAINS_DB, dtype=np.float64)
    np.testing.assert_allclose(
        observed,
        np.broadcast_to(expected, observed.shape),
        atol=1.0,
    )
    np.testing.assert_allclose(metadata.gain_db_start, expected, atol=1.0)
    np.testing.assert_allclose(metadata.gain_db_end, expected, atol=1.0)


def _configuration_readback(radio: DirectUsbLoopbackRadio) -> dict:
    sdr = radio.sdr
    return {
        "sample_rate_hz": int(sdr.sample_rate),
        "rx_bandwidth_hz": int(sdr.rx_rf_bandwidth),
        "tx_bandwidth_hz": int(sdr.tx_rf_bandwidth),
        "rx_lo_hz": int(sdr.rx_lo),
        "tx_lo_hz": int(sdr.tx_lo),
        "rx1_gain_mode": str(sdr.gain_control_mode_chan0),
        "rx2_gain_mode": str(sdr.gain_control_mode_chan1),
        "tx1_gain_db": float(sdr.tx_hardwaregain_chan0),
        "tx2_gain_db": float(sdr.tx_hardwaregain_chan1),
    }


def _tx_core_diagnostics(radio: DirectUsbLoopbackRadio) -> dict:
    """Capture read-only TX core state without masking the RF test result."""

    try:
        device = radio.sdr._ctx.find_device("cf-ad9361-dds-core-lpc")
        if device is None:
            return {"error": "cf-ad9361-dds-core-lpc not found"}
        return {
            name: int(device.reg_read(address))
            for name, address in TX_CORE_REGISTERS.items()
        }
    except Exception as error:  # Diagnostic evidence must not replace RF QC.
        return {"error": f"{type(error).__name__}: {error}"}


def _write_report(radio_report_dir, report: dict) -> None:
    (radio_report_dir / "gain_series_v3_tx2_loopback.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def _gain_values(metadata: RadioMetadataV3) -> np.ndarray:
    return np.asarray(
        [[item.rx1_gain_db, item.rx2_gain_db] for item in metadata.gain_observations],
        dtype=np.float64,
    )


def test_v3_tx2_tone_manual_gain_and_slow_attack_agc(
    attached_plutos, pytestconfig, radio_report_dir
):
    attenuation = pytestconfig.getoption("--radio-tx-loopback-attenuation-db")
    if attenuation is None:
        pytest.fail(
            "--radio-tx-loopback requires " "--radio-tx-loopback-attenuation-db"
        )
    if attenuation < MINIMUM_LOOPBACK_ATTENUATION_DB:
        pytest.fail(
            f"declared loopback attenuation {attenuation:g} dB is below the "
            f"{MINIMUM_LOOPBACK_ATTENUATION_DB:g} dB release-test minimum"
        )

    samples = pytestconfig.getoption("--radio-tx-samples")
    sample_rate_hz = pytestconfig.getoption("--radio-tx-sample-rate")
    bandwidth_hz = pytestconfig.getoption("--radio-tx-bandwidth")
    lo_hz = pytestconfig.getoption("--radio-tx-lo-hz")
    tone_hz = pytestconfig.getoption("--radio-tx-tone-hz")
    nominal_tx_gain = pytestconfig.getoption("--radio-tx-gain-db")
    weak_tx_gain = pytestconfig.getoption("--radio-tx-weak-gain-db")
    strong_tx_gain = pytestconfig.getoption("--radio-tx-strong-gain-db")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    if samples < 16_384:
        pytest.fail("--radio-tx-samples must be at least 16384")
    if not -80 <= weak_tx_gain < nominal_tx_gain < strong_tx_gain <= 0:
        pytest.fail("TX gains must satisfy -80 <= weak < nominal < strong <= 0")

    config = CalibrationConfig(
        frequencies_hz=(lo_hz,),
        gains_db=MANUAL_GAINS_DB,
        repetitions=1,
        sample_rate_hz=sample_rate_hz,
        bandwidth_hz=bandwidth_hz,
        buffer_size=samples,
        tone_offset_hz=tone_hz,
        transient_samples=min(1_024, samples // 16),
        phase_segments=8,
        settle_seconds=0.1,
        frequency_settle_seconds=0.25,
        discard_frames_after_gain=0,
        rf_dc_calibration_policy="never",
        require_preflight_tone=False,
        tx_gain_db=nominal_tx_gain,
        min_quality_valid_per_cell=1,
        setup_label="protocol_v3_release_tx2_loopback",
    )
    thresholds = ToneQualityThresholds(
        min_tone_snr_db=15.0,
        min_tone_dbfs=-70.0,
        max_tone_dbfs=-3.0,
        max_clipping_fraction=0.0,
        min_coherence=0.98,
        max_within_capture_phase_std_deg=5.0,
    )
    report = {
        "attenuation_db": attenuation,
        "lo_hz": lo_hz,
        "sample_rate_hz": sample_rate_hz,
        "bandwidth_hz": bandwidth_hz,
        "tone_hz": tone_hz,
        "radios": [],
    }

    for attached in attached_plutos:
        with DirectUsbLoopbackRadio(
            attached.serial,
            config,
            direct_protocol_version=3,
            direct_receiver_options={
                "gain_observation_interval_samples": interval,
                "gain_observation_capacity": capacity,
            },
        ) as radio:
            radio.configure_frequency(lo_hz, start_tone=False)
            radio.set_gains(*MANUAL_GAINS_DB)

            # Prime after DDS arm before the first direct-USB START.  Keep the
            # tone armed for every active measurement: once direct RX has
            # owned the DMA, trying to prime pyadi again can race the gadget's
            # STOP teardown and correctly returns EBUSY.
            radio.start_tone(
                tx_channel=1,
                tx_gain_db=nominal_tx_gain,
                prime_after_arm=True,
            )
            time.sleep(0.25)
            active_frames = _capture_frames(radio, samples, 2)
            active_metadata = _validate_metadata(active_frames[-1], samples, interval)
            _assert_manual_gain_metadata(active_metadata)
            active_signal = _signal(active_frames[-1], samples)
            tone = analyze_common_tone(
                active_signal,
                sample_rate_hz=sample_rate_hz,
                expected_tone_offset_hz=tone_hz,
                tone_search_width_hz=25_000,
                transient_samples=config.transient_samples,
                phase_segments=config.phase_segments,
                thresholds=thresholds,
            )
            radio_result = {
                "serial": attached.serial,
                "port_path": list(attached.port_path),
                "tone_analysis": tone,
                "tx_core_diagnostics_active": _tx_core_diagnostics(radio),
            }
            report["radios"].append(radio_result)
            # Preserve a partial report before the mandatory assertion. A
            # noise-floor failure is precisely when core status is most useful.
            _write_report(radio_report_dir, report)
            assert tone["quality_valid"], json.dumps(tone, sort_keys=True)
            active_tone_dbfs = _matched_tone_dbfs(
                active_signal,
                sample_rate_hz=sample_rate_hz,
                tone_hz=tone_hz,
            )

            manual_readback = _configuration_readback(radio)
            assert manual_readback["sample_rate_hz"] == sample_rate_hz
            assert manual_readback["rx_bandwidth_hz"] == bandwidth_hz
            assert manual_readback["tx_bandwidth_hz"] == bandwidth_hz
            assert abs(manual_readback["rx_lo_hz"] - lo_hz) < 10
            assert abs(manual_readback["tx_lo_hz"] - lo_hz) < 10
            assert manual_readback["rx1_gain_mode"] == "manual"
            assert manual_readback["rx2_gain_mode"] == "manual"
            assert manual_readback["tx1_gain_db"] <= -79.75
            assert abs(manual_readback["tx2_gain_db"] - nominal_tx_gain) <= 0.25

            # A large, safe cabled level step must make slow-attack AGC reduce
            # gain on both channels. This checks the values are live rather
            # than merely well-formed copies in each v3 header.
            radio.sdr.gain_control_mode_chan0 = "slow_attack"
            radio.sdr.gain_control_mode_chan1 = "slow_attack"
            radio.set_tx_gain(weak_tx_gain)
            time.sleep(0.75)
            weak_frame = _capture_frames(radio, samples, 2)[-1]
            weak_metadata = _validate_metadata(weak_frame, samples, interval)
            radio.set_tx_gain(strong_tx_gain)
            time.sleep(0.75)
            strong_frame = _capture_frames(radio, samples, 3)[-1]
            strong_metadata = _validate_metadata(strong_frame, samples, interval)
            weak_gains = _gain_values(weak_metadata)
            strong_gains = _gain_values(strong_metadata)
            gain_reduction = np.median(weak_gains, axis=0) - np.median(
                strong_gains, axis=0
            )
            assert np.all(gain_reduction >= 1.0), {
                "weak": weak_gains.tolist(),
                "strong": strong_gains.tolist(),
            }

            agc_readback = _configuration_readback(radio)
            assert agc_readback["rx1_gain_mode"] == "slow_attack"
            assert agc_readback["rx2_gain_mode"] == "slow_attack"
            assert agc_readback["sample_rate_hz"] == sample_rate_hz
            assert abs(agc_readback["rx_lo_hz"] - lo_hz) < 10

            # Establish the muted floor last, without re-arming DDS or
            # handing RX DMA back to pyadi.  This validates that STOP really
            # suppresses the same +100 kHz path used above.
            radio.stop_tone()
            time.sleep(0.1)
            muted_frame = _capture_frames(radio, samples, 1)[0]
            muted_metadata = _validate_metadata(muted_frame, samples, interval)
            muted_signal = _signal(muted_frame, samples)
            muted_tone_dbfs = _matched_tone_dbfs(
                muted_signal,
                sample_rate_hz=sample_rate_hz,
                tone_hz=tone_hz,
            )
            assert np.all(active_tone_dbfs - muted_tone_dbfs >= 15.0), (
                muted_tone_dbfs,
                active_tone_dbfs,
            )
            radio_result.update(
                {
                    "muted_tone_dbfs": muted_tone_dbfs.tolist(),
                    "active_tone_dbfs": active_tone_dbfs.tolist(),
                    "manual_configuration": manual_readback,
                    "agc_configuration": agc_readback,
                    "weak_gain_db_median": np.median(weak_gains, axis=0).tolist(),
                    "strong_gain_db_median": np.median(strong_gains, axis=0).tolist(),
                    "agc_gain_reduction_db": gain_reduction.tolist(),
                    "tx_core_diagnostics_muted": _tx_core_diagnostics(radio),
                }
            )
            _write_report(radio_report_dir, report)

    _write_report(radio_report_dir, report)
