"""Explicit TX2-to-RX1/RX2 protocol-v3 release-candidate gate.

This module transmits. It remains skipped unless both ``--radio-hardware`` and
``--radio-tx-loopback`` are present, and it additionally requires the operator
to state the physical attenuation on the command line.
"""

from __future__ import annotations

import json
import math
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
TX_PIPELINE_DEBUG_SELECT = 1 << 0
TX_PIPELINE_DMA_FLAGS = (
    "transfer_request_seen",
    "upstream_valid_seen",
    "timestamp_enabled_seen",
    "fifo_write_seen",
    "fifo_full_seen",
    "fifo_write_reset_busy_seen",
    "fifo_write_possible_seen",
    "fifo_reset_released_seen",
)
TX_PIPELINE_DAC_FLAGS = (
    "downstream_ready_seen",
    "fifo_read_seen",
    "downstream_valid_seen",
    "fifo_nonempty_seen",
    "fifo_read_reset_busy_seen",
    "fifo_read_possible_seen",
    "transfer_start_seen",
    "upack_reset_released_seen",
)


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


def _decode_tx_pipeline_debug(value: int) -> dict:
    dma = (value >> 24) & 0xFF
    dac = (value >> 16) & 0xFF
    return {
        "raw": value,
        "dma_raw": dma,
        "dac_raw": dac,
        "timestamp_discard_count_low16": value & 0xFFFF,
        "dma": {
            name: bool(dma & (1 << bit))
            for bit, name in enumerate(TX_PIPELINE_DMA_FLAGS)
        },
        "dac": {
            name: bool(dac & (1 << bit))
            for bit, name in enumerate(TX_PIPELINE_DAC_FLAGS)
        },
    }


def _tx_core_diagnostics(radio: DirectUsbLoopbackRadio) -> dict:
    """Capture TX core state, including RC7's selectable debug page.

    DAC GPIO output bit 0 is unused by production firmware. RC7 routes it to
    a diagnostics mux only; restoring the original register value in ``finally``
    keeps this probe observational even when a read fails.
    """

    try:
        device = radio.sdr._ctx.find_device("cf-ad9361-dds-core-lpc")
        if device is None:
            return {"error": "cf-ad9361-dds-core-lpc not found"}
        result = {
            name: int(device.reg_read(address))
            for name, address in TX_CORE_REGISTERS.items()
        }
        original_control = result["timestamp_interval_control"]
        try:
            device.reg_write(
                TX_CORE_REGISTERS["timestamp_interval_control"],
                original_control | TX_PIPELINE_DEBUG_SELECT,
            )
            debug_value = int(
                device.reg_read(TX_CORE_REGISTERS["timestamp_discard_count"])
            )
            result["tx_pipeline_debug"] = _decode_tx_pipeline_debug(debug_value)
        except Exception as error:
            result["tx_pipeline_debug_error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                device.reg_write(
                    TX_CORE_REGISTERS["timestamp_interval_control"],
                    original_control,
                )
            except Exception as error:
                result[
                    "tx_pipeline_debug_restore_error"
                ] = f"{type(error).__name__}: {error}"
        return result
    except Exception as error:  # Diagnostic evidence must not replace RF QC.
        return {"error": f"{type(error).__name__}: {error}"}


def _write_report(
    radio_report_dir,
    report: dict,
    filename: str = "gain_series_v3_tx2_loopback.json",
) -> None:
    (radio_report_dir / filename).write_text(json.dumps(report, indent=2) + "\n")


def _gain_values(metadata: RadioMetadataV3) -> np.ndarray:
    return np.asarray(
        [[item.rx1_gain_db, item.rx2_gain_db] for item in metadata.gain_observations],
        dtype=np.float64,
    )


def _single_channel_tone_metrics(
    signal: np.ndarray,
    *,
    sample_rate_hz: int,
    tone_hz: int,
    transient_samples: int,
) -> dict:
    """Measure a digitally looped tone without requiring both RX channels."""

    raw = signal[:, transient_samples:].astype(np.complex128, copy=False)
    raw = raw - np.mean(raw, axis=1, keepdims=True)
    sample_index = np.arange(raw.shape[1], dtype=np.float64)
    carrier = np.exp(2j * np.pi * tone_hz * sample_index / sample_rate_hz)
    coefficient = np.mean(raw * np.conj(carrier)[None, :], axis=1)
    fitted = coefficient[:, None] * carrier[None, :]
    residual_power = np.mean(np.abs(raw - fitted) ** 2, axis=1)
    tone_power = np.abs(coefficient) ** 2
    numerical_floor = np.finfo(np.float64).tiny
    tone_dbfs = 10.0 * np.log10(
        np.maximum(tone_power, numerical_floor) / float(2048**2)
    )
    tone_snr_db = 10.0 * np.log10(
        np.maximum(tone_power, numerical_floor)
        / np.maximum(residual_power, numerical_floor)
    )
    strongest_channel = int(np.argmax(tone_power))
    phase_step = np.angle(
        np.sum(raw[strongest_channel, 1:] * np.conj(raw[strongest_channel, :-1]))
    )
    measured_frequency_hz = float(phase_step * sample_rate_hz / (2.0 * np.pi))
    return {
        "tone_dbfs": tone_dbfs.tolist(),
        "tone_snr_db": tone_snr_db.tolist(),
        "strongest_channel": strongest_channel,
        "measured_frequency_hz": measured_frequency_hz,
        "frequency_error_hz": measured_frequency_hz - tone_hz,
    }


def test_v3_cyclic_tx_reaches_dac_with_timestamping_disabled(
    attached_plutos, pytestconfig, radio_report_dir
):
    """Exercise the DMA/timestamp-FIFO path that FPGA DDS bypasses."""

    attenuation = pytestconfig.getoption("--radio-tx-loopback-attenuation-db")
    if attenuation is None or attenuation < MINIMUM_LOOPBACK_ATTENUATION_DB:
        pytest.fail(
            "cyclic TX requires an explicitly declared attenuated loopback of "
            f"at least {MINIMUM_LOOPBACK_ATTENUATION_DB:g} dB"
        )

    samples = pytestconfig.getoption("--radio-tx-samples")
    sample_rate_hz = pytestconfig.getoption("--radio-tx-sample-rate")
    bandwidth_hz = pytestconfig.getoption("--radio-tx-bandwidth")
    lo_hz = pytestconfig.getoption("--radio-tx-lo-hz")
    tone_hz = int(pytestconfig.getoption("--radio-tx-tone-hz"))
    nominal_tx_gain = pytestconfig.getoption("--radio-tx-gain-db")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    period_samples = sample_rate_hz // math.gcd(sample_rate_hz, abs(tone_hz))
    waveform_samples = period_samples * math.ceil(16_384 / period_samples)
    waveform_index = np.arange(waveform_samples, dtype=np.float64)
    waveform = (
        8192 * np.exp(2j * np.pi * tone_hz * waveform_index / sample_rate_hz)
    ).astype(np.complex64)
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
        setup_label="protocol_v3_internal_cyclic_tx",
    )
    report = {
        "attenuation_db": attenuation,
        "lo_hz": lo_hz,
        "sample_rate_hz": sample_rate_hz,
        "tone_hz": tone_hz,
        "waveform_samples": waveform_samples,
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
            loopback = radio.sdr._ctrl.debug_attrs["loopback"]
            original_loopback = loopback.value
            try:
                radio.configure_frequency(lo_hz, start_tone=False)
                radio.set_gains(*MANUAL_GAINS_DB)
                radio.run_tx_quadrature_calibration()
                loopback.value = "1"
                radio.sdr.disable_dds()
                radio.sdr.tx_destroy_buffer()
                radio.sdr.tx_cyclic_buffer = True
                radio.sdr.tx_enabled_channels = [1]
                radio.sdr.tx_hardwaregain_chan0 = -80
                radio.sdr.tx_hardwaregain_chan1 = nominal_tx_gain
                radio.sdr.tx(waveform)
                radio._prime_iio_rx_dma()
                time.sleep(0.25)
                frame = _capture_frames(radio, samples, 1)[0]
                metadata = _validate_metadata(frame, samples, interval)
                _assert_manual_gain_metadata(metadata)
                metrics = _single_channel_tone_metrics(
                    _signal(frame, samples),
                    sample_rate_hz=sample_rate_hz,
                    tone_hz=tone_hz,
                    transient_samples=config.transient_samples,
                )
                diagnostics = _tx_core_diagnostics(radio)
                result = {
                    "serial": attached.serial,
                    "port_path": list(attached.port_path),
                    "tone_metrics": metrics,
                    "tx_core_diagnostics": diagnostics,
                }
                report["radios"].append(result)
                _write_report(
                    radio_report_dir,
                    report,
                    "gain_series_v3_internal_cyclic_tx.json",
                )

                strongest = metrics["strongest_channel"]
                assert metrics["tone_dbfs"][strongest] >= -25.0, metrics
                assert metrics["tone_snr_db"][strongest] >= 20.0, metrics
                assert abs(metrics["frequency_error_hz"]) <= 250.0, metrics
                assert diagnostics["timestamp_interval_control"] == 0
                assert diagnostics["timestamp_discard_count"] == 0, diagnostics
                debug = diagnostics["tx_pipeline_debug"]
                required_dma = (
                    "transfer_request_seen",
                    "upstream_valid_seen",
                    "fifo_write_seen",
                    "fifo_write_possible_seen",
                    "fifo_reset_released_seen",
                )
                required_dac = (
                    "downstream_ready_seen",
                    "fifo_read_seen",
                    "downstream_valid_seen",
                    "fifo_nonempty_seen",
                    "fifo_read_possible_seen",
                    "transfer_start_seen",
                    "upack_reset_released_seen",
                )
                assert all(debug["dma"][name] for name in required_dma), debug
                assert all(debug["dac"][name] for name in required_dac), debug
                assert not debug["dma"]["timestamp_enabled_seen"], debug
            finally:
                try:
                    radio.sdr.tx_destroy_buffer()
                    radio.sdr.tx_enabled_channels = []
                    radio.sdr.tx_hardwaregain_chan0 = -80
                    radio.sdr.tx_hardwaregain_chan1 = -80
                    radio.sdr.tx_cyclic_buffer = False
                finally:
                    loopback.value = original_loopback

    _write_report(
        radio_report_dir,
        report,
        "gain_series_v3_internal_cyclic_tx.json",
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
        # The cabled AD9361 DDS path can be spur-limited near 10 dB even when
        # its carrier, frequency, coherence, and phase stability are excellent.
        # Keep those independent gates strict; noise-only failures observed in
        # this campaign were below -38 dB SNR and remain far outside this bound.
        min_tone_snr_db=10.0,
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
