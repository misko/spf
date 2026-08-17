"""Opt-in single-boot USB/IP, LO-retune, and gain-mode burn-in."""

from __future__ import annotations

import concurrent.futures
import json
import time
from contextlib import ExitStack

import numpy as np
import pytest

from spf.scripts.mute_pluto_tx import validate_loopback_safety

from spf.bench.dual_rx_phase import ToneQualityThresholds, analyze_common_tone
from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio
from spf.sdrpluto.direct_ip_receiver import PlutoDirectIpReceiver
from spf.sdrpluto.direct_usb_protocol import (
    GainObservationFlags,
    MetadataFlags,
    RadioMetadataV3,
)
from spf.sdrpluto.direct_usb_receiver import iq_payload_to_complex64
from spf.scripts.direct_ip_parallel_ladder import direct_ip_identity
from spf.scripts.mixed_transport_frequency_burn import (
    build_burn_schedule,
    parse_frequency_list,
)


pytestmark = [
    pytest.mark.radio_hardware,
    pytest.mark.radio_gain_series_v3,
    pytest.mark.radio_direct_ip,
    pytest.mark.radio_tx_loopback,
    pytest.mark.radio_soak,
]


def _set_gain_state(radio: DirectUsbLoopbackRadio, state: str) -> dict:
    if state.startswith("manual_"):
        gain = int(state.removeprefix("manual_"))
        radio.sdr.gain_control_mode_chan0 = "manual"
        radio.sdr.gain_control_mode_chan1 = "manual"
        radio.set_gains(gain, gain)
        expected = [gain, gain]
    elif state == "slow_attack":
        radio.sdr.gain_control_mode_chan0 = "slow_attack"
        radio.sdr.gain_control_mode_chan1 = "slow_attack"
        expected = None
    else:
        raise ValueError(f"unsupported gain state: {state}")
    observed_modes = [
        radio.sdr.gain_control_mode_chan0,
        radio.sdr.gain_control_mode_chan1,
    ]
    desired_modes = (
        ["manual", "manual"] if expected is not None else ["slow_attack", "slow_attack"]
    )
    assert observed_modes == desired_modes
    time.sleep(0.05)
    return {"state": state, "expected_gain_db": expected, "modes": observed_modes}


def _capture_transport(
    *,
    radio: DirectUsbLoopbackRadio,
    ip_host: str,
    transport: str,
    samples: int,
    frames: int,
):
    if transport == "usb":
        # Two radios x eight 1 MiB transfers exceed Linux's common 16 MiB
        # usbfs allocation once transfer overhead is included. Keep one bulk
        # transfer queued per radio while preserving a single contiguous
        # firmware stream across all requested frames.
        received = tuple(
            radio.direct.stream_frames(
                samples_per_channel=samples,
                frame_count=frames,
                queue_depth=1,
            )
        )
        return received, {
            "duplicate_fragment_count": 0,
            "expired_frame_count": 0,
            "rejected_frame_count": 0,
            "receive_queue_overflow_count": 0,
        }
    if transport != "ip":
        raise ValueError(f"unsupported transport: {transport}")
    with PlutoDirectIpReceiver(
        remote_host=ip_host,
        protocol_version=3,
        gain_observation_interval_samples=2_048,
        gain_observation_capacity=256,
        minimum_effective_receive_buffer_bytes=8 * 1024 * 1024,
    ) as receiver:
        capture = receiver.capture(
            samples_per_channel=samples,
            frame_count=frames,
        )
    return capture.frames, {
        "duplicate_fragment_count": capture.duplicate_fragment_count,
        "expired_frame_count": capture.expired_frame_count,
        "rejected_frame_count": capture.rejected_frame_count,
        "receive_queue_overflow_count": capture.receive_queue_overflow_count,
    }


def _validate_session(
    frames,
    *,
    samples: int,
    frames_expected: int,
    sample_rate_hz: int,
    tone_hz: int,
    expected_gain_db: list[int] | None,
) -> dict:
    assert len(frames) == frames_expected
    assert [frame.metadata.buffer_sequence for frame in frames] == list(
        range(frames_expected)
    )
    first_samples = [frame.metadata.first_sample_sequence for frame in frames]
    assert first_samples == [
        first_samples[0] + index * samples for index in range(frames_expected)
    ]
    analyses = []
    observation_counts = []
    for frame in frames:
        metadata = frame.metadata
        assert isinstance(metadata, RadioMetadataV3)
        assert metadata.samples_per_channel == samples
        assert metadata.flags & MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
        assert metadata.flags & MetadataFlags.GAIN_OBSERVATIONS_VALID
        assert not metadata.flags & MetadataFlags.GAIN_OBSERVATION_OVERFLOW
        assert not metadata.flags & MetadataFlags.DEVICE_IIO_OVERFLOW
        assert metadata.gain_metadata_valid
        assert metadata.rssi_metadata_valid
        assert metadata.gain_observations
        for observation in metadata.gain_observations:
            required = (
                GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID
            )
            assert observation.flags & required == required
        if expected_gain_db is not None:
            np.testing.assert_array_equal(metadata.gain_db_start, expected_gain_db)
            np.testing.assert_array_equal(metadata.gain_db_end, expected_gain_db)
        signal = iq_payload_to_complex64(frame.iq_payload, samples)
        analysis = analyze_common_tone(
            signal,
            sample_rate_hz=sample_rate_hz,
            expected_tone_offset_hz=tone_hz,
            tone_search_width_hz=25_000,
            transient_samples=1_024,
            phase_segments=8,
            thresholds=ToneQualityThresholds(
                min_tone_snr_db=6.0,
                min_tone_dbfs=-75.0,
                max_tone_dbfs=-1.0,
                min_coherence=0.90,
                max_within_capture_phase_std_deg=8.0,
            ),
        )
        assert analysis["quality_valid"], analysis["quality_reasons"]
        analyses.append(analysis)
        observation_counts.append(len(metadata.gain_observations))
    return {
        "stream_id": frames[0].metadata.stream_id,
        "first_sample_sequence": first_samples[0],
        "last_sample_sequence": first_samples[-1] + samples,
        "gain_observation_counts": observation_counts,
        "minimum_tone_snr_db": min(
            value for analysis in analyses for value in analysis["tone_snr_db"]
        ),
        "minimum_coherence": min(analysis["coherence"] for analysis in analyses),
        "maximum_phase_std_deg": max(
            analysis["within_capture_phase_std_deg"] for analysis in analyses
        ),
        "tone_frequency_error_hz": [
            analysis["tone_frequency_error_hz"] for analysis in analyses
        ],
    }


def test_mixed_usb_ip_frequency_and_gain_state_burn(
    attached_plutos, pytestconfig, radio_report_dir
):
    assert len(attached_plutos) == 2, "mixed burn requires exactly two radios"
    attenuation = pytestconfig.getoption("--radio-tx-loopback-attenuation-db")
    frequencies = parse_frequency_list(
        pytestconfig.getoption("--radio-burn-frequencies")
    )
    epochs = pytestconfig.getoption("--radio-cycles")
    frames_per_session = pytestconfig.getoption("--radio-frames-per-request")
    assert 1 <= frames_per_session <= 16
    schedule = build_burn_schedule(frequencies, epochs=epochs)
    hosts = pytestconfig.getoption("--radio-direct-ip-ladder-host")
    assert len(hosts) == 2 and len(set(hosts)) == 2
    host_by_serial = {direct_ip_identity(host)["serial"]: host for host in hosts}
    serials = {radio.serial for radio in attached_plutos}
    assert set(host_by_serial) == serials

    samples = min(pytestconfig.getoption("--radio-samples"), 131_072)
    sample_rate_hz = pytestconfig.getoption("--radio-tx-sample-rate")
    bandwidth_hz = pytestconfig.getoption("--radio-tx-bandwidth")
    tone_hz = pytestconfig.getoption("--radio-tx-tone-hz")
    tx_gain_db = pytestconfig.getoption("--radio-tx-gain-db")
    validate_loopback_safety(
        physical_attenuation_db=attenuation,
        strongest_tx_gain_db=tx_gain_db,
    )
    config = CalibrationConfig(
        frequencies_hz=frequencies,
        gains_db=(26, 41),
        repetitions=1,
        sample_rate_hz=sample_rate_hz,
        bandwidth_hz=bandwidth_hz,
        buffer_size=samples,
        tone_offset_hz=tone_hz,
        frequency_settle_seconds=0.15,
        tx_gain_db=tx_gain_db,
        require_preflight_tone=False,
        min_quality_valid_per_cell=1,
    )
    report = {
        "purpose": "single_boot_mixed_transport_frequency_gain_state_burn",
        "attenuation_db": attenuation,
        "frequencies_hz": list(frequencies),
        "epochs": epochs,
        "frames_per_session": frames_per_session,
        "samples_per_channel": samples,
        "radios": sorted(serials),
        "hosts_by_serial": host_by_serial,
        "steps": [],
    }
    report_path = radio_report_dir / "mixed_transport_frequency_burn.json"

    with ExitStack() as stack:
        radios = {
            attached.serial: stack.enter_context(
                DirectUsbLoopbackRadio(
                    attached.serial,
                    config,
                    direct_protocol_version=3,
                    direct_receiver_options={
                        "gain_observation_interval_samples": 2_048,
                        "gain_observation_capacity": 256,
                    },
                )
            )
            for attached in attached_plutos
        }
        for step_index, step in enumerate(schedule):
            for serial, radio in radios.items():
                radio.configure_frequency(step.frequency_hz, start_tone=False)
                radio.start_tone(
                    tx_channel=1, tx_gain_db=tx_gain_db, prime_after_arm=True
                )
                assert abs(int(radio.sdr.rx_lo) - step.frequency_hz) < 10
                assert abs(int(radio.sdr.tx_lo) - step.frequency_hz) < 10

            step_report = {
                "step": step_index,
                "epoch": step.epoch,
                "frequency_hz": step.frequency_hz,
                "sessions": [],
            }
            report["steps"].append(step_report)
            try:
                for transport, gain_state in zip(
                    step.transports, step.gain_modes, strict=True
                ):
                    gain_by_serial = {
                        serial: _set_gain_state(radio, gain_state)
                        for serial, radio in radios.items()
                    }
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=2
                    ) as executor:
                        futures = {
                            serial: executor.submit(
                                _capture_transport,
                                radio=radio,
                                ip_host=host_by_serial[serial],
                                transport=transport,
                                samples=samples,
                                frames=frames_per_session,
                            )
                            for serial, radio in radios.items()
                        }
                        captures = {
                            serial: future.result()
                            for serial, future in futures.items()
                        }
                    session = {
                        "transport": transport,
                        "gain_state": gain_state,
                        "radios": {},
                    }
                    for serial, (frames, counters) in captures.items():
                        assert not any(counters.values()), counters
                        session["radios"][serial] = {
                            **gain_by_serial[serial],
                            **counters,
                            **_validate_session(
                                frames,
                                samples=samples,
                                frames_expected=frames_per_session,
                                sample_rate_hz=sample_rate_hz,
                                tone_hz=tone_hz,
                                expected_gain_db=gain_by_serial[serial][
                                    "expected_gain_db"
                                ],
                            ),
                        }
                    step_report["sessions"].append(session)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
            finally:
                for radio in radios.values():
                    radio.stop_tone()
                report_path.write_text(json.dumps(report, indent=2) + "\n")

    assert len(report["steps"]) == len(schedule)
    assert all(len(step["sessions"]) == 3 for step in report["steps"])
