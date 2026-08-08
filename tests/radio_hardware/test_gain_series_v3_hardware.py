"""Opt-in RAM-boot acceptance gates for gain-series protocol v3.

These tests are receive-only. They must remain skipped unless the operator
explicitly supplies ``--radio-gain-series-v3`` and, for IP, ``--radio-direct-ip``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spf.sdrpluto.direct_ip_receiver import PlutoDirectIpReceiver
from spf.sdrpluto.direct_usb_protocol import (
    GainObservationFlags,
    MetadataFeatures,
    MetadataFlags,
    RadioMetadataV3,
)
from spf.sdrpluto.direct_usb_receiver import (
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)


pytestmark = [pytest.mark.radio_hardware, pytest.mark.radio_gain_series_v3]

V3_REQUIRED_FEATURES = (
    MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
    | MetadataFeatures.HEADER_CRC32
    | MetadataFeatures.SAMPLE_SEQUENCE
    | MetadataFeatures.GAIN_DB_ENDPOINTS
    | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
    | MetadataFeatures.GAIN_OBSERVATION_SERIES
    | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
)


def _validate_v3_frame(frame, samples: int) -> dict:
    metadata = frame.metadata
    assert isinstance(metadata, RadioMetadataV3)
    assert metadata.samples_per_channel == samples
    assert metadata.iq_payload_bytes == samples * 8
    assert metadata.features & V3_REQUIRED_FEATURES == V3_REQUIRED_FEATURES
    assert metadata.flags & MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
    assert metadata.flags & MetadataFlags.GAIN_OBSERVATIONS_VALID
    assert not metadata.flags & MetadataFlags.GAIN_OBSERVATION_OVERFLOW
    assert not metadata.flags & MetadataFlags.DEVICE_IIO_OVERFLOW
    assert metadata.gain_observation_overflow_count == 0
    assert metadata.gain_observations

    frame_start = metadata.first_sample_sequence
    frame_end = frame_start + samples
    previous_before = None
    durations = []
    for observation in metadata.gain_observations:
        required = (
            GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID
        )
        assert observation.flags & required == required
        assert observation.sample_sequence_before <= observation.sample_sequence_after
        assert observation.sample_sequence_after >= frame_start
        assert observation.sample_sequence_before < frame_end
        if previous_before is not None:
            assert observation.sample_sequence_before >= previous_before
        previous_before = observation.sample_sequence_before
        assert 0 <= observation.rx1_gain_index <= 0x7F
        assert 0 <= observation.rx2_gain_index <= 0x7F
        assert -128 < observation.rx1_gain_db < 127
        assert -128 < observation.rx2_gain_db < 127
        assert observation.read_duration_ns > 0
        durations.append(observation.read_duration_ns)

    first = metadata.gain_observations[0]
    last = metadata.gain_observations[-1]
    np.testing.assert_array_equal(
        metadata.gain_db_start,
        [first.rx1_gain_db, first.rx2_gain_db],
    )
    np.testing.assert_array_equal(
        metadata.gain_db_end,
        [last.rx1_gain_db, last.rx2_gain_db],
    )
    assert metadata.gain_metadata_valid
    assert metadata.rssi_metadata_valid

    signal = iq_payload_to_complex64(frame.iq_payload, samples)
    assert signal.shape == (2, samples)
    assert signal.dtype == np.complex64
    assert np.isfinite(signal).all()
    assert np.any(signal[0] != 0)
    assert np.any(signal[1] != 0)
    return {
        "stream_id": metadata.stream_id,
        "buffer_sequence": metadata.buffer_sequence,
        "first_sample_sequence": metadata.first_sample_sequence,
        "observation_interval_requested": (metadata.gain_observation_interval_samples),
        "observation_count": len(metadata.gain_observations),
        "observation_read_ns_min": min(durations),
        "observation_read_ns_max": max(durations),
        "observation_read_ns_mean": float(np.mean(durations)),
        "channel_rms": [
            float(np.sqrt(np.mean(np.abs(channel).astype(np.float64) ** 2)))
            for channel in signal
        ],
    }


def _assert_contiguous_frames(frames, samples: int) -> None:
    assert [frame.metadata.buffer_sequence for frame in frames] == list(
        range(len(frames))
    )
    starts = [frame.metadata.first_sample_sequence for frame in frames]
    assert all((right - left) == samples for left, right in zip(starts, starts[1:]))
    assert len({frame.metadata.stream_id for frame in frames}) == 1


def test_v3_usb_gain_observations(attached_plutos, pytestconfig, radio_report_dir):
    samples = pytestconfig.getoption("--radio-samples")
    frames_per_request = pytestconfig.getoption("--radio-frames-per-request")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    assert frames_per_request > 0
    report = {"transport": "direct_usb", "radios": []}
    for radio in attached_plutos:
        with PlutoDirectUsbReceiver(
            serial=radio.serial,
            protocol_version=3,
            gain_observation_interval_samples=interval,
            gain_observation_capacity=capacity,
        ) as receiver:
            assert (
                receiver.capabilities.protocol_min
                <= 3
                <= receiver.capabilities.protocol_max
            )
            assert (
                receiver.capabilities.supported_features & V3_REQUIRED_FEATURES
                == V3_REQUIRED_FEATURES
            )
            frames = list(
                receiver.stream_frames(
                    samples_per_channel=samples,
                    frame_count=frames_per_request,
                    queue_depth=1,
                )
            )
        assert len(frames) == frames_per_request
        _assert_contiguous_frames(frames, samples)
        report["radios"].append(
            {
                "serial": radio.serial,
                "port_path": list(radio.port_path),
                "frames": [_validate_v3_frame(frame, samples) for frame in frames],
            }
        )
    (radio_report_dir / "gain_series_v3_usb.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


@pytest.mark.radio_direct_ip
def test_v3_direct_ip_uses_the_same_inner_frame(pytestconfig, radio_report_dir):
    host = pytestconfig.getoption("--radio-direct-ip-host")
    if not host:
        pytest.fail("--radio-direct-ip-host is required with --radio-direct-ip")
    samples = pytestconfig.getoption("--radio-samples")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    with PlutoDirectIpReceiver(
        remote_host=host,
        protocol_version=3,
        gain_observation_interval_samples=interval,
        gain_observation_capacity=capacity,
    ) as receiver:
        capture = receiver.capture(samples_per_channel=samples, frame_count=1)
    assert capture.duplicate_fragment_count == 0
    assert capture.expired_frame_count == 0
    assert capture.rejected_frame_count == 0
    assert len(capture.frames) == 1
    report = {
        "transport": "direct_ip",
        "host": host,
        "elapsed_seconds": capture.elapsed_seconds,
        "frame": _validate_v3_frame(capture.frames[0], samples),
    }
    (radio_report_dir / "gain_series_v3_direct_ip.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
