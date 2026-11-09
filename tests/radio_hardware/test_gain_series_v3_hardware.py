"""Opt-in RAM-boot acceptance gates for gain-series protocol v3.

These tests are receive-only. They must remain skipped unless the operator
explicitly supplies ``--radio-gain-series-v3`` and, for IP, ``--radio-direct-ip``.
"""

from __future__ import annotations

import concurrent.futures
import json
import socket
import time

import iio
import numpy as np
import pytest

from spf.dataset.v7_data import (
    V7_GAIN_EVENT_CAPACITY,
    V7_GAIN_OBSERVATION_CAPACITY,
    v7rx_2x_keys,
    v7rx_gain_series_scalar_keys,
    v7rx_new_dataset,
    v7rx_sample_time_scalar_keys,
    v7rx_scalar_keys,
)
from spf.sdrpluto.direct_ip_protocol import IpControlFlags
from spf.sdrpluto.sdr_controller import _gain_series_arrays
from spf.sdrpluto.direct_ip_receiver import (
    DEFAULT_DIRECT_IP_CONTROL_PORT,
    PlutoDirectIpReceiver,
)
from spf.sdrpluto.direct_usb_protocol import (
    CapabilityFlags,
    GainObservationFlags,
    MetadataFeatures,
    MetadataFlags,
    RadioMetadataV3,
)
from spf.sdrpluto.direct_usb_receiver import (
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.sample_clock import (
    DEFAULT_SAMPLE_CLOCK_RATE_TOLERANCE_PPM,
    capture_host_realtime_mapping,
    fit_sample_clock,
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


def _sample_clock_report(anchors, frames, sample_rate_hz: float) -> dict:
    reference = frames[0].metadata.first_sample_sequence
    fit = fit_sample_clock(
        [anchor.extend_near(reference) for anchor in anchors],
        nominal_sample_rate_hz=sample_rate_hz,
        maximum_rate_error_ppm=DEFAULT_SAMPLE_CLOCK_RATE_TOLERANCE_PPM,
    )
    first_sample = frames[0].metadata.first_sample_sequence
    end_sample = (
        frames[-1].metadata.first_sample_sequence
        + frames[-1].metadata.samples_per_channel
    )
    realtime = capture_host_realtime_mapping()
    uncertainty_ns = (
        max(fit.uncertainty_ns_at(first_sample), fit.uncertainty_ns_at(end_sample))
        + realtime.uncertainty_ns
    )
    monotonic_start = fit.host_monotonic_ns(first_sample)
    monotonic_end = fit.host_monotonic_ns(end_sample)
    return {
        "sample_counter_end_exclusive": end_sample,
        "sample_time_valid": True,
        "sample_time_monotonic_start_ns": monotonic_start,
        "sample_time_monotonic_end_ns": monotonic_end,
        "sample_time_realtime_start_ns": realtime.realtime_ns(monotonic_start),
        "sample_time_realtime_end_ns": realtime.realtime_ns(monotonic_end),
        "sample_time_uncertainty_ns": uncertainty_ns,
        "sample_time_fitted_rate_hz": fit.fitted_sample_rate_hz,
        "sample_time_rate_tolerance_ppm": fit.maximum_rate_error_ppm,
        "sample_time_anchor_count": fit.anchor_count,
        "sample_time_max_round_trip_ns": fit.maximum_round_trip_ns,
        "maximum_midpoint_residual_ns": fit.maximum_midpoint_residual_ns,
    }


def _iq_power_dbfs(signal: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(signal.astype(np.complex64)) ** 2, axis=1)
    with np.errstate(divide="ignore"):
        return (10.0 * np.log10(power / (2.0 * 2048.0**2))).astype(np.float32)


def _direct_ip_sample_rate(host: str) -> int:
    context = iio.Context(f"ip:{host}")
    try:
        phy = context.find_device("ad9361-phy")
        if phy is None:
            raise RuntimeError("direct-IP radio has no ad9361-phy")
        channel = phy.find_channel("voltage0", True)
        if channel is None or "sampling_frequency" not in channel.attrs:
            raise RuntimeError("direct-IP radio has no RX sampling-frequency control")
        return int(channel.attrs["sampling_frequency"].value)
    finally:
        del context


def _set_direct_ip_sample_rate(host: str, sample_rate_hz: int) -> int:
    context = iio.Context(f"ip:{host}")
    try:
        phy = context.find_device("ad9361-phy")
        if phy is None:
            raise RuntimeError("direct-IP radio has no ad9361-phy")
        channel = phy.find_channel("voltage0", True)
        if channel is None or "sampling_frequency" not in channel.attrs:
            raise RuntimeError("direct-IP radio has no RX sampling-frequency control")
        channel.attrs["sampling_frequency"].value = str(int(sample_rate_hz))
        return int(channel.attrs["sampling_frequency"].value)
    finally:
        del context


def _first_change(metadata: RadioMetadataV3) -> np.ndarray:
    return np.asarray(
        [
            -1
            if metadata.rx1_first_change_sample == 0xFFFFFFFF
            else metadata.rx1_first_change_sample,
            -1
            if metadata.rx2_first_change_sample == 0xFFFFFFFF
            else metadata.rx2_first_change_sample,
        ],
        dtype=np.int32,
    )


def _write_v3_record(
    receiver_z, record_index: int, frame, samples: int, sample_time: dict
) -> None:
    metadata = frame.metadata
    assert isinstance(metadata, RadioMetadataV3)
    signal = iq_payload_to_complex64(frame.iq_payload, samples)
    receiver_z["signal_matrix"][record_index] = signal
    scalar_values = {
        "gain_metadata_valid": metadata.gain_metadata_valid,
        "rssi_metadata_valid": metadata.rssi_metadata_valid,
        "gain_metadata_flags": int(metadata.flags),
        "stream_id": metadata.stream_id,
        "buffer_sequence": metadata.buffer_sequence,
        "sample_sequence": metadata.first_sample_sequence,
        "gain_start_read_duration_ns": metadata.gain_start_read_duration_ns,
        "gain_end_read_duration_ns": metadata.gain_end_read_duration_ns,
        "rssi_start_read_duration_ns": metadata.rssi_start_read_duration_ns,
        "rssi_end_read_duration_ns": metadata.rssi_end_read_duration_ns,
    }
    two_values = {
        "gain_db_start": metadata.gain_db_start,
        "gain_db_end": metadata.gain_db_end,
        "rssi_db_start": metadata.rssi_db_start,
        "rssi_db_end": metadata.rssi_db_end,
        "gain_endpoints_equal": metadata.gain_endpoints_equal,
        "first_gain_change_sample": _first_change(metadata),
        "iq_power_dbfs": _iq_power_dbfs(signal),
    }
    for key in v7rx_scalar_keys:
        receiver_z[key][record_index] = scalar_values[key]
    for key in v7rx_2x_keys:
        receiver_z[key][record_index] = two_values[key]
    for key in v7rx_sample_time_scalar_keys:
        receiver_z[key][record_index] = sample_time[key]
    receiver_z["system_timestamp"][record_index] = time.time()
    receiver_z["rssis"][record_index] = metadata.rssi_db_end
    receiver_z["gains"][record_index] = metadata.gain_db_end

    gain_series = _gain_series_arrays(metadata)
    observation_count = len(metadata.gain_observations)
    event_count = len(metadata.gain_events)
    scalar_series = {
        "gain_observation_count": observation_count,
        "gain_observation_interval_samples": (
            metadata.gain_observation_interval_samples
        ),
        "gain_observation_overflow_count": (metadata.gain_observation_overflow_count),
        "gain_event_count": event_count,
        "gain_event_overflow_count": metadata.gain_event_overflow_count,
    }
    for key in v7rx_gain_series_scalar_keys:
        receiver_z[key][record_index] = scalar_series[key]

    bounds = np.full(
        (V7_GAIN_OBSERVATION_CAPACITY, 2),
        np.iinfo(np.uint64).max,
        dtype=np.uint64,
    )
    indices = np.full((V7_GAIN_OBSERVATION_CAPACITY, 2), 0xFF, dtype=np.uint8)
    gain_db = np.full((V7_GAIN_OBSERVATION_CAPACITY, 2), np.nan, dtype=np.float32)
    valid = np.zeros(V7_GAIN_OBSERVATION_CAPACITY, dtype=np.bool_)
    durations = np.zeros(V7_GAIN_OBSERVATION_CAPACITY, dtype=np.uint32)
    bounds[:observation_count] = gain_series["gain_observation_sample_bounds"]
    indices[:observation_count] = gain_series["gain_observation_index"]
    gain_db[:observation_count] = gain_series["gain_observation_db"]
    valid[:observation_count] = gain_series["gain_observation_valid"]
    durations[:observation_count] = gain_series["gain_observation_read_duration_ns"]
    receiver_z["gain_observation_sample_bounds"][record_index] = bounds
    receiver_z["gain_observation_index"][record_index] = indices
    receiver_z["gain_observation_db"][record_index] = gain_db
    receiver_z["gain_observation_valid"][record_index] = valid
    receiver_z["gain_observation_read_duration_ns"][record_index] = durations

    event_sequences = np.full(
        V7_GAIN_EVENT_CAPACITY, np.iinfo(np.uint64).max, dtype=np.uint64
    )
    event_flags = np.zeros(V7_GAIN_EVENT_CAPACITY, dtype=np.uint16)
    event_sequences[:event_count] = gain_series["gain_event_sample_sequence"]
    event_flags[:event_count] = gain_series["gain_event_flags"]
    receiver_z["gain_event_sample_sequence"][record_index] = event_sequences
    receiver_z["gain_event_flags"][record_index] = event_flags


def test_v3_usb_gain_observations(attached_plutos, pytestconfig, radio_report_dir):
    samples = pytestconfig.getoption("--radio-samples")
    frames_per_request = pytestconfig.getoption("--radio-frames-per-request")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    sample_rate_hz = pytestconfig.getoption("--radio-sample-rate")
    max_uncertainty_ns = int(
        pytestconfig.getoption("--radio-time-anchor-max-uncertainty-ms") * 1_000_000
    )
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
            ), (
                f"radio {radio.serial} is not running protocol-v3 firmware: "
                f"advertised range is v{receiver.capabilities.protocol_min}.."
                f"v{receiver.capabilities.protocol_max}"
            )
            assert (
                receiver.capabilities.supported_features & V3_REQUIRED_FEATURES
                == V3_REQUIRED_FEATURES
            ), f"radio {radio.serial} is missing required protocol-v3 features"
            assert receiver.capabilities.capability_flags & CapabilityFlags.TIME_ANCHOR
            anchors = [receiver.query_time_anchor() for _ in range(8)]
            frames = []
            stream = receiver.stream_frames(
                samples_per_channel=samples,
                frame_count=frames_per_request,
                queue_depth=1,
            )
            for frame in stream:
                frames.append(frame)
                # Exercise EP0 time anchors while the finite bulk stream is
                # still active, matching PPlus for multi-frame requests.
                anchors.append(receiver.query_time_anchor())
        assert len(frames) == frames_per_request
        _assert_contiguous_frames(frames, samples)
        sample_clock = _sample_clock_report(anchors, frames, sample_rate_hz)
        assert sample_clock["sample_time_uncertainty_ns"] <= max_uncertainty_ns
        report["radios"].append(
            {
                "serial": radio.serial,
                "port_path": list(radio.port_path),
                "frames": [_validate_v3_frame(frame, samples) for frame in frames],
                "sample_clock": sample_clock,
            }
        )
    (radio_report_dir / "gain_series_v3_usb.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def test_v3_repeated_fresh_usb_starts(attached_plutos, pytestconfig, radio_report_dir):
    """Catch intermittent loss between the first two blocks of a new stream.

    A long steady stream cannot detect startup-only DMA backlog.  Each cycle
    therefore creates a new receiver, negotiates protocol v3, issues a fresh
    START, and requires the first frames to be exactly contiguous.
    """

    samples = pytestconfig.getoption("--radio-samples")
    frame_count = pytestconfig.getoption("--radio-frames-per-request")
    cycles = pytestconfig.getoption("--radio-cycles")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    assert cycles > 0
    assert frame_count >= 2, "startup continuity requires at least two frames"

    report = {
        "transport": "direct_usb",
        "purpose": "repeated_fresh_stream_start_continuity",
        "cycles_per_radio": cycles,
        "frames_per_cycle": frame_count,
        "radios": [],
    }
    report_path = radio_report_dir / "gain_series_v3_repeated_starts.json"
    for radio in attached_plutos:
        streams = []
        radio_result = {
            "serial": radio.serial,
            "port_path": list(radio.port_path),
            "streams": streams,
        }
        report["radios"].append(radio_result)
        for cycle in range(cycles):
            try:
                with PlutoDirectUsbReceiver(
                    serial=radio.serial,
                    protocol_version=3,
                    gain_observation_interval_samples=interval,
                    gain_observation_capacity=capacity,
                ) as receiver:
                    frames = list(
                        receiver.stream_frames(
                            samples_per_channel=samples,
                            frame_count=frame_count,
                            queue_depth=1,
                        )
                    )
            except Exception as exc:
                streams.append(
                    {
                        "cycle": cycle,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                pytest.fail(
                    f"{radio.serial}: fresh stream cycle {cycle} failed: {exc}",
                    pytrace=True,
                )
            assert len(frames) == frame_count
            _assert_contiguous_frames(frames, samples)
            validated = [_validate_v3_frame(frame, samples) for frame in frames]
            streams.append(
                {
                    "cycle": cycle,
                    "stream_id": validated[0]["stream_id"],
                    "first_sample_sequences": [
                        frame["first_sample_sequence"] for frame in validated
                    ],
                }
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")


def _simultaneous_v3_stream(
    serial: str,
    *,
    samples: int,
    frame_count: int,
    interval: int,
    capacity: int,
    sample_rate_hz: float,
) -> dict:
    with PlutoDirectUsbReceiver(
        serial=serial,
        protocol_version=3,
        gain_observation_interval_samples=interval,
        gain_observation_capacity=capacity,
    ) as receiver:
        anchors = [receiver.query_time_anchor() for _ in range(8)]
        frames = []
        for frame in receiver.stream_frames(
            samples_per_channel=samples,
            frame_count=frame_count,
            queue_depth=1,
        ):
            frames.append(frame)
            anchors.append(receiver.query_time_anchor())
    _assert_contiguous_frames(frames, samples)
    return {
        "serial": serial,
        "frames": [_validate_v3_frame(frame, samples) for frame in frames],
        "sample_clock": _sample_clock_report(anchors, frames, sample_rate_hz),
    }


def test_v3_simultaneous_usb_streams(attached_plutos, pytestconfig, radio_report_dir):
    if len(attached_plutos) < 2:
        pytest.skip("simultaneous protocol-v3 streaming requires two radios")
    samples = pytestconfig.getoption("--radio-samples")
    frame_count = pytestconfig.getoption("--radio-frames-per-request")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    sample_rate_hz = pytestconfig.getoption("--radio-sample-rate")
    max_uncertainty_ns = int(
        pytestconfig.getoption("--radio-time-anchor-max-uncertainty-ms") * 1_000_000
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(attached_plutos)
    ) as executor:
        futures = [
            executor.submit(
                _simultaneous_v3_stream,
                radio.serial,
                samples=samples,
                frame_count=frame_count,
                interval=interval,
                capacity=capacity,
                sample_rate_hz=sample_rate_hz,
            )
            for radio in attached_plutos
        ]
        results = [future.result() for future in futures]
    assert {result["serial"] for result in results} == {
        radio.serial for radio in attached_plutos
    }
    for result in results:
        assert len(result["frames"]) == frame_count
        assert result["sample_clock"]["sample_time_uncertainty_ns"] <= (
            max_uncertainty_ns
        )
    (radio_report_dir / "gain_series_v3_simultaneous_usb.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


@pytest.mark.radio_direct_ip
def test_v3_direct_ip_survives_malformed_control_datagrams(
    pytestconfig, radio_report_dir
):
    host = pytestconfig.getoption("--radio-direct-ip-host")
    if not host:
        pytest.fail("--radio-direct-ip-host is required with --radio-direct-ip")

    # RC11 returned a fatal epoll status for an undersized legacy envelope,
    # terminating the direct-IP daemon. Exercise boundary lengths and unknown
    # magic before proving that a valid capability exchange still succeeds.
    malformed = (
        b"",
        b"x",
        b"xyz",
        b"test",
        bytes.fromhex("504c544f"),  # legacy magic without its 8-byte header
        bytes.fromhex("efbeadde") + bytes(76),
    )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for datagram in malformed:
            probe.sendto(datagram, (host, DEFAULT_DIRECT_IP_CONTROL_PORT))
    time.sleep(0.1)

    with PlutoDirectIpReceiver(remote_host=host) as receiver:
        assert receiver.capabilities.flags & IpControlFlags.FINITE_RX

    (radio_report_dir / "gain_series_v3_direct_ip_malformed.json").write_text(
        json.dumps(
            {
                "host": host,
                "control_port": DEFAULT_DIRECT_IP_CONTROL_PORT,
                "malformed_lengths": [len(item) for item in malformed],
                "valid_capability_query_afterwards": True,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.radio_direct_ip
def test_v3_direct_ip_uses_the_same_inner_frame(
    pytestconfig, direct_ip_transport_profile, radio_report_dir
):
    host = pytestconfig.getoption("--radio-direct-ip-host")
    if not host:
        pytest.fail("--radio-direct-ip-host is required with --radio-direct-ip")
    samples = pytestconfig.getoption("--radio-samples")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    sample_rate_hz = pytestconfig.getoption("--radio-sample-rate")
    max_uncertainty_ns = int(
        pytestconfig.getoption("--radio-time-anchor-max-uncertainty-ms") * 1_000_000
    )
    with PlutoDirectIpReceiver(
        remote_host=host,
        protocol_version=3,
        gain_observation_interval_samples=interval,
        gain_observation_capacity=capacity,
        **direct_ip_transport_profile,
    ) as receiver:
        assert receiver.capabilities.flags & IpControlFlags.TIME_ANCHOR
        anchors = [receiver.query_time_anchor() for _ in range(8)]
        capture = receiver.capture(samples_per_channel=samples, frame_count=1)
        anchors.append(receiver.query_time_anchor())
    assert capture.duplicate_fragment_count == 0
    assert capture.expired_frame_count == 0
    assert capture.rejected_frame_count == 0
    assert len(capture.frames) == 1
    sample_clock = _sample_clock_report(anchors, capture.frames, sample_rate_hz)
    assert sample_clock["sample_time_uncertainty_ns"] <= max_uncertainty_ns
    report = {
        "transport": f"direct_ip_{direct_ip_transport_profile['transport']}",
        "host": host,
        "elapsed_seconds": capture.elapsed_seconds,
        "frame": _validate_v3_frame(capture.frames[0], samples),
        "sample_clock": sample_clock,
    }
    (radio_report_dir / "gain_series_v3_direct_ip.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


@pytest.mark.radio_direct_ip
def test_v3_direct_ip_buffers_a_maximum_finite_burst(
    pytestconfig, direct_ip_transport_profile, radio_report_dir
):
    """Prove capture is decoupled from UDP drain at the production frame size."""

    host = pytestconfig.getoption("--radio-direct-ip-host")
    if not host:
        pytest.fail("--radio-direct-ip-host is required with --radio-direct-ip")
    samples = pytestconfig.getoption("--radio-samples")
    frame_count = pytestconfig.getoption("--radio-frames-per-request")
    cycles = pytestconfig.getoption("--radio-cycles")
    requested_sample_rate_hz = int(
        pytestconfig.getoption("--radio-direct-ip-burst-sample-rate")
    )
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    minimum_mibps = pytestconfig.getoption("--radio-direct-ip-min-payload-mibps")
    minimum_receive_buffer_bytes = int(
        pytestconfig.getoption("--radio-direct-ip-min-receive-buffer-mib") * 1024 * 1024
    )
    assert frame_count == 16, "performance gate must exercise the maximum finite burst"
    assert cycles > 0

    cycle_reports = []
    total_elapsed_seconds = 0.0
    total_payload_bytes = 0
    original_sample_rate_hz = _direct_ip_sample_rate(host)
    try:
        configured_sample_rate_hz = _set_direct_ip_sample_rate(
            host, requested_sample_rate_hz
        )
        assert configured_sample_rate_hz == requested_sample_rate_hz
        with PlutoDirectIpReceiver(
            remote_host=host,
            protocol_version=3,
            gain_observation_interval_samples=interval,
            gain_observation_capacity=capacity,
            minimum_effective_receive_buffer_bytes=minimum_receive_buffer_bytes,
            **direct_ip_transport_profile,
        ) as receiver:
            required_transport = (
                IpControlFlags.BUFFERED_FINITE_RX | IpControlFlags.USB_CLASS_PACING
            )
            assert (
                receiver.capabilities.flags & required_transport == required_transport
            )
            effective_receive_buffer_bytes = (
                receiver.effective_data_receive_buffer_bytes
            )
            transport_flags = int(receiver.capabilities.flags)
            for cycle in range(cycles):
                capture = receiver.capture(
                    samples_per_channel=samples,
                    frame_count=frame_count,
                )
                assert capture.duplicate_fragment_count == 0
                assert capture.expired_frame_count == 0
                assert capture.rejected_frame_count == 0
                assert capture.receive_queue_overflow_count == 0
                assert len(capture.frames) == frame_count
                _assert_contiguous_frames(capture.frames, samples)
                payload_bytes = samples * 8 * frame_count
                cycle_reports.append(
                    {
                        "cycle": cycle,
                        "elapsed_seconds": capture.elapsed_seconds,
                        "payload_mibps": payload_bytes
                        / capture.elapsed_seconds
                        / (1024 * 1024),
                        "first_frame": _validate_v3_frame(capture.frames[0], samples),
                        "last_frame": _validate_v3_frame(capture.frames[-1], samples),
                    }
                )
                total_elapsed_seconds += capture.elapsed_seconds
                total_payload_bytes += payload_bytes
    finally:
        restored_sample_rate_hz = _set_direct_ip_sample_rate(
            host, original_sample_rate_hz
        )
        assert restored_sample_rate_hz == original_sample_rate_hz

    aggregate_payload_mibps = (
        total_payload_bytes / total_elapsed_seconds / (1024 * 1024)
    )

    report = {
        "transport": f"direct_ip_{direct_ip_transport_profile['transport']}",
        "test": "maximum buffered finite burst",
        "host": host,
        "samples_per_channel": samples,
        "requested_sample_rate_hz": requested_sample_rate_hz,
        "configured_sample_rate_hz": configured_sample_rate_hz,
        "restored_sample_rate_hz": restored_sample_rate_hz,
        "frame_count": frame_count,
        "cycles": cycles,
        "payload_bytes": total_payload_bytes,
        "elapsed_seconds": total_elapsed_seconds,
        "payload_mibps": aggregate_payload_mibps,
        "performance_pass": aggregate_payload_mibps >= minimum_mibps,
        "minimum_cycle_payload_mibps": min(
            cycle["payload_mibps"] for cycle in cycle_reports
        ),
        "maximum_cycle_payload_mibps": max(
            cycle["payload_mibps"] for cycle in cycle_reports
        ),
        "minimum_payload_mibps": minimum_mibps,
        "effective_receive_buffer_bytes": effective_receive_buffer_bytes,
        "minimum_receive_buffer_bytes": minimum_receive_buffer_bytes,
        "transport_flags": transport_flags,
        "total_frames": cycles * frame_count,
        "duplicate_fragment_count": 0,
        "expired_frame_count": 0,
        "rejected_frame_count": 0,
        "receive_queue_overflow_count": 0,
        "cycle_reports": cycle_reports,
    }
    (radio_report_dir / "gain_series_v3_direct_ip_buffered_burst.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    assert aggregate_payload_mibps >= minimum_mibps


@pytest.mark.radio_zarr
def test_v3_gain_series_round_trips_through_v7_zarr(
    attached_plutos, pytestconfig, tmp_path
):
    samples = pytestconfig.getoption("--radio-samples")
    frame_count = pytestconfig.getoption("--radio-zarr-frames")
    interval = min(pytestconfig.getoption("--radio-gain-observation-interval"), samples)
    capacity = pytestconfig.getoption("--radio-gain-observation-capacity")
    sample_rate_hz = pytestconfig.getoption("--radio-sample-rate")
    max_uncertainty_ns = int(
        pytestconfig.getoption("--radio-time-anchor-max-uncertainty-ms") * 1_000_000
    )
    assert 0 < frame_count
    assert capacity <= V7_GAIN_OBSERVATION_CAPACITY

    path = tmp_path / "hardware_gain_series_v3.zarr"
    zarr = v7rx_new_dataset(
        filename=str(path),
        timesteps=frame_count,
        buffer_size=samples,
        n_receivers=len(attached_plutos),
        config={
            "data-version": 7,
            "test": "attached-radio protocol-v3 gain-series round trip",
        },
        chunk_size=1,
        compressor=None,
    )
    zarr.attrs["capture_status"] = "in_progress"
    zarr.attrs["capture_records_written_by_receiver"] = [0] * len(attached_plutos)
    zarr.attrs["rx_transport"] = "direct_usb"
    zarr.attrs["direct_usb_protocol_version"] = 3
    expected_by_radio = []
    try:
        for receiver_index, radio in enumerate(attached_plutos):
            receiver_z = zarr[f"receivers/r{receiver_index}"]
            expected_records = []
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
                identity = receiver.query_hardware_identity()
                receiver_z.attrs["sdr_family"] = "pluto"
                receiver_z.attrs["sdr_serial"] = radio.serial
                receiver_z.attrs["usb_port_path"] = list(radio.port_path)
                receiver_z.attrs["gadget_build_id"] = identity.gadget_build_id
                receiver_z.attrs["fpga_device_dna"] = identity.fpga_device_dna
                anchors = [receiver.query_time_anchor() for _ in range(8)]
                for record_index in range(frame_count):
                    stream = receiver.stream_frames(
                        samples_per_channel=samples,
                        frame_count=1,
                        queue_depth=1,
                    )
                    try:
                        frame = next(stream)
                        _validate_v3_frame(frame, samples)
                        # Bracket the yielded frame before advancing the finite
                        # generator into its STOP/cleanup path.
                        anchors.append(receiver.query_time_anchor())
                        with pytest.raises(StopIteration):
                            next(stream)
                    finally:
                        stream.close()
                    anchors = anchors[-32:]
                    sample_time = _sample_clock_report(anchors, [frame], sample_rate_hz)
                    assert (
                        sample_time["sample_time_uncertainty_ns"] <= max_uncertainty_ns
                    )
                    _write_v3_record(
                        receiver_z, record_index, frame, samples, sample_time
                    )
                    metadata = frame.metadata
                    expected_records.append(
                        {
                            "indices": np.asarray(
                                [
                                    [item.rx1_gain_index, item.rx2_gain_index]
                                    for item in metadata.gain_observations
                                ],
                                dtype=np.uint8,
                            ),
                            "bounds": np.asarray(
                                [
                                    [
                                        item.sample_sequence_before,
                                        item.sample_sequence_after,
                                    ]
                                    for item in metadata.gain_observations
                                ],
                                dtype=np.uint64,
                            ),
                            "sample_sequence": metadata.first_sample_sequence,
                            "stream_id": metadata.stream_id,
                        }
                    )
                    progress = list(zarr.attrs["capture_records_written_by_receiver"])
                    progress[receiver_index] = record_index + 1
                    zarr.attrs["capture_records_written_by_receiver"] = progress
            expected_by_radio.append(tuple(expected_records))
        zarr.attrs["capture_status"] = "complete"
    finally:
        zarr.store.close()

    reopened = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        assert reopened.attrs["radio_metadata_schema_version"] == 2
        assert reopened.attrs["gain_series_schema_version"] == 1
        assert reopened.attrs["sample_time_schema_version"] == 1
        assert reopened.attrs["capture_status"] == "complete"
        assert reopened.attrs["rx_transport"] == "direct_usb"
        assert reopened.attrs["direct_usb_protocol_version"] == 3
        for receiver_index, (radio, expected_records) in enumerate(
            zip(attached_plutos, expected_by_radio, strict=True)
        ):
            receiver_z = reopened[f"receivers/r{receiver_index}"]
            assert receiver_z.attrs["sdr_serial"] == radio.serial
            assert tuple(receiver_z.attrs["usb_port_path"]) == radio.port_path
            assert len(receiver_z.attrs["gadget_build_id"]) == 40
            assert receiver_z["signal_matrix"].shape == (
                frame_count,
                2,
                samples,
            )
            counts = receiver_z["gain_observation_count"][:]
            assert np.all(counts > 0)
            assert np.all(counts <= capacity)
            for record_index, expected in enumerate(expected_records):
                count = int(counts[record_index])
                assert count == len(expected["indices"])
                np.testing.assert_array_equal(
                    receiver_z["gain_observation_index"][record_index, :count],
                    expected["indices"],
                )
                np.testing.assert_array_equal(
                    receiver_z["gain_observation_sample_bounds"][record_index, :count],
                    expected["bounds"],
                )
                assert np.all(
                    receiver_z["gain_observation_valid"][record_index, :count]
                )
                assert not np.any(
                    receiver_z["gain_observation_valid"][record_index, count:]
                )
                assert np.all(
                    receiver_z["gain_observation_index"][record_index, count:] == 0xFF
                )
                assert np.all(
                    receiver_z["gain_observation_sample_bounds"][record_index, count:]
                    == np.iinfo(np.uint64).max
                )
            sample_sequences = receiver_z["sample_sequence"][:]
            np.testing.assert_array_equal(
                sample_sequences,
                [item["sample_sequence"] for item in expected_records],
            )
            assert np.all(np.diff(sample_sequences.astype(object)) > 0)
            np.testing.assert_array_equal(
                receiver_z["stream_id"][:],
                [item["stream_id"] for item in expected_records],
            )
            assert len(set(receiver_z["stream_id"][:].tolist())) == frame_count
            np.testing.assert_array_equal(
                receiver_z["buffer_sequence"][:], np.zeros(frame_count)
            )
            for record_index in range(frame_count):
                assert np.any(receiver_z["signal_matrix"][record_index] != 0)
            assert np.all(receiver_z["sample_time_valid"][:])
            np.testing.assert_array_equal(
                receiver_z["sample_counter_end_exclusive"][:],
                receiver_z["sample_sequence"][:] + samples,
            )
            assert np.all(
                receiver_z["sample_time_monotonic_end_ns"][:]
                > receiver_z["sample_time_monotonic_start_ns"][:]
            )
            assert np.all(
                receiver_z["sample_time_uncertainty_ns"][:] <= max_uncertainty_ns
            )
    finally:
        reopened.store.close()
