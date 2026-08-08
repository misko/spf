from __future__ import annotations

import dataclasses
import random
import socket
import struct
import threading

import pytest

from spf.sdrpluto.direct_ip_protocol import (
    IpControlFlags,
    IpControlMessageV1,
    IpControlType,
    fragment_ip_frame,
)
from spf.sdrpluto.direct_ip_receiver import (
    DirectIpTransportError,
    PlutoDirectIpReceiver,
)
from spf.sdrpluto.direct_usb_protocol import (
    FIRST_CHANGE_UNAVAILABLE,
    GainObservationFlags,
    GainObservationV3,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    RadioMetadataV3,
    SampleFormat,
    TIME_ANCHOR_QUERY_MAGIC,
    TimeAnchorFlags,
    TimeAnchorV1,
)


REQUIRED_V3_FEATURES = (
    MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
    | MetadataFeatures.HEADER_CRC32
    | MetadataFeatures.SAMPLE_SEQUENCE
    | MetadataFeatures.GAIN_DB_ENDPOINTS
    | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
    | MetadataFeatures.GAIN_OBSERVATION_SERIES
    | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
)


def _inner_frame(*, stream_id: int, sequence: int, samples: int = 1024) -> bytes:
    first_sample = 1_000_000 + sequence * samples
    observation = GainObservationV3(
        sample_sequence_before=first_sample,
        sample_sequence_after=first_sample + min(samples - 1, 512),
        read_duration_ns=490_000,
        flags=GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID,
        rx1_gain_index=42,
        rx2_gain_index=43,
        rx1_gain_db=20,
        rx2_gain_db=21,
    )
    metadata = RadioMetadataV3(
        features=REQUIRED_V3_FEATURES,
        flags=(
            MetadataFlags.START_VALID
            | MetadataFlags.END_VALID
            | MetadataFlags.SAMPLE_SEQUENCE_VALID
            | MetadataFlags.GAIN_FULL_TABLE_MODE
            | MetadataFlags.GAIN_DB_VALUES
            | MetadataFlags.RSSI_START_VALID
            | MetadataFlags.RSSI_END_VALID
            | MetadataFlags.GAIN_OBSERVATIONS_VALID
            | MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
        ),
        stream_id=stream_id,
        buffer_sequence=sequence,
        first_sample_sequence=first_sample,
        samples_per_channel=samples,
        iq_payload_bytes=samples * 8,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_db_start=20,
        rx2_gain_db_start=21,
        rx1_gain_db_end=20,
        rx2_gain_db_end=21,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx1_rssi_start_qdb=400,
        rx2_rssi_start_qdb=404,
        rx1_rssi_end_qdb=400,
        rx2_rssi_end_qdb=404,
        gain_observation_interval_samples=samples,
        gain_observation_capacity=1,
        gain_observations=(observation,),
    )
    return metadata.pack() + bytes(metadata.iq_payload_bytes)


class _SyntheticIpGadget:
    def __init__(
        self,
        *,
        send_frames: bool = True,
        outer_sequence_delta: int = 0,
        drop_first_start_reply: bool = True,
        bad_started_echo: bool = False,
    ) -> None:
        self.control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control.bind(("127.0.0.1", 0))
        self.control.settimeout(0.1)
        self.control_port = int(self.control.getsockname()[1])
        self.send_frames = send_frames
        self.outer_sequence_delta = outer_sequence_delta
        self.drop_first_start_reply = drop_first_start_reply
        self.bad_started_echo = bad_started_echo
        self.start_request_count = 0
        self.stop_request_count = 0
        self.time_anchor_request_count = 0
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._cached_started: dict[int, IpControlMessageV1] = {}

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.control.close()
        if self._thread.is_alive():
            raise RuntimeError("synthetic direct-IP gadget did not stop")
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    payload, peer = self.control.recvfrom(4096)
                except TimeoutError:
                    continue
                if (
                    len(payload) == 24
                    and struct.unpack_from("<I", payload)[0] == TIME_ANCHOR_QUERY_MAGIC
                ):
                    self.time_anchor_request_count += 1
                    request_id = struct.unpack_from("<Q", payload, 8)[0]
                    anchor = TimeAnchorV1(
                        flags=(
                            TimeAnchorFlags.COUNTER_INTERVAL_VALID
                            | TimeAnchorFlags.MONOTONIC_INTERVAL_VALID
                            | TimeAnchorFlags.COUNTER_LOW32
                            | TimeAnchorFlags.COUNTER_ADVANCED
                        ),
                        request_id=request_id,
                        radio_monotonic_before_ns=1_000_000,
                        sample_counter_before=0xFFFFFFF0,
                        sample_counter_after=0x10,
                        radio_monotonic_after_ns=1_001_000,
                    )
                    self.control.sendto(anchor.pack(), peer)
                    continue
                request = IpControlMessageV1.unpack(payload)
                if request.message_type == IpControlType.QUERY_CAPABILITIES:
                    response = IpControlMessageV1(
                        message_type=IpControlType.CAPABILITIES,
                        request_id=request.request_id,
                        flags=(
                            IpControlFlags.FINITE_RX
                            | IpControlFlags.IDEMPOTENT_REQUESTS
                            | IpControlFlags.TIME_ANCHOR
                        ),
                        protocol_min=2,
                        protocol_max=3,
                        features=REQUIRED_V3_FEATURES,
                        max_samples_per_channel=524_288,
                        max_finite_frames=16,
                    )
                    self.control.sendto(response.pack(), peer)
                elif request.message_type == IpControlType.START_RX:
                    self.start_request_count += 1
                    started = self._cached_started.setdefault(
                        request.request_id,
                        dataclasses.replace(
                            request,
                            message_type=IpControlType.STARTED,
                            samples_per_channel=(
                                request.samples_per_channel + 1
                                if self.bad_started_echo
                                else request.samples_per_channel
                            ),
                            stream_id=77,
                        ),
                    )
                    if self.drop_first_start_reply and self.start_request_count == 1:
                        continue
                    self.control.sendto(started.pack(), peer)
                    if self.send_frames:
                        self._send_frames(request, started.stream_id, peer[0])
                elif request.message_type == IpControlType.STOP_RX:
                    self.stop_request_count += 1
                    stopped = dataclasses.replace(
                        request, message_type=IpControlType.STOPPED
                    )
                    self.control.sendto(stopped.pack(), peer)
                else:
                    raise AssertionError(
                        f"unexpected synthetic request {request.message_type}"
                    )
        except Exception as error:
            self.error = error

    def _send_frames(
        self, request: IpControlMessageV1, stream_id: int, peer_ip: str
    ) -> None:
        data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for sequence in range(request.frame_count):
                frame = _inner_frame(
                    stream_id=stream_id,
                    sequence=sequence,
                    samples=request.samples_per_channel,
                )
                datagrams = list(
                    fragment_ip_frame(
                        frame,
                        stream_id=stream_id,
                        frame_sequence=sequence + self.outer_sequence_delta,
                        max_datagram_bytes=request.max_datagram_bytes,
                    )
                )
                random.Random(sequence).shuffle(datagrams)
                # One harmless duplicate exercises host de-duplication.
                datagrams.insert(1, datagrams[0])
                for datagram in datagrams:
                    data.sendto(datagram, (peer_ip, request.data_port))
        finally:
            data.close()


def test_finite_v3_capture_retries_control_and_parses_common_inner_frames():
    gadget = _SyntheticIpGadget()
    gadget.start()
    try:
        receiver = PlutoDirectIpReceiver(
            remote_host="127.0.0.1",
            remote_control_port=gadget.control_port,
            control_timeout_seconds=0.05,
            frame_timeout_seconds=2,
        )
        with receiver:
            capture = receiver.capture(samples_per_channel=1024, frame_count=2)
        assert [frame.metadata.buffer_sequence for frame in capture.frames] == [0, 1]
        assert all(frame.metadata.stream_id == 77 for frame in capture.frames)
        assert capture.duplicate_fragment_count == 2
        assert capture.expired_frame_count == 0
        assert capture.rejected_frame_count == 0
        assert gadget.start_request_count == 2
        assert gadget.stop_request_count == 1
    finally:
        gadget.close()


def test_direct_ip_time_anchor_uses_common_record_and_host_bracket():
    gadget = _SyntheticIpGadget(drop_first_start_reply=False)
    gadget.start()
    try:
        receiver = PlutoDirectIpReceiver(
            remote_host="127.0.0.1",
            remote_control_port=gadget.control_port,
            control_timeout_seconds=0.05,
        )
        with receiver:
            measurement = receiver.query_time_anchor()
        assert measurement.transport == "direct_ip"
        assert measurement.anchor.sample_counter_before == 0xFFFFFFF0
        assert measurement.anchor.counter_delta == 32
        assert measurement.round_trip_ns >= 0
        assert gadget.time_anchor_request_count == 1
    finally:
        gadget.close()


def test_inner_outer_sequence_mismatch_fails_closed_and_stops_stream():
    gadget = _SyntheticIpGadget(outer_sequence_delta=1, drop_first_start_reply=False)
    gadget.start()
    try:
        receiver = PlutoDirectIpReceiver(
            remote_host="127.0.0.1",
            remote_control_port=gadget.control_port,
            control_timeout_seconds=0.05,
            frame_timeout_seconds=1,
        )
        with receiver:
            with pytest.raises(ProtocolError, match="inner/outer sequence"):
                receiver.capture(samples_per_channel=1024, frame_count=1)
        assert gadget.stop_request_count == 1
    finally:
        gadget.close()


def test_missing_data_times_out_explicitly_and_stops_stream():
    gadget = _SyntheticIpGadget(send_frames=False, drop_first_start_reply=False)
    gadget.start()
    try:
        receiver = PlutoDirectIpReceiver(
            remote_host="127.0.0.1",
            remote_control_port=gadget.control_port,
            control_timeout_seconds=0.05,
            frame_timeout_seconds=0.1,
        )
        with receiver:
            with pytest.raises(DirectIpTransportError, match="timed out"):
                receiver.capture(samples_per_channel=1024, frame_count=1)
        assert gadget.stop_request_count == 1
    finally:
        gadget.close()


def test_receive_buffer_must_be_positive():
    with pytest.raises(ValueError, match="receive buffer"):
        PlutoDirectIpReceiver(remote_host="127.0.0.1", data_receive_buffer_bytes=0)


def test_bad_started_echo_fails_closed_and_stops_assigned_stream():
    gadget = _SyntheticIpGadget(
        send_frames=False,
        drop_first_start_reply=False,
        bad_started_echo=True,
    )
    gadget.start()
    try:
        receiver = PlutoDirectIpReceiver(
            remote_host="127.0.0.1",
            remote_control_port=gadget.control_port,
            control_timeout_seconds=0.05,
            frame_timeout_seconds=1,
        )
        with receiver:
            with pytest.raises(ProtocolError, match="does not echo"):
                receiver.capture(samples_per_channel=1024, frame_count=1)
        assert gadget.stop_request_count == 1
    finally:
        gadget.close()
