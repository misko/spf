from __future__ import annotations

import dataclasses
import random
import struct

import pytest

from spf.sdrpluto.direct_ip_protocol import (
    DEFAULT_UDP_DATAGRAM_BYTES,
    IP_CONTROL_BYTES,
    IP_FRAGMENT_HEADER_BYTES,
    IpControlFlags,
    IpControlMessageV1,
    IpControlType,
    IpFragmentV1,
    IpFrameReassembler,
    ReassembledIpFrame,
    fragment_ip_frame,
    make_ip_capability_query,
    make_ip_start_request,
    make_ip_stop_request,
    reassemble_ip_datagrams,
)
from spf.sdrpluto.direct_usb_protocol import ProtocolError, RxFrameParser
from spf.sdrpluto.direct_usb_protocol import (
    FIRST_CHANGE_UNAVAILABLE,
    GainObservationFlags,
    GainObservationV3,
    MetadataFeatures,
    MetadataFlags,
    RadioMetadataV3,
    SampleFormat,
)


def test_control_capability_query_has_stable_golden_bytes():
    query = make_ip_capability_query(request_id=0x0102030405060708)
    payload = query.pack()
    assert IP_CONTROL_BYTES == 80
    assert payload.hex() == (
        "5349433101000100500000000807060504030201000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000"
    )
    assert IpControlMessageV1.unpack(payload) == query


def test_control_capabilities_round_trip():
    capabilities = IpControlMessageV1(
        message_type=IpControlType.CAPABILITIES,
        request_id=41,
        flags=(
            IpControlFlags.FINITE_RX
            | IpControlFlags.IDEMPOTENT_REQUESTS
            | IpControlFlags.TIME_ANCHOR
        ),
        protocol_min=1,
        protocol_max=3,
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
            | MetadataFeatures.GAIN_DB_ENDPOINTS
            | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.GAIN_OBSERVATION_SERIES
            | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
        ),
        max_samples_per_channel=524_288,
        max_finite_frames=16,
    )
    assert IpControlMessageV1.unpack(capabilities.pack()) == capabilities


def test_v3_start_started_and_stop_control_round_trip():
    features = (
        MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
        | MetadataFeatures.HEADER_CRC32
        | MetadataFeatures.SAMPLE_SEQUENCE
        | MetadataFeatures.GAIN_DB_ENDPOINTS
        | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
        | MetadataFeatures.GAIN_OBSERVATION_SERIES
        | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
    )
    start = make_ip_start_request(
        request_id=100,
        protocol_version=3,
        features=features,
        enabled_scan_mask=0x0F,
        samples_per_channel=524_288,
        frame_count=4,
        gain_observation_interval_samples=32_768,
        gain_observation_capacity=32,
        gain_event_capacity=0,
        data_port=40_000,
    )
    assert IpControlMessageV1.unpack(start.pack()) == start
    started = dataclasses.replace(
        start,
        message_type=IpControlType.STARTED,
        stream_id=0x1122334455667788,
    )
    first_reply = started.pack()
    # A gadget caches the reply by request_id. Retrying the same request must
    # return the exact same assigned stream rather than start a second worker.
    assert started.pack() == first_reply
    assert IpControlMessageV1.unpack(first_reply) == started
    stop = make_ip_stop_request(request_id=101, stream_id=started.stream_id)
    assert IpControlMessageV1.unpack(stop.pack()) == stop
    stopped = dataclasses.replace(stop, message_type=IpControlType.STOPPED)
    assert IpControlMessageV1.unpack(stopped.pack()) == stopped


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"stream_id": 1}, "cannot assign"),
        ({"data_port": 0}, "data port"),
        ({"max_datagram_bytes": IP_FRAGMENT_HEADER_BYTES}, "datagram size"),
        ({"gain_observation_interval_samples": 0}, "observation interval"),
        ({"gain_observation_interval_samples": 524_289}, "observation interval"),
        ({"gain_observation_capacity": 0}, "observation capacity"),
        (
            {
                "features": MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
                | MetadataFeatures.HEADER_CRC32
            },
            "lacks gain-series",
        ),
    ],
)
def test_invalid_v3_start_control_fails_closed(change, message):
    start = make_ip_start_request(
        request_id=1,
        protocol_version=3,
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.GAIN_OBSERVATION_SERIES
            | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
        ),
        enabled_scan_mask=0x0F,
        samples_per_channel=16_384,
        frame_count=1,
        gain_observation_interval_samples=16_384,
        gain_observation_capacity=1,
        data_port=30_433,
    )
    with pytest.raises(ProtocolError, match=message):
        dataclasses.replace(start, **change).pack()


def test_control_identity_and_reserved_semantics_fail_closed():
    payload = bytearray(make_ip_capability_query(request_id=1).pack())
    payload[0:4] = struct.pack("<I", 0)
    with pytest.raises(ProtocolError, match="control magic"):
        IpControlMessageV1.unpack(payload)
    payload = bytearray(make_ip_capability_query(request_id=1).pack())
    payload[4:6] = struct.pack("<H", 2)
    with pytest.raises(ProtocolError, match="control version"):
        IpControlMessageV1.unpack(payload)
    with pytest.raises(ProtocolError, match="non-zero fields"):
        dataclasses.replace(
            make_ip_capability_query(request_id=1), data_port=1234
        ).pack()


def test_fragment_header_size_and_single_datagram_round_trip():
    frame = b"complete inner frame"
    datagrams = fragment_ip_frame(frame, stream_id=7, frame_sequence=11)
    assert IP_FRAGMENT_HEADER_BYTES == 52
    assert len(datagrams) == 1
    fragment = IpFragmentV1.unpack(datagrams[0])
    assert fragment.stream_id == 7
    assert fragment.frame_sequence == 11
    assert fragment.frame_bytes == len(frame)
    assert fragment.fragment_index == 0
    assert fragment.fragment_count == 1
    assert fragment.fragment_offset == 0
    assert fragment.payload == frame
    assert reassemble_ip_datagrams(datagrams) == frame


def test_production_sized_frame_reassembles_out_of_order_with_duplicates():
    frame = bytes(range(251)) * ((4 * 1024 * 1024 // 251) + 1)
    frame = frame[: 4 * 1024 * 1024 + 8_324]
    datagrams = list(fragment_ip_frame(frame, stream_id=0x1234, frame_sequence=99))
    expected_count = (
        len(frame) + (DEFAULT_UDP_DATAGRAM_BYTES - IP_FRAGMENT_HEADER_BYTES) - 1
    ) // (DEFAULT_UDP_DATAGRAM_BYTES - IP_FRAGMENT_HEADER_BYTES)
    assert len(datagrams) == expected_count
    shuffled = datagrams[:]
    random.Random(1234).shuffle(shuffled)
    reassembler = IpFrameReassembler()
    completed = []
    for datagram in shuffled:
        completed.extend(reassembler.feed(datagram, peer=("192.0.2.1", 30433)))
    assert completed == [
        ReassembledIpFrame(stream_id=0x1234, frame_sequence=99, frame=frame)
    ]
    # Late UDP duplicates are ignored for one timeout window and do not create
    # stale partial frames which could exhaust the bounded queue.
    assert reassembler.feed(datagrams[3], peer=("192.0.2.1", 30433)) == []
    assert reassembler.feed(datagrams[-2], peer=("192.0.2.1", 30433)) == []
    assert reassembler.pending_frame_count == 0
    assert reassembler.duplicate_fragment_count == 2


def test_identical_duplicate_before_completion_is_counted_and_ignored():
    frame = bytes(range(256)) * 20
    datagrams = fragment_ip_frame(
        frame, stream_id=5, frame_sequence=6, max_datagram_bytes=256
    )
    reassembler = IpFrameReassembler()
    assert reassembler.feed(datagrams[0]) == []
    assert reassembler.feed(datagrams[0]) == []
    completed = []
    for datagram in datagrams[1:]:
        completed.extend(reassembler.feed(datagram))
    assert completed == [ReassembledIpFrame(stream_id=5, frame_sequence=6, frame=frame)]
    assert reassembler.duplicate_fragment_count == 1


def test_missing_fragment_expires_explicitly():
    datagrams = fragment_ip_frame(
        b"x" * 4096, stream_id=1, frame_sequence=2, max_datagram_bytes=256
    )
    reassembler = IpFrameReassembler(frame_timeout_seconds=0.5)
    for datagram in datagrams[:-1]:
        assert reassembler.feed(datagram, now=10.0) == []
    assert reassembler.pending_frame_count == 1
    assert reassembler.expire(now=10.49) == 0
    assert reassembler.expire(now=10.5) == 1
    assert reassembler.pending_frame_count == 0
    assert reassembler.expired_frame_count == 1


def test_frame_crc_covers_iq_payload_bytes():
    frame = b"metadata" + b"IQ" * 1000
    datagrams = list(
        fragment_ip_frame(frame, stream_id=3, frame_sequence=4, max_datagram_bytes=256)
    )
    damaged = bytearray(datagrams[2])
    damaged[-1] ^= 1
    datagrams[2] = bytes(damaged)
    with pytest.raises(ProtocolError, match="CRC"):
        reassemble_ip_datagrams(datagrams)


def test_conflicting_duplicate_discards_the_whole_frame():
    frame = b"z" * 4096
    datagrams = fragment_ip_frame(
        frame, stream_id=8, frame_sequence=9, max_datagram_bytes=256
    )
    reassembler = IpFrameReassembler()
    reassembler.feed(datagrams[0])
    changed = bytearray(datagrams[0])
    changed[-1] ^= 1
    with pytest.raises(ProtocolError, match="conflicting duplicate"):
        reassembler.feed(changed)
    assert reassembler.pending_frame_count == 0
    assert reassembler.rejected_frame_count == 1


def test_fragment_gap_or_overlap_is_rejected_after_collection():
    frame = b"0123456789" * 100
    datagrams = list(
        fragment_ip_frame(
            frame, stream_id=10, frame_sequence=11, max_datagram_bytes=256
        )
    )
    second = IpFragmentV1.unpack(datagrams[1])
    datagrams[1] = dataclasses.replace(
        second, fragment_offset=second.fragment_offset - 1
    ).pack()
    with pytest.raises(ProtocolError, match="overlap"):
        reassemble_ip_datagrams(datagrams)


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, struct.pack("<I", 0), "magic"),
        (4, struct.pack("<H", 2), "version"),
        (6, struct.pack("<H", 48), "header size"),
    ],
)
def test_bad_fragment_identity_is_rejected(offset, replacement, message):
    datagram = bytearray(fragment_ip_frame(b"abc", stream_id=1, frame_sequence=0)[0])
    datagram[offset : offset + len(replacement)] = replacement
    with pytest.raises(ProtocolError, match=message):
        IpFragmentV1.unpack(datagram)


def test_reassembled_bytes_feed_the_existing_inner_v3_frame_parser():
    first_sample = 1_000_000
    observation = GainObservationV3(
        sample_sequence_before=first_sample - 64,
        sample_sequence_after=first_sample + 14_000,
        read_duration_ns=490_000,
        flags=GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID,
        rx1_gain_index=42,
        rx2_gain_index=43,
        rx1_gain_db=20,
        rx2_gain_db=21,
    )
    metadata = RadioMetadataV3(
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
            | MetadataFeatures.GAIN_DB_ENDPOINTS
            | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.GAIN_OBSERVATION_SERIES
            | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
        ),
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
        stream_id=44,
        buffer_sequence=0,
        first_sample_sequence=first_sample,
        samples_per_channel=16_384,
        iq_payload_bytes=16_384 * 8,
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
        gain_observation_interval_samples=16_384,
        gain_observation_capacity=1,
        gain_observations=(observation,),
    )
    iq = bytes(metadata.iq_payload_bytes)
    inner_frame = metadata.pack() + iq
    reassembled = reassemble_ip_datagrams(
        fragment_ip_frame(inner_frame, stream_id=44, frame_sequence=0)
    )
    parser = RxFrameParser(protocol_version=3)
    frames = parser.feed(reassembled)
    parser.finish()
    assert len(frames) == 1
    assert frames[0].metadata == metadata
    assert frames[0].iq_payload == iq


def test_pending_declared_bytes_are_bounded():
    first = fragment_ip_frame(
        b"a" * 4096,
        stream_id=1,
        frame_sequence=1,
        max_datagram_bytes=256,
    )
    second = fragment_ip_frame(
        b"b" * 4096,
        stream_id=1,
        frame_sequence=2,
        max_datagram_bytes=256,
    )
    reassembler = IpFrameReassembler(max_pending_bytes=4096)
    assert reassembler.feed(first[0]) == []
    assert reassembler.pending_declared_bytes == 4096
    with pytest.raises(ProtocolError, match="pending-byte"):
        reassembler.feed(second[0])


def test_invalid_limits_fail_closed():
    with pytest.raises(ProtocolError, match="frame size"):
        fragment_ip_frame(b"", stream_id=0, frame_sequence=0)
    with pytest.raises(ProtocolError, match="datagram limit"):
        fragment_ip_frame(
            b"x",
            stream_id=0,
            frame_sequence=0,
            max_datagram_bytes=IP_FRAGMENT_HEADER_BYTES,
        )
