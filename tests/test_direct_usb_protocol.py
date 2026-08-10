import dataclasses
import itertools
import struct
import zlib

import numpy as np
import pytest

from spf.sdrpluto.sdr_controller import PPlus

from spf.sdrpluto.direct_usb_protocol import (
    CAPABILITIES_BYTES,
    FIRST_CHANGE_UNAVAILABLE,
    HEADER_BYTES,
    HEADER_BYTES_V2,
    GAIN_OBSERVATION_BYTES,
    RUNTIME_STATUS_BYTES,
    RUNTIME_STATUS_MAGIC,
    RUNTIME_STATUS_VERSION,
    START_REQUEST_BYTES,
    TIME_ANCHOR_BYTES,
    CapabilityFlags,
    GadgetCapabilitiesV1,
    GainMetadataV1,
    GainObservationFlags,
    GainObservationV3,
    RadioMetadataV2,
    RadioMetadataV3,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    ErrorSubsystem,
    RuntimeState,
    RuntimeStatusFlags,
    RuntimeStatusV1,
    TimeAnchorFlags,
    TimeAnchorV1,
    RxFrameParser,
    SampleFormat,
    pack_start_request_v1,
    pack_start_request_v2,
    pack_start_request_v3,
    pack_time_anchor_query,
)
from spf.sdrpluto.sample_clock import HostTimeAnchorMeasurement


def metadata(
    *,
    buffer_sequence=0,
    first_sample_sequence=0,
    samples_per_channel=8,
    stream_id=0x123456789ABCDEF0,
):
    return GainMetadataV1(
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        flags=(
            MetadataFlags.START_VALID
            | MetadataFlags.END_VALID
            | MetadataFlags.SAMPLE_SEQUENCE_VALID
            | MetadataFlags.GAIN_FULL_TABLE_MODE
            | MetadataFlags.RX1_ENDPOINT_CHANGED
        ),
        stream_id=stream_id,
        buffer_sequence=buffer_sequence,
        first_sample_sequence=first_sample_sequence,
        samples_per_channel=samples_per_channel,
        iq_payload_bytes=samples_per_channel * 8,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_start=42,
        rx2_gain_start=43,
        rx1_gain_end=41,
        rx2_gain_end=43,
        gain_start_read_duration_ns=1200,
        gain_end_read_duration_ns=1300,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
    )


def frame(meta):
    return meta.pack() + bytes(range(meta.iq_payload_bytes))


def metadata_v2(
    *,
    buffer_sequence=0,
    first_sample_sequence=0,
    samples_per_channel=8,
    stream_id=0x123456789ABCDEF0,
):
    return RadioMetadataV2(
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
            | MetadataFeatures.GAIN_DB_ENDPOINTS
            | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
        ),
        flags=(
            MetadataFlags.START_VALID
            | MetadataFlags.END_VALID
            | MetadataFlags.SAMPLE_SEQUENCE_VALID
            | MetadataFlags.GAIN_FULL_TABLE_MODE
            | MetadataFlags.GAIN_DB_VALUES
            | MetadataFlags.RSSI_START_VALID
            | MetadataFlags.RSSI_END_VALID
            | MetadataFlags.RX1_ENDPOINT_CHANGED
        ),
        stream_id=stream_id,
        buffer_sequence=buffer_sequence,
        first_sample_sequence=first_sample_sequence,
        samples_per_channel=samples_per_channel,
        iq_payload_bytes=samples_per_channel * 8,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_db_start=42,
        rx2_gain_db_start=43,
        rx1_gain_db_end=41,
        rx2_gain_db_end=43,
        gain_start_read_duration_ns=1200,
        gain_end_read_duration_ns=1300,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx1_rssi_start_qdb=401,
        rx2_rssi_start_qdb=402,
        rx1_rssi_end_qdb=403,
        rx2_rssi_end_qdb=404,
        rssi_start_read_duration_ns=1400,
        rssi_end_read_duration_ns=1500,
    )


def metadata_v3(
    *,
    buffer_sequence=0,
    first_sample_sequence=1_000_000,
    samples_per_channel=32768,
    stream_id=0x123456789ABCDEF0,
):
    observations = (
        GainObservationV3(
            sample_sequence_before=first_sample_sequence - 64,
            sample_sequence_after=first_sample_sequence + 14000,
            read_duration_ns=490_000,
            flags=(
                GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID
            ),
            rx1_gain_index=42,
            rx2_gain_index=43,
            rx1_gain_db=20,
            rx2_gain_db=21,
        ),
        GainObservationV3(
            sample_sequence_before=first_sample_sequence + 32000,
            sample_sequence_after=first_sample_sequence + 32767,
            read_duration_ns=27_000,
            flags=(
                GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID
            ),
            rx1_gain_index=41,
            rx2_gain_index=43,
            rx1_gain_db=19,
            rx2_gain_db=21,
        ),
    )

    return RadioMetadataV3(
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
            | MetadataFlags.RX1_ENDPOINT_CHANGED
            | MetadataFlags.GAIN_OBSERVATIONS_VALID
            | MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
        ),
        stream_id=stream_id,
        buffer_sequence=buffer_sequence,
        first_sample_sequence=first_sample_sequence,
        samples_per_channel=samples_per_channel,
        iq_payload_bytes=samples_per_channel * 8,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_db_start=20,
        rx2_gain_db_start=21,
        rx1_gain_db_end=19,
        rx2_gain_db_end=21,
        gain_start_read_duration_ns=490_000,
        gain_end_read_duration_ns=27_000,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx1_rssi_start_qdb=401,
        rx2_rssi_start_qdb=402,
        rx1_rssi_end_qdb=403,
        rx2_rssi_end_qdb=404,
        rssi_start_read_duration_ns=1400,
        rssi_end_read_duration_ns=1500,
        gain_observation_interval_samples=32768,
        gain_observation_capacity=4,
        gain_observations=observations,
    )


def test_spf_frame_time_fit_uses_observed_fpga_rate_not_device_rate():
    meta = metadata_v3()
    actual_counter_rate = 3_750_000.0
    first_anchor_counter = meta.first_sample_sequence - 100_000
    second_anchor_counter = (
        meta.first_sample_sequence + meta.samples_per_channel + 100_000
    )

    def measurement(request_id, counter):
        host_midpoint = int(
            5_000_000_000 + (counter - first_anchor_counter) * 1e9 / actual_counter_rate
        )
        anchor = TimeAnchorV1(
            flags=(
                TimeAnchorFlags.COUNTER_INTERVAL_VALID
                | TimeAnchorFlags.MONOTONIC_INTERVAL_VALID
                | TimeAnchorFlags.COUNTER_LOW32
                | TimeAnchorFlags.COUNTER_ADVANCED
            ),
            request_id=request_id,
            radio_monotonic_before_ns=request_id * 1_000_000,
            sample_counter_before=counter & 0xFFFFFFFF,
            sample_counter_after=(counter + 2) & 0xFFFFFFFF,
            radio_monotonic_after_ns=request_id * 1_000_000 + 1000,
        )
        return HostTimeAnchorMeasurement(
            anchor=anchor,
            host_monotonic_before_ns=host_midpoint - 100_000,
            host_monotonic_after_ns=host_midpoint + 100_000,
            transport="test",
        )

    radio = PPlus.__new__(PPlus)
    # This deliberately differs by 8x from the FPGA counter rate. The observed
    # anchors are authoritative; the configured rate is only a fallback.
    radio.rx_config = type("RxConfig", (), {"sample_rate": 30_000_000})()
    radio._direct_time_anchors = [
        measurement(1, first_anchor_counter),
        measurement(2, second_anchor_counter),
    ]

    result = radio._fit_direct_sample_time(meta)

    assert result["sample_counter_end_exclusive"] == (
        meta.first_sample_sequence + meta.samples_per_channel
    )
    assert result["sample_time_fitted_rate_hz"] == pytest.approx(
        actual_counter_rate, rel=1e-5
    )
    expected_duration_ns = meta.samples_per_channel * 1e9 / actual_counter_rate
    assert result["sample_time_monotonic_end_ns"] - result[
        "sample_time_monotonic_start_ns"
    ] == pytest.approx(expected_duration_ns, abs=2)
    assert result["sample_time_uncertainty_ns"] < 1_000_000


def test_v3_gain_series_round_trip_and_arbitrary_hardware_sequence():
    expected = metadata_v3()
    packed = expected.pack()
    assert expected.header_bytes == 124 + 4 * GAIN_OBSERVATION_BYTES + 4
    assert len(packed) == expected.header_bytes
    assert RadioMetadataV3.unpack(packed) == expected
    parser = RxFrameParser(protocol_version=3)
    wire = expected.pack() + bytes(expected.iq_payload_bytes)
    parsed = []
    for chunk_start in range(0, len(wire), 137):
        parsed.extend(parser.feed(wire[chunk_start : chunk_start + 137]))
    parser.finish()
    assert len(parsed) == 1
    assert parsed[0].metadata == expected


def test_v3_rejects_observation_outside_frame_and_crc_damage():
    original = metadata_v3()
    with pytest.raises(ProtocolError, match="at least one"):
        dataclasses.replace(original, gain_observations=()).pack()
    outside = dataclasses.replace(
        original.gain_observations[0],
        sample_sequence_before=original.first_sample_sequence - 1000,
        sample_sequence_after=original.first_sample_sequence - 1,
    )
    with pytest.raises(ProtocolError, match="does not overlap"):
        dataclasses.replace(
            original,
            gain_observations=(outside, original.gain_observations[1]),
        ).pack()
    damaged = bytearray(original.pack())
    damaged[130] ^= 1
    with pytest.raises(ProtocolError, match="CRC"):
        RadioMetadataV3.unpack(damaged)


def test_v3_start_request_negotiates_series_shape():
    features = metadata_v3().features
    request = pack_start_request_v3(
        requested_features=features,
        enabled_scan_mask=0x0F,
        samples_per_channel=524288,
        frame_count=1,
        gain_observation_interval_samples=32768,
        gain_observation_capacity=16,
    )
    assert request[:4] == b"SGS3"
    assert struct.unpack_from("<II", request, 24) == (32768, 16)


def test_v2_header_golden_vector_and_legacy_units():
    expected = metadata_v2(buffer_sequence=7, first_sample_sequence=1024)
    packed = expected.pack()
    assert HEADER_BYTES_V2 == 96
    assert packed.hex() == (
        "53474d31020060003700000017840500"
        "f0debc9a785634120700000000000000"
        "00040000000000000800000040000000"
        "0f0000000100022a2b292b00b0040000"
        "14050000ffffffffffffffff91019201"
        "9301940178050000dc0500006ff46dc7"
    )
    parsed = RadioMetadataV2.unpack(packed)
    assert parsed == expected
    assert parsed.gain_db_end == (41.0, 43.0)
    assert parsed.rssi_db_end == (100.75, 101.0)
    assert parsed.gain_endpoints_equal == (False, True)


def test_v2_parser_handles_fragmentation_and_sequence():
    first = metadata_v2()
    second = metadata_v2(buffer_sequence=1, first_sample_sequence=8)
    parser = RxFrameParser(protocol_version=2)
    wire = frame(first) + frame(second)
    parsed = []
    for byte in wire:
        parsed.extend(parser.feed(bytes([byte])))
    assert [item.metadata for item in parsed] == [first, second]
    parser.finish()


def test_v2_rejects_invalid_rssi_and_gain_sentinels():
    with pytest.raises(ProtocolError, match="RSSI_START_VALID"):
        dataclasses.replace(metadata_v2(), rx1_rssi_start_qdb=0xFFFF).pack()
    with pytest.raises(ProtocolError, match="START_VALID"):
        dataclasses.replace(metadata_v2(), rx1_gain_db_start=-128).pack()


def test_v2_invalid_sentinels_present_as_nan_to_host_code():
    valid = metadata_v2()
    invalid = dataclasses.replace(
        valid,
        flags=(
            valid.flags
            & ~MetadataFlags.START_VALID
            & ~MetadataFlags.RSSI_START_VALID
            & ~MetadataFlags.RX1_ENDPOINT_CHANGED
            | MetadataFlags.GAIN_READ_FAILED
            | MetadataFlags.RSSI_READ_FAILED
        ),
        rx1_gain_db_start=-128,
        rx2_gain_db_start=-128,
        rx1_rssi_start_qdb=0xFFFF,
        rx2_rssi_start_qdb=0xFFFF,
    )
    assert all(np.isnan(value) for value in invalid.gain_db_start)
    assert all(np.isnan(value) for value in invalid.rssi_db_start)
    assert RadioMetadataV2.unpack(invalid.pack()) == invalid


def test_v2_raw_state_change_flag_survives_equal_rounded_db():
    original = metadata_v2()
    equal_db_changed_raw_state = dataclasses.replace(
        original,
        rx1_gain_db_end=original.rx1_gain_db_start,
    )
    parsed = RadioMetadataV2.unpack(equal_db_changed_raw_state.pack())
    assert parsed.gain_db_start[0] == parsed.gain_db_end[0]
    assert parsed.gain_endpoints_equal[0] is False


def test_v2_start_request_is_disjoint_from_v1():
    features = (
        MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
        | MetadataFeatures.HEADER_CRC32
        | MetadataFeatures.SAMPLE_SEQUENCE
        | MetadataFeatures.GAIN_DB_ENDPOINTS
        | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
    )
    request = pack_start_request_v2(
        requested_features=features,
        enabled_scan_mask=0x0F,
        samples_per_channel=8,
        frame_count=1,
    )
    assert request[:4] == b"SGS2"
    assert struct.unpack_from("<H", request, 4)[0] == 2


def test_header_size_and_golden_prefix():
    golden_metadata = metadata(buffer_sequence=7, first_sample_sequence=1024)
    packed = golden_metadata.pack()
    assert HEADER_BYTES == 80
    assert len(packed) == HEADER_BYTES
    assert packed.hex() == (
        "53474d3101005000"
        "0700000017040000"
        "f0debc9a78563412"
        "0700000000000000"
        "0004000000000000"
        "0800000040000000"
        "0f0000000100022a"
        "2b292b00b0040000"
        "14050000ffffffff"
        "ffffffff796afe5d"
    )
    assert GainMetadataV1.unpack(packed) == golden_metadata


@pytest.mark.parametrize("split", range(1, HEADER_BYTES + 10))
def test_arbitrary_two_chunk_fragmentation(split):
    expected_meta = metadata()
    wire = frame(expected_meta)
    parser = RxFrameParser()
    assert parser.feed(wire[:split]) == []
    parsed = parser.feed(wire[split:])
    assert len(parsed) == 1
    assert parsed[0].metadata == expected_meta
    assert parsed[0].iq_payload == bytes(range(expected_meta.iq_payload_bytes))
    parser.finish()


def test_single_byte_fragmentation_and_concatenated_frames():
    first = metadata()
    second = metadata(
        buffer_sequence=1,
        first_sample_sequence=first.first_sample_sequence + first.samples_per_channel,
    )
    parser = RxFrameParser()
    parsed = list(
        itertools.chain.from_iterable(
            parser.feed(bytes([byte])) for byte in frame(first) + frame(second)
        )
    )
    assert [item.metadata for item in parsed] == [first, second]


@pytest.mark.parametrize(
    "offset",
    [
        0,  # magic
        4,  # version
        6,  # header size
        16,  # stream ID
        HEADER_BYTES - 1,  # CRC
    ],
)
def test_corrupt_header_is_rejected_and_parser_resets(offset):
    wire = bytearray(frame(metadata()))
    wire[offset] ^= 0x01
    parser = RxFrameParser()
    with pytest.raises(ProtocolError):
        parser.feed(wire)
    assert parser.feed(frame(metadata()))[0].metadata == metadata()


def test_short_stream_rejected_at_finish():
    parser = RxFrameParser()
    parser.feed(frame(metadata())[:-1])
    with pytest.raises(ProtocolError, match="unframed"):
        parser.finish()


def test_buffer_sequence_gap_rejected():
    first = metadata()
    third = metadata(
        buffer_sequence=9,
        first_sample_sequence=first.first_sample_sequence + first.samples_per_channel,
    )
    parser = RxFrameParser()
    parser.feed(frame(first))
    with pytest.raises(ProtocolError, match="buffer sequence discontinuity"):
        parser.feed(frame(third))


def test_sample_sequence_gap_rejected():
    first = metadata()
    second = metadata(buffer_sequence=1, first_sample_sequence=9999)
    parser = RxFrameParser()
    parser.feed(frame(first))
    with pytest.raises(ProtocolError, match="sample sequence discontinuity"):
        parser.feed(frame(second))


def test_stream_change_requires_reset():
    first = metadata()
    replacement = metadata(
        buffer_sequence=0,
        first_sample_sequence=0,
        stream_id=first.stream_id + 1,
    )
    parser = RxFrameParser()
    parser.feed(frame(first))
    with pytest.raises(ProtocolError, match="stream ID changed"):
        parser.feed(frame(replacement))
    parser.reset()
    assert parser.feed(frame(replacement))[0].metadata == replacement


def test_payload_size_must_match_format():
    valid = metadata()
    invalid = dataclasses.replace(valid, iq_payload_bytes=valid.iq_payload_bytes + 2)
    with pytest.raises(ProtocolError, match="payload size mismatch"):
        invalid.pack()


def test_invalid_gain_sentinel_requires_invalid_flag():
    invalid = dataclasses.replace(metadata(), rx1_gain_start=0xFF)
    with pytest.raises(ProtocolError, match="START_VALID"):
        invalid.pack()


def test_equal_endpoints_are_only_a_comparison():
    valid = metadata()
    equal = dataclasses.replace(
        valid,
        flags=valid.flags & ~MetadataFlags.RX1_ENDPOINT_CHANGED,
        rx1_gain_end=valid.rx1_gain_start,
    )
    assert equal.gain_endpoints_equal == (True, True)
    assert equal.gain_metadata_valid
    assert GainMetadataV1.unpack(equal.pack()) == equal


def header_with_recomputed_crc(offset, replacement):
    header = bytearray(metadata().pack())
    header[offset : offset + len(replacement)] = replacement
    header[-4:] = b"\x00" * 4
    struct.pack_into("<I", header, HEADER_BYTES - 4, zlib.crc32(header) & 0xFFFFFFFF)
    return bytes(header)


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, struct.pack("<I", 0xDEADBEEF), "bad metadata magic"),
        (4, struct.pack("<H", 2), "unsupported metadata version"),
        (6, struct.pack("<H", HEADER_BYTES - 1), "unsupported v1 header size"),
        (48, struct.pack("<I", 0x03), "scan elements"),
        (52, struct.pack("<H", 99), "unsupported sample format"),
        (54, b"\x01", "two complex RX channels"),
        (59, b"\x01", "reserved0 must be zero"),
    ],
)
def test_individual_header_fields_rejected_with_valid_crc(offset, replacement, message):
    with pytest.raises(ProtocolError, match=message):
        GainMetadataV1.unpack(header_with_recomputed_crc(offset, replacement))


def test_unknown_feature_and_flag_bits_rejected():
    with pytest.raises(ProtocolError, match="unknown metadata feature"):
        dataclasses.replace(
            metadata(),
            features=MetadataFeatures(int(metadata().features) | (1 << 31)),
        ).pack()
    with pytest.raises(ProtocolError, match="unknown metadata flag"):
        dataclasses.replace(
            metadata(),
            flags=MetadataFlags(int(metadata().flags) | (1 << 31)),
        ).pack()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stream_id", -1, "uint64"),
        ("stream_id", 1 << 64, "uint64"),
        ("samples_per_channel", 1 << 32, "uint32"),
        ("rx1_gain_start", 1 << 8, "uint8"),
    ],
)
def test_out_of_range_integer_is_protocol_error(field, value, message):
    with pytest.raises(ProtocolError, match=message):
        dataclasses.replace(metadata(), **{field: value}).pack()


def test_new_stream_sequences_must_start_at_zero():
    parser = RxFrameParser()
    with pytest.raises(ProtocolError, match="buffer sequence 0"):
        parser.feed(frame(metadata(buffer_sequence=1)))

    parser = RxFrameParser()
    with pytest.raises(ProtocolError, match="sample sequence 0"):
        parser.feed(frame(metadata(first_sample_sequence=1)))


def test_extra_payload_is_not_silently_accepted():
    parser = RxFrameParser()
    parsed = parser.feed(frame(metadata()) + b"unexpected")
    assert len(parsed) == 1
    with pytest.raises(ProtocolError, match="unframed"):
        parser.finish()


def test_complete_frame_fast_path_matches_streaming_parser():
    wire = frame(metadata())
    expected = RxFrameParser().feed(wire)[0]
    assert RxFrameParser().parse_complete_frame(bytearray(wire)) == expected


def test_complete_frame_fast_path_rejects_length_mismatch():
    parser = RxFrameParser()
    with pytest.raises(ProtocolError, match="complete frame length mismatch"):
        parser.parse_complete_frame(frame(metadata()) + b"unexpected")


def test_missing_required_feature_is_rejected():
    with pytest.raises(ProtocolError, match="GAIN_ENDPOINT_SNAPSHOTS"):
        dataclasses.replace(
            metadata(),
            features=metadata().features & ~MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS,
        ).pack()


def test_sample_validity_requires_negotiated_feature():
    with pytest.raises(ProtocolError, match="sample sequence marked valid"):
        dataclasses.replace(
            metadata(),
            features=metadata().features & ~MetadataFeatures.SAMPLE_SEQUENCE,
        ).pack()


def test_invalid_gain_metadata_fails_closed():
    flags = (
        metadata().flags
        & ~MetadataFlags.START_VALID
        & ~MetadataFlags.END_VALID
        & ~MetadataFlags.RX1_ENDPOINT_CHANGED
        | MetadataFlags.GAIN_READ_FAILED
    )
    invalid = dataclasses.replace(
        metadata(),
        flags=flags,
        rx1_gain_start=0xFF,
        rx2_gain_start=0xFF,
        rx1_gain_end=0xFF,
        rx2_gain_end=0xFF,
    )
    assert invalid.gain_metadata_valid is False
    assert invalid.gain_endpoints_equal == (False, False)
    assert GainMetadataV1.unpack(invalid.pack()) == invalid


def test_dummy_gain_metadata_is_never_application_valid():
    dummy = dataclasses.replace(
        metadata(),
        flags=metadata().flags | MetadataFlags.DUMMY_GAINS,
    )
    assert GainMetadataV1.unpack(dummy.pack()) == dummy
    assert dummy.gain_metadata_valid is False
    assert dummy.gain_endpoints_equal == (False, False)


def test_capability_response_matches_c_golden():
    payload = bytes.fromhex(
        "5347435020000100" "0100000007000000" "ffffff1f10000000" "0300000000000000"
    )
    assert len(payload) == CAPABILITIES_BYTES == 32
    assert GadgetCapabilitiesV1.unpack(payload) == GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=0x1FFFFFFF,
        max_finite_frames=16,
        capability_flags=(CapabilityFlags.FINITE_RX | CapabilityFlags.DUMMY_GAINS),
    )


def test_runtime_status_response_matches_c_layout():
    payload = struct.pack(
        "<IHHHHiII16s16sQQ14I",
        RUNTIME_STATUS_MAGIC,
        RUNTIME_STATUS_BYTES,
        RUNTIME_STATUS_VERSION,
        RuntimeState.STREAMING,
        ErrorSubsystem.USB_SUBMIT,
        5,
        int(
            RuntimeStatusFlags.BOOT_ID_VALID
            | RuntimeStatusFlags.PROCESS_NONCE_VALID
            | RuntimeStatusFlags.RX_WORKER_ACTIVE
        ),
        0,
        b"\x11" * 16,
        b"\x22" * 16,
        0x0102030405060708,
        9,
        *range(10, 23),
        0,
    )
    assert len(payload) == RUNTIME_STATUS_BYTES == 128
    status = RuntimeStatusV1.unpack(payload)
    assert status.lifecycle_state is RuntimeState.STREAMING
    assert status.last_error_subsystem is ErrorSubsystem.USB_SUBMIT
    assert status.last_errno == 5
    assert status.boot_id == b"\x11" * 16
    assert status.process_nonce == b"\x22" * 16
    assert status.current_stream_id == 0x0102030405060708
    assert status.last_completed_sequence == 9
    assert status.start_count == 10
    assert status.stop_timeout_count == 21
    assert status.worker_heartbeat_age_ms == 22


def test_time_anchor_records_match_c_golden_and_wrap_safely():
    query = pack_time_anchor_query(request_id=0x0102030405060708)
    assert query.hex() == "5354513118000100080706050403020100000000055b3187"

    anchor = TimeAnchorV1(
        flags=(
            TimeAnchorFlags.COUNTER_INTERVAL_VALID
            | TimeAnchorFlags.MONOTONIC_INTERVAL_VALID
            | TimeAnchorFlags.COUNTER_LOW32
            | TimeAnchorFlags.COUNTER_ADVANCED
        ),
        request_id=7,
        radio_monotonic_before_ns=1000,
        sample_counter_before=0xFFFFFFF0,
        sample_counter_after=0x10,
        radio_monotonic_after_ns=2000,
    )
    payload = anchor.pack()
    assert len(payload) == TIME_ANCHOR_BYTES == 64
    assert payload.hex() == (
        "53544131400001000f000000000000000700000000000000"
        "e803000000000000f0ffffff000000001000000000000000"
        "d0070000000000000000000040df4b9e"
    )
    assert TimeAnchorV1.unpack(payload) == anchor
    assert anchor.counter_delta == 32
    assert anchor.radio_interval_ns == 1000


@pytest.mark.parametrize("offset", [0, 4, 6, 8, 12, 56, 60])
def test_corrupt_time_anchor_is_rejected(offset):
    anchor = TimeAnchorV1(
        flags=(
            TimeAnchorFlags.COUNTER_INTERVAL_VALID
            | TimeAnchorFlags.MONOTONIC_INTERVAL_VALID
            | TimeAnchorFlags.COUNTER_LOW32
            | TimeAnchorFlags.COUNTER_ADVANCED
        ),
        request_id=1,
        radio_monotonic_before_ns=10,
        sample_counter_before=100,
        sample_counter_after=101,
        radio_monotonic_after_ns=20,
    )
    payload = bytearray(anchor.pack())
    payload[offset] ^= 0x80
    with pytest.raises(ProtocolError):
        TimeAnchorV1.unpack(payload)


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, b"BAD!", "magic"),
        (4, b"\x7f\x00", "size"),
        (6, b"\x02\x00", "version"),
        (8, b"\xff\xff", "state or subsystem"),
        (10, b"\xff\xff", "state or subsystem"),
        (16, b"\x00\x00\x00\x80", "flags"),
        (20, b"\x01\x00\x00\x00", "reserved"),
        (124, b"\x01\x00\x00\x00", "reserved"),
    ],
)
def test_corrupt_runtime_status_is_rejected(offset, replacement, message):
    payload = bytearray(
        struct.pack(
            "<IHHHHiII16s16sQQ14I",
            RUNTIME_STATUS_MAGIC,
            RUNTIME_STATUS_BYTES,
            RUNTIME_STATUS_VERSION,
            RuntimeState.IDLE,
            ErrorSubsystem.NONE,
            0,
            int(
                RuntimeStatusFlags.BOOT_ID_VALID
                | RuntimeStatusFlags.PROCESS_NONCE_VALID
            ),
            0,
            b"\x11" * 16,
            b"\x22" * 16,
            0,
            0xFFFFFFFFFFFFFFFF,
            *([0] * 14),
        )
    )
    payload[offset : offset + len(replacement)] = replacement
    with pytest.raises(ProtocolError, match=message):
        RuntimeStatusV1.unpack(payload)


def test_start_request_matches_c_golden():
    request = pack_start_request_v1(
        requested_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        enabled_scan_mask=0x0F,
        samples_per_channel=524288,
        frame_count=1,
    )
    assert len(request) == START_REQUEST_BYTES == 32
    assert request.hex() == (
        "5347533101002000" "070000000f000000" "0000080001000000" "0000000000000000"
    )


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (0, b"\x00\x00\x00\x00"),
        (4, b"\x1f\x00"),
        (6, b"\x02\x00"),
        (8, b"\x00\x00"),
        (10, b"\x01\x00"),
        (12, b"\x00\x00\x00\x80"),
        (16, b"\x00\x00\x00\x00"),
        (20, b"\x00\x00\x00\x00"),
        (24, b"\x00\x00\x00\x80"),
        (28, b"\x01\x00\x00\x00"),
    ],
)
def test_corrupt_capability_fields_are_rejected(offset, replacement):
    payload = bytearray(
        bytes.fromhex(
            "5347435020000100" "0100000007000000" "ffffff1f10000000" "0300000000000000"
        )
    )
    payload[offset : offset + len(replacement)] = replacement
    with pytest.raises(ProtocolError):
        GadgetCapabilitiesV1.unpack(payload)


def test_start_request_rejects_non_finite_or_wrong_layout():
    required = (
        MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
        | MetadataFeatures.HEADER_CRC32
        | MetadataFeatures.SAMPLE_SEQUENCE
    )
    with pytest.raises(ProtocolError, match="feature mask"):
        pack_start_request_v1(
            requested_features=MetadataFeatures.HEADER_CRC32,
            enabled_scan_mask=0x0F,
            samples_per_channel=524288,
            frame_count=1,
        )
    with pytest.raises(ProtocolError, match="scan mask"):
        pack_start_request_v1(
            requested_features=required,
            enabled_scan_mask=0x03,
            samples_per_channel=524288,
            frame_count=1,
        )
    with pytest.raises(ProtocolError, match="frame_count"):
        pack_start_request_v1(
            requested_features=required,
            enabled_scan_mask=0x0F,
            samples_per_channel=524288,
            frame_count=0,
        )
