import dataclasses
import struct
import zlib

import pytest

from spf.direct_radio.tandem_agc import (
    AD9361_TEMPERATURE_FEATURE,
    HEADER_PREFIX_BYTES_V3,
    HEADER_PREFIX_BYTES_V5,
    TEMPERATURE_INVALID,
    TANDEM_METADATA_FEATURE,
    TANDEM_METADATA_VALID_FLAG,
    TANDEM_REQUEST_MAGIC,
    RadioMetadataV5,
    TandemEventDirection,
    TandemEventReason,
    TandemGainTable,
    TandemMode,
    TandemSessionRequestV1,
    TandemState,
)
from spf.direct_radio.usb_protocol import (
    FIRST_CHANGE_UNAVAILABLE,
    GainEventFlags,
    GainEventV3,
    GainObservationFlags,
    GainObservationV3,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    RadioMetadataV3,
    SampleFormat,
)


def _v3_base() -> RadioMetadataV3:
    first = 1000
    observation = GainObservationV3(
        sample_sequence_before=990,
        sample_sequence_after=1099,
        read_duration_ns=100,
        flags=GainObservationFlags.VALID
        | GainObservationFlags.SAMPLE_INTERVAL_VALID,
        rx1_gain_index=20,
        rx2_gain_index=20,
        rx1_gain_db=10,
        rx2_gain_db=10,
    )
    event = GainEventV3(
        sample_sequence=1050,
        flags=GainEventFlags.RX1_CHANGED | GainEventFlags.RX2_CHANGED,
    )
    return RadioMetadataV3(
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
            | MetadataFeatures.FPGA_GAIN_EVENTS
            | MetadataFeatures.GAIN_DB_ENDPOINTS
            | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.GAIN_OBSERVATION_SERIES
            | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
        ),
        flags=(
            MetadataFlags.START_VALID
            | MetadataFlags.END_VALID
            | MetadataFlags.SAMPLE_SEQUENCE_VALID
            | MetadataFlags.FPGA_EVENTS_VALID
            | MetadataFlags.GAIN_FULL_TABLE_MODE
            | MetadataFlags.GAIN_DB_VALUES
            | MetadataFlags.RSSI_START_VALID
            | MetadataFlags.RSSI_END_VALID
            | MetadataFlags.GAIN_OBSERVATIONS_VALID
            | MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
        ),
        stream_id=1,
        buffer_sequence=2,
        first_sample_sequence=first,
        samples_per_channel=100,
        iq_payload_bytes=800,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_db_start=10,
        rx2_gain_db_start=10,
        rx1_gain_db_end=11,
        rx2_gain_db_end=11,
        gain_start_read_duration_ns=100,
        gain_end_read_duration_ns=100,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx1_rssi_start_qdb=100,
        rx2_rssi_start_qdb=100,
        rx1_rssi_end_qdb=100,
        rx2_rssi_end_qdb=100,
        rssi_start_read_duration_ns=100,
        rssi_end_read_duration_ns=100,
        gain_observation_interval_samples=100,
        gain_observation_capacity=1,
        gain_event_capacity=1,
        gain_observations=(observation,),
        gain_events=(event,),
    )


def _v5_frame(
    *,
    event_sequence=7,
    event_flags=0x13,
    rx2_index=21,
    temperature_mdeg_c=43_860,
) -> bytes:
    v3 = bytearray(_v3_base().pack())
    prefix = v3[:HEADER_PREFIX_BYTES_V3]
    struct.pack_into("<H", prefix, 4, 5)
    struct.pack_into("<H", prefix, 6, len(v3) + 56)
    struct.pack_into(
        "<I",
        prefix,
        8,
        struct.unpack_from("<I", prefix, 8)[0]
        | TANDEM_METADATA_FEATURE
        | AD9361_TEMPERATURE_FEATURE,
    )
    struct.pack_into("<I", prefix, 12, struct.unpack_from("<I", prefix, 12)[0] | TANDEM_METADATA_VALID_FLAG)
    extension = struct.pack(
        "<IIIIIIiiiBBBBi3I",
        9,
        int(TandemState.ARMED_AUTO),
        0,
        1,
        int(TandemGainTable.MHZ_1300_4000),
        0x30313A14,
        0,
        62,
        20,
        0,
        76,
        21,
        rx2_index,
        temperature_mdeg_c,
        0,
        0,
        0,
    )
    arrays = bytearray(v3[HEADER_PREFIX_BYTES_V3:-4])
    struct.pack_into("<QIHBB", arrays, 32, 1050, event_sequence, event_flags, 21, rx2_index)
    output = prefix + extension + arrays + bytes(4)
    output[-4:] = struct.pack("<I", zlib.crc32(output))
    assert len(prefix) + len(extension) == HEADER_PREFIX_BYTES_V5
    return bytes(output)


def test_tandem_request_has_stable_104_byte_little_endian_layout():
    request = TandemSessionRequestV1().pack()
    assert len(request) == 104
    assert struct.unpack_from("<IHH", request) == (TANDEM_REQUEST_MAGIC, 1, 104)
    assert struct.unpack_from("<iii", request, 24) == (0, 62, 20)
    assert request[72:] == bytes(32)


def test_tandem_request_rejects_frames_that_can_overrun_event_capacity():
    request = TandemSessionRequestV1()
    assert request.maximum_events_per_frame(65_536) == 22
    request.validate_frame_capacity(65_536)
    assert request.maximum_events_per_frame(524_288) == 171
    with pytest.raises(ValueError, match="cannot cover.*171 AUTO transitions"):
        request.validate_frame_capacity(524_288)

    hold = dataclasses.replace(request, mode=TandemMode.HOLD)
    assert hold.maximum_events_per_frame(524_288) == 0
    hold.validate_frame_capacity(524_288)


def test_tandem_metadata_v5_decodes_temperature_event_and_provenance():
    metadata = RadioMetadataV5.unpack(_v5_frame())
    assert metadata.ownership_epoch == 9
    assert metadata.tandem_state is TandemState.ARMED_AUTO
    assert metadata.gain_table_id is TandemGainTable.MHZ_1300_4000
    assert metadata.samples_per_channel == 100
    assert metadata.header_bytes == HEADER_PREFIX_BYTES_V5 + 32 + 16 + 4
    assert metadata.ad9361_temperature_mdeg_c == 43_860
    assert len(metadata.gain_events) == 1
    event = metadata.gain_events[0]
    assert event.event_sequence == 7
    assert event.direction is TandemEventDirection.INCREASE
    assert event.reason is TandemEventReason.BOTH_LOW_POWER
    assert event.rx1_gain_index == event.rx2_gain_index == 21


def test_tandem_metadata_v5_maps_invalid_temperature_to_none():
    metadata = RadioMetadataV5.unpack(
        _v5_frame(temperature_mdeg_c=TEMPERATURE_INVALID)
    )
    assert metadata.ad9361_temperature_mdeg_c is None


def test_tandem_metadata_v5_rejects_crc_and_unpaired_event():
    damaged = bytearray(_v5_frame())
    damaged[20] ^= 1
    with pytest.raises(ProtocolError, match="CRC"):
        RadioMetadataV5.unpack(damaged)
    with pytest.raises(ProtocolError, match="not paired"):
        RadioMetadataV5.unpack(_v5_frame(rx2_index=22))
    with pytest.raises(ProtocolError, match="unknown flag"):
        RadioMetadataV5.unpack(_v5_frame(event_flags=0x53))


def test_tandem_request_rejects_nonportable_gain_range():
    with pytest.raises(ValueError, match="cross-band"):
        TandemSessionRequestV1(maximum_gain_db=63).pack()
