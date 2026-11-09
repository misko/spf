"""Forward-only tandem-AGC session request and radio metadata v4."""

from __future__ import annotations

import dataclasses
import enum
import struct
import zlib
from typing import Final

from spf.direct_radio.usb_protocol import (
    GAIN_EVENT_BYTES,
    GAIN_OBSERVATION_BYTES,
    HEADER_PREFIX_BYTES_V3,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    RadioMetadataV3,
    VERSION_V3,
)

TANDEM_REQUEST_MAGIC: Final[int] = 0x54465053  # little-endian b"SPFT"
TANDEM_ABI_VERSION: Final[int] = 1
TANDEM_REQUIRED_FEATURES: Final[int] = 0x7
TANDEM_POLICY_FAIL_SESSION: Final[int] = 0
VERSION_V4: Final[int] = 4
TANDEM_METADATA_FEATURE: Final[int] = 1 << 8
TANDEM_METADATA_VALID_FLAG: Final[int] = 1 << 22
HEADER_EXTENSION_BYTES_V4: Final[int] = 56
HEADER_PREFIX_BYTES_V4: Final[int] = (
    HEADER_PREFIX_BYTES_V3 + HEADER_EXTENSION_BYTES_V4
)

_REQUEST_STRUCT: Final[struct.Struct] = struct.Struct("<IHHIIIIiiiIIIIII4BII8I")
_IDENTITY_STRUCT: Final[struct.Struct] = struct.Struct("<IHHII")
_V3_EXTENSION_STRUCT: Final[struct.Struct] = struct.Struct("<IHHHHHHIIII")
_V4_EXTENSION_STRUCT: Final[struct.Struct] = struct.Struct("<IIIIIIiiiBBBB4I")
_EVENT_STRUCT: Final[struct.Struct] = struct.Struct("<QIHBB")
_LEGACY_EVENT_STRUCT: Final[struct.Struct] = struct.Struct("<QHHI")
assert _REQUEST_STRUCT.size == 104
assert _V4_EXTENSION_STRUCT.size == HEADER_EXTENSION_BYTES_V4
assert _EVENT_STRUCT.size == GAIN_EVENT_BYTES


class TandemMode(enum.IntEnum):
    HOLD = 0
    AUTO = 1


class TandemState(enum.IntEnum):
    IDLE = 0
    VALIDATING = 1
    ARMED_HOLD = 2
    ARMED_AUTO = 3
    FAULTED = 4
    RESTORING = 5


class TandemGainTable(enum.IntEnum):
    MHZ_200_1300 = 1
    MHZ_1300_4000 = 2
    MHZ_4000_6000 = 3


class TandemEventDirection(enum.IntEnum):
    INCREASE = 1
    DECREASE = 2


class TandemEventReason(enum.IntEnum):
    LARGE_LMT_OVERLOAD = 0
    LARGE_ADC_OVERLOAD = 1
    SMALL_ADC_INHIBIT = 2
    BOTH_LOW_POWER = 3
    PEER = 4
    CLAMPED = 5
    INITIAL = 6


@dataclasses.dataclass(frozen=True, slots=True)
class TandemSessionRequestV1:
    """The exact 104-byte little-endian request owned by the SPF provider."""

    mode: TandemMode = TandemMode.AUTO
    observation_capacity: int = 64
    event_capacity: int = 64
    minimum_gain_db: int = 0
    maximum_gain_db: int = 62
    initial_gain_db: int = 20
    power_measurement_samples: int = 1024
    low_power_dwell_periods: int = 3
    cooldown_periods: int = 2
    pulse_high_cycles: int = 4
    pulse_low_cycles: int = 4
    detector_blanking_cycles: int = 8
    low_power_threshold: int = 20
    large_lmt_overload_threshold: int = 58
    large_adc_overload_threshold: int = 49
    small_adc_overload_threshold: int = 48

    def maximum_events_per_frame(self, samples_per_channel: int) -> int:
        """Return a conservative AUTO transition bound for one IQ frame."""

        if not isinstance(samples_per_channel, int) or samples_per_channel <= 0:
            raise ValueError("samples per channel must be a positive integer")
        if (
            not isinstance(self.power_measurement_samples, int)
            or self.power_measurement_samples <= 0
            or not isinstance(self.cooldown_periods, int)
            or self.cooldown_periods < 0
        ):
            raise ValueError("tandem period must be positive and cooldown nonnegative")
        if self.mode is TandemMode.HOLD:
            return 0
        minimum_transition_samples = self.power_measurement_samples * (
            self.cooldown_periods + 1
        )
        return 1 + (samples_per_channel - 1) // minimum_transition_samples

    def validate_frame_capacity(self, samples_per_channel: int) -> None:
        maximum_events = self.maximum_events_per_frame(samples_per_channel)
        if maximum_events > self.event_capacity:
            raise ValueError(
                "tandem event capacity "
                f"{self.event_capacity} cannot cover the worst-case "
                f"{maximum_events} AUTO transitions in a "
                f"{samples_per_channel}-sample frame"
            )

    def pack(self) -> bytes:
        values = (
            self.observation_capacity,
            self.event_capacity,
            self.power_measurement_samples,
            self.low_power_dwell_periods,
            self.cooldown_periods,
            self.pulse_high_cycles,
            self.pulse_low_cycles,
            self.detector_blanking_cycles,
        )
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("tandem capacities and timing values must be positive")
        if not 0 <= self.minimum_gain_db <= self.initial_gain_db <= self.maximum_gain_db:
            raise ValueError("tandem gains must be ordered and nonnegative")
        if self.maximum_gain_db > 62:
            raise ValueError("default cross-band tandem gain range ends at 62 dB")
        if self.observation_capacity > 64 or self.event_capacity > 64:
            raise ValueError("firmware tandem capacities cannot exceed 64")
        byte_values = (
            self.low_power_threshold,
            self.large_lmt_overload_threshold,
            self.large_adc_overload_threshold,
            self.small_adc_overload_threshold,
        )
        if any(not isinstance(value, int) or not 0 <= value <= 0xFF for value in byte_values):
            raise ValueError("tandem detector thresholds must fit uint8")
        return _REQUEST_STRUCT.pack(
            TANDEM_REQUEST_MAGIC,
            TANDEM_ABI_VERSION,
            _REQUEST_STRUCT.size,
            TANDEM_REQUIRED_FEATURES,
            int(self.mode),
            self.observation_capacity,
            self.event_capacity,
            self.minimum_gain_db,
            self.maximum_gain_db,
            self.initial_gain_db,
            self.power_measurement_samples,
            self.low_power_dwell_periods,
            self.cooldown_periods,
            self.pulse_high_cycles,
            self.pulse_low_cycles,
            self.detector_blanking_cycles,
            *byte_values,
            TANDEM_POLICY_FAIL_SESSION,
            TANDEM_POLICY_FAIL_SESSION,
            *([0] * 8),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TandemGainEventV1:
    sample_sequence: int
    event_sequence: int
    flags: int
    rx1_gain_index: int
    rx2_gain_index: int

    @property
    def direction(self) -> TandemEventDirection:
        return TandemEventDirection((self.flags >> 4) & 0x3)

    @property
    def reason(self) -> TandemEventReason:
        return TandemEventReason(self.flags & 0xF)

    def pack(self) -> bytes:
        payload = _EVENT_STRUCT.pack(
            self.sample_sequence,
            self.event_sequence,
            self.flags,
            self.rx1_gain_index,
            self.rx2_gain_index,
        )
        self.unpack(payload)
        return payload

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> "TandemGainEventV1":
        if len(payload) != _EVENT_STRUCT.size:
            raise ProtocolError("tandem event record has the wrong size")
        event = cls(*_EVENT_STRUCT.unpack(payload))
        if event.flags & 0xFFC0:
            raise ProtocolError("tandem event has unknown flag bits")
        try:
            event.direction
            event.reason
        except ValueError as exc:
            raise ProtocolError("tandem event direction or reason is invalid") from exc
        if event.rx1_gain_index != event.rx2_gain_index:
            raise ProtocolError("tandem event gains are not paired")
        if event.rx1_gain_index > 0x7F:
            raise ProtocolError("tandem event gain index is invalid")
        return event


@dataclasses.dataclass(frozen=True)
class RadioMetadataV4:
    """Validated v3 observations plus exact tandem ownership and events."""

    base: RadioMetadataV3
    header_bytes: int
    ownership_epoch: int
    tandem_state: TandemState
    tandem_fault_flags: int
    tandem_transition_count: int
    gain_table_id: TandemGainTable
    threshold_provenance: int
    minimum_gain_db: int
    maximum_gain_db: int
    initial_gain_db: int
    minimum_gain_index: int
    maximum_gain_index: int
    rx1_gain_index: int
    rx2_gain_index: int
    gain_events: tuple[TandemGainEventV1, ...]

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    @property
    def features(self) -> MetadataFeatures:
        return MetadataFeatures(int(self.base.features) | TANDEM_METADATA_FEATURE)

    @property
    def flags(self) -> MetadataFlags:
        return MetadataFlags(int(self.base.flags) | TANDEM_METADATA_VALID_FLAG)

    def pack(self) -> bytes:
        v3 = bytearray(self.base.pack())
        if len(self.gain_events) != len(self.base.gain_events):
            raise ProtocolError("v4 and compatibility event counts disagree")
        prefix = v3[:HEADER_PREFIX_BYTES_V3]
        header_bytes = len(v3) + HEADER_EXTENSION_BYTES_V4
        struct.pack_into("<H", prefix, 4, VERSION_V4)
        struct.pack_into("<H", prefix, 6, header_bytes)
        struct.pack_into("<I", prefix, 8, int(self.features))
        struct.pack_into("<I", prefix, 12, int(self.flags))
        extension = _V4_EXTENSION_STRUCT.pack(
            self.ownership_epoch,
            int(self.tandem_state),
            self.tandem_fault_flags,
            self.tandem_transition_count,
            int(self.gain_table_id),
            self.threshold_provenance,
            self.minimum_gain_db,
            self.maximum_gain_db,
            self.initial_gain_db,
            self.minimum_gain_index,
            self.maximum_gain_index,
            self.rx1_gain_index,
            self.rx2_gain_index,
            0,
            0,
            0,
            0,
        )
        arrays = bytearray(v3[HEADER_PREFIX_BYTES_V3:-4])
        event_offset = self.base.gain_observation_capacity * GAIN_OBSERVATION_BYTES
        for index, event in enumerate(self.gain_events):
            offset = event_offset + index * GAIN_EVENT_BYTES
            arrays[offset : offset + GAIN_EVENT_BYTES] = event.pack()
        output = prefix + extension + arrays + bytes(4)
        output[-4:] = struct.pack("<I", zlib.crc32(output) & 0xFFFFFFFF)
        if self.header_bytes not in (0, header_bytes):
            raise ProtocolError("protocol v4 header_bytes does not match capacities")
        return bytes(output)

    @classmethod
    def unpack(cls, header: bytes | bytearray | memoryview) -> "RadioMetadataV4":
        raw = bytes(header)
        if len(raw) < HEADER_PREFIX_BYTES_V4 + 4:
            raise ProtocolError("short protocol v4 metadata header")
        magic, version, header_bytes, features, flags = _IDENTITY_STRUCT.unpack_from(raw)
        if magic != 0x314D4753 or version != VERSION_V4:
            raise ProtocolError("bad protocol v4 identity")
        if len(raw) < header_bytes:
            raise ProtocolError("short protocol v4 metadata payload")
        if (
            not features & TANDEM_METADATA_FEATURE
            or not features & int(MetadataFeatures.FPGA_GAIN_EVENTS)
            or not flags & TANDEM_METADATA_VALID_FLAG
        ):
            raise ProtocolError("protocol v4 tandem session is not valid")
        crc_input = bytearray(raw[:header_bytes])
        received_crc = struct.unpack_from("<I", crc_input, header_bytes - 4)[0]
        crc_input[-4:] = bytes(4)
        if received_crc != zlib.crc32(crc_input) & 0xFFFFFFFF:
            raise ProtocolError("protocol v4 metadata CRC mismatch")

        v3_extension = _V3_EXTENSION_STRUCT.unpack_from(raw, 92)
        observation_count = v3_extension[1]
        observation_capacity = v3_extension[2]
        observation_bytes = v3_extension[3]
        event_count = v3_extension[4]
        event_capacity = v3_extension[5]
        event_bytes = v3_extension[6]
        expected_bytes = (
            HEADER_PREFIX_BYTES_V4
            + observation_capacity * observation_bytes
            + event_capacity * event_bytes
            + 4
        )
        if header_bytes != expected_bytes:
            raise ProtocolError("protocol v4 header size does not match capacities")
        if observation_count > observation_capacity or event_count > event_capacity:
            raise ProtocolError("protocol v4 record count exceeds capacity")
        if observation_bytes != GAIN_OBSERVATION_BYTES or event_bytes != GAIN_EVENT_BYTES:
            raise ProtocolError("protocol v4 record size is unsupported")

        extension_values = _V4_EXTENSION_STRUCT.unpack_from(
            raw, HEADER_PREFIX_BYTES_V3
        )
        if any(extension_values[-4:]):
            raise ProtocolError("protocol v4 reserved fields must be zero")
        (
            ownership_epoch,
            tandem_state,
            tandem_fault_flags,
            tandem_transition_count,
            gain_table_id,
            threshold_provenance,
            minimum_gain_db,
            maximum_gain_db,
            initial_gain_db,
            minimum_gain_index,
            maximum_gain_index,
            rx1_gain_index,
            rx2_gain_index,
            *_reserved,
        ) = extension_values
        if not ownership_epoch or tandem_fault_flags:
            raise ProtocolError("protocol v4 reports an invalid tandem lease")
        try:
            parsed_state = TandemState(tandem_state)
            parsed_table = TandemGainTable(gain_table_id)
        except ValueError as exc:
            raise ProtocolError("protocol v4 tandem provenance is unknown") from exc
        if parsed_state not in (TandemState.ARMED_HOLD, TandemState.ARMED_AUTO):
            raise ProtocolError("protocol v4 tandem lease is not armed")
        if rx1_gain_index != rx2_gain_index:
            raise ProtocolError("protocol v4 endpoint gains are not paired")

        arrays = raw[HEADER_PREFIX_BYTES_V4:header_bytes]
        event_offset = observation_capacity * observation_bytes
        events = []
        previous_sequence = None
        previous_sample = None
        for index in range(event_count):
            offset = event_offset + index * event_bytes
            event = TandemGainEventV1.unpack(arrays[offset : offset + event_bytes])
            if previous_sequence is not None and event.event_sequence != (
                previous_sequence + 1
            ) & 0xFFFFFFFF:
                raise ProtocolError("protocol v4 tandem event sequence has a hole")
            if previous_sample is not None and event.sample_sequence < previous_sample:
                raise ProtocolError("protocol v4 tandem events are not sample ordered")
            events.append(event)
            previous_sequence = event.event_sequence
            previous_sample = event.sample_sequence
        unused_start = event_offset + event_count * event_bytes
        unused_end = event_offset + event_capacity * event_bytes
        if any(arrays[unused_start:unused_end]):
            raise ProtocolError("unused protocol v4 tandem events must be zero")

        synthetic = bytearray(raw[:HEADER_PREFIX_BYTES_V3])
        struct.pack_into("<H", synthetic, 4, VERSION_V3)
        synthetic_bytes = header_bytes - HEADER_EXTENSION_BYTES_V4
        struct.pack_into("<H", synthetic, 6, synthetic_bytes)
        struct.pack_into("<I", synthetic, 8, features & ~TANDEM_METADATA_FEATURE)
        struct.pack_into("<I", synthetic, 12, flags & ~TANDEM_METADATA_VALID_FLAG)
        synthetic.extend(arrays)
        synthetic_event_offset = HEADER_PREFIX_BYTES_V3 + event_offset
        for index, event in enumerate(events):
            _LEGACY_EVENT_STRUCT.pack_into(
                synthetic,
                synthetic_event_offset + index * GAIN_EVENT_BYTES,
                event.sample_sequence,
                3,
                0,
                0,
            )
        synthetic[-4:] = bytes(4)
        struct.pack_into("<I", synthetic, len(synthetic) - 4, zlib.crc32(synthetic))
        base = RadioMetadataV3.unpack(synthetic)

        frame_end = base.first_sample_sequence + base.samples_per_channel
        if any(
            not base.first_sample_sequence <= event.sample_sequence < frame_end
            for event in events
        ):
            raise ProtocolError("protocol v4 tandem event lies outside its IQ frame")
        return cls(
            base=base,
            header_bytes=header_bytes,
            ownership_epoch=ownership_epoch,
            tandem_state=parsed_state,
            tandem_fault_flags=tandem_fault_flags,
            tandem_transition_count=tandem_transition_count,
            gain_table_id=parsed_table,
            threshold_provenance=threshold_provenance,
            minimum_gain_db=minimum_gain_db,
            maximum_gain_db=maximum_gain_db,
            initial_gain_db=initial_gain_db,
            minimum_gain_index=minimum_gain_index,
            maximum_gain_index=maximum_gain_index,
            rx1_gain_index=rx1_gain_index,
            rx2_gain_index=rx2_gain_index,
            gain_events=tuple(events),
        )
