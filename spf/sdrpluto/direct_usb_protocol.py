"""Wire protocol for SPF's Pluto direct-USB RX transport.

The parser in this module is deliberately independent of libusb. USB bulk
completions are byte-stream chunks, not application frame boundaries, so callers
must feed every completion through :class:`RxFrameParser`.
"""

from __future__ import annotations

import dataclasses
import enum
import struct
import zlib
from typing import Final


MAGIC: Final[int] = 0x314D4753  # little-endian bytes: b"SGM1"
VERSION_V1: Final[int] = 1
VERSION_V2: Final[int] = 2
# Kept as the v1 value for source compatibility with the first implementation.
VERSION: Final[int] = VERSION_V1
GAIN_INDEX_INVALID: Final[int] = 0xFF
GAIN_DB_INVALID: Final[int] = -128
RSSI_QDB_INVALID: Final[int] = 0xFFFF
FIRST_CHANGE_UNAVAILABLE: Final[int] = 0xFFFFFFFF
COMMAND_START_LEGACY: Final[int] = 0x10
COMMAND_STOP: Final[int] = 0x11
COMMAND_GET_CAPABILITIES: Final[int] = 0x12
COMMAND_START_RX_V1: Final[int] = 0x13
COMMAND_GET_HARDWARE_IDENTITY: Final[int] = 0x14
COMMAND_GET_STATUS: Final[int] = 0x15
COMMAND_TARGET_RX: Final[int] = 0

CAPABILITIES_MAGIC: Final[int] = 0x50434753  # b"SGCP"
HARDWARE_IDENTITY_MAGIC: Final[int] = 0x31464853  # b"SHF1"
HARDWARE_IDENTITY_VERSION: Final[int] = 1
RUNTIME_STATUS_MAGIC: Final[int] = 0x31545353  # b"SST1"
RUNTIME_STATUS_VERSION: Final[int] = 1
START_REQUEST_MAGIC: Final[int] = 0x31534753  # b"SGS1"
START_REQUEST_MAGIC_V2: Final[int] = 0x32534753  # b"SGS2"
MAX_FINITE_FRAMES: Final[int] = 16
MAX_SAMPLES_PER_CHANNEL: Final[int] = 0xFFFFFFFF // 8


class ProtocolError(ValueError):
    """A received frame violates the negotiated wire protocol."""


class SampleFormat(enum.IntEnum):
    """IQ representation in the USB payload."""

    CS16_LE_TIME_INTERLEAVED = 1


class MetadataFlags(enum.IntFlag):
    START_VALID = 1 << 0
    END_VALID = 1 << 1
    RX1_ENDPOINT_CHANGED = 1 << 2
    RX2_ENDPOINT_CHANGED = 1 << 3
    SAMPLE_SEQUENCE_VALID = 1 << 4
    FPGA_EVENTS_VALID = 1 << 5
    RX1_CHANGED_IN_BUFFER = 1 << 6
    RX2_CHANGED_IN_BUFFER = 1 << 7
    RX1_LOCKED_AT_END = 1 << 8
    RX2_LOCKED_AT_END = 1 << 9
    GAIN_FULL_TABLE_MODE = 1 << 10
    DEVICE_IIO_OVERFLOW = 1 << 11
    GAIN_READ_FAILED = 1 << 12
    FPGA_EVENT_OVERFLOW = 1 << 13
    DUMMY_GAINS = 1 << 14
    RSSI_START_VALID = 1 << 15
    RSSI_END_VALID = 1 << 16
    RSSI_READ_FAILED = 1 << 17
    GAIN_DB_VALUES = 1 << 18


class MetadataFeatures(enum.IntFlag):
    GAIN_ENDPOINT_SNAPSHOTS = 1 << 0
    HEADER_CRC32 = 1 << 1
    SAMPLE_SEQUENCE = 1 << 2
    FPGA_GAIN_EVENTS = 1 << 3
    GAIN_DB_ENDPOINTS = 1 << 4
    RSSI_ENDPOINT_SNAPSHOTS = 1 << 5


class CapabilityFlags(enum.IntFlag):
    FINITE_RX = 1 << 0
    DUMMY_GAINS = 1 << 1
    HARDWARE_IDENTITY = 1 << 2
    STATUS = 1 << 3


class HardwareIdentityFlags(enum.IntFlag):
    FPGA_DEVICE_DNA_VALID = 1 << 0
    GADGET_BUILD_ID_VALID = 1 << 1


class RuntimeStatusFlags(enum.IntFlag):
    BOOT_ID_VALID = 1 << 0
    PROCESS_NONCE_VALID = 1 << 1
    RX_WORKER_ACTIVE = 1 << 2


class RuntimeState(enum.IntEnum):
    IDLE = 0
    STARTING = 1
    STREAMING = 2
    COMPLETE = 3
    STOPPING = 4
    FAILED = 5


class ErrorSubsystem(enum.IntEnum):
    NONE = 0
    CONTROL = 1
    RX_INIT = 2
    IIO_REFILL = 3
    USB_SUBMIT = 4
    USB_COMPLETION = 5
    BUFFER_STARVATION = 6
    GAIN_READ = 7
    RSSI_READ = 8
    STOP_TIMEOUT = 9


KNOWN_FEATURES: Final[MetadataFeatures] = (
    MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
    | MetadataFeatures.HEADER_CRC32
    | MetadataFeatures.SAMPLE_SEQUENCE
    | MetadataFeatures.FPGA_GAIN_EVENTS
    | MetadataFeatures.GAIN_DB_ENDPOINTS
    | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
)
KNOWN_FLAGS: Final[MetadataFlags] = (
    MetadataFlags.START_VALID
    | MetadataFlags.END_VALID
    | MetadataFlags.RX1_ENDPOINT_CHANGED
    | MetadataFlags.RX2_ENDPOINT_CHANGED
    | MetadataFlags.SAMPLE_SEQUENCE_VALID
    | MetadataFlags.FPGA_EVENTS_VALID
    | MetadataFlags.RX1_CHANGED_IN_BUFFER
    | MetadataFlags.RX2_CHANGED_IN_BUFFER
    | MetadataFlags.RX1_LOCKED_AT_END
    | MetadataFlags.RX2_LOCKED_AT_END
    | MetadataFlags.GAIN_FULL_TABLE_MODE
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.FPGA_EVENT_OVERFLOW
    | MetadataFlags.DUMMY_GAINS
    | MetadataFlags.RSSI_START_VALID
    | MetadataFlags.RSSI_END_VALID
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.GAIN_DB_VALUES
)


# CRC-32/ISO-HDLC, as implemented by zlib.crc32:
# poly=0x04C11DB7, reflected input/output, init=0xffffffff,
# xorout=0xffffffff. The CRC covers header_bytes bytes with the final CRC field
# zeroed. Protocol v1 requires header_bytes == HEADER_BYTES.
_HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<IHHIIQQQIIIHBBBBBBIIIII")
HEADER_BYTES: Final[int] = _HEADER_STRUCT.size
assert HEADER_BYTES == 80
HEADER_BYTES_V1: Final[int] = HEADER_BYTES
_HEADER_V2_STRUCT: Final[struct.Struct] = struct.Struct(
    "<IHHIIQQQIIIHBbbbbBIIIIHHHHIII"
)
HEADER_BYTES_V2: Final[int] = _HEADER_V2_STRUCT.size
assert HEADER_BYTES_V2 == 96
_CAPABILITIES_STRUCT: Final[struct.Struct] = struct.Struct("<IHHHHIIIII")
CAPABILITIES_BYTES: Final[int] = _CAPABILITIES_STRUCT.size
assert CAPABILITIES_BYTES == 32
_HARDWARE_IDENTITY_STRUCT: Final[struct.Struct] = struct.Struct("<IHHIIQ40s")
HARDWARE_IDENTITY_BYTES: Final[int] = _HARDWARE_IDENTITY_STRUCT.size
assert HARDWARE_IDENTITY_BYTES == 64
_RUNTIME_STATUS_STRUCT: Final[struct.Struct] = struct.Struct(
    "<IHHHHiII16s16sQQ14I"
)
RUNTIME_STATUS_BYTES: Final[int] = _RUNTIME_STATUS_STRUCT.size
assert RUNTIME_STATUS_BYTES == 128
_START_REQUEST_STRUCT: Final[struct.Struct] = struct.Struct("<IHHIIIIII")
START_REQUEST_BYTES: Final[int] = _START_REQUEST_STRUCT.size
assert START_REQUEST_BYTES == 32


@dataclasses.dataclass(frozen=True, slots=True)
class GadgetCapabilitiesV1:
    protocol_min: int
    protocol_max: int
    supported_features: MetadataFeatures
    max_samples_per_channel: int
    max_finite_frames: int
    capability_flags: CapabilityFlags

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> "GadgetCapabilitiesV1":
        if len(payload) != CAPABILITIES_BYTES:
            raise ProtocolError(
                "capability response size mismatch: "
                f"got {len(payload)}, expected {CAPABILITIES_BYTES}"
            )
        (
            magic,
            response_bytes,
            protocol_min,
            protocol_max,
            reserved0,
            supported_features,
            max_samples_per_channel,
            max_finite_frames,
            capability_flags,
            reserved1,
        ) = _CAPABILITIES_STRUCT.unpack(payload)
        if magic != CAPABILITIES_MAGIC:
            raise ProtocolError(f"bad capability magic: 0x{magic:08x}")
        if response_bytes != CAPABILITIES_BYTES:
            raise ProtocolError(f"unsupported capability size: {response_bytes}")
        if reserved0 != 0 or reserved1 != 0:
            raise ProtocolError("capability reserved fields must be zero")
        if protocol_min <= 0 or protocol_min > protocol_max:
            raise ProtocolError(
                f"invalid gadget protocol range {protocol_min}..{protocol_max}"
            )
        unknown_features = supported_features & ~int(KNOWN_FEATURES)
        if unknown_features:
            raise ProtocolError(
                f"unknown capability feature bits: 0x{unknown_features:08x}"
            )
        known_capability_flags = (
            CapabilityFlags.FINITE_RX
            | CapabilityFlags.DUMMY_GAINS
            | CapabilityFlags.HARDWARE_IDENTITY
            | CapabilityFlags.STATUS
        )
        unknown_capability_flags = capability_flags & ~int(known_capability_flags)
        if unknown_capability_flags:
            raise ProtocolError(
                "unknown capability flag bits: " f"0x{unknown_capability_flags:08x}"
            )
        if not capability_flags & CapabilityFlags.FINITE_RX:
            raise ProtocolError("gadget does not support finite RX")
        if max_samples_per_channel <= 0 or max_finite_frames <= 0:
            raise ProtocolError("gadget reports unusable finite RX limits")
        return cls(
            protocol_min=protocol_min,
            protocol_max=protocol_max,
            supported_features=MetadataFeatures(supported_features),
            max_samples_per_channel=max_samples_per_channel,
            max_finite_frames=max_finite_frames,
            capability_flags=CapabilityFlags(capability_flags),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class HardwareIdentityV1:
    """Passive device identity returned without starting either DMA direction."""

    flags: HardwareIdentityFlags
    fpga_device_dna: int
    gadget_build_id: str

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> "HardwareIdentityV1":
        if len(payload) != HARDWARE_IDENTITY_BYTES:
            raise ProtocolError(
                "hardware identity response size mismatch: "
                f"got {len(payload)}, expected {HARDWARE_IDENTITY_BYTES}"
            )
        (
            magic,
            response_bytes,
            version,
            flags,
            reserved0,
            fpga_device_dna,
            raw_build_id,
        ) = _HARDWARE_IDENTITY_STRUCT.unpack(payload)
        if magic != HARDWARE_IDENTITY_MAGIC:
            raise ProtocolError(f"bad hardware identity magic: 0x{magic:08x}")
        if response_bytes != HARDWARE_IDENTITY_BYTES:
            raise ProtocolError(f"unsupported hardware identity size: {response_bytes}")
        if version != HARDWARE_IDENTITY_VERSION:
            raise ProtocolError(f"unsupported hardware identity version: {version}")
        if reserved0 != 0:
            raise ProtocolError("hardware identity reserved field must be zero")
        known_flags = (
            HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID
            | HardwareIdentityFlags.GADGET_BUILD_ID_VALID
        )
        unknown_flags = flags & ~int(known_flags)
        if unknown_flags:
            raise ProtocolError(
                f"unknown hardware identity flags: 0x{unknown_flags:08x}"
            )
        parsed_flags = HardwareIdentityFlags(flags)
        if parsed_flags & HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID:
            if fpga_device_dna == 0 or fpga_device_dna >> 57:
                raise ProtocolError("FPGA Device DNA is outside the 57-bit range")
        elif fpga_device_dna != 0:
            raise ProtocolError("invalid FPGA Device DNA must be zero")
        try:
            gadget_build_id = raw_build_id.rstrip(b"\x00").decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProtocolError("gadget build ID is not ASCII") from exc
        if parsed_flags & HardwareIdentityFlags.GADGET_BUILD_ID_VALID:
            if len(gadget_build_id) != 40 or any(
                character not in "0123456789abcdef" for character in gadget_build_id
            ):
                raise ProtocolError(
                    "valid gadget build ID must be a lowercase 40-character SHA"
                )
        elif gadget_build_id:
            raise ProtocolError("invalid gadget build ID must be empty")
        return cls(
            flags=parsed_flags,
            fpga_device_dna=fpga_device_dna,
            gadget_build_id=gadget_build_id,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeStatusV1:
    """Passive, volatile health state for one gadget process instance."""

    lifecycle_state: RuntimeState
    last_error_subsystem: ErrorSubsystem
    last_errno: int
    flags: RuntimeStatusFlags
    boot_id: bytes
    process_nonce: bytes
    current_stream_id: int
    last_completed_sequence: int
    start_count: int
    stop_count: int
    completed_frame_count: int
    dropped_frame_count: int
    iio_refill_error_count: int
    usb_submit_error_count: int
    short_write_count: int
    buffer_starvation_count: int
    gain_read_failure_count: int
    rssi_read_failure_count: int
    control_error_count: int
    stop_timeout_count: int
    worker_heartbeat_age_ms: int

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> "RuntimeStatusV1":
        if len(payload) != RUNTIME_STATUS_BYTES:
            raise ProtocolError(
                "runtime status response size mismatch: "
                f"got {len(payload)}, expected {RUNTIME_STATUS_BYTES}"
            )
        values = _RUNTIME_STATUS_STRUCT.unpack(payload)
        (
            magic,
            response_bytes,
            version,
            lifecycle_state,
            last_error_subsystem,
            last_errno,
            flags,
            reserved0,
            boot_id,
            process_nonce,
            current_stream_id,
            last_completed_sequence,
            start_count,
            stop_count,
            completed_frame_count,
            dropped_frame_count,
            iio_refill_error_count,
            usb_submit_error_count,
            short_write_count,
            buffer_starvation_count,
            gain_read_failure_count,
            rssi_read_failure_count,
            control_error_count,
            stop_timeout_count,
            worker_heartbeat_age_ms,
            reserved1,
        ) = values
        if magic != RUNTIME_STATUS_MAGIC:
            raise ProtocolError(f"bad runtime status magic: 0x{magic:08x}")
        if response_bytes != RUNTIME_STATUS_BYTES:
            raise ProtocolError(f"unsupported runtime status size: {response_bytes}")
        if version != RUNTIME_STATUS_VERSION:
            raise ProtocolError(f"unsupported runtime status version: {version}")
        if reserved0 != 0 or reserved1 != 0:
            raise ProtocolError("runtime status reserved fields must be zero")
        known_flags = (
            RuntimeStatusFlags.BOOT_ID_VALID
            | RuntimeStatusFlags.PROCESS_NONCE_VALID
            | RuntimeStatusFlags.RX_WORKER_ACTIVE
        )
        if flags & ~int(known_flags):
            raise ProtocolError(f"unknown runtime status flags: 0x{flags:08x}")
        parsed_flags = RuntimeStatusFlags(flags)
        try:
            parsed_state = RuntimeState(lifecycle_state)
            parsed_subsystem = ErrorSubsystem(last_error_subsystem)
        except ValueError as exc:
            raise ProtocolError("unknown runtime status state or subsystem") from exc
        if not parsed_flags & RuntimeStatusFlags.BOOT_ID_VALID and any(boot_id):
            raise ProtocolError("invalid runtime boot ID must be zero")
        if not parsed_flags & RuntimeStatusFlags.PROCESS_NONCE_VALID and any(
            process_nonce
        ):
            raise ProtocolError("invalid runtime process nonce must be zero")
        if (
            not parsed_flags & RuntimeStatusFlags.RX_WORKER_ACTIVE
            and worker_heartbeat_age_ms != 0
        ):
            raise ProtocolError("inactive worker heartbeat age must be zero")
        return cls(
            lifecycle_state=parsed_state,
            last_error_subsystem=parsed_subsystem,
            last_errno=last_errno,
            flags=parsed_flags,
            boot_id=boot_id,
            process_nonce=process_nonce,
            current_stream_id=current_stream_id,
            last_completed_sequence=last_completed_sequence,
            start_count=start_count,
            stop_count=stop_count,
            completed_frame_count=completed_frame_count,
            dropped_frame_count=dropped_frame_count,
            iio_refill_error_count=iio_refill_error_count,
            usb_submit_error_count=usb_submit_error_count,
            short_write_count=short_write_count,
            buffer_starvation_count=buffer_starvation_count,
            gain_read_failure_count=gain_read_failure_count,
            rssi_read_failure_count=rssi_read_failure_count,
            control_error_count=control_error_count,
            stop_timeout_count=stop_timeout_count,
            worker_heartbeat_age_ms=worker_heartbeat_age_ms,
        )


def pack_start_request_v1(
    *,
    requested_features: MetadataFeatures,
    enabled_scan_mask: int,
    samples_per_channel: int,
    frame_count: int,
) -> bytes:
    required_features = (
        MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
        | MetadataFeatures.HEADER_CRC32
        | MetadataFeatures.SAMPLE_SEQUENCE
    )
    if requested_features != required_features:
        raise ProtocolError(
            f"protocol v1 requires feature mask 0x{int(required_features):08x}"
        )
    if enabled_scan_mask != 0x0F:
        raise ProtocolError("protocol v1 requires scan mask 0x0000000f")
    if not 1 <= samples_per_channel <= MAX_SAMPLES_PER_CHANNEL:
        raise ProtocolError("samples_per_channel is outside the v1 limit")
    if not 1 <= frame_count <= MAX_FINITE_FRAMES:
        raise ProtocolError("frame_count is outside the v1 finite limit")
    return _START_REQUEST_STRUCT.pack(
        START_REQUEST_MAGIC,
        VERSION,
        START_REQUEST_BYTES,
        int(requested_features),
        enabled_scan_mask,
        samples_per_channel,
        frame_count,
        0,
        0,
    )


def pack_start_request_v2(
    *,
    requested_features: MetadataFeatures,
    enabled_scan_mask: int,
    samples_per_channel: int,
    frame_count: int,
) -> bytes:
    required_features = (
        MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
        | MetadataFeatures.HEADER_CRC32
        | MetadataFeatures.SAMPLE_SEQUENCE
        | MetadataFeatures.GAIN_DB_ENDPOINTS
        | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
    )
    if requested_features != required_features:
        raise ProtocolError(
            f"protocol v2 requires feature mask 0x{int(required_features):08x}"
        )
    if enabled_scan_mask != 0x0F:
        raise ProtocolError("protocol v2 requires scan mask 0x0000000f")
    if not 1 <= samples_per_channel <= MAX_SAMPLES_PER_CHANNEL:
        raise ProtocolError("samples_per_channel is outside the v2 limit")
    if not 1 <= frame_count <= MAX_FINITE_FRAMES:
        raise ProtocolError("frame_count is outside the v2 finite limit")
    return _START_REQUEST_STRUCT.pack(
        START_REQUEST_MAGIC_V2,
        VERSION_V2,
        START_REQUEST_BYTES,
        int(requested_features),
        enabled_scan_mask,
        samples_per_channel,
        frame_count,
        0,
        0,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class GainMetadataV1:
    features: MetadataFeatures
    flags: MetadataFlags
    stream_id: int
    buffer_sequence: int
    first_sample_sequence: int
    samples_per_channel: int
    iq_payload_bytes: int
    enabled_scan_mask: int
    sample_format: SampleFormat
    channel_count: int
    rx1_gain_start: int = GAIN_INDEX_INVALID
    rx2_gain_start: int = GAIN_INDEX_INVALID
    rx1_gain_end: int = GAIN_INDEX_INVALID
    rx2_gain_end: int = GAIN_INDEX_INVALID
    gain_start_read_duration_ns: int = 0
    gain_end_read_duration_ns: int = 0
    rx1_first_change_sample: int = FIRST_CHANGE_UNAVAILABLE
    rx2_first_change_sample: int = FIRST_CHANGE_UNAVAILABLE

    @property
    def gain_index_start(self) -> tuple[int, int]:
        return self.rx1_gain_start, self.rx2_gain_start

    @property
    def gain_index_end(self) -> tuple[int, int]:
        return self.rx1_gain_end, self.rx2_gain_end

    @property
    def gain_endpoints_equal(self) -> tuple[bool, bool]:
        """Compare observations; equality does not imply in-buffer stability."""

        if not self.gain_metadata_valid:
            return False, False
        return (
            self.rx1_gain_start == self.rx1_gain_end,
            self.rx2_gain_start == self.rx2_gain_end,
        )

    @property
    def gain_metadata_valid(self) -> bool:
        required = MetadataFlags.START_VALID | MetadataFlags.END_VALID
        return (self.flags & required) == required and not bool(
            self.flags & MetadataFlags.DUMMY_GAINS
        )

    def pack(self) -> bytes:
        _validate_metadata(self)
        fields = _fields_for_pack(self, crc32=0)
        without_crc = _HEADER_STRUCT.pack(*fields)
        crc32 = zlib.crc32(without_crc) & 0xFFFFFFFF
        return _HEADER_STRUCT.pack(*_fields_for_pack(self, crc32=crc32))

    @classmethod
    def unpack(cls, header: bytes | bytearray | memoryview) -> "GainMetadataV1":
        if len(header) < HEADER_BYTES:
            raise ProtocolError(
                f"short metadata header: got {len(header)}, need {HEADER_BYTES}"
            )
        values = _HEADER_STRUCT.unpack_from(header)
        (
            magic,
            version,
            header_bytes,
            features,
            flags,
            stream_id,
            buffer_sequence,
            first_sample_sequence,
            samples_per_channel,
            iq_payload_bytes,
            enabled_scan_mask,
            sample_format,
            channel_count,
            rx1_gain_start,
            rx2_gain_start,
            rx1_gain_end,
            rx2_gain_end,
            reserved0,
            gain_start_read_duration_ns,
            gain_end_read_duration_ns,
            rx1_first_change_sample,
            rx2_first_change_sample,
            received_crc32,
        ) = values

        if magic != MAGIC:
            raise ProtocolError(f"bad metadata magic: 0x{magic:08x}")
        if version != VERSION:
            raise ProtocolError(f"unsupported metadata version: {version}")
        if header_bytes != HEADER_BYTES:
            raise ProtocolError(
                f"unsupported v1 header size: {header_bytes}, expected {HEADER_BYTES}"
            )
        if reserved0 != 0:
            raise ProtocolError(f"reserved0 must be zero, got {reserved0}")

        crc_input = bytearray(memoryview(header)[:header_bytes])
        crc_input[-4:] = b"\x00\x00\x00\x00"
        calculated_crc32 = zlib.crc32(crc_input) & 0xFFFFFFFF
        if received_crc32 != calculated_crc32:
            raise ProtocolError(
                "metadata CRC mismatch: "
                f"received 0x{received_crc32:08x}, "
                f"calculated 0x{calculated_crc32:08x}"
            )

        try:
            parsed_format = SampleFormat(sample_format)
        except ValueError as exc:
            raise ProtocolError(f"unsupported sample format: {sample_format}") from exc

        metadata = cls(
            features=MetadataFeatures(features),
            flags=MetadataFlags(flags),
            stream_id=stream_id,
            buffer_sequence=buffer_sequence,
            first_sample_sequence=first_sample_sequence,
            samples_per_channel=samples_per_channel,
            iq_payload_bytes=iq_payload_bytes,
            enabled_scan_mask=enabled_scan_mask,
            sample_format=parsed_format,
            channel_count=channel_count,
            rx1_gain_start=rx1_gain_start,
            rx2_gain_start=rx2_gain_start,
            rx1_gain_end=rx1_gain_end,
            rx2_gain_end=rx2_gain_end,
            gain_start_read_duration_ns=gain_start_read_duration_ns,
            gain_end_read_duration_ns=gain_end_read_duration_ns,
            rx1_first_change_sample=rx1_first_change_sample,
            rx2_first_change_sample=rx2_first_change_sample,
        )
        _validate_metadata(metadata)
        return metadata


@dataclasses.dataclass(frozen=True, slots=True)
class RadioMetadataV2:
    """Per-frame radio values expressed in the legacy Python units.

    Gain values are integer dB from the active AD9361 full gain table. RSSI is
    positive attenuation magnitude in quarter-dB wire units, matching the
    positive floating-point values historically returned by ``PPlus.rssis()``.
    Endpoint-change flags are derived from raw table indices on the Pluto, so a
    change remains visible even when two indices round to the same integer dB.
    """

    features: MetadataFeatures
    flags: MetadataFlags
    stream_id: int
    buffer_sequence: int
    first_sample_sequence: int
    samples_per_channel: int
    iq_payload_bytes: int
    enabled_scan_mask: int
    sample_format: SampleFormat
    channel_count: int
    rx1_gain_db_start: int = GAIN_DB_INVALID
    rx2_gain_db_start: int = GAIN_DB_INVALID
    rx1_gain_db_end: int = GAIN_DB_INVALID
    rx2_gain_db_end: int = GAIN_DB_INVALID
    gain_start_read_duration_ns: int = 0
    gain_end_read_duration_ns: int = 0
    rx1_first_change_sample: int = FIRST_CHANGE_UNAVAILABLE
    rx2_first_change_sample: int = FIRST_CHANGE_UNAVAILABLE
    rx1_rssi_start_qdb: int = RSSI_QDB_INVALID
    rx2_rssi_start_qdb: int = RSSI_QDB_INVALID
    rx1_rssi_end_qdb: int = RSSI_QDB_INVALID
    rx2_rssi_end_qdb: int = RSSI_QDB_INVALID
    rssi_start_read_duration_ns: int = 0
    rssi_end_read_duration_ns: int = 0

    @property
    def gain_db_start(self) -> tuple[float, float]:
        return tuple(
            float("nan") if value == GAIN_DB_INVALID else float(value)
            for value in (self.rx1_gain_db_start, self.rx2_gain_db_start)
        )

    @property
    def gain_db_end(self) -> tuple[float, float]:
        return tuple(
            float("nan") if value == GAIN_DB_INVALID else float(value)
            for value in (self.rx1_gain_db_end, self.rx2_gain_db_end)
        )

    @property
    def rssi_db_start(self) -> tuple[float, float]:
        return tuple(
            float("nan") if value == RSSI_QDB_INVALID else value / 4.0
            for value in (self.rx1_rssi_start_qdb, self.rx2_rssi_start_qdb)
        )

    @property
    def rssi_db_end(self) -> tuple[float, float]:
        return tuple(
            float("nan") if value == RSSI_QDB_INVALID else value / 4.0
            for value in (self.rx1_rssi_end_qdb, self.rx2_rssi_end_qdb)
        )

    @property
    def gain_metadata_valid(self) -> bool:
        required = (
            MetadataFlags.START_VALID
            | MetadataFlags.END_VALID
            | MetadataFlags.GAIN_DB_VALUES
        )
        return (self.flags & required) == required

    @property
    def rssi_metadata_valid(self) -> bool:
        required = MetadataFlags.RSSI_START_VALID | MetadataFlags.RSSI_END_VALID
        return (self.flags & required) == required

    @property
    def gain_endpoints_equal(self) -> tuple[bool, bool]:
        """Raw-index endpoint comparison; not proof of in-buffer stability."""

        if not self.gain_metadata_valid:
            return False, False
        return (
            not bool(self.flags & MetadataFlags.RX1_ENDPOINT_CHANGED),
            not bool(self.flags & MetadataFlags.RX2_ENDPOINT_CHANGED),
        )

    def pack(self) -> bytes:
        _validate_metadata_v2(self)
        without_crc = _HEADER_V2_STRUCT.pack(*_fields_for_pack_v2(self, crc32=0))
        crc32 = zlib.crc32(without_crc) & 0xFFFFFFFF
        return _HEADER_V2_STRUCT.pack(*_fields_for_pack_v2(self, crc32=crc32))

    @classmethod
    def unpack(cls, header: bytes | bytearray | memoryview) -> "RadioMetadataV2":
        if len(header) < HEADER_BYTES_V2:
            raise ProtocolError(
                f"short metadata header: got {len(header)}, need {HEADER_BYTES_V2}"
            )
        values = _HEADER_V2_STRUCT.unpack_from(header)
        (
            magic,
            version,
            header_bytes,
            features,
            flags,
            stream_id,
            buffer_sequence,
            first_sample_sequence,
            samples_per_channel,
            iq_payload_bytes,
            enabled_scan_mask,
            sample_format,
            channel_count,
            rx1_gain_db_start,
            rx2_gain_db_start,
            rx1_gain_db_end,
            rx2_gain_db_end,
            reserved0,
            gain_start_read_duration_ns,
            gain_end_read_duration_ns,
            rx1_first_change_sample,
            rx2_first_change_sample,
            rx1_rssi_start_qdb,
            rx2_rssi_start_qdb,
            rx1_rssi_end_qdb,
            rx2_rssi_end_qdb,
            rssi_start_read_duration_ns,
            rssi_end_read_duration_ns,
            received_crc32,
        ) = values
        if magic != MAGIC:
            raise ProtocolError(f"bad metadata magic: 0x{magic:08x}")
        if version != VERSION_V2:
            raise ProtocolError(f"unsupported metadata version: {version}")
        if header_bytes != HEADER_BYTES_V2:
            raise ProtocolError(
                f"unsupported v2 header size: {header_bytes}, "
                f"expected {HEADER_BYTES_V2}"
            )
        if reserved0 != 0:
            raise ProtocolError(f"reserved0 must be zero, got {reserved0}")
        crc_input = bytearray(memoryview(header)[:header_bytes])
        crc_input[-4:] = b"\x00\x00\x00\x00"
        calculated_crc32 = zlib.crc32(crc_input) & 0xFFFFFFFF
        if received_crc32 != calculated_crc32:
            raise ProtocolError(
                "metadata CRC mismatch: "
                f"received 0x{received_crc32:08x}, "
                f"calculated 0x{calculated_crc32:08x}"
            )
        try:
            parsed_format = SampleFormat(sample_format)
        except ValueError as exc:
            raise ProtocolError(f"unsupported sample format: {sample_format}") from exc
        metadata = cls(
            features=MetadataFeatures(features),
            flags=MetadataFlags(flags),
            stream_id=stream_id,
            buffer_sequence=buffer_sequence,
            first_sample_sequence=first_sample_sequence,
            samples_per_channel=samples_per_channel,
            iq_payload_bytes=iq_payload_bytes,
            enabled_scan_mask=enabled_scan_mask,
            sample_format=parsed_format,
            channel_count=channel_count,
            rx1_gain_db_start=rx1_gain_db_start,
            rx2_gain_db_start=rx2_gain_db_start,
            rx1_gain_db_end=rx1_gain_db_end,
            rx2_gain_db_end=rx2_gain_db_end,
            gain_start_read_duration_ns=gain_start_read_duration_ns,
            gain_end_read_duration_ns=gain_end_read_duration_ns,
            rx1_first_change_sample=rx1_first_change_sample,
            rx2_first_change_sample=rx2_first_change_sample,
            rx1_rssi_start_qdb=rx1_rssi_start_qdb,
            rx2_rssi_start_qdb=rx2_rssi_start_qdb,
            rx1_rssi_end_qdb=rx1_rssi_end_qdb,
            rx2_rssi_end_qdb=rx2_rssi_end_qdb,
            rssi_start_read_duration_ns=rssi_start_read_duration_ns,
            rssi_end_read_duration_ns=rssi_end_read_duration_ns,
        )
        _validate_metadata_v2(metadata)
        return metadata


@dataclasses.dataclass(frozen=True, slots=True)
class DirectUsbRxFrame:
    metadata: GainMetadataV1 | RadioMetadataV2
    iq_payload: bytes


class RxFrameParser:
    """Incrementally parse strictly ordered, fixed-protocol RX frames.

    A protocol error clears buffered bytes and raises :class:`ProtocolError`.
    The caller must stop and restart the stream; the parser never scans forward
    for a plausible magic value because doing so could silently reassociate IQ
    with the wrong metadata.
    """

    def __init__(self, protocol_version: int = VERSION_V1) -> None:
        if protocol_version not in (VERSION_V1, VERSION_V2):
            raise ProtocolError(f"unsupported parser protocol: {protocol_version}")
        self.protocol_version = protocol_version
        self.header_bytes = (
            HEADER_BYTES_V1 if protocol_version == VERSION_V1 else HEADER_BYTES_V2
        )
        self.metadata_type = (
            GainMetadataV1 if protocol_version == VERSION_V1 else RadioMetadataV2
        )
        self._buffer = bytearray()
        self._expected_stream_id: int | None = None
        self._expected_buffer_sequence: int | None = None
        self._expected_first_sample_sequence: int | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self._expected_stream_id = None
        self._expected_buffer_sequence = None
        self._expected_first_sample_sequence = None

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[DirectUsbRxFrame]:
        if chunk:
            self._buffer.extend(chunk)
        frames: list[DirectUsbRxFrame] = []

        try:
            while len(self._buffer) >= self.header_bytes:
                metadata = self.metadata_type.unpack(self._buffer[: self.header_bytes])
                frame_bytes = self.header_bytes + metadata.iq_payload_bytes
                if len(self._buffer) < frame_bytes:
                    break

                self._validate_sequence(metadata)
                payload = bytes(self._buffer[self.header_bytes : frame_bytes])
                del self._buffer[:frame_bytes]
                frames.append(DirectUsbRxFrame(metadata=metadata, iq_payload=payload))
        except ProtocolError:
            self.reset()
            raise

        return frames

    def finish(self) -> None:
        if self._buffer:
            remaining = len(self._buffer)
            self.reset()
            raise ProtocolError(f"stream ended with {remaining} unframed bytes")

    def _validate_sequence(self, metadata: GainMetadataV1 | RadioMetadataV2) -> None:
        if self._expected_stream_id is None:
            self._expected_stream_id = metadata.stream_id
            if metadata.buffer_sequence != 0:
                raise ProtocolError(
                    "new stream must begin at buffer sequence 0, "
                    f"got {metadata.buffer_sequence}"
                )
        elif metadata.stream_id != self._expected_stream_id:
            raise ProtocolError(
                f"stream ID changed without reset: {self._expected_stream_id} "
                f"-> {metadata.stream_id}"
            )

        if self._expected_buffer_sequence is not None:
            if metadata.buffer_sequence != self._expected_buffer_sequence:
                raise ProtocolError(
                    "buffer sequence discontinuity: "
                    f"expected {self._expected_buffer_sequence}, "
                    f"got {metadata.buffer_sequence}"
                )
        self._expected_buffer_sequence = (
            metadata.buffer_sequence + 1
        ) & 0xFFFFFFFFFFFFFFFF

        if metadata.flags & MetadataFlags.SAMPLE_SEQUENCE_VALID:
            if self._expected_first_sample_sequence is None:
                if metadata.first_sample_sequence != 0:
                    raise ProtocolError(
                        "new stream must begin at sample sequence 0, "
                        f"got {metadata.first_sample_sequence}"
                    )
            else:
                if (
                    metadata.first_sample_sequence
                    != self._expected_first_sample_sequence
                ):
                    raise ProtocolError(
                        "sample sequence discontinuity: "
                        f"expected {self._expected_first_sample_sequence}, "
                        f"got {metadata.first_sample_sequence}"
                    )
            self._expected_first_sample_sequence = (
                metadata.first_sample_sequence + metadata.samples_per_channel
            ) & 0xFFFFFFFFFFFFFFFF
        else:
            self._expected_first_sample_sequence = None


def _fields_for_pack(metadata: GainMetadataV1, crc32: int) -> tuple[int, ...]:
    return (
        MAGIC,
        VERSION,
        HEADER_BYTES,
        int(metadata.features),
        int(metadata.flags),
        metadata.stream_id,
        metadata.buffer_sequence,
        metadata.first_sample_sequence,
        metadata.samples_per_channel,
        metadata.iq_payload_bytes,
        metadata.enabled_scan_mask,
        int(metadata.sample_format),
        metadata.channel_count,
        metadata.rx1_gain_start,
        metadata.rx2_gain_start,
        metadata.rx1_gain_end,
        metadata.rx2_gain_end,
        0,
        metadata.gain_start_read_duration_ns,
        metadata.gain_end_read_duration_ns,
        metadata.rx1_first_change_sample,
        metadata.rx2_first_change_sample,
        crc32,
    )


def _fields_for_pack_v2(metadata: RadioMetadataV2, crc32: int) -> tuple[int, ...]:
    return (
        MAGIC,
        VERSION_V2,
        HEADER_BYTES_V2,
        int(metadata.features),
        int(metadata.flags),
        metadata.stream_id,
        metadata.buffer_sequence,
        metadata.first_sample_sequence,
        metadata.samples_per_channel,
        metadata.iq_payload_bytes,
        metadata.enabled_scan_mask,
        int(metadata.sample_format),
        metadata.channel_count,
        metadata.rx1_gain_db_start,
        metadata.rx2_gain_db_start,
        metadata.rx1_gain_db_end,
        metadata.rx2_gain_db_end,
        0,
        metadata.gain_start_read_duration_ns,
        metadata.gain_end_read_duration_ns,
        metadata.rx1_first_change_sample,
        metadata.rx2_first_change_sample,
        metadata.rx1_rssi_start_qdb,
        metadata.rx2_rssi_start_qdb,
        metadata.rx1_rssi_end_qdb,
        metadata.rx2_rssi_end_qdb,
        metadata.rssi_start_read_duration_ns,
        metadata.rssi_end_read_duration_ns,
        crc32,
    )


def _validate_metadata_v2(metadata: RadioMetadataV2) -> None:
    for name, value, bits in (
        ("features", int(metadata.features), 32),
        ("flags", int(metadata.flags), 32),
        ("stream_id", metadata.stream_id, 64),
        ("buffer_sequence", metadata.buffer_sequence, 64),
        ("first_sample_sequence", metadata.first_sample_sequence, 64),
        ("samples_per_channel", metadata.samples_per_channel, 32),
        ("iq_payload_bytes", metadata.iq_payload_bytes, 32),
        ("enabled_scan_mask", metadata.enabled_scan_mask, 32),
        ("sample_format", int(metadata.sample_format), 16),
        ("channel_count", metadata.channel_count, 8),
        ("gain_start_read_duration_ns", metadata.gain_start_read_duration_ns, 32),
        ("gain_end_read_duration_ns", metadata.gain_end_read_duration_ns, 32),
        ("rx1_first_change_sample", metadata.rx1_first_change_sample, 32),
        ("rx2_first_change_sample", metadata.rx2_first_change_sample, 32),
        ("rx1_rssi_start_qdb", metadata.rx1_rssi_start_qdb, 16),
        ("rx2_rssi_start_qdb", metadata.rx2_rssi_start_qdb, 16),
        ("rx1_rssi_end_qdb", metadata.rx1_rssi_end_qdb, 16),
        ("rx2_rssi_end_qdb", metadata.rx2_rssi_end_qdb, 16),
        ("rssi_start_read_duration_ns", metadata.rssi_start_read_duration_ns, 32),
        ("rssi_end_read_duration_ns", metadata.rssi_end_read_duration_ns, 32),
    ):
        _validate_uint(name, value, bits)
    for name in (
        "rx1_gain_db_start",
        "rx2_gain_db_start",
        "rx1_gain_db_end",
        "rx2_gain_db_end",
    ):
        value = getattr(metadata, name)
        if not isinstance(value, int) or not -128 <= value <= 127:
            raise ProtocolError(f"{name} is outside int8: {value!r}")

    unknown_features = int(metadata.features) & ~int(KNOWN_FEATURES)
    if unknown_features:
        raise ProtocolError(f"unknown metadata feature bits: 0x{unknown_features:08x}")
    unknown_flags = int(metadata.flags) & ~int(KNOWN_FLAGS)
    if unknown_flags:
        raise ProtocolError(f"unknown metadata flag bits: 0x{unknown_flags:08x}")
    required_features = (
        MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
        | MetadataFeatures.HEADER_CRC32
        | MetadataFeatures.GAIN_DB_ENDPOINTS
        | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
    )
    if metadata.features & required_features != required_features:
        raise ProtocolError("protocol v2 is missing required gain/RSSI features")
    if metadata.stream_id == 0:
        raise ProtocolError("stream_id must be nonzero")
    if metadata.samples_per_channel <= 0:
        raise ProtocolError("samples_per_channel must be positive")
    if metadata.channel_count != 2:
        raise ProtocolError(
            f"protocol v2 requires two complex RX channels, got {metadata.channel_count}"
        )
    if metadata.enabled_scan_mask != 0x0F:
        raise ProtocolError(
            "protocol v2 requires RX1 I/Q and RX2 I/Q scan elements "
            f"(mask 0x0f), got 0x{metadata.enabled_scan_mask:08x}"
        )
    if metadata.sample_format != SampleFormat.CS16_LE_TIME_INTERLEAVED:
        raise ProtocolError(f"unsupported sample format: {metadata.sample_format}")
    expected_payload_bytes = metadata.samples_per_channel * 8
    if metadata.iq_payload_bytes != expected_payload_bytes:
        raise ProtocolError(
            f"payload size mismatch: got {metadata.iq_payload_bytes}, "
            f"expected {expected_payload_bytes}"
        )

    start_valid = bool(metadata.flags & MetadataFlags.START_VALID)
    end_valid = bool(metadata.flags & MetadataFlags.END_VALID)
    starts = (metadata.rx1_gain_db_start, metadata.rx2_gain_db_start)
    ends = (metadata.rx1_gain_db_end, metadata.rx2_gain_db_end)
    if start_valid != all(gain != GAIN_DB_INVALID for gain in starts):
        raise ProtocolError("START_VALID disagrees with start gain sentinels")
    if end_valid != all(gain != GAIN_DB_INVALID for gain in ends):
        raise ProtocolError("END_VALID disagrees with end gain sentinels")
    if (start_valid or end_valid) and not bool(
        metadata.flags & MetadataFlags.GAIN_DB_VALUES
    ):
        raise ProtocolError("valid v2 gain values require GAIN_DB_VALUES")
    if (
        start_valid
        and end_valid
        and not bool(metadata.flags & MetadataFlags.GAIN_FULL_TABLE_MODE)
    ):
        raise ProtocolError("valid v2 gains require full gain-table mode")
    gain_failed = bool(metadata.flags & MetadataFlags.GAIN_READ_FAILED)
    if gain_failed != (not start_valid or not end_valid):
        raise ProtocolError("GAIN_READ_FAILED disagrees with gain validity")
    if metadata.flags & (
        MetadataFlags.RX1_ENDPOINT_CHANGED | MetadataFlags.RX2_ENDPOINT_CHANGED
    ) and not (start_valid and end_valid):
        raise ProtocolError("gain endpoint-change flag requires valid endpoints")
    if metadata.flags & MetadataFlags.DUMMY_GAINS:
        raise ProtocolError("protocol v2 does not accept dummy gains")

    rssi_start_valid = bool(metadata.flags & MetadataFlags.RSSI_START_VALID)
    rssi_end_valid = bool(metadata.flags & MetadataFlags.RSSI_END_VALID)
    rssi_starts = (metadata.rx1_rssi_start_qdb, metadata.rx2_rssi_start_qdb)
    rssi_ends = (metadata.rx1_rssi_end_qdb, metadata.rx2_rssi_end_qdb)
    if rssi_start_valid != all(value != RSSI_QDB_INVALID for value in rssi_starts):
        raise ProtocolError("RSSI_START_VALID disagrees with RSSI sentinels")
    if rssi_end_valid != all(value != RSSI_QDB_INVALID for value in rssi_ends):
        raise ProtocolError("RSSI_END_VALID disagrees with RSSI sentinels")
    for value in (*rssi_starts, *rssi_ends):
        if value != RSSI_QDB_INVALID and value > 511:
            raise ProtocolError(f"RSSI quarter-dB value is out of range: {value}")
    rssi_failed = bool(metadata.flags & MetadataFlags.RSSI_READ_FAILED)
    if rssi_failed != (not rssi_start_valid or not rssi_end_valid):
        raise ProtocolError("RSSI_READ_FAILED disagrees with RSSI validity")

    sample_feature = bool(metadata.features & MetadataFeatures.SAMPLE_SEQUENCE)
    sample_valid = bool(metadata.flags & MetadataFlags.SAMPLE_SEQUENCE_VALID)
    if sample_valid and not sample_feature:
        raise ProtocolError("sample sequence marked valid without negotiated feature")
    if not sample_valid and metadata.first_sample_sequence != 0:
        raise ProtocolError("invalid sample sequence must be zero")

    fpga_feature = bool(metadata.features & MetadataFeatures.FPGA_GAIN_EVENTS)
    fpga_flags = (
        MetadataFlags.FPGA_EVENTS_VALID
        | MetadataFlags.RX1_CHANGED_IN_BUFFER
        | MetadataFlags.RX2_CHANGED_IN_BUFFER
        | MetadataFlags.RX1_LOCKED_AT_END
        | MetadataFlags.RX2_LOCKED_AT_END
        | MetadataFlags.FPGA_EVENT_OVERFLOW
    )
    if metadata.flags & fpga_flags and not fpga_feature:
        raise ProtocolError("FPGA flags present without negotiated FPGA feature")
    events_valid = bool(metadata.flags & MetadataFlags.FPGA_EVENTS_VALID)
    for channel, first_change, changed_flag in (
        ("RX1", metadata.rx1_first_change_sample, MetadataFlags.RX1_CHANGED_IN_BUFFER),
        ("RX2", metadata.rx2_first_change_sample, MetadataFlags.RX2_CHANGED_IN_BUFFER),
    ):
        changed = bool(metadata.flags & changed_flag)
        if changed:
            if not events_valid or first_change >= metadata.samples_per_channel:
                raise ProtocolError(f"{channel} event position is invalid")
        elif first_change != FIRST_CHANGE_UNAVAILABLE:
            raise ProtocolError(f"{channel} event position present without change flag")


def _validate_metadata(metadata: GainMetadataV1) -> None:
    for name, value, bits in (
        ("features", int(metadata.features), 32),
        ("flags", int(metadata.flags), 32),
        ("stream_id", metadata.stream_id, 64),
        ("buffer_sequence", metadata.buffer_sequence, 64),
        ("first_sample_sequence", metadata.first_sample_sequence, 64),
        ("samples_per_channel", metadata.samples_per_channel, 32),
        ("iq_payload_bytes", metadata.iq_payload_bytes, 32),
        ("enabled_scan_mask", metadata.enabled_scan_mask, 32),
        ("sample_format", int(metadata.sample_format), 16),
        ("channel_count", metadata.channel_count, 8),
        ("gain_start_read_duration_ns", metadata.gain_start_read_duration_ns, 32),
        ("gain_end_read_duration_ns", metadata.gain_end_read_duration_ns, 32),
        ("rx1_first_change_sample", metadata.rx1_first_change_sample, 32),
        ("rx2_first_change_sample", metadata.rx2_first_change_sample, 32),
    ):
        _validate_uint(name, value, bits)

    unknown_features = int(metadata.features) & ~int(KNOWN_FEATURES)
    if unknown_features:
        raise ProtocolError(f"unknown metadata feature bits: 0x{unknown_features:08x}")
    unknown_flags = int(metadata.flags) & ~int(KNOWN_FLAGS)
    if unknown_flags:
        raise ProtocolError(f"unknown metadata flag bits: 0x{unknown_flags:08x}")

    required_features = (
        MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS | MetadataFeatures.HEADER_CRC32
    )
    if metadata.features & required_features != required_features:
        raise ProtocolError(
            "protocol v1 requires GAIN_ENDPOINT_SNAPSHOTS and HEADER_CRC32"
        )
    if metadata.features & MetadataFeatures.HEADER_CRC32 == 0:
        raise ProtocolError("protocol v1 requires HEADER_CRC32")
    if metadata.stream_id == 0:
        raise ProtocolError("stream_id must be nonzero")
    if metadata.samples_per_channel <= 0:
        raise ProtocolError("samples_per_channel must be positive")
    if metadata.channel_count != 2:
        raise ProtocolError(
            f"protocol v1 requires two complex RX channels, got {metadata.channel_count}"
        )
    if metadata.enabled_scan_mask != 0x0F:
        raise ProtocolError(
            "protocol v1 requires RX1 I/Q and RX2 I/Q scan elements "
            f"(mask 0x0f), got 0x{metadata.enabled_scan_mask:08x}"
        )
    if metadata.sample_format != SampleFormat.CS16_LE_TIME_INTERLEAVED:
        raise ProtocolError(f"unsupported sample format: {metadata.sample_format}")

    expected_payload_bytes = metadata.samples_per_channel * metadata.channel_count * 4
    if metadata.iq_payload_bytes != expected_payload_bytes:
        raise ProtocolError(
            f"payload size mismatch: got {metadata.iq_payload_bytes}, "
            f"expected {expected_payload_bytes}"
        )

    for name in (
        "rx1_gain_start",
        "rx2_gain_start",
        "rx1_gain_end",
        "rx2_gain_end",
    ):
        gain = getattr(metadata, name)
        _validate_uint(name, gain, 8)
        if gain != GAIN_INDEX_INVALID and not 0 <= gain <= 0x7F:
            raise ProtocolError(f"{name} is not a seven-bit gain index: {gain}")

    start_valid = bool(metadata.flags & MetadataFlags.START_VALID)
    end_valid = bool(metadata.flags & MetadataFlags.END_VALID)
    starts = (metadata.rx1_gain_start, metadata.rx2_gain_start)
    ends = (metadata.rx1_gain_end, metadata.rx2_gain_end)
    if start_valid != all(gain != GAIN_INDEX_INVALID for gain in starts):
        raise ProtocolError("START_VALID disagrees with start gain sentinels")
    if end_valid != all(gain != GAIN_INDEX_INVALID for gain in ends):
        raise ProtocolError("END_VALID disagrees with end gain sentinels")

    dummy_gains = bool(metadata.flags & MetadataFlags.DUMMY_GAINS)
    full_table = bool(metadata.flags & MetadataFlags.GAIN_FULL_TABLE_MODE)
    if start_valid and end_valid and not (full_table or dummy_gains):
        raise ProtocolError("valid real gain indices require full gain-table mode")

    gain_read_failed = bool(metadata.flags & MetadataFlags.GAIN_READ_FAILED)
    if not dummy_gains and gain_read_failed != (not start_valid or not end_valid):
        raise ProtocolError("GAIN_READ_FAILED disagrees with endpoint validity flags")

    if bool(metadata.flags & MetadataFlags.RX1_ENDPOINT_CHANGED) != (
        start_valid and end_valid and metadata.rx1_gain_start != metadata.rx1_gain_end
    ):
        raise ProtocolError("RX1 endpoint-change flag disagrees with gain indices")
    if bool(metadata.flags & MetadataFlags.RX2_ENDPOINT_CHANGED) != (
        start_valid and end_valid and metadata.rx2_gain_start != metadata.rx2_gain_end
    ):
        raise ProtocolError("RX2 endpoint-change flag disagrees with gain indices")

    sample_feature = bool(metadata.features & MetadataFeatures.SAMPLE_SEQUENCE)
    sample_valid = bool(metadata.flags & MetadataFlags.SAMPLE_SEQUENCE_VALID)
    if sample_valid and not sample_feature:
        raise ProtocolError("sample sequence marked valid without negotiated feature")
    if not sample_valid and metadata.first_sample_sequence != 0:
        raise ProtocolError("invalid sample sequence must be zero")

    fpga_feature = bool(metadata.features & MetadataFeatures.FPGA_GAIN_EVENTS)
    fpga_flags = (
        MetadataFlags.FPGA_EVENTS_VALID
        | MetadataFlags.RX1_CHANGED_IN_BUFFER
        | MetadataFlags.RX2_CHANGED_IN_BUFFER
        | MetadataFlags.RX1_LOCKED_AT_END
        | MetadataFlags.RX2_LOCKED_AT_END
        | MetadataFlags.FPGA_EVENT_OVERFLOW
    )
    if metadata.flags & fpga_flags and not fpga_feature:
        raise ProtocolError("FPGA flags present without negotiated FPGA feature")

    events_valid = bool(metadata.flags & MetadataFlags.FPGA_EVENTS_VALID)
    for channel, first_change, changed_flag in (
        (
            "RX1",
            metadata.rx1_first_change_sample,
            MetadataFlags.RX1_CHANGED_IN_BUFFER,
        ),
        (
            "RX2",
            metadata.rx2_first_change_sample,
            MetadataFlags.RX2_CHANGED_IN_BUFFER,
        ),
    ):
        changed = bool(metadata.flags & changed_flag)
        if not events_valid and first_change != FIRST_CHANGE_UNAVAILABLE:
            raise ProtocolError(
                f"{channel} event position present without valid events"
            )
        if changed:
            if not events_valid:
                raise ProtocolError(
                    f"{channel} change flag present without valid events"
                )
            if first_change >= metadata.samples_per_channel:
                raise ProtocolError(f"{channel} event position is outside the payload")
        elif first_change != FIRST_CHANGE_UNAVAILABLE:
            raise ProtocolError(f"{channel} event position present without change flag")


def _validate_uint(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise ProtocolError(f"{name} is outside uint{bits}: {value!r}")
