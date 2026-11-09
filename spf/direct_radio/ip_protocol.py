"""Reliable framing for SPF radio frames transported in UDP datagrams.

The inner radio frame is byte-for-byte identical to the direct-USB frame
defined in :mod:`spf.direct_radio.usb_protocol`.  This module only defines
the outer UDP fragmentation envelope and bounded host-side reassembly.

UDP delivery is allowed to reorder or duplicate datagrams.  Missing,
conflicting, overlapping, corrupt, or expired fragments discard the complete
radio frame; partial IQ is never exposed to callers.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import struct
import time
import zlib
from array import array
from collections.abc import Hashable, Iterable
from typing import Final

from spf.direct_radio.usb_protocol import (
    KNOWN_FEATURES,
    MAX_FINITE_FRAMES,
    MAX_GAIN_EVENTS,
    MAX_GAIN_OBSERVATIONS,
    MAX_SAMPLES_PER_CHANNEL,
    VERSION_V1,
    VERSION_V2,
    VERSION_V3,
    MetadataFeatures,
    ProtocolError,
)


IP_FRAGMENT_MAGIC: Final[int] = 0x31504953  # little-endian bytes: b"SIP1"
IP_FRAGMENT_VERSION: Final[int] = 1
MAX_UDP_DATAGRAM_BYTES: Final[int] = 65_507
DEFAULT_UDP_DATAGRAM_BYTES: Final[int] = 1_472
MAX_IP_FRAME_BYTES: Final[int] = 16 * 1024 * 1024
MAX_IP_FRAGMENT_COUNT: Final[int] = 65_536

IP_CONTROL_MAGIC: Final[int] = 0x31434953  # little-endian bytes: b"SIC1"
IP_CONTROL_VERSION: Final[int] = 1


class IpControlType(enum.IntEnum):
    QUERY_CAPABILITIES = 1
    CAPABILITIES = 2
    START_RX = 3
    STARTED = 4
    STOP_RX = 5
    STOPPED = 6
    ERROR = 7


class IpControlFlags(enum.IntFlag):
    FINITE_RX = 1 << 0
    IDEMPOTENT_REQUESTS = 1 << 1
    TIME_ANCHOR = 1 << 2
    QUERY_TRANSPORT_CAPABILITIES = 1 << 3
    BUFFERED_FINITE_RX = 1 << 4
    USB_CLASS_PACING = 1 << 5
    # Deliver radio frames over a TCP stream instead of UDP datagrams.  A
    # gadget advertises this bit only in reply to a query which set it, so a
    # host predating the bit never sees it and never fails its own strict
    # flag validation.
    TCP_DATA_TRANSPORT = 1 << 6
    # Drop whole frames rather than stalling when the host falls behind.
    # Requires TCP_DATA_TRANSPORT: the admission test needs a send queue whose
    # depth can be measured, which UDP does not provide.
    DROP_STALE_FRAMES = 1 << 7


KNOWN_IP_CONTROL_FLAGS: Final[IpControlFlags] = (
    IpControlFlags.FINITE_RX
    | IpControlFlags.IDEMPOTENT_REQUESTS
    | IpControlFlags.TIME_ANCHOR
    | IpControlFlags.QUERY_TRANSPORT_CAPABILITIES
    | IpControlFlags.BUFFERED_FINITE_RX
    | IpControlFlags.USB_CLASS_PACING
    | IpControlFlags.TCP_DATA_TRANSPORT
    | IpControlFlags.DROP_STALE_FRAMES
)

# Flags a host may set on QUERY_CAPABILITIES to ask about an optional feature.
# A gadget which does not know one of these rejects the whole request, which is
# how a new host discovers old firmware.
QUERY_PROBE_FLAGS: Final[IpControlFlags] = (
    IpControlFlags.QUERY_TRANSPORT_CAPABILITIES | IpControlFlags.TCP_DATA_TRANSPORT
)

# Flags which select per-stream behaviour on START_RX.  Everything else in
# IpControlFlags describes a gadget capability and belongs only in CAPABILITIES.
START_TRANSPORT_FLAGS: Final[IpControlFlags] = (
    IpControlFlags.BUFFERED_FINITE_RX
    | IpControlFlags.USB_CLASS_PACING
    | IpControlFlags.TCP_DATA_TRANSPORT
    | IpControlFlags.DROP_STALE_FRAMES
)


# One fixed control record keeps the Pluto C implementation simple.  Fields
# which do not apply to a message type must be zero.  STARTED echoes all START
# parameters and adds the assigned non-zero stream_id, making a retried request
# safe to identify and validate.
_IP_CONTROL_STRUCT: Final[struct.Struct] = struct.Struct("<IHHIQiIHHQIIIIIIHHHHQ")
IP_CONTROL_BYTES: Final[int] = _IP_CONTROL_STRUCT.size
assert IP_CONTROL_BYTES == 80


@dataclasses.dataclass(frozen=True, slots=True)
class IpControlMessageV1:
    message_type: IpControlType
    request_id: int
    status: int = 0
    flags: IpControlFlags = IpControlFlags(0)
    protocol_min: int = 0
    protocol_max: int = 0
    features: MetadataFeatures = MetadataFeatures(0)
    max_samples_per_channel: int = 0
    max_finite_frames: int = 0
    enabled_scan_mask: int = 0
    samples_per_channel: int = 0
    frame_count: int = 0
    gain_observation_interval_samples: int = 0
    gain_observation_capacity: int = 0
    gain_event_capacity: int = 0
    data_port: int = 0
    max_datagram_bytes: int = 0
    stream_id: int = 0

    def pack(self) -> bytes:
        _validate_control_message(self)
        return _IP_CONTROL_STRUCT.pack(
            IP_CONTROL_MAGIC,
            IP_CONTROL_VERSION,
            int(self.message_type),
            IP_CONTROL_BYTES,
            self.request_id,
            self.status,
            int(self.flags),
            self.protocol_min,
            self.protocol_max,
            int(self.features),
            self.max_samples_per_channel,
            self.max_finite_frames,
            self.enabled_scan_mask,
            self.samples_per_channel,
            self.frame_count,
            self.gain_observation_interval_samples,
            self.gain_observation_capacity,
            self.gain_event_capacity,
            self.data_port,
            self.max_datagram_bytes,
            self.stream_id,
        )

    @classmethod
    def unpack(cls, payload: bytes | bytearray | memoryview) -> "IpControlMessageV1":
        if len(payload) != IP_CONTROL_BYTES:
            raise ProtocolError(
                "direct-IP control size mismatch: "
                f"got {len(payload)}, expected {IP_CONTROL_BYTES}"
            )
        (
            magic,
            version,
            message_type,
            message_bytes,
            request_id,
            status,
            flags,
            protocol_min,
            protocol_max,
            features,
            max_samples_per_channel,
            max_finite_frames,
            enabled_scan_mask,
            samples_per_channel,
            frame_count,
            gain_observation_interval_samples,
            gain_observation_capacity,
            gain_event_capacity,
            data_port,
            max_datagram_bytes,
            stream_id,
        ) = _IP_CONTROL_STRUCT.unpack(payload)
        if magic != IP_CONTROL_MAGIC:
            raise ProtocolError(f"bad direct-IP control magic: 0x{magic:08x}")
        if version != IP_CONTROL_VERSION:
            raise ProtocolError(f"unsupported direct-IP control version: {version}")
        if message_bytes != IP_CONTROL_BYTES:
            raise ProtocolError(f"unsupported direct-IP control size: {message_bytes}")
        try:
            parsed_type = IpControlType(message_type)
        except ValueError as exc:
            raise ProtocolError(
                f"unknown direct-IP control message type: {message_type}"
            ) from exc
        message = cls(
            message_type=parsed_type,
            request_id=request_id,
            status=status,
            flags=IpControlFlags(flags),
            protocol_min=protocol_min,
            protocol_max=protocol_max,
            features=MetadataFeatures(features),
            max_samples_per_channel=max_samples_per_channel,
            max_finite_frames=max_finite_frames,
            enabled_scan_mask=enabled_scan_mask,
            samples_per_channel=samples_per_channel,
            frame_count=frame_count,
            gain_observation_interval_samples=gain_observation_interval_samples,
            gain_observation_capacity=gain_observation_capacity,
            gain_event_capacity=gain_event_capacity,
            data_port=data_port,
            max_datagram_bytes=max_datagram_bytes,
            stream_id=stream_id,
        )
        _validate_control_message(message, strict_flags=False)
        return message


def make_ip_capability_query(
    *,
    request_id: int,
    transport_capabilities: bool = False,
    tcp_transport: bool = False,
) -> IpControlMessageV1:
    flags = IpControlFlags(0)
    if transport_capabilities:
        flags |= IpControlFlags.QUERY_TRANSPORT_CAPABILITIES
    if tcp_transport:
        flags |= IpControlFlags.TCP_DATA_TRANSPORT
    return IpControlMessageV1(
        message_type=IpControlType.QUERY_CAPABILITIES,
        request_id=request_id,
        flags=flags,
    )


def make_ip_start_request(
    *,
    request_id: int,
    protocol_version: int,
    features: MetadataFeatures,
    enabled_scan_mask: int,
    samples_per_channel: int,
    frame_count: int,
    data_port: int,
    max_datagram_bytes: int = DEFAULT_UDP_DATAGRAM_BYTES,
    gain_observation_interval_samples: int = 0,
    gain_observation_capacity: int = 0,
    gain_event_capacity: int = 0,
    transport_flags: IpControlFlags = IpControlFlags(0),
) -> IpControlMessageV1:
    return IpControlMessageV1(
        message_type=IpControlType.START_RX,
        request_id=request_id,
        flags=transport_flags,
        protocol_min=protocol_version,
        protocol_max=protocol_version,
        features=features,
        enabled_scan_mask=enabled_scan_mask,
        samples_per_channel=samples_per_channel,
        frame_count=frame_count,
        gain_observation_interval_samples=gain_observation_interval_samples,
        gain_observation_capacity=gain_observation_capacity,
        gain_event_capacity=gain_event_capacity,
        data_port=data_port,
        max_datagram_bytes=max_datagram_bytes,
    )


def make_ip_stop_request(*, request_id: int, stream_id: int) -> IpControlMessageV1:
    return IpControlMessageV1(
        message_type=IpControlType.STOP_RX,
        request_id=request_id,
        stream_id=stream_id,
    )


class IpFragmentFlags(enum.IntFlag):
    FIRST = 1 << 0
    LAST = 1 << 1


# The frame CRC covers the complete inner metadata + IQ byte string.  In
# particular, it protects IQ bytes which are intentionally outside the inner
# metadata-header CRC.
_IP_FRAGMENT_STRUCT: Final[struct.Struct] = struct.Struct("<IHHIQQIIIIII")
IP_FRAGMENT_HEADER_BYTES: Final[int] = _IP_FRAGMENT_STRUCT.size
assert IP_FRAGMENT_HEADER_BYTES == 52


@dataclasses.dataclass(frozen=True, slots=True)
class IpFragmentV1:
    flags: IpFragmentFlags
    stream_id: int
    frame_sequence: int
    frame_bytes: int
    frame_crc32: int
    fragment_index: int
    fragment_count: int
    fragment_offset: int
    payload: bytes

    def pack(self) -> bytes:
        _validate_fragment(self)
        header = _IP_FRAGMENT_STRUCT.pack(
            IP_FRAGMENT_MAGIC,
            IP_FRAGMENT_VERSION,
            IP_FRAGMENT_HEADER_BYTES,
            int(self.flags),
            self.stream_id,
            self.frame_sequence,
            self.frame_bytes,
            self.frame_crc32,
            self.fragment_index,
            self.fragment_count,
            self.fragment_offset,
            len(self.payload),
        )
        return header + self.payload

    @classmethod
    def unpack(cls, datagram: bytes | bytearray | memoryview) -> "IpFragmentV1":
        if not IP_FRAGMENT_HEADER_BYTES < len(datagram) <= MAX_UDP_DATAGRAM_BYTES:
            raise ProtocolError(
                "direct-IP datagram size is outside the supported range"
            )
        (
            magic,
            version,
            header_bytes,
            flags,
            stream_id,
            frame_sequence,
            frame_bytes,
            frame_crc32,
            fragment_index,
            fragment_count,
            fragment_offset,
            fragment_bytes,
        ) = _IP_FRAGMENT_STRUCT.unpack_from(datagram)
        if magic != IP_FRAGMENT_MAGIC:
            raise ProtocolError(f"bad direct-IP magic: 0x{magic:08x}")
        if version != IP_FRAGMENT_VERSION:
            raise ProtocolError(f"unsupported direct-IP version: {version}")
        if header_bytes != IP_FRAGMENT_HEADER_BYTES:
            raise ProtocolError(f"unsupported direct-IP header size: {header_bytes}")
        if len(datagram) != header_bytes + fragment_bytes:
            raise ProtocolError("direct-IP fragment length mismatch")
        fragment = cls(
            flags=IpFragmentFlags(flags),
            stream_id=stream_id,
            frame_sequence=frame_sequence,
            frame_bytes=frame_bytes,
            frame_crc32=frame_crc32,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
            fragment_offset=fragment_offset,
            payload=bytes(memoryview(datagram)[header_bytes:]),
        )
        _validate_fragment(fragment)
        return fragment


@dataclasses.dataclass(frozen=True, slots=True)
class ReassembledIpFrame:
    stream_id: int
    frame_sequence: int
    frame: bytes | bytearray


def fragment_ip_frame(
    frame: bytes | bytearray | memoryview,
    *,
    stream_id: int,
    frame_sequence: int,
    max_datagram_bytes: int = DEFAULT_UDP_DATAGRAM_BYTES,
) -> tuple[bytes, ...]:
    """Split one complete inner radio frame into versioned UDP datagrams."""

    frame_view = memoryview(frame)
    frame_bytes = len(frame_view)
    if not 1 <= frame_bytes <= MAX_IP_FRAME_BYTES:
        raise ProtocolError("direct-IP frame size is outside the supported range")
    if not IP_FRAGMENT_HEADER_BYTES < max_datagram_bytes <= MAX_UDP_DATAGRAM_BYTES:
        raise ProtocolError("direct-IP datagram limit is outside the UDP range")
    _validate_uint("stream_id", stream_id, 64)
    _validate_uint("frame_sequence", frame_sequence, 64)
    payload_capacity = max_datagram_bytes - IP_FRAGMENT_HEADER_BYTES
    fragment_count = math.ceil(frame_bytes / payload_capacity)
    if fragment_count > MAX_IP_FRAGMENT_COUNT:
        raise ProtocolError("direct-IP frame requires too many fragments")
    frame_crc32 = zlib.crc32(frame_view) & 0xFFFFFFFF
    datagrams = []
    for fragment_index in range(fragment_count):
        offset = fragment_index * payload_capacity
        payload = bytes(frame_view[offset : offset + payload_capacity])
        flags = IpFragmentFlags(0)
        if fragment_index == 0:
            flags |= IpFragmentFlags.FIRST
        if fragment_index == fragment_count - 1:
            flags |= IpFragmentFlags.LAST
        datagrams.append(
            IpFragmentV1(
                flags=flags,
                stream_id=stream_id,
                frame_sequence=frame_sequence,
                frame_bytes=frame_bytes,
                frame_crc32=frame_crc32,
                fragment_index=fragment_index,
                fragment_count=fragment_count,
                fragment_offset=offset,
                payload=payload,
            ).pack()
        )
    return tuple(datagrams)


@dataclasses.dataclass(slots=True)
class _PartialIpFrame:
    frame_bytes: int
    frame_crc32: int
    fragment_count: int
    first_seen: float
    frame: bytearray
    fragment_offsets: array
    fragment_lengths: array
    received_fragment_count: int = 0


class IpFrameReassembler:
    """Bounded UDP frame reassembler with explicit loss accounting."""

    def __init__(
        self,
        *,
        frame_timeout_seconds: float = 2.0,
        max_pending_frames: int = 8,
        max_pending_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if frame_timeout_seconds <= 0:
            raise ValueError("frame_timeout_seconds must be positive")
        if max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be positive")
        if max_pending_bytes <= 0:
            raise ValueError("max_pending_bytes must be positive")
        self.frame_timeout_seconds = float(frame_timeout_seconds)
        self.max_pending_frames = int(max_pending_frames)
        self.max_pending_bytes = int(max_pending_bytes)
        self._pending: dict[tuple[Hashable | None, int, int], _PartialIpFrame] = {}
        self._recently_completed: dict[
            tuple[Hashable | None, int, int], tuple[int, int, float]
        ] = {}
        self._pending_declared_bytes = 0
        self._feed_count = 0
        self._last_expiry_check = 0.0
        self.completed_frame_count = 0
        self.expired_frame_count = 0
        self.rejected_frame_count = 0
        self.duplicate_fragment_count = 0

    @property
    def pending_frame_count(self) -> int:
        return len(self._pending)

    @property
    def pending_declared_bytes(self) -> int:
        return self._pending_declared_bytes

    def reset(self) -> None:
        self._pending.clear()
        self._recently_completed.clear()
        self._pending_declared_bytes = 0

    def expire(self, *, now: float | None = None) -> int:
        """Discard timed-out partial frames and return the number expired."""

        current = time.monotonic() if now is None else float(now)
        expired = [
            key
            for key, partial in self._pending.items()
            if current - partial.first_seen >= self.frame_timeout_seconds
        ]
        for key in expired:
            self._remove_pending(key)
        completed_expiry = [
            key
            for key, (_, _, completed_at) in self._recently_completed.items()
            if current - completed_at >= self.frame_timeout_seconds
        ]
        for key in completed_expiry:
            del self._recently_completed[key]
        self.expired_frame_count += len(expired)
        return len(expired)

    def feed(
        self,
        datagram: bytes | bytearray | memoryview,
        *,
        peer: Hashable | None = None,
        now: float | None = None,
    ) -> list[ReassembledIpFrame]:
        """Consume one datagram and return zero or one complete radio frames."""

        self._feed_count += 1
        if now is None:
            # A production frame contains roughly 3,000 MTU-safe datagrams.
            # Reading the clock and scanning both dictionaries for every packet
            # consumed a measurable share of a Pi 4 core.  A 256-packet cadence
            # remains far below the multi-second frame timeout while keeping the
            # per-datagram hot path allocation-light.
            if self._last_expiry_check == 0.0 or self._feed_count % 256 == 0:
                current = time.monotonic()
                self.expire(now=current)
                self._last_expiry_check = current
            else:
                current = self._last_expiry_check
        else:
            current = float(now)
            self.expire(now=current)
            self._last_expiry_check = current

        (
            flags,
            stream_id,
            frame_sequence,
            frame_bytes,
            frame_crc32,
            fragment_index,
            fragment_count,
            fragment_offset,
            payload,
        ) = _unpack_ip_fragment_view(datagram)
        key = (peer, stream_id, frame_sequence)
        completed = self._recently_completed.get(key)
        if completed is not None:
            completed_crc32, completed_fragment_count, _ = completed
            if (
                completed_crc32 != frame_crc32
                or completed_fragment_count != fragment_count
            ):
                self.rejected_frame_count += 1
                raise ProtocolError("conflicting late direct-IP fragment")
            self.duplicate_fragment_count += 1
            return []
        partial = self._pending.get(key)
        if partial is None:
            if len(self._pending) >= self.max_pending_frames:
                self.rejected_frame_count += 1
                raise ProtocolError("direct-IP pending-frame limit exceeded")
            if self._pending_declared_bytes + frame_bytes > self.max_pending_bytes:
                self.rejected_frame_count += 1
                raise ProtocolError("direct-IP pending-byte limit exceeded")
            partial = _PartialIpFrame(
                frame_bytes=frame_bytes,
                frame_crc32=frame_crc32,
                fragment_count=fragment_count,
                first_seen=current,
                frame=bytearray(frame_bytes),
                fragment_offsets=array("I", [0xFFFFFFFF]) * fragment_count,
                fragment_lengths=array("I", [0]) * fragment_count,
            )
            self._pending[key] = partial
            self._pending_declared_bytes += frame_bytes
        elif (
            partial.frame_bytes != frame_bytes
            or partial.frame_crc32 != frame_crc32
            or partial.fragment_count != fragment_count
        ):
            self._remove_pending(key)
            self.rejected_frame_count += 1
            raise ProtocolError("conflicting direct-IP frame description")

        existing_offset = partial.fragment_offsets[fragment_index]
        if existing_offset != 0xFFFFFFFF:
            existing_length = partial.fragment_lengths[fragment_index]
            existing_payload = memoryview(partial.frame)[
                existing_offset : existing_offset + existing_length
            ]
            if (
                existing_offset != fragment_offset
                or existing_length != len(payload)
                or existing_payload != payload
            ):
                self._remove_pending(key)
                self.rejected_frame_count += 1
                raise ProtocolError("conflicting duplicate direct-IP fragment")
            self.duplicate_fragment_count += 1
            return []
        fragment_end = fragment_offset + len(payload)
        partial.frame[fragment_offset:fragment_end] = payload
        partial.fragment_offsets[fragment_index] = fragment_offset
        partial.fragment_lengths[fragment_index] = len(payload)
        partial.received_fragment_count += 1
        if partial.received_fragment_count != partial.fragment_count:
            return []

        try:
            frame = _assemble_complete_frame(partial)
        except ProtocolError:
            self.rejected_frame_count += 1
            raise
        finally:
            self._remove_pending(key)
        self.completed_frame_count += 1
        self._recently_completed[key] = (
            partial.frame_crc32,
            partial.fragment_count,
            current,
        )
        return [
            ReassembledIpFrame(
                stream_id=stream_id,
                frame_sequence=frame_sequence,
                frame=frame,
            )
        ]

    def _remove_pending(self, key: tuple[Hashable | None, int, int]) -> None:
        partial = self._pending.pop(key)
        self._pending_declared_bytes -= partial.frame_bytes


def reassemble_ip_datagrams(
    datagrams: Iterable[bytes],
    *,
    peer: Hashable | None = None,
) -> bytes:
    """Convenience helper for a known-complete collection of datagrams."""

    reassembler = IpFrameReassembler()
    frames: list[ReassembledIpFrame] = []
    for datagram in datagrams:
        frames.extend(reassembler.feed(datagram, peer=peer))
    if reassembler.pending_frame_count:
        raise ProtocolError("direct-IP datagram collection is incomplete")
    if len(frames) != 1:
        raise ProtocolError(f"expected one direct-IP frame, received {len(frames)}")
    return bytes(frames[0].frame)


def _assemble_complete_frame(partial: _PartialIpFrame) -> bytearray:
    expected_offset = 0
    for fragment_index in range(partial.fragment_count):
        offset = partial.fragment_offsets[fragment_index]
        payload_bytes = partial.fragment_lengths[fragment_index]
        if offset == 0xFFFFFFFF:
            raise ProtocolError("direct-IP frame has a missing fragment index")
        if offset != expected_offset:
            relation = "overlap" if offset < expected_offset else "gap"
            raise ProtocolError(f"direct-IP frame has a fragment {relation}")
        expected_offset += payload_bytes
    if expected_offset != partial.frame_bytes:
        raise ProtocolError("direct-IP fragments do not cover the declared frame")
    calculated_crc32 = zlib.crc32(partial.frame) & 0xFFFFFFFF
    if calculated_crc32 != partial.frame_crc32:
        raise ProtocolError("direct-IP frame CRC mismatch")
    return partial.frame


def _unpack_ip_fragment_view(
    datagram: bytes | bytearray | memoryview,
) -> tuple[int, int, int, int, int, int, int, int, memoryview]:
    """Parse one fragment without copying its payload or allocating a dataclass."""

    if not IP_FRAGMENT_HEADER_BYTES < len(datagram) <= MAX_UDP_DATAGRAM_BYTES:
        raise ProtocolError("direct-IP datagram size is outside the supported range")
    (
        magic,
        version,
        header_bytes,
        flags,
        stream_id,
        frame_sequence,
        frame_bytes,
        frame_crc32,
        fragment_index,
        fragment_count,
        fragment_offset,
        fragment_bytes,
    ) = _IP_FRAGMENT_STRUCT.unpack_from(datagram)
    if magic != IP_FRAGMENT_MAGIC:
        raise ProtocolError(f"bad direct-IP magic: 0x{magic:08x}")
    if version != IP_FRAGMENT_VERSION:
        raise ProtocolError(f"unsupported direct-IP version: {version}")
    if header_bytes != IP_FRAGMENT_HEADER_BYTES:
        raise ProtocolError(f"unsupported direct-IP header size: {header_bytes}")
    if len(datagram) != header_bytes + fragment_bytes:
        raise ProtocolError("direct-IP fragment length mismatch")
    # The struct already constrains every numeric field to its unsigned wire
    # width.  Keep the remaining semantic checks inline: constructing an enum
    # dataclass and running generic integer validators for every UDP packet was
    # over 80% of host reassembly CPU on the Pi 4.
    if flags & ~int(IpFragmentFlags.FIRST | IpFragmentFlags.LAST):
        raise ProtocolError("unknown direct-IP fragment flags")
    if not 1 <= frame_bytes <= MAX_IP_FRAME_BYTES:
        raise ProtocolError("direct-IP frame size is outside the supported range")
    if not 1 <= fragment_count <= MAX_IP_FRAGMENT_COUNT:
        raise ProtocolError("direct-IP fragment count is outside the supported range")
    if fragment_index >= fragment_count:
        raise ProtocolError("direct-IP fragment index is outside the frame")
    if (
        fragment_bytes == 0
        or fragment_offset > frame_bytes
        or fragment_bytes > frame_bytes - fragment_offset
    ):
        raise ProtocolError("direct-IP fragment range is outside the frame")
    if bool(flags & int(IpFragmentFlags.FIRST)) != (fragment_index == 0):
        raise ProtocolError("direct-IP FIRST flag does not match fragment index")
    if bool(flags & int(IpFragmentFlags.LAST)) != (
        fragment_index == fragment_count - 1
    ):
        raise ProtocolError("direct-IP LAST flag does not match fragment index")
    payload = memoryview(datagram)[header_bytes:]
    return (
        flags,
        stream_id,
        frame_sequence,
        frame_bytes,
        frame_crc32,
        fragment_index,
        fragment_count,
        fragment_offset,
        payload,
    )


def _validate_fragment(fragment: IpFragmentV1) -> None:
    for name, value, bits in (
        ("flags", int(fragment.flags), 32),
        ("stream_id", fragment.stream_id, 64),
        ("frame_sequence", fragment.frame_sequence, 64),
        ("frame_bytes", fragment.frame_bytes, 32),
        ("frame_crc32", fragment.frame_crc32, 32),
        ("fragment_index", fragment.fragment_index, 32),
        ("fragment_count", fragment.fragment_count, 32),
        ("fragment_offset", fragment.fragment_offset, 32),
    ):
        _validate_uint(name, value, bits)
    unknown_flags = int(fragment.flags) & ~int(
        IpFragmentFlags.FIRST | IpFragmentFlags.LAST
    )
    if unknown_flags:
        raise ProtocolError(f"unknown direct-IP fragment flags: 0x{unknown_flags:08x}")
    if not 1 <= fragment.frame_bytes <= MAX_IP_FRAME_BYTES:
        raise ProtocolError("direct-IP frame size is outside the supported range")
    if not 1 <= fragment.fragment_count <= MAX_IP_FRAGMENT_COUNT:
        raise ProtocolError("direct-IP fragment count is outside the supported range")
    if fragment.fragment_index >= fragment.fragment_count:
        raise ProtocolError("direct-IP fragment index is outside the frame")
    if not fragment.payload:
        raise ProtocolError("direct-IP fragments cannot have empty payloads")
    if fragment.fragment_offset + len(fragment.payload) > fragment.frame_bytes:
        raise ProtocolError("direct-IP fragment extends past the frame")
    expected_first = fragment.fragment_index == 0
    expected_last = fragment.fragment_index == fragment.fragment_count - 1
    if bool(fragment.flags & IpFragmentFlags.FIRST) != expected_first:
        raise ProtocolError("direct-IP FIRST flag disagrees with fragment index")
    if bool(fragment.flags & IpFragmentFlags.LAST) != expected_last:
        raise ProtocolError("direct-IP LAST flag disagrees with fragment index")


def _validate_control_message(
    message: IpControlMessageV1, *, strict_flags: bool = True
) -> None:
    _validate_uint("request_id", message.request_id, 64)
    if not -(1 << 31) <= message.status <= (1 << 31) - 1:
        raise ProtocolError("direct-IP control status is outside int32")
    unknown_flags = int(message.flags) & ~int(KNOWN_IP_CONTROL_FLAGS)
    if unknown_flags and (
        strict_flags or message.message_type != IpControlType.CAPABILITIES
    ):
        # Conservative in what we send, liberal in what we accept.  An
        # unrecognised capability bit from a newer gadget must not fail the
        # host's open(): that is precisely how a firmware upgrade would break
        # every already-deployed client.  Requests stay strict, so we never
        # transmit a bit the peer cannot have agreed to.
        raise ProtocolError(f"unknown direct-IP control flags: 0x{unknown_flags:08x}")
    unknown_features = int(message.features) & ~int(KNOWN_FEATURES)
    if unknown_features:
        raise ProtocolError(
            f"unknown direct-IP metadata features: 0x{unknown_features:016x}"
        )
    for name, value, bits in (
        ("flags", int(message.flags), 32),
        ("protocol_min", message.protocol_min, 16),
        ("protocol_max", message.protocol_max, 16),
        ("features", int(message.features), 64),
        ("max_samples_per_channel", message.max_samples_per_channel, 32),
        ("max_finite_frames", message.max_finite_frames, 32),
        ("enabled_scan_mask", message.enabled_scan_mask, 32),
        ("samples_per_channel", message.samples_per_channel, 32),
        ("frame_count", message.frame_count, 32),
        (
            "gain_observation_interval_samples",
            message.gain_observation_interval_samples,
            32,
        ),
        ("gain_observation_capacity", message.gain_observation_capacity, 16),
        ("gain_event_capacity", message.gain_event_capacity, 16),
        ("data_port", message.data_port, 16),
        ("max_datagram_bytes", message.max_datagram_bytes, 16),
        ("stream_id", message.stream_id, 64),
    ):
        _validate_uint(name, value, bits)

    message_type = message.message_type
    if message_type != IpControlType.ERROR and message.status != 0:
        raise ProtocolError("successful direct-IP control message has non-zero status")
    if message_type == IpControlType.ERROR:
        if message.status == 0:
            raise ProtocolError("direct-IP ERROR message requires non-zero status")
        return

    zero_except_flags = (
        message.protocol_min,
        message.protocol_max,
        int(message.features),
        message.max_samples_per_channel,
        message.max_finite_frames,
        message.enabled_scan_mask,
        message.samples_per_channel,
        message.frame_count,
        message.gain_observation_interval_samples,
        message.gain_observation_capacity,
        message.gain_event_capacity,
        message.data_port,
        message.max_datagram_bytes,
        message.stream_id,
    )
    if message_type == IpControlType.QUERY_CAPABILITIES:
        if int(message.flags) & ~int(QUERY_PROBE_FLAGS) or any(zero_except_flags):
            raise ProtocolError("direct-IP capability query has non-zero fields")
        return

    if message_type == IpControlType.CAPABILITIES:
        if not message.flags & IpControlFlags.FINITE_RX:
            raise ProtocolError("direct-IP gadget does not advertise finite RX")
        if message.flags & IpControlFlags.QUERY_TRANSPORT_CAPABILITIES:
            raise ProtocolError("direct-IP capability response retained query flags")
        if (
            message.flags & IpControlFlags.USB_CLASS_PACING
            and not message.flags & IpControlFlags.BUFFERED_FINITE_RX
        ):
            raise ProtocolError("direct-IP pacing requires buffered finite RX")
        if (
            message.flags & IpControlFlags.DROP_STALE_FRAMES
            and not message.flags & IpControlFlags.TCP_DATA_TRANSPORT
        ):
            raise ProtocolError("direct-IP frame drop requires the TCP transport")
        if not (
            VERSION_V1 <= message.protocol_min <= message.protocol_max <= VERSION_V3
        ):
            raise ProtocolError("direct-IP capability protocol range is invalid")
        if not 1 <= message.max_samples_per_channel <= MAX_SAMPLES_PER_CHANNEL:
            raise ProtocolError("direct-IP maximum sample count is invalid")
        if not 1 <= message.max_finite_frames <= MAX_FINITE_FRAMES:
            raise ProtocolError("direct-IP maximum frame count is invalid")
        irrelevant = (
            message.enabled_scan_mask,
            message.samples_per_channel,
            message.frame_count,
            message.gain_observation_interval_samples,
            message.gain_observation_capacity,
            message.gain_event_capacity,
            message.data_port,
            message.max_datagram_bytes,
            message.stream_id,
        )
        if any(irrelevant):
            raise ProtocolError("direct-IP capability response has request fields")
        return

    if message_type in (IpControlType.START_RX, IpControlType.STARTED):
        if int(message.flags) & ~int(START_TRANSPORT_FLAGS):
            raise ProtocolError("direct-IP START has capability-only flags")
        if (
            message.flags & IpControlFlags.USB_CLASS_PACING
            and not message.flags & IpControlFlags.BUFFERED_FINITE_RX
        ):
            raise ProtocolError("direct-IP pacing requires buffered finite RX")
        if (
            message.flags & IpControlFlags.DROP_STALE_FRAMES
            and not message.flags & IpControlFlags.TCP_DATA_TRANSPORT
        ):
            raise ProtocolError("direct-IP frame drop requires the TCP transport")
        if (
            message.protocol_min != message.protocol_max
            or message.protocol_min
            not in (
                VERSION_V1,
                VERSION_V2,
                VERSION_V3,
            )
        ):
            raise ProtocolError("direct-IP START protocol selection is invalid")
        if message.max_samples_per_channel or message.max_finite_frames:
            raise ProtocolError("direct-IP START has capability-only limits")
        if message.enabled_scan_mask != 0x0F:
            raise ProtocolError("direct-IP START requires the dual-RX scan mask")
        if not 1 <= message.samples_per_channel <= MAX_SAMPLES_PER_CHANNEL:
            raise ProtocolError("direct-IP START sample count is invalid")
        if not 1 <= message.frame_count <= MAX_FINITE_FRAMES:
            raise ProtocolError("direct-IP START frame count is invalid")
        if not 1 <= message.data_port <= 0xFFFF:
            raise ProtocolError("direct-IP START data port is invalid")
        if not (
            IP_FRAGMENT_HEADER_BYTES
            < message.max_datagram_bytes
            <= MAX_UDP_DATAGRAM_BYTES
        ):
            raise ProtocolError("direct-IP START datagram size is invalid")
        if message.protocol_min == VERSION_V3:
            required = (
                MetadataFeatures.GAIN_OBSERVATION_SERIES
                | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
            )
            if message.features & required != required:
                raise ProtocolError("direct-IP v3 START lacks gain-series features")
            if not (
                1
                <= message.gain_observation_interval_samples
                <= message.samples_per_channel
            ):
                raise ProtocolError("direct-IP v3 observation interval is invalid")
            if not (1 <= message.gain_observation_capacity <= MAX_GAIN_OBSERVATIONS):
                raise ProtocolError("direct-IP v3 observation capacity is invalid")
            if not 0 <= message.gain_event_capacity <= MAX_GAIN_EVENTS:
                raise ProtocolError("direct-IP v3 event capacity is invalid")
        elif (
            message.gain_observation_interval_samples
            or message.gain_observation_capacity
            or message.gain_event_capacity
        ):
            raise ProtocolError("direct-IP v1/v2 START has v3-only fields")
        if message_type == IpControlType.START_RX and message.stream_id:
            raise ProtocolError("direct-IP START request cannot assign a stream ID")
        if message_type == IpControlType.STARTED and not message.stream_id:
            raise ProtocolError("direct-IP STARTED response requires a stream ID")
        return

    if message_type in (IpControlType.STOP_RX, IpControlType.STOPPED):
        if not message.stream_id:
            raise ProtocolError("direct-IP STOP requires a stream ID")
        if message.flags or any(zero_except_flags[:-1]):
            raise ProtocolError("direct-IP STOP has unrelated fields")
        return

    raise ProtocolError(f"unsupported direct-IP control message: {message_type}")


def _validate_uint(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"{name} must be an integer")
    if not 0 <= value <= (1 << bits) - 1:
        raise ProtocolError(f"{name} is outside uint{bits}: {value}")
