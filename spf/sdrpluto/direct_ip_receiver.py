"""Finite-frame client for the SPF direct-IP radio transport."""

from __future__ import annotations

import dataclasses
import secrets
import socket
import time
from contextlib import suppress

from spf.sdrpluto.direct_ip_protocol import (
    DEFAULT_UDP_DATAGRAM_BYTES,
    IpControlMessageV1,
    IpControlType,
    IpFrameReassembler,
    make_ip_capability_query,
    make_ip_start_request,
    make_ip_stop_request,
)
from spf.sdrpluto.direct_usb_protocol import (
    VERSION_V2,
    VERSION_V3,
    DirectUsbRxFrame,
    MetadataFeatures,
    ProtocolError,
    RxFrameParser,
)


DEFAULT_DIRECT_IP_CONTROL_PORT = 30_432
DEFAULT_CONTROL_TIMEOUT_SECONDS = 0.5
DEFAULT_FRAME_TIMEOUT_SECONDS = 10.0
DEFAULT_CONTROL_ATTEMPTS = 3
DEFAULT_DATA_RECEIVE_BUFFER_BYTES = 16 * 1024 * 1024
MAX_CONTROL_DATAGRAM_BYTES = 4096


class DirectIpTransportError(RuntimeError):
    """The direct-IP socket/control layer failed below radio framing."""


@dataclasses.dataclass(frozen=True, slots=True)
class DirectIpCapture:
    remote_host: str
    remote_control_port: int
    capabilities: IpControlMessageV1
    frames: tuple[DirectUsbRxFrame, ...]
    elapsed_seconds: float
    duplicate_fragment_count: int
    expired_frame_count: int
    rejected_frame_count: int


class PlutoDirectIpReceiver:
    """Receive bounded radio frames using acknowledged UDP control.

    PyADI/libiio remains responsible for configuring the selected radio before
    this object starts streaming.  The gadget returns the exact same inner
    frame used by direct USB; only the outer UDP fragmentation differs.
    """

    def __init__(
        self,
        *,
        remote_host: str,
        remote_control_port: int = DEFAULT_DIRECT_IP_CONTROL_PORT,
        local_host: str = "0.0.0.0",
        protocol_version: int = VERSION_V3,
        max_datagram_bytes: int = DEFAULT_UDP_DATAGRAM_BYTES,
        gain_observation_interval_samples: int = 32_768,
        gain_observation_capacity: int = 32,
        gain_event_capacity: int = 0,
        data_receive_buffer_bytes: int = DEFAULT_DATA_RECEIVE_BUFFER_BYTES,
        control_timeout_seconds: float = DEFAULT_CONTROL_TIMEOUT_SECONDS,
        frame_timeout_seconds: float = DEFAULT_FRAME_TIMEOUT_SECONDS,
        control_attempts: int = DEFAULT_CONTROL_ATTEMPTS,
    ) -> None:
        if not remote_host:
            raise ValueError("remote_host is required")
        if not 1 <= remote_control_port <= 0xFFFF:
            raise ValueError("remote control port is invalid")
        if protocol_version not in (VERSION_V2, VERSION_V3):
            raise ValueError("direct IP supports SPF protocol v2 or v3")
        if control_timeout_seconds <= 0 or frame_timeout_seconds <= 0:
            raise ValueError("direct-IP timeouts must be positive")
        if control_attempts <= 0:
            raise ValueError("direct-IP control attempts must be positive")
        if data_receive_buffer_bytes <= 0:
            raise ValueError("direct-IP receive buffer must be positive")
        self.remote_host = remote_host
        self.remote_control_port = int(remote_control_port)
        self.local_host = local_host
        self.protocol_version = int(protocol_version)
        self.max_datagram_bytes = int(max_datagram_bytes)
        self.gain_observation_interval_samples = int(gain_observation_interval_samples)
        self.gain_observation_capacity = int(gain_observation_capacity)
        self.gain_event_capacity = int(gain_event_capacity)
        self.data_receive_buffer_bytes = int(data_receive_buffer_bytes)
        self.control_timeout_seconds = float(control_timeout_seconds)
        self.frame_timeout_seconds = float(frame_timeout_seconds)
        self.control_attempts = int(control_attempts)
        self._control_socket: socket.socket | None = None
        self._data_socket: socket.socket | None = None
        self._remote_ip: str | None = None
        self._capabilities: IpControlMessageV1 | None = None
        self._next_request_id = secrets.randbits(64) or 1

    @property
    def capabilities(self) -> IpControlMessageV1:
        if self._capabilities is None:
            raise RuntimeError("direct-IP receiver is not open")
        return self._capabilities

    @property
    def local_data_port(self) -> int:
        if self._data_socket is None:
            raise RuntimeError("direct-IP receiver is not open")
        return int(self._data_socket.getsockname()[1])

    def open(self) -> None:
        if self._control_socket is not None or self._data_socket is not None:
            raise RuntimeError("direct-IP receiver is already open")
        addresses = socket.getaddrinfo(
            self.remote_host,
            self.remote_control_port,
            family=socket.AF_INET,
            type=socket.SOCK_DGRAM,
        )
        if not addresses:
            raise DirectIpTransportError("direct-IP host did not resolve")
        remote = addresses[0][4]
        control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            control.connect(remote)
            control.settimeout(self.control_timeout_seconds)
            data.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_RCVBUF,
                self.data_receive_buffer_bytes,
            )
            data.bind((self.local_host, 0))
            data.settimeout(min(0.25, self.frame_timeout_seconds))
            self._control_socket = control
            self._data_socket = data
            self._remote_ip = remote[0]
            query = make_ip_capability_query(request_id=self._request_id())
            capabilities = self._exchange_control(
                query, expected_type=IpControlType.CAPABILITIES
            )
            self._capabilities = capabilities
        except Exception:
            control.close()
            data.close()
            self._control_socket = None
            self._data_socket = None
            self._remote_ip = None
            self._capabilities = None
            raise

    def close(self) -> None:
        if self._control_socket is not None:
            self._control_socket.close()
        if self._data_socket is not None:
            self._data_socket.close()
        self._control_socket = None
        self._data_socket = None
        self._remote_ip = None
        self._capabilities = None

    def capture(self, *, samples_per_channel: int, frame_count: int) -> DirectIpCapture:
        capabilities = self.capabilities
        self._require_control_socket()
        data = self._require_data_socket()
        if not (
            capabilities.protocol_min
            <= self.protocol_version
            <= capabilities.protocol_max
        ):
            raise ProtocolError(
                f"gadget does not support requested protocol v{self.protocol_version}"
            )
        if not 1 <= samples_per_channel <= capabilities.max_samples_per_channel:
            raise ProtocolError("direct-IP sample request exceeds capability")
        if not 1 <= frame_count <= capabilities.max_finite_frames:
            raise ProtocolError("direct-IP frame request exceeds capability")
        features = self._required_features()
        if capabilities.features & features != features:
            raise ProtocolError("direct-IP gadget is missing required features")
        start_request = make_ip_start_request(
            request_id=self._request_id(),
            protocol_version=self.protocol_version,
            features=features,
            enabled_scan_mask=0x0F,
            samples_per_channel=samples_per_channel,
            frame_count=frame_count,
            gain_observation_interval_samples=(
                min(
                    self.gain_observation_interval_samples,
                    samples_per_channel,
                )
                if self.protocol_version == VERSION_V3
                else 0
            ),
            gain_observation_capacity=(
                self.gain_observation_capacity
                if self.protocol_version == VERSION_V3
                else 0
            ),
            gain_event_capacity=(
                self.gain_event_capacity if self.protocol_version == VERSION_V3 else 0
            ),
            data_port=self.local_data_port,
            max_datagram_bytes=self.max_datagram_bytes,
        )
        started = self._exchange_control(
            start_request, expected_type=IpControlType.STARTED
        )
        parser = RxFrameParser(protocol_version=self.protocol_version)
        reassembler = IpFrameReassembler(
            frame_timeout_seconds=self.frame_timeout_seconds
        )
        frames: list[DirectUsbRxFrame] = []
        start_time = time.monotonic()
        deadline = start_time + self.frame_timeout_seconds
        capture_error: Exception | None = None
        try:
            expected_started = dataclasses.replace(
                start_request,
                message_type=IpControlType.STARTED,
                stream_id=started.stream_id,
            )
            if started != expected_started:
                raise ProtocolError("direct-IP STARTED response does not echo request")
            while len(frames) < frame_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reassembler.expire(now=time.monotonic())
                    raise DirectIpTransportError(
                        "direct-IP finite RX timed out: "
                        f"received {len(frames)}/{frame_count} frames, "
                        f"pending={reassembler.pending_frame_count}"
                    )
                data.settimeout(min(0.25, remaining))
                try:
                    datagram, peer = data.recvfrom(self.max_datagram_bytes)
                except TimeoutError:
                    continue
                if self._remote_ip is not None and peer[0] != self._remote_ip:
                    raise ProtocolError(
                        f"direct-IP data arrived from unexpected peer {peer[0]}"
                    )
                for outer in reassembler.feed(datagram, peer=peer[0]):
                    inner_frames = parser.feed(outer.frame)
                    if len(inner_frames) != 1:
                        raise ProtocolError(
                            "one direct-IP outer frame must contain one inner frame"
                        )
                    inner = inner_frames[0]
                    if inner.metadata.stream_id != outer.stream_id:
                        raise ProtocolError("direct-IP inner/outer stream mismatch")
                    if inner.metadata.buffer_sequence != outer.frame_sequence:
                        raise ProtocolError("direct-IP inner/outer sequence mismatch")
                    if outer.stream_id != started.stream_id:
                        raise ProtocolError(
                            "direct-IP frame has unnegotiated stream ID"
                        )
                    frames.append(inner)
                    if len(frames) > frame_count:
                        raise ProtocolError("direct-IP gadget returned extra frames")
            parser.finish()
            if reassembler.pending_frame_count:
                raise ProtocolError("direct-IP capture ended with partial frames")
        except Exception as error:
            capture_error = error
            raise
        finally:
            stop = make_ip_stop_request(
                request_id=self._request_id(), stream_id=started.stream_id
            )
            try:
                stopped = self._exchange_control(
                    stop, expected_type=IpControlType.STOPPED
                )
                if stopped != dataclasses.replace(
                    stop, message_type=IpControlType.STOPPED
                ):
                    raise ProtocolError(
                        "direct-IP STOPPED response does not echo request"
                    )
            except Exception:
                if capture_error is None:
                    raise

        return DirectIpCapture(
            remote_host=self.remote_host,
            remote_control_port=self.remote_control_port,
            capabilities=capabilities,
            frames=tuple(frames),
            elapsed_seconds=time.monotonic() - start_time,
            duplicate_fragment_count=reassembler.duplicate_fragment_count,
            expired_frame_count=reassembler.expired_frame_count,
            rejected_frame_count=reassembler.rejected_frame_count,
        )

    def _required_features(self) -> MetadataFeatures:
        features = (
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
            | MetadataFeatures.GAIN_DB_ENDPOINTS
            | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
        )
        if self.protocol_version == VERSION_V3:
            features |= (
                MetadataFeatures.GAIN_OBSERVATION_SERIES
                | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
            )
        return features

    def _request_id(self) -> int:
        value = self._next_request_id
        self._next_request_id = (value + 1) & 0xFFFFFFFFFFFFFFFF
        if self._next_request_id == 0:
            self._next_request_id = 1
        return value

    def _exchange_control(
        self,
        request: IpControlMessageV1,
        *,
        expected_type: IpControlType,
    ) -> IpControlMessageV1:
        control = self._require_control_socket()
        payload = request.pack()
        last_error: Exception | None = None
        for _attempt in range(self.control_attempts):
            control.send(payload)
            try:
                response_payload = control.recv(MAX_CONTROL_DATAGRAM_BYTES)
            except TimeoutError as error:
                last_error = error
                continue
            response = IpControlMessageV1.unpack(response_payload)
            if response.request_id != request.request_id:
                last_error = ProtocolError(
                    "direct-IP response request ID does not match"
                )
                continue
            if response.message_type == IpControlType.ERROR:
                raise DirectIpTransportError(
                    f"direct-IP gadget rejected control request: {response.status}"
                )
            if response.message_type != expected_type:
                last_error = ProtocolError(
                    "unexpected direct-IP control response: "
                    f"{response.message_type.name}"
                )
                continue
            return response
        raise DirectIpTransportError(
            f"direct-IP {request.message_type.name} was not acknowledged after "
            f"{self.control_attempts} attempts"
        ) from last_error

    def _require_control_socket(self) -> socket.socket:
        if self._control_socket is None:
            raise RuntimeError("direct-IP receiver is not open")
        return self._control_socket

    def _require_data_socket(self) -> socket.socket:
        if self._data_socket is None:
            raise RuntimeError("direct-IP receiver is not open")
        return self._data_socket

    def __enter__(self) -> "PlutoDirectIpReceiver":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        with suppress(Exception):
            self.close()
