"""Strict finite-frame client for the SPF Pluto direct-USB gadget."""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import logging
import time
from collections.abc import Callable, Generator, Iterable, Iterator
from typing import Any

import numpy as np
import usb1

from spf.sdrpluto.direct_usb_protocol import (
    CAPABILITIES_BYTES,
    COMMAND_GET_CAPABILITIES,
    COMMAND_GET_HARDWARE_IDENTITY,
    COMMAND_GET_STATUS,
    COMMAND_GET_TIME_ANCHOR,
    COMMAND_START_RX_V1,
    COMMAND_STOP,
    COMMAND_TARGET_RX,
    HEADER_BYTES,
    HEADER_BYTES_V2,
    HEADER_PREFIX_BYTES_V3,
    GAIN_EVENT_BYTES,
    GAIN_OBSERVATION_BYTES,
    HARDWARE_IDENTITY_BYTES,
    RUNTIME_STATUS_BYTES,
    TIME_ANCHOR_BYTES,
    VERSION_V1,
    VERSION_V2,
    VERSION_V3,
    CapabilityFlags,
    DirectUsbRxFrame,
    GadgetCapabilitiesV1,
    HardwareIdentityV1,
    HardwareIdentityFlags,
    RuntimeStatusFlags,
    RuntimeStatusV1,
    TimeAnchorV1,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    RxFrameParser,
    pack_start_request_v1,
    pack_start_request_v2,
    pack_start_request_v3,
)


PLUTO_VENDOR_ID = 0x0456
PLUTO_PRODUCT_ID = 0xB673
USB_CLASS_VENDOR_SPECIFIC = 0xFF
USB_TRANSFER_TYPE_MASK = 0x03
USB_CONTROL_TIMEOUT_MS = 1_000
USB_BULK_TIMEOUT_MS = 10_000
DEFAULT_BULK_CHUNK_BYTES = 1024 * 1024
DEFAULT_RECONNECT_ATTEMPTS = 20
DEFAULT_RECONNECT_DELAY_SECONDS = 0.25
ORPHAN_DRAIN_TIMEOUT_MS = 50
MAX_ORPHAN_DRAIN_TRANSFERS = 32
# The wire capability is deliberately UINT32-wide, not an allocation promise.
# Eight MiB covers Rover's 4,194,400-byte production frame while remaining one
# bounded transfer under the Pi's default 16 MiB usbfs budget.
MAX_ORPHAN_DRAIN_BYTES = 8 * 1024 * 1024


class DirectUsbNotFoundError(RuntimeError):
    """The selected direct-USB gadget is not currently enumerated."""


class DirectUsbTransportError(RuntimeError):
    """A claimed direct-USB connection failed below the framing layer."""


class DirectUsbTransferTimeoutError(DirectUsbTransportError):
    """A finite bulk request missed its host deadline without valid framing."""

    def __init__(
        self,
        *,
        pending_transfer_count: int,
        requested_transfer_count: int,
        timeout_ms: int,
        serial: str,
        port_path: tuple[int, ...],
    ) -> None:
        self.pending_transfer_count = int(pending_transfer_count)
        self.requested_transfer_count = int(requested_transfer_count)
        self.timeout_ms = int(timeout_ms)
        self.serial = serial
        self.port_path = tuple(port_path)
        super().__init__(
            "direct USB finite RX timed out "
            f"after {self.timeout_ms} ms with "
            f"{self.pending_transfer_count}/{self.requested_transfer_count} "
            "queued transfers pending for "
            f"serial={self.serial!r} port_path={self.port_path!r}"
        )


class DirectUsbStreamDiscontinuityError(RuntimeError):
    """A radio was re-attested, but its new stream needs a new artifact."""


class DirectUsbRecoveryError(RuntimeError):
    """Bounded recovery could not restore one valid finite RX request."""


@dataclasses.dataclass(frozen=True, slots=True)
class RecoveryAttestationDifference:
    """One fail-closed mismatch observed before a recovered stream START."""

    field: str
    expected: Any
    observed: Any


class DirectUsbRecoveryAttestationError(DirectUsbRecoveryError):
    """The re-enumerated device cannot be proven equivalent to the original."""

    def __init__(self, differences: Iterable[RecoveryAttestationDifference]):
        self.differences = tuple(differences)
        if not self.differences:
            raise ValueError("at least one recovery attestation difference is required")
        details = "; ".join(
            f"{difference.field}: expected={difference.expected!r}, "
            f"observed={difference.observed!r}"
            for difference in self.differences
        )
        super().__init__(f"direct USB recovery attestation failed: {details}")


@dataclasses.dataclass(frozen=True, slots=True)
class DirectUsbIdentity:
    serial: str
    bus: int
    address: int
    port_path: tuple[int, ...]
    interface: int
    bulk_in_endpoint: int
    bulk_out_endpoint: int


@dataclasses.dataclass(frozen=True, slots=True)
class DirectUsbCapture:
    identity: DirectUsbIdentity
    capabilities: GadgetCapabilitiesV1
    frames: tuple[DirectUsbRxFrame, ...]
    elapsed_seconds: float
    recovered_after_transport_loss: bool = False
    transport_loss_summary: str | None = None


def iq_payload_to_complex64(
    payload: bytes | bytearray | memoryview,
    samples_per_channel: int,
) -> np.ndarray:
    expected_bytes = samples_per_channel * 8
    if len(payload) != expected_bytes:
        raise ProtocolError(
            f"IQ payload size mismatch: got {len(payload)}, expected {expected_bytes}"
        )
    iq = np.frombuffer(payload, dtype="<i2").reshape(samples_per_channel, 4)
    signal_matrix = np.empty((2, samples_per_channel), dtype=np.complex64)
    signal_matrix[0] = iq[:, 0] + 1j * iq[:, 1]
    signal_matrix[1] = iq[:, 2] + 1j * iq[:, 3]
    return signal_matrix


class PlutoDirectUsbReceiver:
    """Own one custom gadget interface and make bounded RX requests.

    Bulk-IN transfers are queued before START and completed through libusb's
    event context. The finite request bounds both device work and host memory.
    """

    def __init__(
        self,
        *,
        serial: str | None = None,
        port_path: tuple[int, ...] | None = None,
        bulk_chunk_bytes: int = DEFAULT_BULK_CHUNK_BYTES,
        protocol_version: int = VERSION_V1,
        gain_observation_interval_samples: int = 32768,
        gain_observation_capacity: int = 32,
        gain_event_capacity: int = 0,
        reconnect_attempts: int = DEFAULT_RECONNECT_ATTEMPTS,
        reconnect_delay_seconds: float = DEFAULT_RECONNECT_DELAY_SECONDS,
        reconnect_attestor: Callable[[], None] | None = None,
    ) -> None:
        if not serial and not port_path:
            raise ValueError("serial or physical USB port_path is required")
        if bulk_chunk_bytes <= 0:
            raise ValueError("bulk_chunk_bytes must be positive")
        if protocol_version not in (VERSION_V1, VERSION_V2, VERSION_V3):
            raise ValueError(f"unsupported direct USB protocol {protocol_version}")
        if gain_observation_interval_samples <= 0:
            raise ValueError("gain observation interval must be positive")
        if gain_observation_capacity <= 0:
            raise ValueError("gain observation capacity must be positive")
        if gain_event_capacity < 0:
            raise ValueError("gain event capacity must be non-negative")
        if reconnect_attempts <= 0:
            raise ValueError("reconnect_attempts must be positive")
        if reconnect_delay_seconds < 0:
            raise ValueError("reconnect_delay_seconds must be non-negative")
        self.requested_serial = serial
        self.requested_port_path = port_path
        self.bulk_chunk_bytes = bulk_chunk_bytes
        self.protocol_version = protocol_version
        self.gain_observation_interval_samples = gain_observation_interval_samples
        self.gain_observation_capacity = gain_observation_capacity
        self.gain_event_capacity = gain_event_capacity
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.reconnect_attestor = reconnect_attestor
        self._context: usb1.USBContext | None = None
        self._handle: usb1.USBDeviceHandle | None = None
        self._identity: DirectUsbIdentity | None = None
        self._capabilities: GadgetCapabilitiesV1 | None = None
        self._recovery_hardware_identity: HardwareIdentityV1 | None = None
        self._recovery_runtime_status: RuntimeStatusV1 | None = None
        self._terminal_recovery_error: Exception | None = None
        self._detached_kernel_driver = False
        self._active_stream: Generator[DirectUsbRxFrame, None, None] | None = None
        self._next_time_anchor_request_id = 1

    @property
    def identity(self) -> DirectUsbIdentity:
        if self._identity is None:
            raise RuntimeError("direct USB receiver is not open")
        return self._identity

    @property
    def capabilities(self) -> GadgetCapabilitiesV1:
        if self._capabilities is None:
            raise RuntimeError("direct USB capabilities have not been queried")
        return self._capabilities

    def __enter__(self) -> "PlutoDirectUsbReceiver":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        if self._handle is not None:
            raise RuntimeError("direct USB receiver is already open")
        context = usb1.USBContext()
        context.open()
        try:
            for device in context.getDeviceIterator(skip_on_error=True):
                if (
                    device.getVendorID() != PLUTO_VENDOR_ID
                    or device.getProductID() != PLUTO_PRODUCT_ID
                ):
                    continue
                candidate_port_path = tuple(device.getPortNumberList())
                if (
                    self.requested_port_path is not None
                    and candidate_port_path != self.requested_port_path
                ):
                    continue
                try:
                    candidate_serial = device.getSerialNumber()
                except usb1.USBError:
                    continue
                if (
                    self.requested_serial is not None
                    and candidate_serial != self.requested_serial
                ):
                    continue

                for interface, bulk_in, bulk_out in _candidate_interfaces(device):
                    handle = device.open()
                    claimed = False
                    detached_kernel_driver = False
                    try:
                        if handle.kernelDriverActive(interface):
                            handle.detachKernelDriver(interface)
                            detached_kernel_driver = True
                        handle.claimInterface(interface)
                        claimed = True
                        payload = handle.controlRead(
                            usb1.ENDPOINT_IN
                            | usb1.TYPE_VENDOR
                            | usb1.RECIPIENT_INTERFACE,
                            COMMAND_GET_CAPABILITIES,
                            COMMAND_TARGET_RX,
                            interface,
                            CAPABILITIES_BYTES,
                            timeout=USB_CONTROL_TIMEOUT_MS,
                        )
                        capabilities = GadgetCapabilitiesV1.unpack(payload)
                        self._quiesce_rx_endpoint(
                            handle=handle,
                            interface=interface,
                            bulk_in_endpoint=bulk_in,
                            capabilities=capabilities,
                        )
                    except (usb1.USBError, ProtocolError, DirectUsbTransportError):
                        if claimed:
                            with contextlib.suppress(usb1.USBError):
                                handle.releaseInterface(interface)
                        if detached_kernel_driver:
                            with contextlib.suppress(usb1.USBError):
                                handle.attachKernelDriver(interface)
                        handle.close()
                        continue

                    self._context = context
                    self._handle = handle
                    self._capabilities = capabilities
                    self._detached_kernel_driver = detached_kernel_driver
                    self._identity = DirectUsbIdentity(
                        serial=candidate_serial,
                        bus=device.getBusNumber(),
                        address=device.getDeviceAddress(),
                        port_path=candidate_port_path,
                        interface=interface,
                        bulk_in_endpoint=bulk_in,
                        bulk_out_endpoint=bulk_out,
                    )
                    # Bus/address are transient. Pin both durable identities after
                    # the first successful open so a reconnect cannot claim a
                    # different Pluto that happens to enumerate first.
                    self.requested_serial = candidate_serial
                    self.requested_port_path = candidate_port_path
                    return
        except Exception:
            context.close()
            raise
        context.close()
        raise DirectUsbNotFoundError(
            "matching SPF direct-USB gadget was not found for "
            f"serial={self.requested_serial!r} "
            f"port_path={self.requested_port_path!r}"
        )

    @staticmethod
    def _quiesce_rx_endpoint(
        *,
        handle,
        interface: int,
        bulk_in_endpoint: int,
        capabilities: GadgetCapabilitiesV1,
    ) -> None:
        """Stop an orphaned producer and discard its bounded endpoint backlog.

        A host killed between START and STOP releases its interface, but a
        completed FunctionFS AIO write can remain queued on bulk-IN. The next
        process would otherwise splice that old transfer into its first frame.
        Interface claiming excludes a live competing host, so draining here
        can only discard data whose owner no longer exists.
        """

        handle.controlWrite(
            usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
            COMMAND_STOP,
            COMMAND_TARGET_RX,
            interface,
            b"",
            timeout=USB_CONTROL_TIMEOUT_MS,
        )
        maximum_frame_bytes = min(
            MAX_ORPHAN_DRAIN_BYTES,
            HEADER_BYTES_V2 + capabilities.max_samples_per_channel * 8,
        )
        drain_limit = min(
            MAX_ORPHAN_DRAIN_TRANSFERS,
            max(1, capabilities.max_finite_frames + 1),
        )
        drained = []
        try:
            for _attempt in range(drain_limit):
                try:
                    stale = handle.bulkRead(
                        bulk_in_endpoint,
                        maximum_frame_bytes,
                        timeout=ORPHAN_DRAIN_TIMEOUT_MS,
                    )
                except usb1.USBErrorTimeout:
                    if drained:
                        logging.warning(
                            "discarded %s orphaned direct-USB RX transfer(s), "
                            "%s bytes total before opening a new stream",
                            len(drained),
                            sum(drained),
                        )
                    return
                except usb1.USBErrorOverflow:
                    # The bounded buffer consumed/discarded the head of an
                    # unexpectedly large orphaned write. Continue until a
                    # timeout proves that its endpoint tail is empty.
                    drained.append(maximum_frame_bytes)
                    continue
                drained.append(len(stale))
        finally:
            with contextlib.suppress(usb1.USBError, AttributeError):
                handle.clearHalt(bulk_in_endpoint)
        raise DirectUsbTransportError(
            "direct USB bulk-IN remained non-empty after discarding "
            f"{drain_limit} orphaned transfers"
        )

    def close(self) -> None:
        active_stream = self._active_stream
        self._active_stream = None
        if active_stream is not None:
            active_stream.close()
        handle = self._handle
        identity = self._identity
        context = self._context
        self._handle = None
        self._identity = None
        self._capabilities = None
        self._context = None
        detached_kernel_driver = self._detached_kernel_driver
        self._detached_kernel_driver = False
        if handle is not None:
            if identity is not None:
                with contextlib.suppress(usb1.USBError):
                    handle.releaseInterface(identity.interface)
                if detached_kernel_driver:
                    with contextlib.suppress(usb1.USBError):
                        handle.attachKernelDriver(identity.interface)
            handle.close()
        if context is not None:
            context.close()

    def capture(
        self,
        *,
        samples_per_channel: int,
        frame_count: int = 1,
    ) -> DirectUsbCapture:
        if self._active_stream is not None:
            raise RuntimeError("a direct USB frame stream is already active")
        if self._terminal_recovery_error is not None:
            raise DirectUsbRecoveryError(
                "direct USB receiver is terminal after failed recovery; "
                "create a new receiver/capture"
            ) from self._terminal_recovery_error
        try:
            return self._capture_once(
                samples_per_channel=samples_per_channel,
                frame_count=frame_count,
            )
        except (usb1.USBError, DirectUsbTransportError) as error:
            failed_identity = self._identity
            logging.warning(
                "direct USB transport lost for serial=%s port_path=%s: %s; "
                "starting bounded rediscovery",
                self.requested_serial,
                self.requested_port_path,
                error,
            )
            try:
                self._recover_connection(error)
            except DirectUsbRecoveryError as recovery_error:
                self._mark_terminal_recovery_failure(recovery_error)
                raise
            try:
                self._attest_recovered_connection(failed_identity)
            except DirectUsbRecoveryAttestationError as attestation_error:
                self._mark_terminal_recovery_failure(attestation_error)
                raise
            try:
                capture = self._capture_once(
                    samples_per_channel=samples_per_channel,
                    frame_count=frame_count,
                )
            except ProtocolError as protocol_error:
                # A newly rediscovered device that returns invalid framing or
                # sequence state is not safe to reuse in this capture. The
                # parser can reject sequence zero before the explicit check
                # below, so latch both paths terminally.
                self._mark_terminal_recovery_failure(protocol_error)
                raise
            except (usb1.USBError, DirectUsbTransportError) as retry_error:
                recovery_error = DirectUsbRecoveryError(
                    "direct USB transport failed again after rediscovery for "
                    f"serial={self.requested_serial!r} "
                    f"port_path={self.requested_port_path!r}"
                )
                self._mark_terminal_recovery_failure(recovery_error)
                raise recovery_error from retry_error

            metadata = capture.frames[0].metadata
            if metadata.buffer_sequence != 0 or (
                metadata.flags & MetadataFlags.SAMPLE_SEQUENCE_VALID
                and metadata.first_sample_sequence != 0
            ):
                sequence_error = ProtocolError(
                    "recovered direct USB START did not begin a new stream epoch"
                )
                self._mark_terminal_recovery_failure(sequence_error)
                raise sequence_error
            logging.warning(
                "direct USB transport recovered for serial=%s port_path=%s "
                "address=%s->%s stream_id=%s; new stream epoch, phase "
                "continuity is not retained",
                capture.identity.serial,
                capture.identity.port_path,
                None if failed_identity is None else failed_identity.address,
                capture.identity.address,
                metadata.stream_id,
            )
            return dataclasses.replace(
                capture,
                recovered_after_transport_loss=True,
                transport_loss_summary=f"{type(error).__name__}: {error}",
            )

    def stream_frames(
        self,
        *,
        samples_per_channel: int,
        frame_count: int,
        queue_depth: int = 1,
    ) -> Generator[DirectUsbRxFrame, None, None]:
        """Yield a bounded frame group under one START/STOP lifecycle.

        Only ``queue_depth`` USB transfers are resident at once. A completed
        transfer is copied and immediately resubmitted for the next frame,
        avoiding both per-frame radio teardown and usbfs memory growth. A
        transport loss terminates the stream: callers must start a new capture
        artifact rather than append a restarted stream epoch.
        """

        if self._active_stream is not None:
            raise RuntimeError("a direct USB frame stream is already active")
        if self._terminal_recovery_error is not None:
            raise DirectUsbRecoveryError(
                "direct USB receiver is terminal after failed recovery; "
                "close it and construct a new receiver"
            ) from self._terminal_recovery_error
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        self._require_handle()
        self.identity
        self._prepare_rx_request(
            samples_per_channel=samples_per_channel,
            frame_count=frame_count,
        )

        def managed_stream() -> Generator[DirectUsbRxFrame, None, None]:
            try:
                yield from self._stream_frames_once(
                    samples_per_channel=samples_per_channel,
                    frame_count=frame_count,
                    queue_depth=queue_depth,
                )
            except (usb1.USBError, DirectUsbTransportError) as error:
                raise DirectUsbStreamDiscontinuityError(
                    "direct USB frame stream lost transport continuity; "
                    "start a new capture artifact"
                ) from error
            finally:
                self._active_stream = None

        stream = managed_stream()
        self._active_stream = stream
        return stream

    def _stream_frames_once(
        self,
        *,
        samples_per_channel: int,
        frame_count: int,
        queue_depth: int,
    ) -> Generator[DirectUsbRxFrame, None, None]:
        handle = self._require_handle()
        identity = self.identity
        request, frame_bytes = self._prepare_rx_request(
            samples_per_channel=samples_per_channel,
            frame_count=frame_count,
        )
        parser = RxFrameParser(protocol_version=self.protocol_version)
        chunks = None
        try:
            if self._context is None:
                chunks = self._stream_sync_for_test(
                    handle=handle,
                    identity=identity,
                    request=request,
                    frame_count=frame_count,
                    frame_bytes=frame_bytes,
                )
            else:
                chunks = self._stream_queued(
                    handle=handle,
                    identity=identity,
                    request=request,
                    frame_count=frame_count,
                    frame_bytes=frame_bytes,
                    queue_depth=queue_depth,
                )
            yielded = 0
            for chunk in chunks:
                if not chunk:
                    raise ProtocolError("zero-length direct USB bulk read")
                frames = parser.feed(chunk)
                if len(frames) != 1:
                    raise ProtocolError(
                        "one fixed direct USB transfer did not contain exactly one frame"
                    )
                yielded += 1
                yield frames[0]
            if yielded != frame_count:
                raise ProtocolError(
                    f"gadget returned {yielded} frames, expected {frame_count}"
                )
            parser.finish()
        finally:
            try:
                if chunks is not None:
                    close_chunks = getattr(chunks, "close", None)
                    if close_chunks is not None:
                        close_chunks()
            finally:
                with contextlib.suppress(usb1.USBError):
                    handle.controlWrite(
                        usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
                        COMMAND_STOP,
                        COMMAND_TARGET_RX,
                        identity.interface,
                        b"",
                        timeout=USB_CONTROL_TIMEOUT_MS,
                    )

    def _prepare_rx_request(
        self, *, samples_per_channel: int, frame_count: int
    ) -> tuple[bytes, int]:
        capabilities = self.capabilities
        if samples_per_channel <= 0:
            raise ProtocolError("sample request must be positive")
        if frame_count <= 0:
            raise ProtocolError("frame request must be positive")
        if samples_per_channel > capabilities.max_samples_per_channel:
            raise ProtocolError("sample request exceeds gadget capability")
        if frame_count > capabilities.max_finite_frames:
            raise ProtocolError("frame request exceeds gadget capability")
        if not (
            capabilities.protocol_min
            <= self.protocol_version
            <= capabilities.protocol_max
        ):
            raise ProtocolError(
                f"gadget does not support requested protocol v{self.protocol_version}"
            )

        features = (
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        )
        if self.protocol_version in (VERSION_V2, VERSION_V3):
            features |= (
                MetadataFeatures.GAIN_DB_ENDPOINTS
                | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
            )
        if self.protocol_version == VERSION_V3:
            features |= (
                MetadataFeatures.GAIN_OBSERVATION_SERIES
                | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
            )
        if features & capabilities.supported_features != features:
            raise ProtocolError("gadget is missing required metadata features")
        if self.protocol_version == VERSION_V1:
            request = pack_start_request_v1(
                requested_features=features,
                enabled_scan_mask=0x0F,
                samples_per_channel=samples_per_channel,
                frame_count=frame_count,
            )
            header_bytes = HEADER_BYTES
        elif self.protocol_version == VERSION_V2:
            request = pack_start_request_v2(
                requested_features=features,
                enabled_scan_mask=0x0F,
                samples_per_channel=samples_per_channel,
                frame_count=frame_count,
            )
            header_bytes = HEADER_BYTES_V2
        else:
            request = pack_start_request_v3(
                requested_features=features,
                enabled_scan_mask=0x0F,
                samples_per_channel=samples_per_channel,
                frame_count=frame_count,
                gain_observation_interval_samples=(
                    min(
                        self.gain_observation_interval_samples,
                        samples_per_channel,
                    )
                ),
                gain_observation_capacity=self.gain_observation_capacity,
                gain_event_capacity=self.gain_event_capacity,
            )
            header_bytes = (
                HEADER_PREFIX_BYTES_V3
                + self.gain_observation_capacity * GAIN_OBSERVATION_BYTES
                + self.gain_event_capacity * GAIN_EVENT_BYTES
                + 4
            )
        return request, header_bytes + samples_per_channel * 8

    def _mark_terminal_recovery_failure(self, error: Exception) -> None:
        self._terminal_recovery_error = error
        with contextlib.suppress(usb1.USBError, AttributeError):
            self.close()

    def _capture_once(
        self,
        *,
        samples_per_channel: int,
        frame_count: int,
    ) -> DirectUsbCapture:
        handle = self._require_handle()
        identity = self.identity
        capabilities = self.capabilities
        request, frame_bytes = self._prepare_rx_request(
            samples_per_channel=samples_per_channel,
            frame_count=frame_count,
        )
        parser = RxFrameParser(protocol_version=self.protocol_version)
        bulk_read_bytes = max(self.bulk_chunk_bytes, frame_bytes)
        start = time.monotonic()
        try:
            if self._context is None:
                chunks = self._capture_sync_for_test(
                    handle=handle,
                    identity=identity,
                    request=request,
                    frame_count=frame_count,
                    bulk_read_bytes=bulk_read_bytes,
                )
            else:
                chunks = self._capture_queued(
                    handle=handle,
                    identity=identity,
                    request=request,
                    frame_count=frame_count,
                    frame_bytes=frame_bytes,
                )
            frames: list[DirectUsbRxFrame] = []
            for chunk in chunks:
                if not chunk:
                    raise ProtocolError("zero-length direct USB bulk read")
                frames.extend(parser.feed(chunk))
                if len(frames) > frame_count:
                    raise ProtocolError("gadget returned more frames than requested")
            if len(frames) != frame_count:
                raise ProtocolError(
                    f"gadget returned {len(frames)} frames, expected {frame_count}"
                )
            parser.finish()
        finally:
            with contextlib.suppress(usb1.USBError):
                handle.controlWrite(
                    usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
                    COMMAND_STOP,
                    COMMAND_TARGET_RX,
                    identity.interface,
                    b"",
                    timeout=USB_CONTROL_TIMEOUT_MS,
                )
        return DirectUsbCapture(
            identity=identity,
            capabilities=capabilities,
            frames=tuple(frames),
            elapsed_seconds=time.monotonic() - start,
        )

    def _stream_sync_for_test(
        self,
        *,
        handle,
        identity,
        request,
        frame_count,
        frame_bytes,
    ) -> Iterator[bytes]:
        handle.controlWrite(
            usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
            COMMAND_START_RX_V1,
            COMMAND_TARGET_RX,
            identity.interface,
            request,
            timeout=USB_CONTROL_TIMEOUT_MS,
        )
        for _index in range(frame_count):
            yield bytes(
                handle.bulkRead(
                    identity.bulk_in_endpoint,
                    frame_bytes,
                    timeout=USB_BULK_TIMEOUT_MS,
                )
            )

    def _recover_connection(self, original_error: Exception) -> None:
        with contextlib.suppress(usb1.USBError, AttributeError):
            self.close()

        last_error: Exception = original_error
        for attempt in range(1, self.reconnect_attempts + 1):
            if self.reconnect_delay_seconds:
                time.sleep(self.reconnect_delay_seconds)
            try:
                self.open()
            except (DirectUsbNotFoundError, usb1.USBError) as error:
                last_error = error
                logging.warning(
                    "direct USB rediscovery %s/%s failed for serial=%s "
                    "port_path=%s: %s",
                    attempt,
                    self.reconnect_attempts,
                    self.requested_serial,
                    self.requested_port_path,
                    error,
                )
                continue
            return

        raise DirectUsbRecoveryError(
            "direct USB gadget did not reappear after "
            f"{self.reconnect_attempts} bounded attempts for "
            f"serial={self.requested_serial!r} "
            f"port_path={self.requested_port_path!r}"
        ) from last_error

    def pin_recovery_hardware_identity(self) -> HardwareIdentityV1 | None:
        """Pin valid passive identity fields for subsequent reconnects.

        Older gadgets do not expose this capability. In that case USB serial
        and physical port remain the durable identity boundary.
        """

        if self.capabilities.capability_flags & CapabilityFlags.STATUS:
            self._recovery_runtime_status = self.query_runtime_status()
        else:
            self._recovery_runtime_status = None

        if not (self.capabilities.capability_flags & CapabilityFlags.HARDWARE_IDENTITY):
            self._recovery_hardware_identity = None
            return None
        identity = self.query_hardware_identity()
        self._recovery_hardware_identity = identity
        return identity

    def _attest_recovered_connection(
        self, failed_identity: DirectUsbIdentity | None
    ) -> None:
        differences: list[RecoveryAttestationDifference] = []
        recovered_identity = self.identity
        if failed_identity is None:
            differences.append(
                RecoveryAttestationDifference(
                    field="usb_identity",
                    expected="identity pinned before transport loss",
                    observed=None,
                )
            )
        else:
            if recovered_identity.serial != failed_identity.serial:
                differences.append(
                    RecoveryAttestationDifference(
                        field="usb_serial",
                        expected=failed_identity.serial,
                        observed=recovered_identity.serial,
                    )
                )
            if recovered_identity.port_path != failed_identity.port_path:
                differences.append(
                    RecoveryAttestationDifference(
                        field="usb_port_path",
                        expected=failed_identity.port_path,
                        observed=recovered_identity.port_path,
                    )
                )

        expected_hardware = self._recovery_hardware_identity
        if expected_hardware is not None:
            try:
                observed_hardware = self.query_hardware_identity()
            except Exception as error:
                differences.append(
                    RecoveryAttestationDifference(
                        field="hardware_identity",
                        expected="readable passive identity",
                        observed=f"{type(error).__name__}: {error}",
                    )
                )
            else:
                fields = (
                    (
                        HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID,
                        "fpga_device_dna",
                    ),
                    (
                        HardwareIdentityFlags.GADGET_BUILD_ID_VALID,
                        "gadget_build_id",
                    ),
                )
                for flag, field in fields:
                    if not expected_hardware.flags & flag:
                        continue
                    expected = getattr(expected_hardware, field)
                    observed = (
                        getattr(observed_hardware, field)
                        if observed_hardware.flags & flag
                        else None
                    )
                    if observed != expected:
                        differences.append(
                            RecoveryAttestationDifference(
                                field=field,
                                expected=expected,
                                observed=observed,
                            )
                        )

        expected_runtime = self._recovery_runtime_status
        if expected_runtime is not None:
            try:
                observed_runtime = self.query_runtime_status()
            except Exception as error:
                differences.append(
                    RecoveryAttestationDifference(
                        field="runtime_status",
                        expected="readable passive status",
                        observed=f"{type(error).__name__}: {error}",
                    )
                )
            else:
                runtime_fields = (
                    (RuntimeStatusFlags.BOOT_ID_VALID, "boot_id"),
                    (RuntimeStatusFlags.PROCESS_NONCE_VALID, "process_nonce"),
                )
                for flag, field in runtime_fields:
                    if not expected_runtime.flags & flag:
                        continue
                    expected = getattr(expected_runtime, field)
                    observed = (
                        getattr(observed_runtime, field)
                        if observed_runtime.flags & flag
                        else None
                    )
                    if observed != expected:
                        differences.append(
                            RecoveryAttestationDifference(
                                field=field,
                                expected=expected.hex(),
                                observed=(None if observed is None else observed.hex()),
                            )
                        )

        if differences:
            raise DirectUsbRecoveryAttestationError(iter(differences))
        if self.reconnect_attestor is None:
            raise DirectUsbRecoveryAttestationError(
                iter(
                    (
                        RecoveryAttestationDifference(
                            field="iio_rx_configuration",
                            expected="configured reconnect attestor",
                            observed=None,
                        ),
                    )
                )
            )
        try:
            self.reconnect_attestor()
        except DirectUsbRecoveryAttestationError:
            raise
        except Exception as error:
            raise DirectUsbRecoveryAttestationError(
                iter(
                    (
                        RecoveryAttestationDifference(
                            field="iio_rx_configuration",
                            expected="readable matching configuration",
                            observed=f"{type(error).__name__}: {error}",
                        ),
                    )
                )
            ) from error

    def query_hardware_identity(self) -> HardwareIdentityV1:
        """Read passive identity data without starting RX or TX."""

        handle = self._require_handle()
        identity = self.identity
        if not (self.capabilities.capability_flags & CapabilityFlags.HARDWARE_IDENTITY):
            raise ProtocolError(
                "gadget does not advertise passive hardware identity support"
            )
        payload = handle.controlRead(
            usb1.ENDPOINT_IN | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
            COMMAND_GET_HARDWARE_IDENTITY,
            COMMAND_TARGET_RX,
            identity.interface,
            HARDWARE_IDENTITY_BYTES,
            timeout=USB_CONTROL_TIMEOUT_MS,
        )
        return HardwareIdentityV1.unpack(payload)

    def query_runtime_status(self) -> RuntimeStatusV1:
        """Read volatile gadget health without starting RX or TX."""

        handle = self._require_handle()
        identity = self.identity
        if not (self.capabilities.capability_flags & CapabilityFlags.STATUS):
            raise ProtocolError("gadget does not advertise runtime status support")
        payload = handle.controlRead(
            usb1.ENDPOINT_IN | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
            COMMAND_GET_STATUS,
            COMMAND_TARGET_RX,
            identity.interface,
            RUNTIME_STATUS_BYTES,
            timeout=USB_CONTROL_TIMEOUT_MS,
        )
        return RuntimeStatusV1.unpack(payload)

    def query_time_anchor(self):
        """Bracket one coherent FPGA-counter observation on the host clock."""

        from spf.sdrpluto.sample_clock import HostTimeAnchorMeasurement

        handle = self._require_handle()
        identity = self.identity
        if not (self.capabilities.capability_flags & CapabilityFlags.TIME_ANCHOR):
            raise ProtocolError("gadget does not advertise time-anchor support")
        request_id = self._next_time_anchor_request_id
        self._next_time_anchor_request_id = (request_id + 1) & 0xFFFF
        if self._next_time_anchor_request_id == 0:
            self._next_time_anchor_request_id = 1
        host_before_ns = time.monotonic_ns()
        payload = handle.controlRead(
            usb1.ENDPOINT_IN | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
            COMMAND_GET_TIME_ANCHOR,
            request_id,
            identity.interface,
            TIME_ANCHOR_BYTES,
            timeout=USB_CONTROL_TIMEOUT_MS,
        )
        host_after_ns = time.monotonic_ns()
        anchor = TimeAnchorV1.unpack(payload)
        if anchor.request_id != request_id:
            raise ProtocolError(
                "time-anchor response request ID does not match USB request"
            )
        return HostTimeAnchorMeasurement(
            anchor=anchor,
            host_monotonic_before_ns=host_before_ns,
            host_monotonic_after_ns=host_after_ns,
            transport="direct_usb",
        )

    def _capture_sync_for_test(
        self,
        *,
        handle,
        identity,
        request,
        frame_count,
        bulk_read_bytes,
    ) -> list[bytes]:
        """Fallback used only by isolated tests without a USB event context."""

        handle.controlWrite(
            usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
            COMMAND_START_RX_V1,
            COMMAND_TARGET_RX,
            identity.interface,
            request,
            timeout=USB_CONTROL_TIMEOUT_MS,
        )
        return [
            bytes(
                handle.bulkRead(
                    identity.bulk_in_endpoint,
                    bulk_read_bytes,
                    timeout=USB_BULK_TIMEOUT_MS,
                )
            )
            for _ in range(frame_count)
        ]

    def _capture_queued(
        self,
        *,
        handle,
        identity,
        request,
        frame_count,
        frame_bytes,
    ) -> list[bytes]:
        """Queue bounded bulk-IN transfers before starting device DMA."""

        context = self._context
        if context is None:
            raise RuntimeError("direct USB event context is not open")

        chunks: list[bytes | None] = [None] * frame_count
        pending: set[int] = set()
        protocol_errors: list[str] = []
        transport_errors: list[str] = []
        timeout_error = None
        transfers = []

        def completed(transfer) -> None:
            index = transfer.getUserData()
            status = transfer.getStatus()
            if status == usb1.TRANSFER_COMPLETED:
                actual = transfer.getActualLength()
                if actual != frame_bytes:
                    protocol_errors.append(
                        f"bulk transfer {index} completed with {actual} bytes, "
                        f"expected {frame_bytes}"
                    )
                else:
                    chunks[index] = bytes(transfer.getBuffer()[:actual])
            else:
                transport_errors.append(
                    f"bulk transfer {index} failed with libusb status {status}"
                )
            pending.discard(index)

        try:
            for index in range(frame_count):
                # python-libusb1 3.x accepts ``short_is_error`` here, while
                # the Debian 12 wrapper (2.0.1) only accepts ``iso_packets``.
                # The callback below rejects every short transfer by checking
                # its exact byte count, so the portable no-argument form
                # preserves the same fail-closed behavior.
                transfer = handle.getTransfer()
                transfer.setBulk(
                    identity.bulk_in_endpoint,
                    frame_bytes,
                    callback=completed,
                    user_data=index,
                    timeout=USB_BULK_TIMEOUT_MS,
                )
                transfers.append(transfer)
                transfer.submit()
                pending.add(index)

            handle.controlWrite(
                usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
                COMMAND_START_RX_V1,
                COMMAND_TARGET_RX,
                identity.interface,
                request,
                timeout=USB_CONTROL_TIMEOUT_MS,
            )
            deadline = time.monotonic() + USB_BULK_TIMEOUT_MS / 1000.0 + 1.0
            while pending and not protocol_errors and not transport_errors:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timeout_error = DirectUsbTransferTimeoutError(
                        pending_transfer_count=len(pending),
                        requested_transfer_count=frame_count,
                        timeout_ms=USB_BULK_TIMEOUT_MS,
                        serial=identity.serial,
                        port_path=identity.port_path,
                    )
                    break
                context.handleEventsTimeout(min(0.1, remaining))
        finally:
            if pending:
                for index in tuple(pending):
                    with contextlib.suppress(usb1.USBError):
                        transfers[index].cancel()
                cancel_deadline = time.monotonic() + 1.0
                while pending and time.monotonic() < cancel_deadline:
                    with contextlib.suppress(usb1.USBError):
                        context.handleEventsTimeout(0.05)
            # USBTransfer retains its C transfer buffer and a callback reference
            # cycle until explicitly closed. A one-frame START per Rover snapshot
            # otherwise leaks roughly one full framed transfer on every rx().
            for transfer in transfers:
                if not transfer.isSubmitted():
                    transfer.close()
            # python-libusb1 constructs its receive ctypes array with
            # ``from_buffer(bytearray)``. Even after USBTransfer.close(), that
            # array and its managed memoryview form an unreachable cycle around
            # the multi-megabyte bytearray. It is not reference-counted away,
            # and the normal GC allocation threshold may not run during a
            # numeric capture loop. Collect generation zero here so completed
            # frame buffers are reclaimed deterministically.
            gc.collect(0)

        if transport_errors:
            raise DirectUsbTransportError("; ".join(transport_errors))
        if timeout_error is not None:
            raise timeout_error
        if protocol_errors:
            raise ProtocolError("; ".join(protocol_errors))
        if pending or any(chunk is None for chunk in chunks):
            raise ProtocolError("queued direct USB transfer cleanup was incomplete")
        return [chunk for chunk in chunks if chunk is not None]

    def _stream_queued(
        self,
        *,
        handle,
        identity,
        request,
        frame_count,
        frame_bytes,
        queue_depth,
    ) -> Iterator[bytes]:
        """Yield frames from a rolling, bounded asynchronous transfer queue."""

        context = self._context
        if context is None:
            raise RuntimeError("direct USB event context is not open")
        queue_depth = min(queue_depth, frame_count)
        ready: dict[int, bytes] = {}
        assignments: dict[int, int] = {}
        pending: set[int] = set()
        protocol_errors: list[str] = []
        transport_errors: list[str] = []
        transfers = []
        next_submit_index = queue_depth

        def make_completed(slot: int):
            def completed(transfer) -> None:
                nonlocal next_submit_index
                frame_index = assignments[slot]
                status = transfer.getStatus()
                pending.discard(slot)
                if status == usb1.TRANSFER_COMPLETED:
                    actual = transfer.getActualLength()
                    if actual != frame_bytes:
                        protocol_errors.append(
                            f"bulk transfer {frame_index} completed with {actual} "
                            f"bytes, expected {frame_bytes}"
                        )
                    else:
                        ready[frame_index] = bytes(transfer.getBuffer()[:actual])
                else:
                    transport_errors.append(
                        f"bulk transfer {frame_index} failed with libusb status "
                        f"{status}"
                    )

                if (
                    not protocol_errors
                    and not transport_errors
                    and next_submit_index < frame_count
                ):
                    assignments[slot] = next_submit_index
                    next_submit_index += 1
                    try:
                        transfer.submit()
                    except Exception as error:
                        transport_errors.append(
                            "failed to resubmit rolling bulk transfer "
                            f"{assignments[slot]}: {type(error).__name__}: {error}"
                        )
                    else:
                        pending.add(slot)

            return completed

        try:
            for slot in range(queue_depth):
                transfer = handle.getTransfer()
                transfer.setBulk(
                    identity.bulk_in_endpoint,
                    frame_bytes,
                    callback=make_completed(slot),
                    user_data=slot,
                    timeout=USB_BULK_TIMEOUT_MS,
                )
                transfers.append(transfer)
                assignments[slot] = slot
                transfer.submit()
                pending.add(slot)

            handle.controlWrite(
                usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_INTERFACE,
                COMMAND_START_RX_V1,
                COMMAND_TARGET_RX,
                identity.interface,
                request,
                timeout=USB_CONTROL_TIMEOUT_MS,
            )

            for frame_index in range(frame_count):
                deadline = time.monotonic() + USB_BULK_TIMEOUT_MS / 1000.0 + 1.0
                while (
                    frame_index not in ready
                    and not protocol_errors
                    and not transport_errors
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DirectUsbTransferTimeoutError(
                            pending_transfer_count=len(pending),
                            requested_transfer_count=frame_count,
                            timeout_ms=USB_BULK_TIMEOUT_MS,
                            serial=identity.serial,
                            port_path=identity.port_path,
                        )
                    context.handleEventsTimeout(min(0.1, remaining))
                if transport_errors:
                    raise DirectUsbTransportError("; ".join(transport_errors))
                if protocol_errors:
                    raise ProtocolError("; ".join(protocol_errors))
                yield ready.pop(frame_index)
        finally:
            if pending:
                for slot in tuple(pending):
                    with contextlib.suppress(usb1.USBError):
                        transfers[slot].cancel()
                cancel_deadline = time.monotonic() + 1.0
                while pending and time.monotonic() < cancel_deadline:
                    with contextlib.suppress(usb1.USBError):
                        context.handleEventsTimeout(0.05)
            for transfer in transfers:
                if not transfer.isSubmitted():
                    transfer.close()
            gc.collect(0)
            if pending:
                raise ProtocolError(
                    "rolling direct USB transfer cleanup was incomplete: "
                    f"{len(pending)} transfer(s) remain submitted"
                )

    def _require_handle(self) -> usb1.USBDeviceHandle:
        if self._handle is None:
            raise RuntimeError("direct USB receiver is not open")
        return self._handle


def _candidate_interfaces(
    device: usb1.USBDevice,
) -> Iterator[tuple[int, int, int]]:
    for setting in device.iterSettings():
        if setting.getClass() != USB_CLASS_VENDOR_SPECIFIC:
            continue
        bulk_in = None
        bulk_out = None
        for endpoint in setting.iterEndpoints():
            if (
                endpoint.getAttributes() & USB_TRANSFER_TYPE_MASK
            ) != usb1.TRANSFER_TYPE_BULK:
                continue
            address = endpoint.getAddress()
            if address & usb1.ENDPOINT_IN:
                bulk_in = address
            else:
                bulk_out = address
        if bulk_in is not None and bulk_out is not None:
            yield setting.getNumber(), bulk_in, bulk_out
