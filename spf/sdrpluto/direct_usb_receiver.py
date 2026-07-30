"""Strict finite-frame client for the SPF Pluto direct-USB gadget."""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import logging
import time
from collections.abc import Iterator

import numpy as np
import usb1

from spf.sdrpluto.direct_usb_protocol import (
    CAPABILITIES_BYTES,
    COMMAND_GET_CAPABILITIES,
    COMMAND_GET_HARDWARE_IDENTITY,
    COMMAND_START_RX_V1,
    COMMAND_STOP,
    COMMAND_TARGET_RX,
    HEADER_BYTES,
    HEADER_BYTES_V2,
    HARDWARE_IDENTITY_BYTES,
    VERSION_V1,
    VERSION_V2,
    CapabilityFlags,
    DirectUsbRxFrame,
    GadgetCapabilitiesV1,
    HardwareIdentityV1,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    RxFrameParser,
    pack_start_request_v1,
    pack_start_request_v2,
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


class DirectUsbNotFoundError(RuntimeError):
    """The selected direct-USB gadget is not currently enumerated."""


class DirectUsbTransportError(RuntimeError):
    """A claimed direct-USB connection failed below the framing layer."""


class DirectUsbRecoveryError(RuntimeError):
    """Bounded recovery could not restore one valid finite RX request."""


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
        reconnect_attempts: int = DEFAULT_RECONNECT_ATTEMPTS,
        reconnect_delay_seconds: float = DEFAULT_RECONNECT_DELAY_SECONDS,
    ) -> None:
        if not serial and not port_path:
            raise ValueError("serial or physical USB port_path is required")
        if bulk_chunk_bytes <= 0:
            raise ValueError("bulk_chunk_bytes must be positive")
        if protocol_version not in (VERSION_V1, VERSION_V2):
            raise ValueError(f"unsupported direct USB protocol {protocol_version}")
        if reconnect_attempts <= 0:
            raise ValueError("reconnect_attempts must be positive")
        if reconnect_delay_seconds < 0:
            raise ValueError("reconnect_delay_seconds must be non-negative")
        self.requested_serial = serial
        self.requested_port_path = port_path
        self.bulk_chunk_bytes = bulk_chunk_bytes
        self.protocol_version = protocol_version
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._context: usb1.USBContext | None = None
        self._handle: usb1.USBDeviceHandle | None = None
        self._identity: DirectUsbIdentity | None = None
        self._capabilities: GadgetCapabilitiesV1 | None = None
        self._detached_kernel_driver = False

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
                    except (usb1.USBError, ProtocolError):
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

    def close(self) -> None:
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
            self._recover_connection(error)
            try:
                capture = self._capture_once(
                    samples_per_channel=samples_per_channel,
                    frame_count=frame_count,
                )
            except (usb1.USBError, DirectUsbTransportError) as retry_error:
                raise DirectUsbRecoveryError(
                    "direct USB transport failed again after rediscovery for "
                    f"serial={self.requested_serial!r} "
                    f"port_path={self.requested_port_path!r}"
                ) from retry_error

            metadata = capture.frames[0].metadata
            if metadata.buffer_sequence != 0 or (
                metadata.flags & MetadataFlags.SAMPLE_SEQUENCE_VALID
                and metadata.first_sample_sequence != 0
            ):
                raise ProtocolError(
                    "recovered direct USB START did not begin a new stream epoch"
                )
            logging.warning(
                "direct USB transport recovered for serial=%s port_path=%s "
                "address=%s->%s stream_id=%s",
                capture.identity.serial,
                capture.identity.port_path,
                None if failed_identity is None else failed_identity.address,
                capture.identity.address,
                metadata.stream_id,
            )
            return capture

    def _capture_once(
        self,
        *,
        samples_per_channel: int,
        frame_count: int,
    ) -> DirectUsbCapture:
        handle = self._require_handle()
        identity = self.identity
        capabilities = self.capabilities
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
        if self.protocol_version == VERSION_V2:
            features |= (
                MetadataFeatures.GAIN_DB_ENDPOINTS
                | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
            )
        if features & capabilities.supported_features != features:
            raise ProtocolError("gadget is missing required metadata features")
        pack_start_request = (
            pack_start_request_v1
            if self.protocol_version == VERSION_V1
            else pack_start_request_v2
        )
        request = pack_start_request(
            requested_features=features,
            enabled_scan_mask=0x0F,
            samples_per_channel=samples_per_channel,
            frame_count=frame_count,
        )
        parser = RxFrameParser(protocol_version=self.protocol_version)
        header_bytes = (
            HEADER_BYTES if self.protocol_version == VERSION_V1 else HEADER_BYTES_V2
        )
        frame_bytes = header_bytes + samples_per_channel * 8
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
                    protocol_errors.append(
                        f"timed out with {len(pending)} queued transfers pending"
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
        if protocol_errors:
            raise ProtocolError("; ".join(protocol_errors))
        if pending or any(chunk is None for chunk in chunks):
            raise ProtocolError("queued direct USB transfer cleanup was incomplete")
        return [chunk for chunk in chunks if chunk is not None]

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
