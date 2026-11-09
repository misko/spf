import numpy as np
import pytest
import struct
import usb1

from spf.sdrpluto.direct_usb_protocol import (
    COMMAND_STOP,
    COMMAND_GET_STATUS,
    HARDWARE_IDENTITY_BYTES,
    HARDWARE_IDENTITY_MAGIC,
    HARDWARE_IDENTITY_VERSION,
    RUNTIME_STATUS_BYTES,
    RUNTIME_STATUS_MAGIC,
    RUNTIME_STATUS_VERSION,
    FIRST_CHANGE_UNAVAILABLE,
    HEADER_BYTES,
    HEADER_BYTES_V2,
    CapabilityFlags,
    GadgetCapabilitiesV1,
    GainMetadataV1,
    HardwareIdentityFlags,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    RxFrameParser,
    ErrorSubsystem,
    RuntimeState,
    RuntimeStatusFlags,
    TimeAnchorFlags,
    TimeAnchorV1,
    SampleFormat,
)
from spf.sdrpluto.direct_usb_receiver import (
    DirectUsbStreamDiscontinuityError,
    DirectUsbTransferTimeoutError,
    DirectUsbNotFoundError,
    DirectUsbRecoveryError,
    DirectUsbTransportError,
    DirectUsbIdentity,
    DEFAULT_RECONNECT_ATTEMPTS,
    DEFAULT_RECONNECT_DELAY_SECONDS,
    MAX_ORPHAN_DRAIN_BYTES,
    MAX_ORPHAN_DRAIN_TRANSFERS,
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)
from spf.sdrpluto.sdr_controller import PPlus


def test_iq_payload_layout_rx1_iq_rx2_iq():
    raw = np.asarray(
        [
            [1, -2, 3, -4],
            [5, -6, 7, -8],
        ],
        dtype="<i2",
    )
    signal = iq_payload_to_complex64(raw.tobytes(), samples_per_channel=2)
    assert signal.shape == (2, 2)
    assert signal.dtype == np.complex64
    np.testing.assert_array_equal(signal[0], [1 - 2j, 5 - 6j])
    np.testing.assert_array_equal(signal[1], [3 - 4j, 7 - 8j])


def test_iq_payload_size_is_strict():
    with pytest.raises(ProtocolError, match="IQ payload size mismatch"):
        iq_payload_to_complex64(b"\x00" * 7, samples_per_channel=1)


def test_receiver_requires_stable_identity():
    with pytest.raises(ValueError, match="serial or physical"):
        PlutoDirectUsbReceiver()


def test_receiver_rejects_invalid_bulk_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        PlutoDirectUsbReceiver(serial="test", bulk_chunk_bytes=0)


class _FakeHandle:
    def __init__(self, wire):
        self.wire = wire
        self.control_writes = []
        self.control_reads = []
        self.bulk_read_sizes = []

    def controlWrite(self, *args, **kwargs):
        self.control_writes.append((args, kwargs))

    def controlRead(self, *args, **kwargs):
        self.control_reads.append((args, kwargs))
        return self.wire

    def bulkRead(self, endpoint, size, timeout):
        self.bulk_read_sizes.append((endpoint, size, timeout))
        wire, self.wire = self.wire, b""
        return wire

    def releaseInterface(self, interface):
        pass

    def close(self):
        pass


class _DisconnectedHandle(_FakeHandle):
    def controlWrite(self, *args, **kwargs):
        raise usb1.USBErrorNoDevice()

    def releaseInterface(self, interface):
        pass

    def close(self):
        pass


class _OrphanedEndpointHandle:
    def __init__(self, stale_transfers):
        self.stale_transfers = list(stale_transfers)
        self.events = []

    def controlWrite(self, *args, **kwargs):
        self.events.append(("stop", args, kwargs))

    def bulkRead(self, endpoint, size, timeout):
        self.events.append(("drain", endpoint, size, timeout))
        if not self.stale_transfers:
            raise usb1.USBErrorTimeout()
        return self.stale_transfers.pop(0)

    def clearHalt(self, endpoint):
        self.events.append(("clear", endpoint))


def _finite_capabilities(maximum_frames=16, *, status=False):
    return GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=2,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=524288,
        max_finite_frames=maximum_frames,
        capability_flags=(
            CapabilityFlags.FINITE_RX
            | (CapabilityFlags.STATUS if status else CapabilityFlags(0))
        ),
    )


def _runtime_status_payload(
    state=RuntimeState.IDLE, flags=RuntimeStatusFlags(0)
):
    return struct.pack(
        "<IHHHHiII16s16sQQ14I",
        RUNTIME_STATUS_MAGIC,
        RUNTIME_STATUS_BYTES,
        RUNTIME_STATUS_VERSION,
        state,
        ErrorSubsystem.NONE,
        0,
        int(flags),
        0,
        bytes(16),
        bytes(16),
        0,
        0xFFFFFFFFFFFFFFFF,
        *([0] * 14),
    )


class _StatusFenceHandle(_OrphanedEndpointHandle):
    def __init__(self, payload):
        super().__init__([])
        self.payload = payload

    def controlRead(self, *args, **kwargs):
        self.events.append(("status", args, kwargs))
        return self.payload


class _SequencedStatusFenceHandle(_StatusFenceHandle):
    def __init__(self, payloads):
        super().__init__(b"")
        self.payloads = list(payloads)

    def controlRead(self, *args, **kwargs):
        self.events.append(("status", args, kwargs))
        return self.payloads.pop(0)


def test_quiesce_skips_stop_when_status_is_already_idle():
    handle = _StatusFenceHandle(_runtime_status_payload())

    PlutoDirectUsbReceiver._quiesce_rx_endpoint(
        handle=handle,
        interface=6,
        bulk_in_endpoint=0x89,
        capabilities=_finite_capabilities(status=True),
    )

    assert [event[0] for event in handle.events] == [
        "status",
        "drain",
        "clear",
    ]
    assert handle.events[0][1][1] == COMMAND_GET_STATUS
    assert handle.events[0][2]["timeout"] == 1_000


def test_quiesce_stops_active_worker_and_fences_idle_before_draining():
    handle = _SequencedStatusFenceHandle(
        [
            _runtime_status_payload(
                RuntimeState.STREAMING, RuntimeStatusFlags.RX_WORKER_ACTIVE
            ),
            _runtime_status_payload(),
        ]
    )

    PlutoDirectUsbReceiver._quiesce_rx_endpoint(
        handle=handle,
        interface=6,
        bulk_in_endpoint=0x89,
        capabilities=_finite_capabilities(status=True),
    )

    assert [event[0] for event in handle.events] == [
        "status",
        "stop",
        "status",
        "drain",
        "clear",
    ]


def test_stop_fence_rejects_non_idle_worker_status():
    handle = _StatusFenceHandle(
        _runtime_status_payload(
            RuntimeState.STREAMING, RuntimeStatusFlags.RX_WORKER_ACTIVE
        )
    )

    with pytest.raises(DirectUsbTransportError, match="did not reach an idle"):
        PlutoDirectUsbReceiver._stop_rx_and_fence(
            handle=handle,
            interface=6,
            capabilities=_finite_capabilities(status=True),
        )
    assert [event[0] for event in handle.events] == ["stop", "status"]
    assert handle.events[1][2]["timeout"] == 4_000


def test_usb_time_anchor_brackets_control_exchange():
    anchor = TimeAnchorV1(
        flags=(
            TimeAnchorFlags.COUNTER_INTERVAL_VALID
            | TimeAnchorFlags.MONOTONIC_INTERVAL_VALID
            | TimeAnchorFlags.COUNTER_LOW32
            | TimeAnchorFlags.COUNTER_ADVANCED
        ),
        request_id=1,
        radio_monotonic_before_ns=1000,
        sample_counter_before=100,
        sample_counter_after=103,
        radio_monotonic_after_ns=1200,
    )
    receiver = PlutoDirectUsbReceiver(serial="test", protocol_version=3)
    receiver._handle = _FakeHandle(anchor.pack())
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x81,
        bulk_out_endpoint=0x01,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=3,
        protocol_max=3,
        supported_features=MetadataFeatures.HARDWARE_SAMPLE_COUNTER,
        max_samples_per_channel=524288,
        max_finite_frames=16,
        capability_flags=(CapabilityFlags.FINITE_RX | CapabilityFlags.TIME_ANCHOR),
    )

    measurement = receiver.query_time_anchor()

    assert measurement.anchor == anchor
    assert measurement.transport == "direct_usb"
    assert measurement.round_trip_ns >= 0
    assert receiver._handle.control_writes == []
    control_args, control_kwargs = receiver._handle.control_reads[0]
    assert control_args[2] == 1  # 16-bit request ID in wValue
    assert control_args[3] == 6  # FunctionFS interface remains in wIndex
    assert control_kwargs["timeout"] > 0


def test_open_quiesce_stops_and_drains_orphaned_bulk_data():
    handle = _OrphanedEndpointHandle([b"old-frame", b"old-tail"])

    PlutoDirectUsbReceiver._quiesce_rx_endpoint(
        handle=handle,
        interface=6,
        bulk_in_endpoint=0x89,
        capabilities=_finite_capabilities(),
    )

    assert [event[0] for event in handle.events] == [
        "stop",
        "drain",
        "drain",
        "drain",
        "clear",
    ]
    assert handle.events[0][1][1] == COMMAND_STOP
    assert handle.events[1][2] == min(
        MAX_ORPHAN_DRAIN_BYTES,
        HEADER_BYTES_V2 + _finite_capabilities().max_samples_per_channel * 8,
    )


def test_open_quiesce_fails_if_orphaned_backlog_does_not_end():
    handle = _OrphanedEndpointHandle([b"stale"] * MAX_ORPHAN_DRAIN_TRANSFERS)

    with pytest.raises(DirectUsbTransportError, match="remained non-empty"):
        PlutoDirectUsbReceiver._quiesce_rx_endpoint(
            handle=handle,
            interface=6,
            bulk_in_endpoint=0x89,
            capabilities=_finite_capabilities(
                maximum_frames=MAX_ORPHAN_DRAIN_TRANSFERS
            ),
        )

    assert [event[0] for event in handle.events].count("drain") == (
        MAX_ORPHAN_DRAIN_TRANSFERS
    )
    assert handle.events[-1] == ("clear", 0x89)


def _valid_wire_frame(samples=8, sequence=0):
    metadata = GainMetadataV1(
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
        ),
        stream_id=123,
        buffer_sequence=sequence,
        first_sample_sequence=sequence * samples,
        samples_per_channel=samples,
        iq_payload_bytes=samples * 8,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_start=42,
        rx2_gain_start=43,
        rx1_gain_end=42,
        rx2_gain_end=43,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
    )
    return metadata.pack() + bytes(samples * 8)


def test_capture_requests_a_complete_framed_transfer_and_stops():
    wire = _valid_wire_frame()
    handle = _FakeHandle(wire)
    receiver = PlutoDirectUsbReceiver(serial="test", bulk_chunk_bytes=16)
    receiver._handle = handle
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=CapabilityFlags.FINITE_RX,
    )

    capture = receiver.capture(samples_per_channel=8)

    assert len(capture.frames) == 1
    assert handle.bulk_read_sizes[0][1] == HEADER_BYTES + 8 * 8
    assert len(handle.control_writes) == 2


class _SequentialFakeHandle(_FakeHandle):
    def __init__(self, wires):
        super().__init__(b"")
        self.wires = list(wires)

    def bulkRead(self, endpoint, size, timeout):
        self.bulk_read_sizes.append((endpoint, size, timeout))
        return self.wires.pop(0)


def test_frame_stream_uses_one_start_stop_for_multiple_sync_frames():
    handle = _SequentialFakeHandle(
        [_valid_wire_frame(sequence=index) for index in range(3)]
    )
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = _finite_capabilities()

    frames = list(
        receiver.stream_frames(
            samples_per_channel=8,
            frame_count=3,
            queue_depth=1,
        )
    )

    assert [frame.metadata.buffer_sequence for frame in frames] == [0, 1, 2]
    assert len(handle.control_writes) == 2
    assert receiver._active_stream is None


def test_close_stops_an_active_frame_stream():
    handle = _SequentialFakeHandle(
        [_valid_wire_frame(sequence=index) for index in range(2)]
    )
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = _finite_capabilities()
    stream = receiver.stream_frames(samples_per_channel=8, frame_count=2)
    assert next(stream).metadata.buffer_sequence == 0

    receiver.close()

    assert len(handle.control_writes) == 2
    assert receiver._active_stream is None


def test_capture_rediscovers_same_radio_after_usb_address_changes(monkeypatch):
    disconnected = _DisconnectedHandle(b"")
    attest_calls = []
    receiver = PlutoDirectUsbReceiver(
        serial="test",
        port_path=(1,),
        reconnect_attestor=lambda: attest_calls.append(True),
    )
    receiver._handle = disconnected
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=CapabilityFlags.FINITE_RX,
    )
    replacement = _FakeHandle(_valid_wire_frame())
    open_calls = []

    def reopen():
        open_calls.append(True)
        receiver._handle = replacement
        receiver._identity = DirectUsbIdentity(
            serial="test",
            bus=1,
            address=9,
            port_path=(1,),
            interface=6,
            bulk_in_endpoint=0x89,
            bulk_out_endpoint=0x07,
        )
        receiver._capabilities = GadgetCapabilitiesV1(
            protocol_min=1,
            protocol_max=1,
            supported_features=(
                MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
                | MetadataFeatures.HEADER_CRC32
                | MetadataFeatures.SAMPLE_SEQUENCE
            ),
            max_samples_per_channel=1024,
            max_finite_frames=16,
            capability_flags=CapabilityFlags.FINITE_RX,
        )

    monkeypatch.setattr(receiver, "open", reopen)

    capture = receiver.capture(samples_per_channel=8)

    assert len(open_calls) == 1
    assert len(attest_calls) == 1
    assert capture.identity.serial == "test"
    assert capture.identity.port_path == (1,)
    assert capture.identity.address == 9
    assert capture.frames[0].metadata.buffer_sequence == 0
    assert capture.frames[0].metadata.first_sample_sequence == 0


def test_capture_rediscovery_is_bounded(monkeypatch):
    disconnected = _DisconnectedHandle(b"")
    receiver = PlutoDirectUsbReceiver(
        serial="test",
        port_path=(1,),
        reconnect_attempts=3,
        reconnect_delay_seconds=0,
    )
    receiver._handle = disconnected
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=CapabilityFlags.FINITE_RX,
    )
    open_calls = []

    def missing():
        open_calls.append(True)
        raise DirectUsbNotFoundError("still disconnected")

    monkeypatch.setattr(receiver, "open", missing)

    with pytest.raises(DirectUsbRecoveryError, match="3 bounded attempts"):
        receiver.capture(samples_per_channel=8)

    assert len(open_calls) == 3


def test_default_rediscovery_budget_covers_firmware_watchdog_and_reenumeration():
    assert (
        DEFAULT_RECONNECT_ATTEMPTS * DEFAULT_RECONNECT_DELAY_SECONDS
        >= 15.0
    )


def test_protocol_error_fails_closed_without_transport_rediscovery(monkeypatch):
    handle = _FakeHandle(b"\x00" * (HEADER_BYTES + 8 * 8))
    receiver = PlutoDirectUsbReceiver(serial="test", port_path=(1,))
    receiver._handle = handle
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=CapabilityFlags.FINITE_RX,
    )
    open_calls = []
    monkeypatch.setattr(receiver, "open", lambda: open_calls.append(True))

    with pytest.raises(ProtocolError, match="magic"):
        receiver.capture(samples_per_channel=8)

    assert open_calls == []


def test_hardware_identity_query_is_read_only_and_does_not_start_streaming():
    payload = struct.pack(
        "<IHHIIQ40s",
        HARDWARE_IDENTITY_MAGIC,
        HARDWARE_IDENTITY_BYTES,
        HARDWARE_IDENTITY_VERSION,
        int(
            HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID
            | HardwareIdentityFlags.GADGET_BUILD_ID_VALID
        ),
        0,
        0x123456789ABCD,
        b"c" * 40,
    )
    handle = _FakeHandle(payload)
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=2,
        supported_features=MetadataFeatures(0x37),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=(
            CapabilityFlags.FINITE_RX | CapabilityFlags.HARDWARE_IDENTITY
        ),
    )

    identity = receiver.query_hardware_identity()

    assert identity.fpga_device_dna == 0x123456789ABCD
    assert identity.gadget_build_id == "c" * 40
    assert handle.control_writes == []
    assert handle.bulk_read_sizes == []


def test_runtime_status_query_is_read_only_and_does_not_start_streaming():
    payload = struct.pack(
        "<IHHHHiII16s16sQQ14I",
        RUNTIME_STATUS_MAGIC,
        RUNTIME_STATUS_BYTES,
        RUNTIME_STATUS_VERSION,
        RuntimeState.IDLE,
        ErrorSubsystem.NONE,
        0,
        int(RuntimeStatusFlags.BOOT_ID_VALID | RuntimeStatusFlags.PROCESS_NONCE_VALID),
        0,
        b"\x11" * 16,
        b"\x22" * 16,
        0,
        0xFFFFFFFFFFFFFFFF,
        *([0] * 14),
    )
    handle = _FakeHandle(payload)
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=2,
        supported_features=MetadataFeatures(0x37),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=CapabilityFlags.FINITE_RX | CapabilityFlags.STATUS,
    )

    status = receiver.query_runtime_status()

    assert status.lifecycle_state is RuntimeState.IDLE
    assert status.boot_id == b"\x11" * 16
    assert handle.control_writes == []
    assert handle.bulk_read_sizes == []


class _FakeTransfer:
    def __init__(self, wire, event_log):
        self.wire = wire
        self.event_log = event_log
        self.callback = None
        self.user_data = None
        self.cancelled = False
        self.closed = False

    def setBulk(
        self,
        endpoint,
        buffer_or_len,
        callback=None,
        user_data=None,
        timeout=0,
    ):
        assert len(self.wire) == buffer_or_len
        self.callback = callback
        self.user_data = user_data

    def submit(self):
        self.event_log.append(f"submit:{self.user_data}")

    def cancel(self):
        self.cancelled = True
        self.event_log.append(f"cancel:{self.user_data}")

    def getUserData(self):
        return self.user_data

    def getStatus(self):
        return usb1.TRANSFER_CANCELLED if self.cancelled else usb1.TRANSFER_COMPLETED

    def getActualLength(self):
        return len(self.wire)

    def getBuffer(self):
        return self.wire

    def isSubmitted(self):
        return False

    def close(self):
        self.closed = True
        self.callback = None


class _FakeAsyncContext:
    def __init__(self, handle):
        self.handle = handle

    def handleEventsTimeout(self, timeout):
        transfer = self.handle.transfers.pop(0)
        transfer.callback(transfer)


class _FakeAsyncHandle:
    def __init__(self, wires):
        self.wires = list(wires)
        self.transfers = []
        self.event_log = []
        self.control_writes = []

    def getTransfer(self, **kwargs):
        transfer = _FakeTransfer(
            self.wires[len(self.transfers)],
            self.event_log,
        )
        self.transfers.append(transfer)
        return transfer

    def controlWrite(self, *args, **kwargs):
        self.event_log.append("control")
        self.control_writes.append((args, kwargs))


class _FailSecondSubmitHandle(_FakeAsyncHandle):
    def getTransfer(self, **kwargs):
        transfer = super().getTransfer(**kwargs)
        if len(self.transfers) == 2:

            def fail_submit():
                self.event_log.append("submit-failed:1")
                raise RuntimeError("synthetic transfer allocation failure")

            transfer.submit = fail_submit
        return transfer


class _RollingTransfer:
    def __init__(self, handle):
        self.handle = handle
        self.callback = None
        self.user_data = None
        self.wire = b""
        self.submitted = False
        self.cancelled = False
        self.closed = False

    def setBulk(
        self,
        endpoint,
        buffer_or_len,
        callback=None,
        user_data=None,
        timeout=0,
    ):
        self.expected_bytes = buffer_or_len
        self.callback = callback
        self.user_data = user_data

    def submit(self):
        self.wire = self.handle.wires.pop(0)
        assert len(self.wire) == self.expected_bytes
        self.submitted = True
        self.cancelled = False
        self.handle.pending.append(self)
        self.handle.event_log.append("submit")

    def cancel(self):
        self.cancelled = True
        self.handle.event_log.append("cancel")

    def getStatus(self):
        return usb1.TRANSFER_CANCELLED if self.cancelled else usb1.TRANSFER_COMPLETED

    def getActualLength(self):
        return len(self.wire)

    def getBuffer(self):
        return self.wire

    def isSubmitted(self):
        return self.submitted

    def close(self):
        self.closed = True
        self.callback = None


class _RollingAsyncHandle:
    def __init__(self, wires):
        self.wires = list(wires)
        self.pending = []
        self.transfers = []
        self.event_log = []
        self.control_writes = []

    def getTransfer(self):
        transfer = _RollingTransfer(self)
        self.transfers.append(transfer)
        return transfer

    def controlWrite(self, *args, **kwargs):
        self.event_log.append("control")
        self.control_writes.append((args, kwargs))


class _RollingAsyncContext:
    def __init__(self, handle):
        self.handle = handle

    def handleEventsTimeout(self, timeout):
        transfer = self.handle.pending.pop(0)
        transfer.submitted = False
        transfer.callback(transfer)


def test_frame_stream_reuses_one_rolling_transfer():
    handle = _RollingAsyncHandle(
        [_valid_wire_frame(sequence=index) for index in range(4)]
    )
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._context = _RollingAsyncContext(handle)
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = _finite_capabilities()

    stream = receiver.stream_frames(
        samples_per_channel=8,
        frame_count=4,
        queue_depth=1,
    )
    assert next(stream).metadata.buffer_sequence == 0
    # The same transfer is already resubmitted before control returns to the
    # consumer, keeping the finite gadget stream flowing with bounded memory.
    assert len(handle.transfers) == 1
    assert handle.event_log[:3] == ["submit", "control", "submit"]
    remaining = list(stream)

    assert [frame.metadata.buffer_sequence for frame in remaining] == [1, 2, 3]
    assert handle.event_log.count("submit") == 4
    assert handle.event_log.count("control") == 2
    assert handle.transfers[0].closed


def test_closing_rolling_frame_stream_cancels_pending_transfer_before_stop():
    handle = _RollingAsyncHandle(
        [_valid_wire_frame(sequence=index) for index in range(3)]
    )
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._context = _RollingAsyncContext(handle)
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = _finite_capabilities()
    stream = receiver.stream_frames(
        samples_per_channel=8,
        frame_count=3,
        queue_depth=1,
    )
    assert next(stream).metadata.buffer_sequence == 0

    stream.close()

    assert handle.event_log.count("cancel") == 1
    assert handle.event_log[-1] == "control"
    assert len(handle.control_writes) == 2
    assert handle.transfers[0].closed
    assert receiver._active_stream is None


def test_capture_queues_bulk_transfers_before_start_and_parses_in_order():
    handle = _FakeAsyncHandle(
        [_valid_wire_frame(sequence=0), _valid_wire_frame(sequence=1)]
    )
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._context = _FakeAsyncContext(handle)
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=CapabilityFlags.FINITE_RX,
    )

    capture = receiver.capture(samples_per_channel=8, frame_count=2)

    assert [frame.metadata.buffer_sequence for frame in capture.frames] == [0, 1]
    assert handle.event_log[:3] == ["submit:0", "submit:1", "control"]
    assert all(transfer.closed for transfer in handle.transfers)


class _NeverCompletesContext:
    def handleEventsTimeout(self, timeout):
        return None


def test_queued_deadline_is_classified_as_transport_timeout(monkeypatch):
    handle = _FakeAsyncHandle([_valid_wire_frame(sequence=0)])
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._context = _NeverCompletesContext()
    identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    times = iter([0.0, 12.0, 12.0, 14.0])
    monkeypatch.setattr(
        "spf.sdrpluto.direct_usb_receiver.time.monotonic", lambda: next(times)
    )

    with pytest.raises(DirectUsbTransferTimeoutError) as raised:
        receiver._capture_queued(
            handle=handle,
            identity=identity,
            request=b"request",
            frame_count=1,
            frame_bytes=len(_valid_wire_frame(sequence=0)),
        )

    assert raised.value.pending_transfer_count == 1
    assert raised.value.requested_transfer_count == 1
    assert raised.value.timeout_ms == 10_000
    assert raised.value.serial == "test"
    assert raised.value.port_path == (1,)


def test_spf_rejects_an_attested_restarted_stream_before_using_its_iq():
    radio = PPlus.__new__(PPlus)
    radio.rx_config = type(
        "RxConfig",
        (),
        {"buffer_size": 8, "direct_usb_frame_count_per_request": 4},
    )()
    radio._direct_frame_stream = None
    radio._direct_frames_remaining = 0

    def failed_stream(**_kwargs):
        def frames():
            raise DirectUsbStreamDiscontinuityError(
                "direct USB frame stream lost transport continuity; "
                "start a new capture artifact"
            )
            yield

        return frames()

    radio.direct_rx = type(
        "RecoveredReceiver",
        (),
        {"stream_frames": lambda self, **kwargs: failed_stream(**kwargs)},
    )()

    with pytest.raises(DirectUsbStreamDiscontinuityError, match="new capture artifact"):
        radio._capture_direct_frame()
    assert radio._direct_frame_stream is None
    assert radio._direct_frames_remaining == 0


def test_spf_returns_one_frame_at_a_time_from_one_finite_stream_group():
    frames = []
    parser = RxFrameParser(protocol_version=1)
    for sequence in range(4):
        frames.extend(parser.feed(_valid_wire_frame(sequence=sequence)))
    parser.finish()

    class GroupedReceiver:
        def __init__(self):
            self.requests = []

        def stream_frames(self, **kwargs):
            self.requests.append(kwargs)

            def grouped_frames():
                yield from frames

            return grouped_frames()

    radio = PPlus.__new__(PPlus)
    radio.rx_config = type(
        "RxConfig",
        (),
        {"buffer_size": 8, "direct_usb_frame_count_per_request": 4},
    )()
    radio.direct_rx = GroupedReceiver()
    radio._direct_frame_stream = None
    radio._direct_frames_remaining = 0

    results = [radio._capture_direct_frame() for _index in range(4)]

    assert [metadata.buffer_sequence for _signal, metadata in results] == [0, 1, 2, 3]
    assert radio.direct_rx.requests == [
        {"samples_per_channel": 8, "frame_count": 4, "queue_depth": 1}
    ]
    assert radio._direct_frame_stream is None
    assert radio._direct_frames_remaining == 0


def test_spf_takes_final_frame_anchor_before_finite_stream_teardown():
    events = []
    parser = RxFrameParser(protocol_version=1)
    frames = parser.feed(_valid_wire_frame(sequence=0))
    parser.finish()

    class OrderedReceiver:
        def stream_frames(self, **_kwargs):
            def one_frame():
                try:
                    events.append("yield")
                    yield frames[0]
                finally:
                    events.append("stop")

            return one_frame()

    radio = PPlus.__new__(PPlus)
    radio.rx_config = type(
        "RxConfig",
        (),
        {"buffer_size": 8, "direct_usb_frame_count_per_request": 1},
    )()
    radio.direct_rx = OrderedReceiver()
    radio._direct_frame_stream = None
    radio._direct_frames_remaining = 0
    radio._refresh_direct_time_anchors = lambda: events.append("anchor")
    radio._fit_direct_sample_time = lambda _metadata: events.append("fit")

    radio._capture_direct_frame()

    assert events == ["anchor", "yield", "anchor", "fit", "stop"]


def test_partial_transfer_submission_is_cancelled_before_start():
    handle = _FailSecondSubmitHandle(
        [_valid_wire_frame(sequence=0), _valid_wire_frame(sequence=1)]
    )
    receiver = PlutoDirectUsbReceiver(serial="test")
    receiver._handle = handle
    receiver._context = _FakeAsyncContext(handle)
    receiver._identity = DirectUsbIdentity(
        serial="test",
        bus=1,
        address=2,
        port_path=(1,),
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )
    receiver._capabilities = GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=CapabilityFlags.FINITE_RX,
    )

    with pytest.raises(RuntimeError, match="allocation failure"):
        receiver.capture(samples_per_channel=8, frame_count=2)

    assert len(handle.control_writes) == 1
    assert handle.control_writes[0][0][1] == COMMAND_STOP
    assert handle.event_log == [
        "submit:0",
        "submit-failed:1",
        "cancel:0",
        "control",
    ]
