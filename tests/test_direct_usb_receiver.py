import numpy as np
import pytest
import struct
import usb1

from spf.sdrpluto.direct_usb_protocol import (
    COMMAND_STOP,
    HARDWARE_IDENTITY_BYTES,
    HARDWARE_IDENTITY_MAGIC,
    HARDWARE_IDENTITY_VERSION,
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
    SampleFormat,
)
from spf.sdrpluto.direct_usb_receiver import (
    DirectUsbStreamDiscontinuityError,
    DirectUsbTransferTimeoutError,
    DirectUsbNotFoundError,
    DirectUsbRecoveryError,
    DirectUsbTransportError,
    DirectUsbIdentity,
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
        self.bulk_read_sizes = []

    def controlWrite(self, *args, **kwargs):
        self.control_writes.append((args, kwargs))

    def controlRead(self, *args, **kwargs):
        return self.wire

    def bulkRead(self, endpoint, size, timeout):
        self.bulk_read_sizes.append((endpoint, size, timeout))
        wire, self.wire = self.wire, b""
        return wire


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


def _finite_capabilities(maximum_frames=16):
    return GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=2,
        supported_features=MetadataFeatures(0),
        max_samples_per_channel=524288,
        max_finite_frames=maximum_frames,
        capability_flags=CapabilityFlags.FINITE_RX,
    )


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
    radio.rx_config = type("RxConfig", (), {"buffer_size": 8})()
    radio.direct_rx = type(
        "RecoveredReceiver",
        (),
        {
            "capture": lambda self, **_kwargs: type(
                "RecoveredCapture",
                (),
                {
                    "recovered_after_transport_loss": True,
                    "transport_loss_summary": "USB device re-enumerated",
                    "frames": (),
                },
            )()
        },
    )()

    with pytest.raises(DirectUsbStreamDiscontinuityError, match="new capture artifact"):
        radio._capture_direct_frame()


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
