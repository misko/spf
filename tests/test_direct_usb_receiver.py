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
    DirectUsbIdentity,
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)


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
    receiver = PlutoDirectUsbReceiver(serial="test", port_path=(1,))
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
    assert capture.identity.serial == "test"
    assert capture.identity.port_path == (1,)
    assert capture.identity.address == 9
    assert capture.frames[0].metadata.buffer_sequence == 0
    assert capture.frames[0].metadata.first_sample_sequence == 0


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
