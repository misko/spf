from types import SimpleNamespace

import pytest
import usb1

from spf.sdrpluto import sdr_controller
from spf.sdrpluto.direct_usb_protocol import (
    CapabilityFlags,
    GadgetCapabilitiesV1,
    GainMetadataV1,
    HardwareIdentityFlags,
    HardwareIdentityV1,
    ErrorSubsystem,
    RuntimeState,
    RuntimeStatusFlags,
    RuntimeStatusV1,
    MetadataFeatures,
    MetadataFlags,
    ProtocolError,
    SampleFormat,
)
from spf.sdrpluto.direct_usb_receiver import (
    DirectUsbIdentity,
    DirectUsbRecoveryAttestationError,
    DirectUsbRecoveryError,
    PlutoDirectUsbReceiver,
    RecoveryAttestationDifference,
)
from spf.sdrpluto.sdr_controller import PPlus


SERIAL = "104000bac4950008230026001b440a003a"
PORT_PATH = (1, 4, 5)
GADGET_SHA = "a" * 40
FPGA_DNA = 0x123456789ABCD


class _FakeHandle:
    def __init__(self, wire):
        self.wire = wire
        self.control_writes = []

    def controlWrite(self, *args, **kwargs):
        self.control_writes.append((args, kwargs))

    def controlRead(self, *args, **kwargs):
        return self.wire

    def bulkRead(self, endpoint, size, timeout):
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
        rx1_first_change_sample=0xFFFFFFFF,
        rx2_first_change_sample=0xFFFFFFFF,
    )
    return metadata.pack() + bytes(samples * 8)


def _identity(*, bus=1, address=2, serial=SERIAL, port_path=PORT_PATH):
    return DirectUsbIdentity(
        serial=serial,
        bus=bus,
        address=address,
        port_path=port_path,
        interface=6,
        bulk_in_endpoint=0x89,
        bulk_out_endpoint=0x07,
    )


def _capabilities(*, hardware_identity=False, runtime_status=False):
    flags = CapabilityFlags.FINITE_RX
    if hardware_identity:
        flags |= CapabilityFlags.HARDWARE_IDENTITY
    if runtime_status:
        flags |= CapabilityFlags.STATUS
    return GadgetCapabilitiesV1(
        protocol_min=1,
        protocol_max=1,
        supported_features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
        ),
        max_samples_per_channel=1024,
        max_finite_frames=16,
        capability_flags=flags,
    )


def _hardware_identity(*, dna=FPGA_DNA, build_id=GADGET_SHA):
    return HardwareIdentityV1(
        flags=(
            HardwareIdentityFlags.FPGA_DEVICE_DNA_VALID
            | HardwareIdentityFlags.GADGET_BUILD_ID_VALID
        ),
        fpga_device_dna=dna,
        gadget_build_id=build_id,
    )


def _runtime_status(*, boot_id=b"\x11" * 16, process_nonce=b"\x22" * 16):
    return RuntimeStatusV1(
        lifecycle_state=RuntimeState.IDLE,
        last_error_subsystem=ErrorSubsystem.NONE,
        last_errno=0,
        flags=(
            RuntimeStatusFlags.BOOT_ID_VALID
            | RuntimeStatusFlags.PROCESS_NONCE_VALID
        ),
        boot_id=boot_id,
        process_nonce=process_nonce,
        current_stream_id=0,
        last_completed_sequence=0xFFFFFFFFFFFFFFFF,
        start_count=0,
        stop_count=0,
        completed_frame_count=0,
        dropped_frame_count=0,
        iio_refill_error_count=0,
        usb_submit_error_count=0,
        short_write_count=0,
        buffer_starvation_count=0,
        gain_read_failure_count=0,
        rssi_read_failure_count=0,
        control_error_count=0,
        stop_timeout_count=0,
        worker_heartbeat_age_ms=0,
    )


def test_recovery_rejects_changed_gadget_process_nonce(monkeypatch):
    attestor_calls = []
    receiver = _disconnected_receiver(
        attestor=lambda: attestor_calls.append("configured")
    )
    receiver._identity = _identity(address=9)
    receiver._capabilities = _capabilities(runtime_status=True)
    receiver._recovery_runtime_status = _runtime_status()
    monkeypatch.setattr(
        receiver,
        "query_runtime_status",
        lambda: _runtime_status(process_nonce=b"\x33" * 16),
    )

    with pytest.raises(DirectUsbRecoveryAttestationError) as error:
        receiver._attest_recovered_connection(_identity())

    assert [difference.field for difference in error.value.differences] == [
        "process_nonce"
    ]
    assert attestor_calls == []


class _RecoveryHandle(_FakeHandle):
    def __init__(self, wire, event_log):
        super().__init__(wire)
        self.event_log = event_log

    def controlWrite(self, *args, **kwargs):
        self.event_log.append("start_or_stop")
        return super().controlWrite(*args, **kwargs)


def _disconnected_receiver(*, attestor):
    receiver = PlutoDirectUsbReceiver(
        serial=SERIAL,
        port_path=PORT_PATH,
        reconnect_attempts=1,
        reconnect_delay_seconds=0,
        reconnect_attestor=attestor,
    )
    receiver._handle = _DisconnectedHandle(b"")
    receiver._identity = _identity()
    receiver._capabilities = _capabilities()
    return receiver


def test_recovery_attestation_precedes_start_and_changed_address_is_allowed(
    monkeypatch,
):
    events = []
    receiver = _disconnected_receiver(attestor=lambda: events.append("iio_config"))
    receiver._recovery_hardware_identity = _hardware_identity()
    replacement = _RecoveryHandle(_valid_wire_frame(), events)

    def reopen():
        events.append("rediscover")
        receiver._handle = replacement
        receiver._identity = _identity(address=9)
        receiver._capabilities = _capabilities(hardware_identity=True)

    monkeypatch.setattr(receiver, "open", reopen)
    monkeypatch.setattr(
        receiver,
        "query_hardware_identity",
        lambda: (events.append("firmware_identity") or _hardware_identity()),
    )

    capture = receiver.capture(samples_per_channel=8)

    assert capture.identity.address == 9
    assert capture.recovered_after_transport_loss is True
    assert "USBErrorNoDevice" in capture.transport_loss_summary
    assert events[:4] == [
        "rediscover",
        "firmware_identity",
        "iio_config",
        "start_or_stop",
    ]


@pytest.mark.parametrize(
    ("field", "recovered"),
    (
        ("usb_serial", _identity(serial="different")),
        ("usb_port_path", _identity(port_path=(1, 4, 6))),
    ),
)
def test_durable_usb_identity_mismatch_rejects_before_start(
    monkeypatch, field, recovered
):
    attest_calls = []
    receiver = _disconnected_receiver(attestor=lambda: attest_calls.append(True))
    replacement = _FakeHandle(_valid_wire_frame())

    def reopen():
        receiver._handle = replacement
        receiver._identity = recovered
        receiver._capabilities = _capabilities()

    monkeypatch.setattr(receiver, "open", reopen)

    with pytest.raises(DirectUsbRecoveryAttestationError) as raised:
        receiver.capture(samples_per_channel=8)

    assert [difference.field for difference in raised.value.differences] == [field]
    assert replacement.control_writes == []
    assert attest_calls == []


@pytest.mark.parametrize(
    ("field", "observed"),
    (
        ("fpga_device_dna", _hardware_identity(dna=FPGA_DNA + 1)),
        ("gadget_build_id", _hardware_identity(build_id="b" * 40)),
    ),
)
def test_firmware_identity_mismatch_rejects_before_start(monkeypatch, field, observed):
    attest_calls = []
    receiver = _disconnected_receiver(attestor=lambda: attest_calls.append(True))
    receiver._recovery_hardware_identity = _hardware_identity()
    replacement = _FakeHandle(_valid_wire_frame())

    def reopen():
        receiver._handle = replacement
        receiver._identity = _identity(address=9)
        receiver._capabilities = _capabilities(hardware_identity=True)

    monkeypatch.setattr(receiver, "open", reopen)
    monkeypatch.setattr(receiver, "query_hardware_identity", lambda: observed)

    with pytest.raises(DirectUsbRecoveryAttestationError) as raised:
        receiver.capture(samples_per_channel=8)

    assert [difference.field for difference in raised.value.differences] == [field]
    assert replacement.control_writes == []
    assert attest_calls == []


def test_missing_iio_attestor_fails_closed_before_start(monkeypatch):
    receiver = _disconnected_receiver(attestor=None)
    replacement = _FakeHandle(_valid_wire_frame())

    def reopen():
        receiver._handle = replacement
        receiver._identity = _identity(address=9)
        receiver._capabilities = _capabilities()

    monkeypatch.setattr(receiver, "open", reopen)

    with pytest.raises(
        DirectUsbRecoveryAttestationError, match="configured reconnect attestor"
    ):
        receiver.capture(samples_per_channel=8)

    assert replacement.control_writes == []


def test_iio_configuration_mismatch_rejects_before_start(monkeypatch):
    difference = RecoveryAttestationDifference(
        field="rx_lo",
        expected=2_467_000_000,
        observed=2_412_000_000,
    )

    def mismatch():
        raise DirectUsbRecoveryAttestationError((difference,))

    receiver = _disconnected_receiver(attestor=mismatch)
    replacement = _FakeHandle(_valid_wire_frame())

    def reopen():
        receiver._handle = replacement
        receiver._identity = _identity(address=9)
        receiver._capabilities = _capabilities()

    monkeypatch.setattr(receiver, "open", reopen)

    with pytest.raises(DirectUsbRecoveryAttestationError) as raised:
        receiver.capture(samples_per_channel=8)

    assert raised.value.differences == (difference,)
    assert replacement.control_writes == []
    with pytest.raises(DirectUsbRecoveryError, match="terminal"):
        receiver.capture(samples_per_channel=8)
    assert replacement.control_writes == []


def test_invalid_sequence_zero_is_rejected_after_attested_restart(monkeypatch):
    receiver = _disconnected_receiver(attestor=lambda: None)
    replacement = _FakeHandle(_valid_wire_frame(sequence=1))

    def reopen():
        receiver._handle = replacement
        receiver._identity = _identity(address=9)
        receiver._capabilities = _capabilities()

    monkeypatch.setattr(receiver, "open", reopen)

    with pytest.raises(ProtocolError, match="sequence 0"):
        receiver.capture(samples_per_channel=8)
    with pytest.raises(DirectUsbRecoveryError, match="terminal"):
        receiver.capture(samples_per_channel=8)


class _Value:
    def __init__(self, value):
        self.value = value


class _FakeContext:
    def __init__(self, serial):
        self.attrs = {"hw_serial": _Value(serial)}
        self._context = object()
        self.destroy_calls = 0

    def __del__(self):
        if self._context is not None:
            self.destroy_calls += 1


class _RxChannel:
    def __init__(self, fir):
        self.attrs = {"filter_fir_en": _Value(str(fir))}


class _FakeCtrl:
    def __init__(self, *, fir_rx1=1, fir_rx2=1, debug_attr=1, register_bit=1):
        self.channels = {
            "voltage0": _RxChannel(fir_rx1),
            "voltage1": _RxChannel(fir_rx2),
        }
        self.debug_attrs = {
            "adi,rx1-rx2-phase-inversion-enable": _Value(str(debug_attr))
        }
        self.register_bit = register_bit

    def find_channel(self, name, is_output):
        assert not is_output
        return self.channels[name]

    def reg_read(self, address):
        assert address == 0x22
        return self.register_bit << 6


class _FreshSdr:
    def __init__(
        self,
        *,
        serial=SERIAL,
        rx_lo=2_467_000_000,
        sample_rate=3_000_000,
        rx_rf_bandwidth=3_000_000,
        gain_mode_rx1="manual",
        gain_mode_rx2="manual",
        gain_rx1=26,
        gain_rx2=41,
        fir_rx1=1,
        fir_rx2=1,
        debug_attr=1,
        register_bit=1,
        gains_readable=True,
    ):
        self._ctx = _FakeContext(serial)
        self.rx_lo = rx_lo
        self.sample_rate = sample_rate
        self.rx_rf_bandwidth = rx_rf_bandwidth
        self.gain_control_mode_chan0 = gain_mode_rx1
        self.gain_control_mode_chan1 = gain_mode_rx2
        self._gain_rx1 = gain_rx1
        self._gain_rx2 = gain_rx2
        self.gains_readable = gains_readable
        self._ctrl = _FakeCtrl(
            fir_rx1=fir_rx1,
            fir_rx2=fir_rx2,
            debug_attr=debug_attr,
            register_bit=register_bit,
        )

    @property
    def rx_hardwaregain_chan0(self):
        if not self.gains_readable:
            raise AssertionError("instantaneous AGC gain must not be read")
        return self._gain_rx1

    @property
    def rx_hardwaregain_chan1(self):
        if not self.gains_readable:
            raise AssertionError("instantaneous AGC gain must not be read")
        return self._gain_rx2


def _pplus(*, gain_modes=("manual", "manual")):
    pplus = object.__new__(PPlus)
    pplus.uri = "usb:1.4.5"
    pplus.rx_config = SimpleNamespace(
        lo=2_467_000_000,
        sample_rate=3_000_000,
        rf_bandwidth=3_000_000,
        gain_control_modes=list(gain_modes),
        gains=[26, 41],
        filter_fir_en=1,
        phase_inversion_debug_attr=1,
        phase_inversion_register_bit=1,
    )
    pplus.direct_rx = SimpleNamespace(identity=_identity(address=9))
    pplus.sdr = _FreshSdr()
    pplus._scan_iio_contexts = lambda: {
        "usb:1.9.5": f"Pluto, serial={SERIAL}",
    }
    return pplus


@pytest.mark.parametrize(
    ("field", "overrides"),
    (
        ("iio_serial", {"serial": "different"}),
        ("rx_lo", {"rx_lo": 2_412_000_000}),
        ("sample_rate", {"sample_rate": 2_000_000}),
        ("rx_rf_bandwidth", {"rx_rf_bandwidth": 2_000_000}),
        ("gain_control_mode_rx1", {"gain_mode_rx1": "slow_attack"}),
        ("gain_control_mode_rx2", {"gain_mode_rx2": "slow_attack"}),
        ("manual_gain_rx1_db", {"gain_rx1": 25}),
        ("manual_gain_rx2_db", {"gain_rx2": 40}),
        ("filter_fir_en_rx1", {"fir_rx1": 0}),
        ("filter_fir_en_rx2", {"fir_rx2": 0}),
        ("phase_inversion_debug_attr", {"debug_attr": 0}),
        ("phase_inversion_register_bit", {"register_bit": 0}),
    ),
)
def test_pplus_rejects_each_observable_configuration_mismatch(
    monkeypatch, field, overrides
):
    pplus = _pplus()
    fresh = _FreshSdr(**overrides)
    monkeypatch.setattr(sdr_controller.adi, "ad9361", lambda uri: fresh)

    with pytest.raises(DirectUsbRecoveryAttestationError) as raised:
        pplus._attest_direct_usb_reconnect()

    assert field in [difference.field for difference in raised.value.differences]
    assert pplus.sdr is not fresh
    assert fresh._ctx is None


def test_pplus_matching_configuration_retains_fresh_iio_context(monkeypatch):
    pplus = _pplus()
    stale = pplus.sdr
    stale_context = stale._ctx
    fresh = _FreshSdr()
    opened_uris = []

    def open_fresh(uri):
        opened_uris.append(uri)
        return fresh

    monkeypatch.setattr(sdr_controller.adi, "ad9361", open_fresh)

    pplus._attest_direct_usb_reconnect()

    assert opened_uris == ["usb:1.9.5"]
    assert pplus.uri == "usb:1.9.5"
    assert pplus.sdr is fresh
    assert stale._ctx is None
    assert stale_context.destroy_calls == 1
    assert fresh._ctx is not None


def test_pplus_does_not_compare_instantaneous_gain_in_agc_modes(monkeypatch):
    pplus = _pplus(gain_modes=("slow_attack", "fast_attack"))
    fresh = _FreshSdr(
        gain_mode_rx1="slow_attack",
        gain_mode_rx2="fast_attack",
        gains_readable=False,
    )
    monkeypatch.setattr(sdr_controller.adi, "ad9361", lambda uri: fresh)

    pplus._attest_direct_usb_reconnect()

    assert pplus.sdr is fresh


def test_pplus_iio_unavailable_is_a_structured_fail_closed_error(monkeypatch):
    pplus = _pplus()

    def unavailable(uri):
        raise OSError("No such device")

    monkeypatch.setattr(sdr_controller.adi, "ad9361", unavailable)

    with pytest.raises(DirectUsbRecoveryAttestationError) as raised:
        pplus._attest_direct_usb_reconnect()

    assert raised.value.differences == (
        RecoveryAttestationDifference(
            field="iio_context",
            expected="readable context at usb:1.9.5",
            observed="OSError: No such device",
        ),
    )


def test_pplus_fails_closed_when_recovered_iio_uri_is_not_the_same_usb_device():
    pplus = _pplus()
    pplus._scan_iio_contexts = lambda: {
        "usb:1.4.5": f"stale address, serial={SERIAL}",
        "usb:1.9.5": "different serial, serial=OTHER",
    }

    with pytest.raises(DirectUsbRecoveryAttestationError) as raised:
        pplus._attest_direct_usb_reconnect()

    assert raised.value.differences[0].field == "iio_uri"
