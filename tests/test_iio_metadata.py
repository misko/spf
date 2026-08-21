import errno
import sys
import time
import types

import numpy as np
import pytest

from spf.direct_radio import iio_metadata
from spf.direct_radio.iio_metadata import IioMetadataRx
from spf.direct_radio.tandem_agc import (
    RadioMetadataV5,
    TandemGainTable,
    TandemState,
)
from spf.direct_radio.usb_protocol import (
    FIRST_CHANGE_UNAVAILABLE,
    GainObservationFlags,
    GainObservationV3,
    MetadataFeatures,
    MetadataFlags,
    RadioMetadataV3,
    SampleFormat,
)


def _metadata(samples=1024, first_sample_sequence=1_000_000):
    observation = GainObservationV3(
        sample_sequence_before=first_sample_sequence - 32,
        sample_sequence_after=first_sample_sequence + samples,
        read_duration_ns=12_000,
        flags=(GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID),
        rx1_gain_index=42,
        rx2_gain_index=43,
        rx1_gain_db=20,
        rx2_gain_db=21,
    )
    base = RadioMetadataV3(
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
            | MetadataFeatures.FPGA_GAIN_EVENTS
            | MetadataFeatures.GAIN_DB_ENDPOINTS
            | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.GAIN_OBSERVATION_SERIES
            | MetadataFeatures.HARDWARE_SAMPLE_COUNTER
        ),
        flags=(
            MetadataFlags.START_VALID
            | MetadataFlags.END_VALID
            | MetadataFlags.SAMPLE_SEQUENCE_VALID
            | MetadataFlags.GAIN_FULL_TABLE_MODE
            | MetadataFlags.GAIN_DB_VALUES
            | MetadataFlags.RSSI_START_VALID
            | MetadataFlags.RSSI_END_VALID
            | MetadataFlags.GAIN_OBSERVATIONS_VALID
            | MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
        ),
        stream_id=7,
        buffer_sequence=3,
        first_sample_sequence=first_sample_sequence,
        samples_per_channel=samples,
        iq_payload_bytes=samples * 8,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_db_start=20,
        rx2_gain_db_start=21,
        rx1_gain_db_end=20,
        rx2_gain_db_end=21,
        gain_start_read_duration_ns=12_000,
        gain_end_read_duration_ns=12_000,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx1_rssi_start_qdb=200,
        rx2_rssi_start_qdb=204,
        rx1_rssi_end_qdb=208,
        rx2_rssi_end_qdb=212,
        rssi_start_read_duration_ns=13_000,
        rssi_end_read_duration_ns=13_000,
        gain_observation_interval_samples=samples,
        gain_observation_capacity=1,
        gain_event_capacity=64,
        gain_observations=(observation,),
    )
    return RadioMetadataV5(
        base=base,
        header_bytes=base.header_bytes + 56,
        ownership_epoch=7,
        tandem_state=TandemState.ARMED_AUTO,
        tandem_fault_flags=0,
        tandem_transition_count=0,
        gain_table_id=TandemGainTable.MHZ_1300_4000,
        threshold_provenance=0x30313A14,
        minimum_gain_db=0,
        maximum_gain_db=62,
        initial_gain_db=20,
        minimum_gain_index=0,
        maximum_gain_index=76,
        rx1_gain_index=42,
        rx2_gain_index=42,
        gain_events=(),
        ad9361_temperature_mdeg_c=43_860,
    )


class _FakeRxAdc:
    def __init__(self, metadata, sample_rate, events):
        self.metadata_frames = [metadata.pack()]
        self.sample_rate = sample_rate
        self.start_ns = time.monotonic_ns()
        self.start_counter = metadata.first_sample_sequence - sample_rate // 10
        self.register_reads = 0
        self.events = events

    def reg_read(self, register):
        assert register == 0x800000B8
        self.register_reads += 1
        elapsed_ns = time.monotonic_ns() - self.start_ns
        return (
            self.start_counter + elapsed_ns * self.sample_rate // 1_000_000_000
        ) & 0xFFFFFFFF


class _FakeMetadataBuffer:
    def __init__(self, device, samples, request, metadata_capacity):
        assert samples == 1024
        assert len(request) == 104
        assert metadata_capacity == 64 * 1024
        device.events.append("metadata_construct")
        self.device = device
        self.metadata = None
        self.refills = 0
        self.close_calls = 0
        device.owner.metadata_buffers.append(self)

    def refill(self):
        self.device.events.append("metadata_refill")
        self.metadata = self.device.metadata_frames.pop(0)
        self.refills += 1
        return self.metadata

    def close(self):
        self.close_calls += 1
        self.device.events.append("metadata_close_start")
        self.device.events.append("metadata_close_end")


class _FakeOrdinaryBuffer:
    def __init__(self, events):
        self.events = events
        self.close_calls = 0

    def refill(self):
        self.events.append("ordinary_refill")

    def close(self):
        self.close_calls += 1
        self.events.append("ordinary_close_start")
        self.events.append("ordinary_close_end")


class _FakePyadiSdr:
    def __init__(
        self,
        metadata,
        sample_rate,
        *,
        ordinary_signal=None,
        metadata_signal=None,
    ):
        self.events = []
        self._rxadc = _FakeRxAdc(metadata, sample_rate, self.events)
        self._rxadc.owner = self
        self._rxbuf = None
        self.ordinary_buffer_creations = 0
        self.ordinary_buffers = []
        self.metadata_buffers = []
        indices = np.arange(1024, dtype=np.float32)
        self.ordinary_signal = (
            ordinary_signal
            if ordinary_signal is not None
            else [
                (indices + 1j * (indices + 1)).astype(np.complex64),
                ((indices + 2) + 1j * (indices + 3)).astype(np.complex64),
            ]
        )
        self.metadata_signal = (
            metadata_signal
            if metadata_signal is not None
            else [
                ((indices + 11) + 1j * (indices + 13)).astype(np.complex64),
                ((indices + 17) + 1j * (indices + 19)).astype(np.complex64),
            ]
        )

    def rx_destroy_buffer(self):
        if isinstance(self._rxbuf, _FakeOrdinaryBuffer):
            self.events.append("ordinary_destroy")
        elif isinstance(self._rxbuf, _FakeMetadataBuffer):
            self.events.append("metadata_destroy")
        self._rxbuf = None

    def _rx_init_channels(self):
        self.ordinary_buffer_creations += 1
        self.events.append("ordinary_construct")
        self._rxbuf = _FakeOrdinaryBuffer(self.events)
        self.ordinary_buffers.append(self._rxbuf)

    def rx(self):
        if self._rxbuf is None:
            self._rx_init_channels()
        self._rxbuf.refill()
        if isinstance(self._rxbuf, _FakeMetadataBuffer):
            return self.metadata_signal
        return self.ordinary_signal


def test_iio_metadata_adapter_reuses_pyadi_iq_and_returns_capture_time(monkeypatch):
    metadata = _metadata()
    fake_sdr = _FakePyadiSdr(metadata, sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=_FakeMetadataBuffer)
    )

    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )
    receiver.open()
    signal, parsed, capture_time = receiver.capture()

    assert fake_sdr.ordinary_buffer_creations == 1
    assert fake_sdr.events.count("ordinary_refill") == 1
    np.testing.assert_array_equal(signal, np.vstack(fake_sdr.metadata_signal))
    assert not np.array_equal(
        np.vstack(fake_sdr.ordinary_signal), np.vstack(fake_sdr.metadata_signal)
    )
    assert fake_sdr.events[:5] == [
        "ordinary_construct",
        "ordinary_refill",
        "ordinary_destroy",
        "ordinary_close_start",
        "ordinary_close_end",
    ]
    assert fake_sdr.events[5:7] == [
        "metadata_construct",
        "metadata_refill",
    ]
    assert fake_sdr.events.index("ordinary_close_end") < fake_sdr.events.index(
        "metadata_construct"
    )
    assert parsed == metadata
    assert parsed.buffer_sequence == 3
    assert capture_time["sample_time_valid"] is True
    assert capture_time["sample_counter_end_exclusive"] == 1_001_024
    assert (
        capture_time["sample_time_monotonic_end_ns"]
        > capture_time["sample_time_monotonic_start_ns"]
    )
    assert capture_time["sample_time_uncertainty_ns"] >= 0
    assert fake_sdr._rxbuf.refills == 1
    assert fake_sdr._rxadc.register_reads == 9

    metadata_buffer = fake_sdr._rxbuf
    receiver.close()
    receiver.close()
    assert fake_sdr._rxbuf is None
    assert fake_sdr.ordinary_buffers[0].close_calls == 1
    assert metadata_buffer.close_calls == 1


def test_iio_metadata_adapter_cleans_up_failed_ordinary_prime(monkeypatch):
    class FailingPrimeSdr(_FakePyadiSdr):
        def rx(self):
            if self._rxbuf is None:
                self._rx_init_channels()
            if isinstance(self._rxbuf, _FakeOrdinaryBuffer):
                self._rxbuf.refill()
                raise OSError(errno.EIO, "ordinary prime failed")
            return super().rx()

    fake_sdr = FailingPrimeSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=_FakeMetadataBuffer)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    with pytest.raises(OSError, match="ordinary prime failed"):
        receiver.open()

    assert fake_sdr._rxbuf is None
    assert fake_sdr.events == [
        "ordinary_construct",
        "ordinary_refill",
        "ordinary_destroy",
        "ordinary_close_start",
        "ordinary_close_end",
    ]
    assert fake_sdr.ordinary_buffers[0].close_calls == 1


def test_iio_metadata_adapter_rejects_wrong_ordinary_prime_shape(monkeypatch):
    wrong_shape = [np.ones(1024, dtype=np.complex64)]
    fake_sdr = _FakePyadiSdr(
        _metadata(),
        sample_rate=2_000_000,
        ordinary_signal=wrong_shape,
    )
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=_FakeMetadataBuffer)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    with pytest.raises(RuntimeError, match="dual-channel complex IQ"):
        receiver.open()

    assert fake_sdr._rxbuf is None
    assert "metadata_construct" not in fake_sdr.events
    assert fake_sdr.ordinary_buffers[0].close_calls == 1


def test_iio_metadata_adapter_rejects_known_radio_20_constant_prime(monkeypatch):
    held_rx0 = np.full(1024, 1549 + 137j, dtype=np.complex64)
    held_rx1 = np.zeros(1024, dtype=np.complex64)
    fake_sdr = _FakePyadiSdr(
        _metadata(),
        sample_rate=2_000_000,
        ordinary_signal=[held_rx0, held_rx1],
    )
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=_FakeMetadataBuffer)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    with pytest.raises(RuntimeError) as raised:
        receiver.open()

    assert str(raised.value) == "ordinary IIO prime has a constant IQ component"
    assert "metadata_construct" not in fake_sdr.events
    assert fake_sdr._rxbuf is None
    assert fake_sdr.ordinary_buffers[0].close_calls == 1


@pytest.mark.parametrize("component_index", range(4))
def test_iio_metadata_adapter_rejects_each_constant_prime_component(
    monkeypatch, component_index
):
    indices = np.arange(1024, dtype=np.float32)
    signal = np.vstack(
        (
            indices + 1j * (indices + 1),
            (indices + 2) + 1j * (indices + 3),
        )
    ).astype(np.complex64)
    components = (
        signal[0].real,
        signal[0].imag,
        signal[1].real,
        signal[1].imag,
    )
    components[component_index][:] = 23
    fake_sdr = _FakePyadiSdr(
        _metadata(),
        sample_rate=2_000_000,
        ordinary_signal=[signal[0], signal[1]],
    )
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=_FakeMetadataBuffer)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    with pytest.raises(RuntimeError) as raised:
        receiver.open()

    assert str(raised.value) == "ordinary IIO prime has a constant IQ component"
    assert "metadata_construct" not in fake_sdr.events
    assert fake_sdr.ordinary_buffers[0].close_calls == 1


def test_iio_metadata_adapter_closes_metadata_buffer_when_anchor_setup_fails(
    monkeypatch,
):
    fake_sdr = _FakePyadiSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=_FakeMetadataBuffer)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    def fail_anchor(*, initial):
        assert initial is True
        raise OSError(errno.EIO, "anchor setup failed")

    monkeypatch.setattr(receiver, "_refresh_time_anchors", fail_anchor)

    with pytest.raises(OSError, match="anchor setup failed"):
        receiver.open()

    metadata_buffer = next(
        item
        for item in fake_sdr.ordinary_buffers + fake_sdr.metadata_buffers
        if isinstance(item, _FakeMetadataBuffer)
    )
    assert metadata_buffer.close_calls == 1
    assert fake_sdr._rxbuf is None
    assert receiver.is_open is False


def test_iio_metadata_adapter_retains_legacy_buffer_cleanup_fallback(monkeypatch):
    class LegacyOrdinaryBuffer:
        def __init__(self, events):
            self.events = events

        def refill(self):
            self.events.append("legacy_ordinary_refill")

    class LegacyMetadataBuffer:
        def __init__(self, device, samples, request, metadata_capacity):
            assert samples == 1024
            assert len(request) == 104
            assert metadata_capacity == 64 * 1024
            self.device = device
            self.metadata = None

        def refill(self):
            self.metadata = self.device.metadata_frames.pop(0)
            return self.metadata

    class LegacyBufferSdr(_FakePyadiSdr):
        def _rx_init_channels(self):
            self.ordinary_buffer_creations += 1
            self._rxbuf = LegacyOrdinaryBuffer(self.events)

        def rx(self):
            if self._rxbuf is None:
                self._rx_init_channels()
            self._rxbuf.refill()
            if isinstance(self._rxbuf, LegacyMetadataBuffer):
                return self.metadata_signal
            return self.ordinary_signal

    fake_sdr = LegacyBufferSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=LegacyMetadataBuffer)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    receiver.open()
    receiver.close()

    assert fake_sdr._rxbuf is None
    assert receiver.is_open is False


def test_iio_metadata_adapter_retries_bounded_ebusy_open(monkeypatch):
    class BusyThenOpen(_FakeMetadataBuffer):
        attempts = 0

        def __init__(self, device, samples, request, metadata_capacity):
            type(self).attempts += 1
            device.events.append("metadata_open_attempt")
            if self.attempts < iio_metadata._METADATA_OPEN_MAX_ATTEMPTS:
                raise OSError(errno.EBUSY, "prior buffer teardown is incomplete")
            super().__init__(device, samples, request, metadata_capacity)

    delays = []
    monkeypatch.setattr(iio_metadata.time, "sleep", delays.append)
    fake_sdr = _FakePyadiSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=BusyThenOpen)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    receiver.open()

    assert BusyThenOpen.attempts == iio_metadata._METADATA_OPEN_MAX_ATTEMPTS
    assert delays.count(iio_metadata._METADATA_OPEN_RETRY_DELAY_SECONDS) == 2
    assert fake_sdr.events.index("ordinary_destroy") < fake_sdr.events.index(
        "metadata_construct"
    )
    receiver.close()


@pytest.mark.parametrize("failure_errno", [errno.EIO, errno.ETIMEDOUT])
def test_iio_metadata_adapter_does_not_retry_non_ebusy_open(monkeypatch, failure_errno):
    class FailingOpen:
        attempts = 0

        def __init__(self, device, samples, request, metadata_capacity):
            type(self).attempts += 1
            raise OSError(failure_errno, "metadata open failed")

    delays = []
    monkeypatch.setattr(iio_metadata.time, "sleep", delays.append)
    fake_sdr = _FakePyadiSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=FailingOpen)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    with pytest.raises(OSError) as raised:
        receiver.open()

    assert raised.value.errno == failure_errno
    assert FailingOpen.attempts == 1
    assert iio_metadata._METADATA_OPEN_RETRY_DELAY_SECONDS not in delays
    assert fake_sdr._rxbuf is None


def test_iio_metadata_adapter_exhausts_bounded_ebusy_open(monkeypatch):
    class AlwaysBusy:
        attempts = 0

        def __init__(self, device, samples, request, metadata_capacity):
            type(self).attempts += 1
            raise OSError(errno.EBUSY, "metadata owner remains active")

    delays = []
    monkeypatch.setattr(iio_metadata.time, "sleep", delays.append)
    fake_sdr = _FakePyadiSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=AlwaysBusy)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )

    with pytest.raises(OSError) as raised:
        receiver.open()

    assert raised.value.errno == errno.EBUSY
    assert AlwaysBusy.attempts == iio_metadata._METADATA_OPEN_MAX_ATTEMPTS
    assert delays.count(iio_metadata._METADATA_OPEN_RETRY_DELAY_SECONDS) == 2
    assert fake_sdr._rxbuf is None


def test_iio_metadata_adapter_requires_patched_python_binding(monkeypatch):
    fake_sdr = _FakePyadiSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(sys.modules, "iio", types.SimpleNamespace())
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )
    try:
        receiver.open()
    except RuntimeError as error:
        assert "exact patched SPF tandem libiio 0.25" in str(error)
    else:
        raise AssertionError("stock Python binding was accepted")


def test_iio_metadata_adapter_retries_only_typed_startup_discards(monkeypatch):
    class StartupDiscardSdr(_FakePyadiSdr):
        def __init__(self, metadata, sample_rate):
            super().__init__(metadata, sample_rate)
            self.attempts = 0

        def rx(self):
            if isinstance(self._rxbuf, _FakeMetadataBuffer):
                self.attempts += 1
                if self.attempts <= 2:
                    raise OSError(errno.EAGAIN, "startup frame lacks metadata")
            return super().rx()

    fake_sdr = StartupDiscardSdr(_metadata(), sample_rate=2_000_000)
    monkeypatch.setitem(
        sys.modules, "iio", types.SimpleNamespace(MetadataBuffer=_FakeMetadataBuffer)
    )
    receiver = IioMetadataRx(
        fake_sdr,
        sample_rate_hz=2_000_000,
        samples_per_channel=1024,
    )
    receiver.open()
    _signal, parsed, _capture_time = receiver.capture()
    assert parsed.buffer_sequence == 3
    assert fake_sdr.attempts == 3
    receiver.close()


def test_pplus_rx_with_metadata_does_not_poll_host_gain_or_rssi():
    from spf.sdrpluto.sdr_controller import PPlus

    metadata = _metadata()
    signal = np.zeros((2, 1024), dtype=np.complex64)
    sample_time = {
        "sample_counter_end_exclusive": 1_001_024,
        "sample_time_valid": True,
        "sample_time_monotonic_start_ns": 10,
        "sample_time_monotonic_end_ns": 20,
        "sample_time_realtime_start_ns": 30,
        "sample_time_realtime_end_ns": 40,
        "sample_time_uncertainty_ns": 5,
        "sample_time_fitted_rate_hz": 2_000_000.0,
        "sample_time_anchor_count": 9,
        "sample_time_max_round_trip_ns": 100,
        "sample_time_rate_tolerance_ppm": 100.0,
    }

    class Adapter:
        def capture(self):
            return signal, metadata, sample_time

    class HostAttributesMustNotBeRead:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected host-side attribute read: {name}")

    radio = PPlus.__new__(PPlus)
    radio.rx_config = types.SimpleNamespace(rx_transport="iio")
    radio._iio_metadata_rx = Adapter()
    radio._last_direct_gains = None
    radio._last_direct_rssis = None
    radio._last_direct_metadata = None
    radio._last_direct_sample_time = None
    radio.sdr = HostAttributesMustNotBeRead()

    frame = radio.rx_with_metadata()
    np.testing.assert_array_equal(frame.signal_matrix, signal)
    np.testing.assert_array_equal(frame.gains, [20.0, 21.0])
    np.testing.assert_array_equal(frame.rssis, [52.0, 53.0])
    assert frame.buffer_sequence == 3
    assert frame.sample_sequence == 1_000_000
    assert frame.sample_time_valid is True
    assert frame.gain_observation_index.tolist() == [[42, 43]]
    assert frame.ad9361_temperature_mdeg_c == 43_860
    radio.sdr = None
