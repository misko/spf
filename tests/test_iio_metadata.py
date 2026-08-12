import sys
import time
import types

import numpy as np

from spf.direct_radio.iio_metadata import IioMetadataRx
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
        flags=(
            GainObservationFlags.VALID
            | GainObservationFlags.SAMPLE_INTERVAL_VALID
        ),
        rx1_gain_index=42,
        rx2_gain_index=43,
        rx1_gain_db=20,
        rx2_gain_db=21,
    )
    return RadioMetadataV3(
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
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
        gain_observations=(observation,),
    )


class _FakeRxAdc:
    def __init__(self, metadata, sample_rate):
        self.metadata_frames = [metadata.pack()]
        self.sample_rate = sample_rate
        self.start_ns = time.monotonic_ns()
        self.start_counter = metadata.first_sample_sequence - sample_rate // 10
        self.register_reads = 0

    def reg_read(self, register):
        assert register == 0x800000B8
        self.register_reads += 1
        elapsed_ns = time.monotonic_ns() - self.start_ns
        return (self.start_counter + elapsed_ns * self.sample_rate // 1_000_000_000) & 0xFFFFFFFF


class _FakeMetadataBuffer:
    def __init__(self, device, samples, metadata_capacity):
        assert samples == 1024
        assert metadata_capacity == 64 * 1024
        self.device = device
        self.metadata = None
        self.refills = 0

    def refill(self):
        self.metadata = self.device.metadata_frames.pop(0)
        self.refills += 1
        return self.metadata


class _FakePyadiSdr:
    def __init__(self, metadata, sample_rate):
        self._rxadc = _FakeRxAdc(metadata, sample_rate)
        self._rxbuf = None
        self.ordinary_buffer_creations = 0

    def rx_destroy_buffer(self):
        self._rxbuf = None

    def _rx_init_channels(self):
        self.ordinary_buffer_creations += 1
        self._rxbuf = object()

    def rx(self):
        self._rxbuf.refill()
        return [
            np.zeros(1024, dtype=np.complex64),
            np.zeros(1024, dtype=np.complex64),
        ]


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
    assert len(signal) == 2
    assert parsed == metadata
    assert parsed.buffer_sequence == 3
    assert capture_time["sample_time_valid"] is True
    assert capture_time["sample_counter_end_exclusive"] == 1_001_024
    assert capture_time["sample_time_monotonic_end_ns"] > capture_time[
        "sample_time_monotonic_start_ns"
    ]
    assert capture_time["sample_time_uncertainty_ns"] >= 0
    assert fake_sdr._rxbuf.refills == 1
    assert fake_sdr._rxadc.register_reads == 9

    receiver.close()
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
        assert "0.25 or 0.26" in str(error)
    else:
        raise AssertionError("stock Python binding was accepted")


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
    radio.sdr = None
