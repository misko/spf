import numpy as np
import pytest

from spf.data_collector import ThreadedRXRawV6, ThreadedRXRawV7
from spf.sdrpluto.sdr_controller import (
    PlutoRxBuffer,
    ReceiverConfig,
    rx_config_from_receiver_yaml,
)


class _DirectOnlyPPlus:
    def __init__(self):
        self.rx_config = ReceiverConfig(
            lo=5_766_000_000,
            rf_bandwidth=3_000_000,
            sample_rate=30_000_000,
            intermediate=100_000,
            uri="pluto://usb:test",
            buffer_size=8,
            rx_spacing=0.05075,
            rx_transport="direct_usb",
        )
        self.calls = 0

    def rx_with_metadata(self):
        self.calls += 1
        return PlutoRxBuffer(
            signal_matrix=np.ones((2, 8), dtype=np.complex64),
            rssis=np.full(2, np.nan),
            gains=np.full(2, np.nan),
            gain_index_start=np.array([42, 43], dtype=np.uint8),
            gain_index_end=np.array([41, 43], dtype=np.uint8),
            gain_metadata_valid=True,
            gain_endpoints_equal=np.array([False, True]),
            gain_metadata_flags=0x417,
            stream_id=123,
            buffer_sequence=9,
            sample_sequence=72,
            gain_start_read_duration_ns=1100,
            gain_end_read_duration_ns=1200,
            first_gain_change_sample=np.array([-1, -1], dtype=np.int32),
            iq_power_dbfs=np.array([-30.0, -30.0], dtype=np.float32),
            gain_db_start=np.array([20.0, 40.0], dtype=np.float32),
            gain_db_end=np.array([20.0, 40.0], dtype=np.float32),
            rssi_db_start=np.array([80.0, 81.0], dtype=np.float32),
            rssi_db_end=np.array([80.25, 81.25], dtype=np.float32),
            rssi_metadata_valid=True,
            rssi_start_read_duration_ns=1300,
            rssi_end_read_duration_ns=1400,
        )

    def rssis(self):
        raise AssertionError("direct collector must not read remote RSSI")

    def gains(self):
        raise AssertionError("direct collector must not read remote gain")


def test_direct_collector_hot_path_preserves_metadata_without_attribute_reads():
    pplus = _DirectOnlyPPlus()
    reader = ThreadedRXRawV6(
        pplus=pplus,
        time_offset=0,
        nthetas=65,
    )
    snapshot = reader.get_data()

    assert pplus.calls == 1
    assert snapshot.signal_matrix.shape == (2, 8)
    assert snapshot.gain_metadata_valid
    np.testing.assert_array_equal(snapshot.gain_index_start, [42, 43])
    np.testing.assert_array_equal(snapshot.gain_index_end, [41, 43])
    np.testing.assert_array_equal(snapshot.gain_endpoints_equal, [False, True])
    assert snapshot.stream_id == 123
    assert snapshot.buffer_sequence == 9
    assert snapshot.sample_sequence == 72


def test_v7_collector_preserves_gain_rssi_and_stream_metadata():
    reader = ThreadedRXRawV7(
        pplus=_DirectOnlyPPlus(),
        time_offset=0,
        nthetas=65,
    )

    snapshot = reader.get_data()

    np.testing.assert_array_equal(snapshot.gains, [np.nan, np.nan])
    np.testing.assert_array_equal(snapshot.rssis, [np.nan, np.nan])
    np.testing.assert_array_equal(snapshot.gain_db_end, [20.0, 40.0])
    np.testing.assert_array_equal(snapshot.rssi_db_end, [80.25, 81.25])
    assert snapshot.gain_metadata_valid
    assert snapshot.rssi_metadata_valid
    assert snapshot.stream_id == 123
    assert snapshot.buffer_sequence == 9
    assert snapshot.sample_sequence == 72


def test_direct_collector_does_not_hide_a_failed_frame_with_retry():
    pplus = _DirectOnlyPPlus()

    def fail():
        pplus.calls += 1
        raise RuntimeError("synthetic USB sequence failure")

    pplus.rx_with_metadata = fail
    reader = ThreadedRXRawV6(
        pplus=pplus,
        time_offset=0,
        nthetas=65,
    )

    with pytest.raises(RuntimeError, match="synthetic USB sequence failure"):
        reader.get_rx()
    assert pplus.calls == 1


def test_receiver_yaml_supports_per_channel_manual_gains():
    receiver = {
        "f-carrier": 5_766_000_000,
        "bandwidth": 3_000_000,
        "f-sampling": 30_000_000,
        "rx-gain": -3,
        "rx-gains": [20, 40],
        "rx-gain-mode": "slow_attack",
        "rx-gain-modes": ["manual", "manual"],
        "buffer-size": 524288,
        "f-intermediate": 100000,
        "receiver-uri": "pluto://usb:test",
        "antenna-spacing-m": 0.05075,
        "theta-in-pis": 0,
        "rx-buffers": 4,
    }

    config = rx_config_from_receiver_yaml(receiver)

    assert config.gains == [20, 40]
    assert config.gain_control_modes == ["manual", "manual"]


def test_receiver_yaml_legacy_gain_is_still_applied_to_both_channels():
    receiver = {
        "f-carrier": 5_766_000_000,
        "bandwidth": 3_000_000,
        "f-sampling": 30_000_000,
        "rx-gain": -3,
        "rx-gain-mode": "slow_attack",
        "buffer-size": 524288,
        "f-intermediate": 100000,
        "receiver-uri": "pluto://usb:test",
        "antenna-spacing-m": 0.05075,
        "theta-in-pis": 0,
        "rx-buffers": 4,
    }

    config = rx_config_from_receiver_yaml(receiver)

    assert config.gains == [-3, -3]
    assert config.gain_control_modes == ["slow_attack", "slow_attack"]
