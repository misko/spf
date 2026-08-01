from types import SimpleNamespace

import numpy as np
import pytest

from spf.scripts.validate_direct_usb_v7_zarr import _validate_receiver


SAMPLES = 524_288


def _receiver():
    sample_index = np.arange(SAMPLES, dtype=np.float32)
    signal = np.empty((1, 2, SAMPLES), dtype=np.complex64)
    signal[0, 0] = sample_index + 1j * (sample_index % 17)
    signal[0, 1] = (sample_index % 31) + 1j * (sample_index % 23)
    gain = np.array([[42.0, 43.0]], dtype=np.float32)
    rssi = np.array([[80.0, 81.0]], dtype=np.float32)
    return SimpleNamespace(
        signal_matrix=signal,
        gain_metadata_valid=np.ones(1, dtype=bool),
        rssi_metadata_valid=np.ones(1, dtype=bool),
        gain_db_start=gain.copy(),
        gain_db_end=gain.copy(),
        rssi_db_start=rssi.copy(),
        rssi_db_end=rssi.copy(),
        gains=gain.copy(),
        rssis=rssi.copy(),
        gain_metadata_flags=np.zeros(1, dtype=np.uint16),
        gain_endpoints_equal=np.ones((1, 2), dtype=bool),
        stream_id=np.array([1], dtype=np.uint64),
        buffer_sequence=np.array([0], dtype=np.uint64),
        sample_sequence=np.array([0], dtype=np.uint64),
        system_timestamp=np.array([1.0], dtype=np.float64),
        attrs={"sdr_serial": "radio-a", "usb_port_path": [1, 2]},
    )


def test_receiver_validation_accepts_two_active_distinct_channels():
    report = _validate_receiver(_receiver(), expected_frames=1)

    assert report["frames"] == 1


def test_receiver_validation_rejects_duplicated_rx_channel():
    receiver = _receiver()
    receiver.signal_matrix[0, 1] = receiver.signal_matrix[0, 0]

    with pytest.raises(ValueError, match="duplicated RX channels"):
        _validate_receiver(receiver, expected_frames=1)


@pytest.mark.parametrize("channel", (0, 1))
def test_receiver_validation_rejects_stuck_nonzero_channel(channel):
    receiver = _receiver()
    receiver.signal_matrix[0, channel] = np.complex64(7 + 3j)

    with pytest.raises(ValueError, match=rf"channel {channel} is constant"):
        _validate_receiver(receiver, expected_frames=1)
