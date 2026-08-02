from types import SimpleNamespace

import numpy as np

from spf.sdrpluto.direct_usb_protocol import (
    FIRST_CHANGE_UNAVAILABLE,
    DirectUsbRxFrame,
    MetadataFeatures,
    MetadataFlags,
    RadioMetadataV2,
    SampleFormat,
)
from spf.sdrpluto.sdr_controller import PPlus


class _OneFrameReceiver:
    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def capture(self, *, samples_per_channel, frame_count):
        self.calls += 1
        assert samples_per_channel == 8
        assert frame_count == 1
        return SimpleNamespace(
            frames=(self.frame,),
            recovered_after_transport_loss=False,
            transport_loss_summary=None,
        )


def _metadata_v2():
    return RadioMetadataV2(
        features=(
            MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
            | MetadataFeatures.HEADER_CRC32
            | MetadataFeatures.SAMPLE_SEQUENCE
            | MetadataFeatures.GAIN_DB_ENDPOINTS
            | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
        ),
        flags=(
            MetadataFlags.START_VALID
            | MetadataFlags.END_VALID
            | MetadataFlags.SAMPLE_SEQUENCE_VALID
            | MetadataFlags.GAIN_FULL_TABLE_MODE
            | MetadataFlags.GAIN_DB_VALUES
            | MetadataFlags.RSSI_START_VALID
            | MetadataFlags.RSSI_END_VALID
        ),
        stream_id=1,
        buffer_sequence=0,
        first_sample_sequence=0,
        samples_per_channel=8,
        iq_payload_bytes=64,
        enabled_scan_mask=0x0F,
        sample_format=SampleFormat.CS16_LE_TIME_INTERLEAVED,
        channel_count=2,
        rx1_gain_db_start=20,
        rx2_gain_db_start=40,
        rx1_gain_db_end=20,
        rx2_gain_db_end=40,
        rx1_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx2_first_change_sample=FIRST_CHANGE_UNAVAILABLE,
        rx1_rssi_start_qdb=401,
        rx2_rssi_start_qdb=402,
        rx1_rssi_end_qdb=403,
        rx2_rssi_end_qdb=404,
    )


def _pplus_with_v2_frame():
    pplus = object.__new__(PPlus)
    pplus.rx_config = SimpleNamespace(
        rx_transport="direct_usb",
        buffer_size=8,
        direct_usb_require_gain_metadata=True,
    )
    pplus._last_direct_gains = None
    pplus._last_direct_rssis = None
    pplus._last_direct_metadata = None
    payload = np.zeros((8, 4), dtype="<i2").tobytes()
    frame = DirectUsbRxFrame(metadata=_metadata_v2(), iq_payload=payload)
    pplus.direct_rx = _OneFrameReceiver(frame)
    # Any accidental legacy IIO read would fail: this object has no .sdr.
    return pplus


def test_existing_rx_gains_rssis_interface_uses_same_frame_without_iio_reads():
    pplus = _pplus_with_v2_frame()

    signal_matrix = pplus.rx()
    gains = pplus.gains()
    rssis = pplus.rssis()

    assert pplus.direct_rx.calls == 1
    assert signal_matrix.shape == (2, 8)
    assert gains.dtype == np.float64
    assert rssis.dtype == np.float64
    np.testing.assert_array_equal(gains, [20.0, 40.0])
    np.testing.assert_array_equal(rssis, [100.75, 101.0])


def test_cached_values_are_not_exposed_for_mutation():
    pplus = _pplus_with_v2_frame()
    pplus.rx()
    gains = pplus.gains()
    rssis = pplus.rssis()
    gains[0] = -999
    rssis[0] = -999
    np.testing.assert_array_equal(pplus.gains(), [20.0, 40.0])
    np.testing.assert_array_equal(pplus.rssis(), [100.75, 101.0])


def test_v2_metadata_result_preserves_legacy_and_stream_values():
    pplus = _pplus_with_v2_frame()

    result = pplus.rx_with_metadata()

    np.testing.assert_array_equal(result.gains, [20.0, 40.0])
    np.testing.assert_array_equal(result.rssis, [100.75, 101.0])
    np.testing.assert_array_equal(result.gain_db_start, [20.0, 40.0])
    np.testing.assert_array_equal(result.gain_db_end, [20.0, 40.0])
    np.testing.assert_array_equal(result.rssi_db_start, [100.25, 100.5])
    np.testing.assert_array_equal(result.rssi_db_end, [100.75, 101.0])
    assert result.gain_metadata_valid
    assert result.rssi_metadata_valid
    assert result.stream_id == 1
    assert result.buffer_sequence == 0
    assert result.sample_sequence == 0


def test_compatibility_values_fail_closed_before_first_frame():
    pplus = _pplus_with_v2_frame()
    for accessor in (pplus.gains, pplus.rssis):
        try:
            accessor()
        except RuntimeError as exc:
            assert "call rx() before" in str(exc)
        else:
            raise AssertionError("uncached direct metadata was accepted")
