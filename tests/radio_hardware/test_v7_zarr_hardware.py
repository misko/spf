"""Hardware-backed protocol-v2 to V7 Zarr round-trip gate."""

from __future__ import annotations

import time

import numpy as np
import pytest

from spf.dataset.v7_data import v7rx_2x_keys, v7rx_scalar_keys, v7rx_new_dataset
from spf.sdrpluto.direct_usb_protocol import RadioMetadataV2
from spf.sdrpluto.direct_usb_receiver import (
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


pytestmark = [pytest.mark.radio_hardware, pytest.mark.radio_zarr]


def _iq_power_dbfs(signal: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(signal.astype(np.complex64)) ** 2, axis=1)
    with np.errstate(divide="ignore"):
        return (10.0 * np.log10(power / (2.0 * 2048.0**2))).astype(np.float32)


def _first_change(metadata: RadioMetadataV2) -> np.ndarray:
    return np.asarray(
        [
            -1
            if metadata.rx1_first_change_sample == 0xFFFFFFFF
            else metadata.rx1_first_change_sample,
            -1
            if metadata.rx2_first_change_sample == 0xFFFFFFFF
            else metadata.rx2_first_change_sample,
        ],
        dtype=np.int32,
    )


def test_v2_frames_round_trip_through_v7_zarr(attached_plutos, pytestconfig, tmp_path):
    samples = pytestconfig.getoption("--radio-samples")
    frame_count = pytestconfig.getoption("--radio-zarr-frames")
    assert frame_count > 0
    frames_by_radio = []
    for radio in attached_plutos:
        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            # Match the current collector lifecycle exactly: each Zarr record
            # is one finite START/STOP request. Queuing many 4 MiB records in
            # one libusb request is a different workload and can exceed the
            # host usbfs transfer-memory limit before streaming begins.
            radio_frames = []
            for _ in range(frame_count):
                capture = receiver.capture(
                    samples_per_channel=samples,
                    frame_count=1,
                )
                assert len(capture.frames) == 1
                radio_frames.append(capture.frames[0])
            frames_by_radio.append(tuple(radio_frames))

    path = tmp_path / "hardware_v7.zarr"
    zarr = v7rx_new_dataset(
        filename=str(path),
        timesteps=frame_count,
        buffer_size=samples,
        n_receivers=len(attached_plutos),
        config={
            "data-version": 7,
            "test": "attached-radio protocol-v2 Zarr round trip",
        },
        chunk_size=1,
        compressor=None,
    )
    zarr.attrs["capture_status"] = "in_progress"
    zarr.attrs["capture_records_written_by_receiver"] = [0] * len(attached_plutos)
    try:
        for receiver_index, (radio, radio_frames) in enumerate(
            zip(attached_plutos, frames_by_radio, strict=True)
        ):
            receiver = zarr[f"receivers/r{receiver_index}"]
            receiver.attrs["sdr_family"] = "pluto"
            receiver.attrs["sdr_serial"] = radio.serial
            receiver.attrs["usb_port_path"] = list(radio.port_path)
            for frame_index, frame in enumerate(radio_frames):
                metadata = frame.metadata
                assert isinstance(metadata, RadioMetadataV2)
                assert metadata.gain_metadata_valid
                assert metadata.rssi_metadata_valid
                signal = iq_payload_to_complex64(frame.iq_payload, samples)
                receiver["signal_matrix"][frame_index] = signal

                scalar_values = {
                    "gain_metadata_valid": metadata.gain_metadata_valid,
                    "rssi_metadata_valid": metadata.rssi_metadata_valid,
                    "gain_metadata_flags": int(metadata.flags),
                    "stream_id": metadata.stream_id,
                    "buffer_sequence": metadata.buffer_sequence,
                    "sample_sequence": metadata.first_sample_sequence,
                    "gain_start_read_duration_ns": (
                        metadata.gain_start_read_duration_ns
                    ),
                    "gain_end_read_duration_ns": metadata.gain_end_read_duration_ns,
                    "rssi_start_read_duration_ns": (
                        metadata.rssi_start_read_duration_ns
                    ),
                    "rssi_end_read_duration_ns": metadata.rssi_end_read_duration_ns,
                }
                two_values = {
                    "gain_db_start": metadata.gain_db_start,
                    "gain_db_end": metadata.gain_db_end,
                    "rssi_db_start": metadata.rssi_db_start,
                    "rssi_db_end": metadata.rssi_db_end,
                    "gain_endpoints_equal": metadata.gain_endpoints_equal,
                    "first_gain_change_sample": _first_change(metadata),
                    "iq_power_dbfs": _iq_power_dbfs(signal),
                }
                for key in v7rx_scalar_keys:
                    receiver[key][frame_index] = scalar_values[key]
                for key in v7rx_2x_keys:
                    receiver[key][frame_index] = two_values[key]
                receiver["system_timestamp"][frame_index] = time.time()
                receiver["rssis"][frame_index] = metadata.rssi_db_end
                receiver["gains"][frame_index] = metadata.gain_db_end

        zarr.attrs["capture_records_written_by_receiver"] = [frame_count] * len(
            attached_plutos
        )
        zarr.attrs["capture_status"] = "complete"
    finally:
        zarr.store.close()

    reopened = zarr_open_from_lmdb_store(str(path), mode="r")
    try:
        assert reopened.attrs["radio_metadata_schema_version"] == 2
        assert reopened.attrs["capture_status"] == "complete"
        assert reopened.attrs["capture_records_written_by_receiver"] == [
            frame_count
        ] * len(attached_plutos)
        for receiver_index, radio in enumerate(attached_plutos):
            receiver = reopened[f"receivers/r{receiver_index}"]
            assert receiver.attrs["sdr_serial"] == radio.serial
            assert tuple(receiver.attrs["usb_port_path"]) == radio.port_path
            assert receiver["signal_matrix"].shape == (frame_count, 2, samples)
            assert receiver["signal_matrix"].dtype == np.dtype("complex64")
            assert receiver["gain_metadata_valid"][:].all()
            assert receiver["rssi_metadata_valid"][:].all()
            assert np.isfinite(receiver["gain_db_end"][:]).all()
            assert np.isfinite(receiver["rssi_db_end"][:]).all()
            assert np.any(receiver["signal_matrix"][:] != 0)
            # Every record is a new finite stream generation today, so its
            # within-stream buffer/sample sequences both begin at zero. The
            # stream ID disambiguates those generations in V7.
            np.testing.assert_array_equal(
                receiver["buffer_sequence"][:], np.zeros(frame_count)
            )
            np.testing.assert_array_equal(
                receiver["sample_sequence"][:], np.zeros(frame_count)
            )
            assert len(set(receiver["stream_id"][:].tolist())) == frame_count
    finally:
        reopened.store.close()
