from pathlib import Path

import numpy as np

from spf.dataset.v7_data import v7rx_keys, v7rx_new_dataset
from spf.scripts.recover_interrupted_v7 import (
    SOURCE_HASH_ALGORITHM,
    _sha256_file,
    recover_capture,
    valid_receiver_prefix,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


SAMPLES = 524288


class _Receiver(dict):
    def __getattr__(self, name):
        return self[name]


def _signal_frame():
    sample = np.arange(SAMPLES, dtype=np.float32)
    frame = np.empty((2, SAMPLES), dtype=np.complex64)
    frame[0] = sample + 1j * (sample % 17)
    frame[1] = (sample % 31) + 1j * (sample % 23)
    return frame


def _fake_receiver():
    receiver = _Receiver({name: None for name in v7rx_keys()})
    signal = np.zeros((2, 2, SAMPLES), dtype=np.complex64)
    signal[0] = _signal_frame()
    receiver.update(
        {
            "signal_matrix": signal,
            "system_timestamp": np.array([1000.0, 0.0]),
            "gain_metadata_valid": np.array([True, False]),
            "rssi_metadata_valid": np.array([True, False]),
            "gain_metadata_flags": np.array([0, 0], dtype=np.uint32),
            "gain_db_start": np.array([[40.0, 41.0], [0.0, 0.0]]),
            "gain_db_end": np.array([[40.0, 41.0], [0.0, 0.0]]),
            "rssi_db_start": np.array([[80.0, 81.0], [0.0, 0.0]]),
            "rssi_db_end": np.array([[80.0, 81.0], [0.0, 0.0]]),
            "gains": np.array([[40.0, 41.0], [0.0, 0.0]]),
            "rssis": np.array([[80.0, 81.0], [0.0, 0.0]]),
            "gain_endpoints_equal": np.array([[True, True], [False, False]]),
            "stream_id": np.array([1, 0], dtype=np.uint64),
            "buffer_sequence": np.array([0, 0], dtype=np.uint64),
            "sample_sequence": np.array([0, 0], dtype=np.uint64),
        }
    )
    return receiver


def _write_partial(path: Path):
    zarr = v7rx_new_dataset(
        filename=str(path),
        timesteps=2,
        buffer_size=SAMPLES,
        n_receivers=1,
        config={"data-version": 7, "n-records-per-receiver": 2},
        chunk_size=1,
        compressor=None,
    )
    zarr.attrs["capture_status"] = "incomplete"
    zarr.attrs["capture_records_written_by_receiver"] = [1]
    receiver = zarr.receivers.r0
    receiver.attrs["sdr_serial"] = "radio-a"
    fake = _fake_receiver()
    for key in v7rx_keys():
        if fake[key] is not None:
            receiver[key][:] = fake[key]
    zarr.store.close()


def test_valid_prefix_stops_before_unwritten_row_and_rejects_duplicate_channels():
    receiver = _fake_receiver()

    assert valid_receiver_prefix(receiver) == (1, "missing system timestamp")

    receiver["signal_matrix"][0, 1] = receiver["signal_matrix"][0, 0]
    assert valid_receiver_prefix(receiver) == (0, "duplicated RX channels")


def test_recovery_copies_verified_prefix_without_modifying_source(tmp_path):
    source = tmp_path / "capture.zarr.tmp"
    output = tmp_path / "capture.recovered.zarr"
    _write_partial(source)
    source_hash = _sha256_file(source / "data.mdb")
    validation_calls = []

    def validate(path, frames, receivers):
        validation_calls.append((path, frames, receivers))
        return {"status": "pass", "receiver_count": receivers}

    report = recover_capture(
        source,
        output,
        reason="unit-test interrupted capture",
        strict_validator=validate,
    )

    assert validation_calls == [
        (output.with_name(output.name + ".recovery.tmp"), 1, 1)
    ]
    assert report["common_prefix_records"] == 1
    assert report["detected_valid_records_by_receiver"] == [1]
    assert report["source_data_sha256"] == source_hash
    assert report["source_data_hash_algorithm"] == SOURCE_HASH_ALGORITHM
    assert _sha256_file(source / "data.mdb") == source_hash
    assert source.is_dir()
    assert output.is_dir()
    assert output.with_name(output.name + ".recovery.json").is_file()

    recovered = zarr_open_from_lmdb_store(str(output), mode="r")
    try:
        assert recovered.attrs["capture_status"] == "recovered_incomplete"
        assert recovered.attrs["capture_records_written_by_receiver"] == [1]
        assert recovered.attrs["recovery_source_data_sha256"] == source_hash
        assert (
            recovered.attrs["recovery_source_data_hash_algorithm"]
            == SOURCE_HASH_ALGORITHM
        )
        assert recovered.receivers.r0.signal_matrix.shape == (1, 2, SAMPLES)
        assert recovered.receivers.r0.attrs["sdr_serial"] == "radio-a"
    finally:
        recovered.store.close()
