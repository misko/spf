import tempfile

import numpy as np
import pytest
from compress_pickle import dump

from spf.dataset.spf_dataset import SessionsDatasetSimulated
from spf.dataset.spf_generate import generate_session_and_dump
from spf.dataset.v4_data import v4rx_2xf64_keys, v4rx_f64_keys, v4rx_new_dataset
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys, v5rx_new_dataset
from spf.dataset.v6_data import (
    v6rx_2x_keys,
    v6rx_scalar_keys,
    v6rx_new_dataset,
)
from spf.dataset.v7_data import (
    v7rx_2x_keys,
    v7rx_keys,
    v7rx_sample_time_scalar_keys,
    v7rx_scalar_keys,
    v7rx_new_dataset,
)
from spf.rf import get_peaks_for_2rx
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store, zarr_shrink
from spf.utils import dotdict, random_signal_matrix


@pytest.fixture
def default_args():
    return dotdict(
        {
            "carrier_frequency": 2.4e9,
            "signal_frequency": 100e3,
            "sampling_frequency": 10e6,
            "array_type": "linear",  # "circular"],
            "elements": 11,
            "random_silence": False,
            "detector_noise": 1e-4,
            "random_emitter_timing": False,
            "sources": 2,
            "seed": 0,
            "beam_former_spacing": 256 + 1,
            "width": 128,
            "detector_trajectory": "bounce",
            "detector_speed": 10.0,
            "source_speed": 0.0,
            "sigma_noise": 1.0,
            "time_steps": 1024,
            "time_interval": 0.3,
            "readings_per_snapshot": 3,
            "sessions": 2,
            "reference": False,
            "cpus": 2,
            "live": False,
            "profile": False,
            "fixed_detector": None,  #
        }
    )


def test_data_generation(default_args):
    with tempfile.TemporaryDirectory() as tmp:
        args = default_args
        args.output = tmp
        dump(
            args,
            "/".join([args.output, "args.pkl"]),
            compression="lzma",
        )
        result = [  # noqa
            generate_session_and_dump((args, session_idx))
            for session_idx in range(args.sessions)
        ]
        ds = SessionsDatasetSimulated(root_dir=tmp, snapshots_per_session=1024)
        session = ds[1]
        dump(
            session,
            "/".join([args.output, "onesession.pkl"]),
            compression="lzma",
        )


def test_closeness_to_ground_truth(default_args):
    with tempfile.TemporaryDirectory() as tmp:
        args = default_args
        args.output = tmp
        args.live = True
        args.sources = 1
        args.elements = 2
        args.detector_speed = 0.0
        args.source_speed = 10.0
        args.sigma_noise = 0.0
        args.detector_noise = 0.0
        args.beam_former_spacing = 4096 + 1
        dump(
            args,
            "/".join([args.output, "args.pkl"]),
            compression="lzma",
        )
        ds = SessionsDatasetSimulated(root_dir=tmp, snapshots_per_session=1024)
        session = ds[1]
        peaks_at_t = np.array(
            [
                get_peaks_for_2rx(bf_out)
                for bf_out in session["beam_former_outputs_at_t"]
            ]
        )
        peaks_at_t_in_radians = (
            2 * (peaks_at_t / args.beam_former_spacing - 0.5) * np.pi
        )
        peaks_at_t_in_radians_adjusted = (
            peaks_at_t_in_radians + session["detector_orientation_at_t"]
        )
        ground_truth = (
            session["detector_orientation_at_t"] + session["source_theta_at_t"]
        )
        deviation = (
            np.abs(peaks_at_t_in_radians_adjusted - ground_truth).min(axis=1).mean()
        )
        assert deviation < 0.01


def test_live_data_generation(default_args):
    with tempfile.TemporaryDirectory() as tmp:
        args = default_args
        args.output = tmp
        args.live = True
        dump(
            args,
            "/".join([args.output, "args.pkl"]),
            compression="lzma",
        )
        result = [  # noqa
            generate_session_and_dump((args, session_idx))
            for session_idx in range(args.sessions)
        ]
        ds = SessionsDatasetSimulated(root_dir=tmp, snapshots_per_session=1024)
        session = ds[1]
        dump(
            session,
            "/".join([args.output, "onesession.pkl"]),
            compression="lzma",
        )


def testv4_data_create():
    with tempfile.TemporaryDirectory() as tmp:
        timesteps = 11
        buffer_size = 2**13
        z = v4rx_new_dataset(
            tmp + "/testdata",
            timesteps=timesteps,
            buffer_size=buffer_size,
            n_receivers=2,
            config="test_config",
        )
        for time_idx in range(timesteps):
            for receiver_idx in range(2):
                z.receivers[f"r{receiver_idx}"].signal_matrix[
                    time_idx, :
                ] = random_signal_matrix(2 * buffer_size).reshape(2, buffer_size)
                for k in v4rx_f64_keys:
                    z.receivers[f"r{receiver_idx}"][k][time_idx] = np.random.rand()
                for k in v4rx_2xf64_keys:
                    z.receivers[f"r{receiver_idx}"][k][time_idx, :] = np.random.rand()


def testv5_data_create():
    with tempfile.TemporaryDirectory() as tmp:
        timesteps = 11
        buffer_size = 2**13
        fn = tmp + "/testdata"
        z = v5rx_new_dataset(
            fn,
            timesteps=timesteps,
            buffer_size=buffer_size,
            n_receivers=2,
            config="test_config",
        )
        for time_idx in range(timesteps):
            for receiver_idx in range(2):
                z.receivers[f"r{receiver_idx}"].signal_matrix[
                    time_idx, :
                ] = random_signal_matrix(2 * buffer_size).reshape(2, buffer_size)
                for k in v5rx_f64_keys:
                    z.receivers[f"r{receiver_idx}"][k][time_idx] = np.random.rand()
                for k in v5rx_2xf64_keys:
                    z.receivers[f"r{receiver_idx}"][k][time_idx, :] = np.random.rand()

        z1 = z.receivers.r1.signal_matrix[:]
        z.store.close()
        z = None

        zarr_shrink(fn)

        z = zarr_open_from_lmdb_store(fn)
        assert np.isclose(z1, z.receivers.r1.signal_matrix[:]).all()


def testv6_data_create_and_metadata_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        fn = tmp + "/testdata"
        z = v6rx_new_dataset(
            fn,
            timesteps=3,
            buffer_size=16,
            n_receivers=1,
            config={"data-version": 6},
        )
        receiver = z.receivers.r0
        expected_scalar = {
            "gain_metadata_valid": True,
            "gain_metadata_flags": 0x413,
            "stream_id": 0x123456789ABC,
            "buffer_sequence": 7,
            "sample_sequence": 7 * 16,
            "gain_start_read_duration_ns": 1200,
            "gain_end_read_duration_ns": 1300,
        }
        expected_2x = {
            "gain_index_start": [42, 43],
            "gain_index_end": [41, 43],
            "gain_endpoints_equal": [False, True],
            "first_gain_change_sample": [-1, -1],
            "iq_power_dbfs": [-18.5, -19.0],
        }
        for key in v6rx_scalar_keys:
            receiver[key][1] = expected_scalar[key]
        for key in v6rx_2x_keys:
            receiver[key][1] = expected_2x[key]

        for key in v6rx_scalar_keys:
            assert receiver[key][1] == expected_scalar[key]
        for key in v6rx_2x_keys:
            np.testing.assert_allclose(receiver[key][1], expected_2x[key])


def testv7_data_create_and_radio_metadata_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        z = v7rx_new_dataset(
            tmp + "/testdata",
            timesteps=3,
            buffer_size=16,
            n_receivers=1,
            config={"data-version": 7},
        )
        receiver = z.receivers.r0
        expected_scalar = {
            "gain_metadata_valid": True,
            "rssi_metadata_valid": True,
            "gain_metadata_flags": 0x58013,
            "stream_id": 0x123456789ABC,
            "buffer_sequence": 7,
            "sample_sequence": 7 * 16,
            "gain_start_read_duration_ns": 1200,
            "gain_end_read_duration_ns": 1300,
            "rssi_start_read_duration_ns": 1400,
            "rssi_end_read_duration_ns": 1500,
        }
        expected_2x = {
            "gain_db_start": [20.0, 40.0],
            "gain_db_end": [21.0, 40.0],
            "rssi_db_start": [80.25, 81.5],
            "rssi_db_end": [80.5, 81.75],
            "gain_endpoints_equal": [False, True],
            "first_gain_change_sample": [-1, -1],
            "iq_power_dbfs": [-18.5, -19.0],
        }
        expected_sample_time = {
            "sample_counter_end_exclusive": 7 * 16 + 16,
            "sample_time_valid": True,
            "sample_time_monotonic_start_ns": 123_000_000,
            "sample_time_monotonic_end_ns": 123_016_000,
            "sample_time_realtime_start_ns": 1_723_000_000_000_000_000,
            "sample_time_realtime_end_ns": 1_723_000_000_000_016_000,
            "sample_time_uncertainty_ns": 225_000,
            "sample_time_fitted_rate_hz": 1_000_000.25,
            "sample_time_anchor_count": 8,
            "sample_time_max_round_trip_ns": 310_000,
            "sample_time_rate_tolerance_ppm": 100.0,
        }
        for key in v7rx_scalar_keys:
            receiver[key][1] = expected_scalar[key]
        for key in v7rx_2x_keys:
            receiver[key][1] = expected_2x[key]
        for key in v7rx_sample_time_scalar_keys:
            receiver[key][1] = expected_sample_time[key]

        assert z.attrs["radio_metadata_schema_version"] == 2
        assert z.attrs["gain_series_schema_version"] == 1
        assert z.attrs["sample_time_schema_version"] == 1
        assert "gain_observation_index" not in v7rx_keys()
        assert "gain_observation_index" in v7rx_keys(include_gain_series=True)
        assert "sample_time_valid" not in v7rx_keys()
        assert "sample_time_valid" in v7rx_keys(include_sample_time=True)
        receiver["gain_observation_count"][1] = 2
        receiver["gain_observation_interval_samples"][1] = 32768
        receiver["gain_observation_sample_bounds"][1, :2] = [
            [0x100000000, 0x100001D4C],
            [0x100008000, 0x100009D4C],
        ]
        receiver["gain_observation_index"][1, :2] = [[42, 43], [41, 43]]
        receiver["gain_observation_db"][1, :2] = [[20.0, 21.0], [19.0, 21.0]]
        receiver["gain_observation_valid"][1, :2] = [True, True]
        receiver["gain_observation_read_duration_ns"][1, :2] = [500000, 510000]
        assert receiver["gain_observation_count"][1] == 2
        assert receiver["gain_observation_interval_samples"][1] == 32768
        np.testing.assert_array_equal(
            receiver["gain_observation_index"][1, :2], [[42, 43], [41, 43]]
        )
        np.testing.assert_allclose(
            receiver["gain_observation_db"][1, :2], [[20.0, 21.0], [19.0, 21.0]]
        )
        for key in v7rx_scalar_keys:
            assert receiver[key][1] == expected_scalar[key]
        for key in v7rx_2x_keys:
            np.testing.assert_allclose(receiver[key][1], expected_2x[key])
        for key in v7rx_sample_time_scalar_keys:
            assert receiver[key][1] == expected_sample_time[key]


def test_lmdb_async_resume_matches_initial_capture_write_flags():
    with tempfile.TemporaryDirectory() as tmp:
        path = tmp + "/testdata"
        z = zarr_open_from_lmdb_store(path, mode="w", map_size=2**20)
        z.create_dataset("value", shape=(1,), dtype=np.int32)
        z["value"][0] = 7
        initial_flags = z.store.db.flags()
        z.store.close()

        z = zarr_open_from_lmdb_store(
            path,
            mode="rw",
            map_size=2**20,
            map_async=True,
        )
        resumed_flags = z.store.db.flags()
        assert resumed_flags["writemap"] is True
        assert resumed_flags["map_async"] is True
        assert resumed_flags["sync"] is False
        assert resumed_flags["metasync"] is False
        for flag in ("writemap", "map_async", "sync", "metasync"):
            assert resumed_flags[flag] == initial_flags[flag]
        assert z["value"][0] == 7
        z["value"][0] = 8
        z.store.close()

        z = zarr_open_from_lmdb_store(path, mode="r", map_size=2**20)
        assert z["value"][0] == 8
        z.store.close()
