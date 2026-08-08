"""SPF v7 schema for direct-USB v2 endpoints and bounded v3 gain series."""

import numpy as np

from spf.dataset.v4_data import v4rx_2xf64_keys, v4rx_f64_keys
from spf.scripts.zarr_utils import zarr_new_dataset


v7rx_f64_keys = list(v4rx_f64_keys)
v7rx_2xf64_keys = list(v4rx_2xf64_keys)

v7rx_scalar_keys = {
    "gain_metadata_valid": np.bool_,
    "rssi_metadata_valid": np.bool_,
    "gain_metadata_flags": np.uint32,
    "stream_id": np.uint64,
    "buffer_sequence": np.uint64,
    "sample_sequence": np.uint64,
    "gain_start_read_duration_ns": np.uint32,
    "gain_end_read_duration_ns": np.uint32,
    "rssi_start_read_duration_ns": np.uint32,
    "rssi_end_read_duration_ns": np.uint32,
}

v7rx_2x_keys = {
    "gain_db_start": np.float32,
    "gain_db_end": np.float32,
    "rssi_db_start": np.float32,
    "rssi_db_end": np.float32,
    "gain_endpoints_equal": np.bool_,
    "first_gain_change_sample": np.int32,
    "iq_power_dbfs": np.float32,
}

V7_GAIN_OBSERVATION_CAPACITY = 256
V7_GAIN_EVENT_CAPACITY = 64

v7rx_gain_series_scalar_keys = {
    "gain_observation_count": np.uint16,
    "gain_observation_interval_samples": np.uint32,
    "gain_observation_overflow_count": np.uint32,
    "gain_event_count": np.uint16,
    "gain_event_overflow_count": np.uint32,
}


def v7rx_gain_series_keys():
    return list(v7rx_gain_series_scalar_keys) + [
        "gain_observation_sample_bounds",
        "gain_observation_index",
        "gain_observation_db",
        "gain_observation_valid",
        "gain_observation_read_duration_ns",
        "gain_event_sample_sequence",
        "gain_event_flags",
    ]


def v7rx_keys(*, include_gain_series=False):
    """Return required fields without breaking protocol-v2 V7 captures.

    The gain series is an optional, independently versioned V7 extension.
    Callers inspecting a store should request it only when the root attribute
    ``gain_series_schema_version`` is present and supported.
    """

    keys = (
        v7rx_f64_keys
        + v7rx_2xf64_keys
        + list(v7rx_scalar_keys)
        + list(v7rx_2x_keys)
        + ["signal_matrix"]
    )
    if include_gain_series:
        keys += v7rx_gain_series_keys()
    return keys


def v7rx_new_dataset(
    filename,
    timesteps,
    buffer_size,
    n_receivers,
    config,
    chunk_size=1024,
    compressor=None,
):
    z = zarr_new_dataset(
        filename=filename,
        timesteps=timesteps,
        buffer_size=buffer_size,
        n_receivers=n_receivers,
        keys_f64=v7rx_f64_keys,
        keys_2xf64=v7rx_2xf64_keys,
        chunk_size=chunk_size,
        compressor=compressor,
        config=config,
    )
    # Preserve the endpoint gain/RSSI schema contract for existing V7 readers.
    # The bounded observation/event arrays are an independently versioned,
    # backward-compatible extension.
    z.attrs["radio_metadata_schema_version"] = 2
    z.attrs["gain_series_schema_version"] = 1
    for receiver_idx in range(n_receivers):
        receiver_z = z[f"receivers/r{receiver_idx}"]
        for key, dtype in v7rx_scalar_keys.items():
            receiver_z.create_dataset(
                key,
                shape=(timesteps,),
                dtype=dtype,
                chunks=(timesteps,),
                compressor=None,
            )
        for key, dtype in v7rx_2x_keys.items():
            receiver_z.create_dataset(
                key,
                shape=(timesteps, 2),
                dtype=dtype,
                chunks=(timesteps, 2),
                compressor=None,
            )
        for key, dtype in v7rx_gain_series_scalar_keys.items():
            receiver_z.create_dataset(
                key,
                shape=(timesteps,),
                dtype=dtype,
                chunks=(timesteps,),
                compressor=None,
            )
        row_chunk = min(timesteps, 512)
        receiver_z.create_dataset(
            "gain_observation_sample_bounds",
            shape=(timesteps, V7_GAIN_OBSERVATION_CAPACITY, 2),
            dtype=np.uint64,
            chunks=(row_chunk, V7_GAIN_OBSERVATION_CAPACITY, 2),
            compressor=compressor,
        )
        receiver_z.create_dataset(
            "gain_observation_index",
            shape=(timesteps, V7_GAIN_OBSERVATION_CAPACITY, 2),
            dtype=np.uint8,
            chunks=(row_chunk, V7_GAIN_OBSERVATION_CAPACITY, 2),
            compressor=compressor,
        )
        receiver_z.create_dataset(
            "gain_observation_db",
            shape=(timesteps, V7_GAIN_OBSERVATION_CAPACITY, 2),
            dtype=np.float32,
            chunks=(row_chunk, V7_GAIN_OBSERVATION_CAPACITY, 2),
            compressor=compressor,
        )
        receiver_z.create_dataset(
            "gain_observation_valid",
            shape=(timesteps, V7_GAIN_OBSERVATION_CAPACITY),
            dtype=np.bool_,
            chunks=(row_chunk, V7_GAIN_OBSERVATION_CAPACITY),
            compressor=compressor,
        )
        receiver_z.create_dataset(
            "gain_observation_read_duration_ns",
            shape=(timesteps, V7_GAIN_OBSERVATION_CAPACITY),
            dtype=np.uint32,
            chunks=(row_chunk, V7_GAIN_OBSERVATION_CAPACITY),
            compressor=compressor,
        )
        receiver_z.create_dataset(
            "gain_event_sample_sequence",
            shape=(timesteps, V7_GAIN_EVENT_CAPACITY),
            dtype=np.uint64,
            chunks=(row_chunk, V7_GAIN_EVENT_CAPACITY),
            compressor=compressor,
        )
        receiver_z.create_dataset(
            "gain_event_flags",
            shape=(timesteps, V7_GAIN_EVENT_CAPACITY),
            dtype=np.uint16,
            chunks=(row_chunk, V7_GAIN_EVENT_CAPACITY),
            compressor=compressor,
        )
    return z
