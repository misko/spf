"""SPF v7 capture schema for direct-USB protocol v2 radio metadata."""

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


def v7rx_keys():
    return (
        v7rx_f64_keys
        + v7rx_2xf64_keys
        + list(v7rx_scalar_keys)
        + list(v7rx_2x_keys)
        + ["signal_matrix"]
    )


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
    z.attrs["radio_metadata_schema_version"] = 2
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
    return z
