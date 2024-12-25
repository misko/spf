import os
import shutil

from spf.scripts.zarr_utils import (
    compare_and_copy,
    zarr_new_dataset,
    zarr_open_from_lmdb_store,
)

v5rx_f64_keys = [
    "system_timestamp",
    "tx_pos_x_mm",
    "tx_pos_y_mm",
    "rx_pos_x_mm",
    "rx_pos_y_mm",
    "rx_theta_in_pis",
    "rx_spacing",
    "rx_lo",
    "rx_bandwidth",
    "rx_heading_in_pis",
]


v5rx_2xf64_keys = [
    "avg_phase_diff",
    "rssis",
    "gains",
]


def v5rx_keys():
    return v5rx_f64_keys + v5rx_2xf64_keys + ["signal_matrix"]


def v5rx_new_dataset(
    filename,
    timesteps,
    buffer_size,
    n_receivers,
    config,
    chunk_size=1024,
    compressor=None,
    skip_signal_matrix=False,
    remove_if_exists=True,
):
    return zarr_new_dataset(
        filename=filename,
        timesteps=timesteps,
        buffer_size=buffer_size,
        n_receivers=n_receivers,
        keys_f64=v5rx_f64_keys,
        keys_2xf64=v5rx_2xf64_keys,
        chunk_size=chunk_size,
        compressor=compressor,
        skip_signal_matrix=skip_signal_matrix,
        config=config,
        remove_if_exists=remove_if_exists,
    )
