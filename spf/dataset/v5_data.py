from spf.utils import zarr_new_dataset

v5rx_f64_keys = [
    "system_timestamp",
    "tx_pos_x_mm",
    "tx_pos_y_mm",
    "rx_pos_x_mm",
    "rx_pos_y_mm",
    "rx_theta_in_pis",
    "rx_spacing",
]


v5rx_2xf64_keys = [
    "avg_phase_diff",
    "rssis",
    "gains",
]


def v5rx_keys():
    return v5rx_f64_keys + v5rx_2xf64_keys + ["signal_matrix"]


def v5rx_new_dataset(
    filename, timesteps, buffer_size, n_receivers, chunk_size=4096, compressor=None
):
    return zarr_new_dataset(
        filename=filename,
        timesteps=timesteps,
        buffer_size=buffer_size,
        n_receivers=n_receivers,
        keys_f64=v5rx_f64_keys,
        keys_2xf64=v5rx_2xf64_keys,
    )
