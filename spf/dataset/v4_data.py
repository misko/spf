from spf.utils import zarr_new_dataset

v4rx_f64_keys = [
    "system_timestamp",
    "gps_timestamp",
    "gps_lat",
    "gps_long",
    "heading",
    "rx_lo",
    "rx_bandwidth",
]
v4rx_2xf64_keys = [
    "avg_phase_diff",
    "rssis",
    "gains",
]


def v4rx_keys():
    return v4rx_f64_keys + v4rx_2xf64_keys + ["signal_matrix"]


def v4rx_new_dataset(
    filename, timesteps, buffer_size, n_receivers, chunk_size=1024, compressor=None
):
    return zarr_new_dataset(
        filename=filename,
        timesteps=timesteps,
        buffer_size=buffer_size,
        n_receivers=n_receivers,
        keys_f64=v4rx_f64_keys,
        keys_2xf64=v4rx_2xf64_keys,
        chunk_size=chunk_size,
        compressor=compressor,
    )
