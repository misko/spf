import zarr
from numcodecs import Blosc

v4rx_f64_keys = [
    "system_timestamps",
    "gps_timestamps",
    "lat",
    "long",
    "heading",
    "avg_phase_diff",
    "rssi",
    "gain",
]


def v4rx_keys():
    return v4rx_f64_keys + ["signal_matrix"]


def v4rx_new_dataset(
    filename, timesteps, buffer_size, n_receivers, chunk_size=4096, compressor=None
):
    z = zarr.open(
        filename,
        mode="w",
    )
    if compressor is None:
        compressor = Blosc(
            cname="zstd",
            clevel=1,
            shuffle=Blosc.BITSHUFFLE,
        )
    z.create_group("receivers")
    for receiver_idx in range(n_receivers):
        receiver_z = z["receivers"].create_group(f"r{receiver_idx}")
        receiver_z.create_dataset(
            "signal_matrix",
            shape=(timesteps, 2, buffer_size),
            chunks=(1, 1, 1024 * chunk_size),
            dtype="complex128",
            compressor=compressor,
        )
        for key in v4rx_f64_keys:
            receiver_z.create_dataset(
                key,
                shape=(timesteps,),
                chunks=(1024 * chunk_size),
                dtype="float64",
                compressor=compressor,
            )
    return z
