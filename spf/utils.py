import numpy as np
import zarr
from numcodecs import Blosc


class dotdict(dict):
    __getattr__ = dict.get

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


def is_pi():
    try:
        import RPi.GPIO as GPIO  # noqa

        return True
    except (RuntimeError, ImportError):
        return False


def random_signal_matrix(n):
    return np.random.uniform(-1, 1, (n,)) + 1.0j * np.random.uniform(-1, 1, (n,))


def zarr_new_dataset(
    filename,
    timesteps,
    buffer_size,
    n_receivers,
    keys_f64,
    keys_2xf64,
    chunk_size=4096,
    compressor=None,
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
        for key in keys_f64:
            receiver_z.create_dataset(
                key,
                shape=(timesteps,),
                dtype="float64",
            )
        for key in keys_2xf64:
            receiver_z.create_dataset(
                key,
                shape=(timesteps, 2),
                dtype="float64",
            )
    return z
