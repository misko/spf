import os
from contextlib import contextmanager
from datetime import datetime

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


def random_signal_matrix(n, rng=None):
    if rng is not None:
        return rng.uniform(-1, 1, (n,)) + 1.0j * rng.uniform(-1, 1, (n,))
    return np.random.uniform(-1, 1, (n,)) + 1.0j * np.random.uniform(-1, 1, (n,))


def zarr_remove_if_exists(zarr_fn):
    for fn in ["data.mdb", "lock.mdb"]:
        if os.path.exists(zarr_fn + "/" + fn):
            os.remove(zarr_fn + "/" + fn)


def zarr_shrink(filename):
    store = zarr.LMDBStore(filename, map_size=2**38, writemap=True, map_async=True)
    store.db.set_mapsize(1)
    print(store.db.info())
    store.close()


@contextmanager
def zarr_open_from_lmdb_store_cm(filename, mode="r"):
    try:
        z = zarr_open_from_lmdb_store(filename, mode)
        yield z
    finally:
        z.store.close()


def zarr_open_from_lmdb_store(filename, mode="r"):
    if mode == "r":
        store = zarr.LMDBStore(
            filename,
            map_size=2**38,
            writemap=False,
            readonly=True,
            max_readers=1024 * 1024,
        )
    elif mode == "w":
        store = zarr.LMDBStore(filename, map_size=2**38, writemap=True, map_async=True)
    else:
        raise NotImplementedError
    return zarr.open(
        store=store,
        mode=mode,
    )


def zarr_new_dataset(
    filename,
    timesteps,
    buffer_size,
    n_receivers,
    keys_f64,
    keys_2xf64,
    chunk_size=512,  # tested , blosc1 / chunk_size=512 / buffer_size (2^18~20) = seems pretty good
    compressor=None,
):
    zarr_remove_if_exists(filename)
    z = zarr_open_from_lmdb_store(filename, mode="w")
    if compressor is None:
        compressor = Blosc(
            cname="zstd",
            clevel=1,
            shuffle=Blosc.BITSHUFFLE,
        )
    else:
        raise NotImplementedError
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


class DataVersionNotImplemented(NotImplementedError):
    pass


def filenames_from_time_in_seconds(
    time_in_seconds, temp_dir_name, yaml_config, data_version, tag, craft
):
    os.makedirs(temp_dir_name, exist_ok=True)
    dt = datetime.fromtimestamp(time_in_seconds)
    date_str = dt.strftime("%Y_%m_%d_%H_%M_%S")

    output_files_prefix = f"{craft}_{date_str}_nRX{len(yaml_config['receivers'])}_{yaml_config['routine']}"
    if tag != "":
        output_files_prefix += f"_tag_{tag}"

    filename_log = f"{temp_dir_name}/{output_files_prefix}.log.tmp"
    filename_yaml = f"{temp_dir_name}/{output_files_prefix}.yaml.tmp"
    if data_version == (2, 3):
        filename_data = f"{temp_dir_name}/{output_files_prefix}.npy.tmp"
    elif data_version in (4, 5):
        filename_data = f"{temp_dir_name}/{output_files_prefix}.zarr.tmp"
    else:
        raise NotImplementedError
    temp_filenames = {
        "log": filename_log,
        "yaml": filename_yaml,
        "data": filename_data,
    }
    final_filenames = {k: v.replace(".tmp", "") for k, v in temp_filenames.items()}

    return temp_filenames, final_filenames
