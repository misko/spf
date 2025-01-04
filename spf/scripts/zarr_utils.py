import logging
import os
import shutil
from contextlib import contextmanager

import numpy as np
import yaml
import zarr
from numcodecs import Blosc

from spf.utils import SEGMENTATION_VERSION


@contextmanager
def zarr_open_from_lmdb_store_cm(filename, mode="r", readahead=False):
    f = None
    try:
        if mode == "r":
            f = open(filename + "/data.mdb", "rb")
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
        z = zarr_open_from_lmdb_store(filename, mode, readahead=readahead)
        yield z
    finally:
        if f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            f.close()
        z.store.close()


def new_yarr_dataset(
    filename,
    n_receivers,
    all_windows_stats_shape,
    windowed_beamformer_shape,
    weighted_beamformer_shape,
    downsampled_segmentation_mask_shape,
    mean_phase_shape,
    # compressor=None,
):
    zarr_remove_if_exists(filename)
    z = zarr_open_from_lmdb_store(filename, mode="w", map_size=2**32)
    z.create_dataset("version", dtype=np.float32, shape=(1))
    z["version"][0] = SEGMENTATION_VERSION
    compressor = Blosc(
        cname="zstd",
        clevel=1,
        shuffle=Blosc.BITSHUFFLE,
    )
    for receiver_idx in range(n_receivers):
        receiver_z = z.create_group(f"r{receiver_idx}")
        receiver_z.create_dataset(
            "all_windows_stats",
            shape=all_windows_stats_shape,
            chunks=(16, -1, -1),
            dtype="float16",
            compressor=None,
        )
        receiver_z.create_dataset(
            "weighted_windows_stats",
            shape=all_windows_stats_shape[:-1],
            chunks=(16, -1),
            dtype="float32",
            compressor=None,
        )
        receiver_z.create_dataset(
            "windowed_beamformer",
            shape=windowed_beamformer_shape,
            chunks=(16, -1, -1),
            dtype="float16",
            compressor=compressor,
        )
        receiver_z.create_dataset(
            "weighted_beamformer",
            shape=weighted_beamformer_shape,
            chunks=(16, -1),
            dtype="float32",
            compressor=None,
        )
        receiver_z.create_dataset(
            "downsampled_segmentation_mask",
            shape=downsampled_segmentation_mask_shape,
            chunks=(16, -1),
            dtype="bool",
            compressor=None,
        )
        receiver_z.create_dataset(
            "mean_phase",
            shape=mean_phase_shape,
            chunks=(-1),
            dtype="float32",
            compressor=None,
        )
    return z


def zarr_open_from_lmdb_store(filename, mode="r", readahead=False, map_size=2**37):
    if mode == "r":
        store = zarr.LMDBStore(
            filename,
            map_size=map_size,
            writemap=False,
            readonly=True,
            max_readers=1024,  # 1024 * 1024,
            lock=False,
            meminit=False,
            readahead=readahead,
        )
    elif mode == "rw":
        store = zarr.LMDBStore(
            filename,
            map_size=map_size,
            writemap=False,
            readonly=False,
            sync=True,
            max_readers=32,  # 1024 * 1024,
            lock=True,
            meminit=False,
            readahead=readahead,
        )
    elif mode == "w":
        store = zarr.LMDBStore(
            filename, map_size=map_size, writemap=True, map_async=True
        )
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
    config,
    chunk_size=512,  # tested , blosc1 / chunk_size=512 / buffer_size (2^18~20) = seems pretty good
    compressor=None,
    skip_signal_matrix=False,
    remove_if_exists=True,
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

    z.create_dataset("config", dtype=str, shape=(1))
    if isinstance(config, dict):
        config = yaml.dump(config)
    if isinstance(config, zarr.core.Array) and len(config.shape) > 0:
        config = config[0]
    if not isinstance(config, int) and (isinstance(config, str) or config.shape != ()):
        z["config"][0] = config
        assert z["config"][0] == config

    z.create_group("receivers")
    for receiver_idx in range(n_receivers):
        receiver_z = z["receivers"].create_group(f"r{receiver_idx}")
        if not skip_signal_matrix:
            receiver_z.create_dataset(
                "signal_matrix",
                shape=(timesteps, 2, buffer_size),
                # chunks=(1, 1, 1024 * chunk_size),
                chunks=(1, 2, buffer_size // 2),
                dtype="complex64",
                compressor=compressor,
            )
        for key in keys_f64:
            receiver_z.create_dataset(
                key,
                shape=(timesteps,),
                dtype="float64",
                chunks=(timesteps,),
                compressor=None,
            )
        for key in keys_2xf64:
            receiver_z.create_dataset(
                key,
                shape=(timesteps, 2),
                dtype="float64",
                chunks=(timesteps, 2),
                compressor=None,
            )
    return z


def zarr_remove_if_exists(zarr_fn):
    for fn in ["data.mdb", "lock.mdb"]:
        if os.path.exists(zarr_fn + "/" + fn):
            os.remove(zarr_fn + "/" + fn)


def zarr_shrink(filename):
    store = zarr.LMDBStore(filename, map_size=2**37, writemap=True, map_async=True)
    store.db.set_mapsize(1)
    print(store.db.info())
    store.close()


def compare_and_check(prefix, src, dst, skip_signal_matrix=False):
    if isinstance(src, zarr.hierarchy.Group):
        for key in src.keys():
            if not skip_signal_matrix or key != "signal_matrix":
                compare_and_check(
                    prefix + "/" + key, src[key], dst[key], skip_signal_matrix
                )
    else:
        if prefix == "/config":
            if not src.shape == ():
                if dst[:] != src[:]:
                    raise ValueError(f"{prefix} does not match!")
        else:
            for x in range(src.shape[0]):
                if not (dst[x] == src[x]).all():
                    raise ValueError(f"{prefix} does not match!")
    return True


def compare_and_copy(prefix, src, dst, skip_signal_matrix=False, copy_up_to_idx=None):
    if isinstance(src, zarr.hierarchy.Group):
        for key in src.keys():
            if not skip_signal_matrix or key != "signal_matrix":
                if key == "rx_heading" and key not in dst:
                    dst_key = "rx_heading_in_pis"
                else:
                    dst_key = key
                compare_and_copy(
                    prefix + "/" + key,
                    src[key],
                    dst[dst_key],
                    skip_signal_matrix,
                    copy_up_to_idx=copy_up_to_idx,
                )
    else:
        if prefix == "/config":
            if src.shape != ():
                dst[:] = src[:]
        else:
            for x in range(src.shape[0] if copy_up_to_idx is None else copy_up_to_idx):
                dst[x] = src[
                    x
                ]  # TODO why cant we just copy the whole thing at once? # too big?


def truncate_zarr(zarr_fn, f64_keys, f64x2_keys):
    partial_zarr = zarr_open_from_lmdb_store(zarr_fn, readahead=True, mode="r")
    missing_entries = max(
        [
            (partial_zarr[f"receivers/r{r_idx}/system_timestamp"][:] == 0).sum()
            for r_idx in [0, 1]
        ]
    )
    valid_entries = (
        partial_zarr["receivers/r1/system_timestamp"].shape[0] - missing_entries
    )
    buffer_size = partial_zarr["receivers/r0/signal_matrix"].shape[-1]
    config = partial_zarr["config"]

    bkup_zarr_fn = f"{zarr_fn}.bkup"
    assert not os.path.exists(bkup_zarr_fn)
    shutil.move(zarr_fn, bkup_zarr_fn)

    n_receivers = 2
    logging.info(f"valid entries {valid_entries}")
    new_zarr = zarr_new_dataset(
        zarr_fn,
        valid_entries,
        buffer_size,
        n_receivers,
        f64_keys,
        f64x2_keys,
        config,
        chunk_size=512,  # tested , blosc1 / chunk_size=512 / buffer_size (2^18~20) = seems pretty good
        compressor=None,
        skip_signal_matrix=False,
    )
    compare_and_copy(
        "",
        partial_zarr,
        new_zarr,
        skip_signal_matrix=False,
        copy_up_to_idx=valid_entries,
    )
