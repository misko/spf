import math
import os
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import yaml
import zarr
from deepdiff.diff import DeepDiff
from numcodecs import Blosc
from torch.utils.data import BatchSampler, DistributedSampler

SEGMENTATION_VERSION = 3.11
warnings.simplefilter(action="ignore", category=FutureWarning)


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


@torch.jit.script
def torch_random_signal_matrix(n: int):
    return (torch.rand((n,), dtype=torch.float32) - 0.5) * 2 + 1.0j * (
        torch.rand((n,), dtype=torch.float32) - 0.5
    )


def zarr_remove_if_exists(zarr_fn):
    for fn in ["data.mdb", "lock.mdb"]:
        if os.path.exists(zarr_fn + "/" + fn):
            os.remove(zarr_fn + "/" + fn)


def zarr_shrink(filename):
    store = zarr.LMDBStore(filename, map_size=2**37, writemap=True, map_async=True)
    store.db.set_mapsize(1)
    print(store.db.info())
    store.close()


@contextmanager
def zarr_open_from_lmdb_store_cm(filename, mode="r", readahead=False):
    try:
        z = zarr_open_from_lmdb_store(filename, mode, readahead=readahead)
        yield z
    finally:
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
            max_readers=32,  # 1024 * 1024,
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


class DataVersionNotImplemented(NotImplementedError):
    pass


def rx_spacing_to_str(rx_spacing):
    return f"{rx_spacing:0.6f}"


def filenames_from_time_in_seconds(
    time_in_seconds, temp_dir_name, yaml_config, data_version, tag, craft
):
    os.makedirs(temp_dir_name, exist_ok=True)
    dt = datetime.fromtimestamp(time_in_seconds)
    date_str = dt.strftime("%Y_%m_%d_%H_%M_%S")

    output_files_prefix = f"{craft}_{date_str}_nRX{len(yaml_config['receivers'])}_{yaml_config['routine']}_spacing{str(yaml_config['receivers'][0]['antenna-spacing-m']).replace('.' , 'p')}"
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


def identical_datasets(dsA, dsB, skip_signal_matrix=False):
    assert len(dsA) == len(dsB)
    for idx in range(len(dsA)):
        assert len(dsA[idx]) == len(dsB[idx])
        for r_idx in range(len(dsA[idx])):
            eA = dsA[idx][r_idx]
            eB = dsB[idx][r_idx]
            eA_keys = set(eA.keys())
            eB_keys = set(eB.keys())
            if skip_signal_matrix:
                eA_keys -= set(["signal_matrix", "abs_signal_and_phase_diff"])
                eB_keys -= set(["signal_matrix", "abs_signal_and_phase_diff"])
            assert eA_keys == eB_keys
            for key in eA_keys:
                if isinstance(eA[key], torch.Tensor):
                    assert torch.isclose(eA[key], eB[key]).all()
                else:
                    assert len(DeepDiff(eA[key], eB[key])) == 0


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


def compare_and_copy(prefix, src, dst, skip_signal_matrix=False):
    if isinstance(src, zarr.hierarchy.Group):
        for key in src.keys():
            if not skip_signal_matrix or key != "signal_matrix":
                compare_and_copy(
                    prefix + "/" + key, src[key], dst[key], skip_signal_matrix
                )
    else:
        if prefix == "/config":
            if src.shape != ():
                dst[:] = src[:]
        else:
            for x in range(src.shape[0]):
                dst[x] = src[
                    x
                ]  # TODO why cant we just copy the whole thing at once? # too big?


# from fair-chem repo
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size, **kwargs):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset=dataset, **kwargs)

        self.start_iter = 0
        self.batch_size = batch_size
        assert self.batch_size > 0, "batch_size not set for the sampler"
        # logging.info(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # TODO: For very large datasets, even virtual datasets this might slow down
        # or not work correctly. The issue is that we enumerate the full list of all
        # samples in a single epoch, and manipulate this list directly. A better way
        # of doing this would be to keep this sequence strictly as an iterator
        # that stores the current state (instead of the full sequence)
        distributed_sampler_sequence = super().__iter__()
        if self.start_iter > 0:
            for i, _ in enumerate(distributed_sampler_sequence):
                if i == self.start_iter * self.batch_size - 1:
                    break
        return distributed_sampler_sequence

    def set_epoch_and_start_iteration(self, epoch, start_iter):
        self.set_epoch(epoch)
        self.start_iter = start_iter


class StatefulBatchsampler(BatchSampler):
    def __init__(self, dataset, batch_size, seed=0, shuffle=False, drop_last=False):
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            drop_last=False,
            batch_size=batch_size,
            seed=seed,
        )
        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)

    def set_epoch_and_start_iteration(self, epoch: int, start_iteration: int) -> None:
        self.sampler.set_epoch_and_start_iteration(epoch, start_iteration)

    def __iter__(self):
        # for x in super().__iter__():
        #     print("YIELD", x)
        #     yield x
        yield from super().__iter__()


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@torch.jit.script
def to_bin(x: torch.Tensor, bins: int):
    return ((x / (2 * torch.pi) + 0.5) * bins).to(torch.long)
