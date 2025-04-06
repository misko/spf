import functools
import hashlib
import math
import os
import warnings
from datetime import datetime

import numpy as np
import torch
import yaml
from deepdiff.diff import DeepDiff
from torch.utils.data import BatchSampler, DistributedSampler

SEGMENTATION_VERSION = 3.6
warnings.simplefilter(action="ignore", category=FutureWarning)


@functools.cache
def running_on_pi_zero_2():
    """
    Returns True if we're running on a Raspberry Pi Zero 2,
    determined by reading /proc/device-tree/model.
    """
    model_file = "/proc/device-tree/model"
    if not os.path.exists(model_file):
        return False

    try:
        with open(model_file, "r") as f:
            model_str = f.read()
            return "Raspberry Pi Zero 2" in model_str
    except IOError:
        return False


def no_op_script(obj=None, *args, **kwargs):
    # Return obj unchanged (ignore JIT compilation).
    return obj


if running_on_pi_zero_2():
    # If we are on Pi Zero 2, define a dummy (no-op) decorator
    torch.jit.script = no_op_script
    print("Detected Pi Zero 2: Disabling torch JIT script.")


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
def torch_random_signal_matrix(n: int, generator: torch.Generator):
    return (
        torch.rand((n,), dtype=torch.float32, generator=generator) - 0.5
    ) * 2 + 1.0j * (torch.rand((n,), dtype=torch.float32, generator=generator) - 0.5)


class DataVersionNotImplemented(NotImplementedError):
    pass


def rx_spacing_to_str(rx_spacing):
    return f"{rx_spacing:0.5f}"


@functools.cache
def get_md5_of_file(fn, cache_md5=True):
    if os.path.exists(fn + ".md5"):
        try:
            return open(fn + ".md5", "r").readlines()[0].strip()
        except:
            pass
    hash_md5 = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    md5 = hash_md5.hexdigest()
    if cache_md5:
        with open(fn + ".md5", "w") as f:
            f.write(md5)
    return md5


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
    x = x.clamp(min=-torch.pi, max=torch.pi - 1e-5)
    # assert (x >= -torch.pi).all() and (x <= torch.pi).all()
    bins = ((x / (2 * torch.pi) + 0.5) * bins).to(torch.long)
    bins[x.isnan()] = 0
    return bins


def load_config(yaml_fn):
    yaml_config = yaml.safe_load(open(yaml_fn, "r"))
    int_fields = ["f-carrier", "f-sampling", "f-intermediate", "bandwidth"]
    for receiver_config in yaml_config["receivers"]:
        for int_field in int_fields:
            if int_field in receiver_config:
                receiver_config[int_field] = int(receiver_config[int_field])
    return yaml_config
