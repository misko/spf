###
# Experiment 1 : wall drones , receiver center is x=135.27 cm, y=264.77cm, dist between is 6cm
###

import logging
import multiprocessing
import os
import pickle
import time
from contextlib import contextmanager
from enum import Enum
from functools import cache
from multiprocessing import Queue
from typing import Dict, List

import numpy as np
import torch
import yaml
from compress_pickle import load
from tensordict import TensorDict
from torch.utils.data import Dataset

from spf.dataset.segmentation import (
    DEFAULT_SEGMENT_ARGS,
    mean_phase_from_simple_segmentation,
    mp_segment_zarr,
    segment_session,
)
from spf.dataset.spf_generate import generate_session
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys
from spf.rf import (
    phase_diff_to_theta,
    precompute_steering_vectors,
    speed_of_light,
    torch_get_phase_diff,
    torch_pi_norm,
)
from spf.s3_utils import (
    b2_download_folder_cache,
    b2_file_to_local_with_cache,
    b2path_to_bucket_and_path,
)
from spf.scripts.train_utils import global_config_to_keys_used, load_config_from_fn
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.sdr_controller import rx_config_from_receiver_yaml
from spf.utils import SEGMENTATION_VERSION, load_config, rx_spacing_to_str, to_bin


# from Stackoverflow
def yaml_as_dict(my_file):
    my_dict = {}
    with open(my_file, "r") as fp:
        docs = yaml.safe_load_all(fp)
        for doc in docs:
            for key, value in doc.items():
                my_dict[key] = value
    return my_dict


def pos_to_rel(p, width):
    return 2 * (p / width - 0.5)


def rel_to_pos(r, width):
    return ((r / 2) + 0.5) * width


def encode_vehicle_type(vehicle_type):
    if vehicle_type == "wallarray":
        return -0.5
    elif vehicle_type == "rover":
        return 0.5
    assert 1 == 0


# from stackoverflow
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# target_thetas (N,1)
@torch.jit.script
def v5_thetas_to_targets(
    target_thetas: torch.Tensor, nthetas: int, range_in_rad: float, sigma: float = 1
):
    if target_thetas.ndim == 1:
        target_thetas = target_thetas.reshape(-1, 1)
    p = torch.exp(
        -(
            (
                (
                    target_thetas
                    - torch.linspace(
                        -torch.pi * range_in_rad / 2,
                        torch.pi * range_in_rad / 2,
                        nthetas,
                        device=target_thetas.device,
                    ).reshape(1, -1)
                )
                / sigma
            )
            ** 2
        )
    )
    return p / torch.sum(p, dim=1, keepdim=True)
    # return torch.nn.functional.normalize(p, p=1, dim=1)


# Batch is a list over items in the batch
# which contains a list over receivers (2)
# which is a dictionary of str to tensor
def v5_collate_keys_fast(keys: List[str], batch: List[List[Dict[str, torch.Tensor]]]):
    d = {}
    for key in keys:
        d[key] = torch.vstack(
            [x[key] for paired_sample in batch for x in paired_sample]
        )
        if key == "windowed_beamformer" or key == "all_windows_stats":
            d[key] = d[key].to(torch.float32)
    # d["random_rotations"] = torch_pi_norm(
    #     torch.rand(d["y_rad"].shape[0], 1) * 2 * torch.pi
    # )
    return TensorDict(d, batch_size=d["y_rad"].shape)


# @torch.jit.script
def v5_collate_all_fast(batch):
    d = {}
    for key in batch[0][0].keys():
        d[key] = torch.vstack(
            [x[key] for paired_sample in batch for x in paired_sample]
        )
    return d


def v5_collate_beamsegnet(batch):
    n_windows = batch[0][0]["all_windows_stats"].shape[1]
    y_rad_list = []
    y_phi_list = []
    simple_segmentation_list = []
    all_window_stats_list = []
    windowed_beamformers_list = []
    downsampled_segmentation_mask_list = []
    x_list = []
    segmentation_mask_list = []
    rx_spacing_list = []
    craft_y_rad_list = []
    receiver_idx_list = []
    rx_pos_list = []
    for paired_sample in batch:
        for x in paired_sample:
            y_rad_list.append(x["y_rad"])
            y_phi_list.append(x["y_phi"])
            rx_pos_list.append(x["rx_pos_xy"])
            craft_y_rad_list.append(x["craft_y_rad"])
            rx_spacing_list.append(x["rx_spacing"].reshape(-1, 1))
            # simple_segmentation_list += x["simple_segmentations"]
            all_window_stats_list.append(x["all_windows_stats"])  # .astype(np.float32)
            windowed_beamformers_list.append(
                x["windowed_beamformer"]  # .astype(np.float32)
            )
            downsampled_segmentation_mask_list.append(
                x["downsampled_segmentation_mask"]
            )
            receiver_idx_list.append(x["receiver_idx"].expand_as(x["y_rad"]))

            if "x" in batch[0][0]:
                x_list.append(x["x"])
                segmentation_mask_list.append(v5_segmentation_mask(x))
    d = {
        "y_rad": torch.vstack(y_rad_list),
        "y_phi": torch.vstack(y_phi_list),
        "rx_pos_xy": torch.vstack(rx_pos_list),
        "receiver_idx": torch.vstack(receiver_idx_list),
        "craft_y_rad": torch.vstack(craft_y_rad_list),
        "rx_spacing": torch.vstack(rx_spacing_list),
        # "simple_segmentation": simple_segmentation_list,
        "all_windows_stats": torch.from_numpy(np.vstack(all_window_stats_list)),
        "windowed_beamformer": torch.from_numpy(np.vstack(windowed_beamformers_list)),
        "downsampled_segmentation_mask": torch.vstack(
            downsampled_segmentation_mask_list
        ),
    }
    if "x" in batch[0][0]:
        d["x"] = torch.vstack(x_list)
        d["segmentation_mask"] = torch.vstack(segmentation_mask_list)
    return d


# using a spacing of at most max_offset and of at least 1
# get n random indexes, such that sum of the idxs is uniform
# across n,max_offset*n
def get_idx_for_rand_session(max_offset, n):
    target_length = np.random.randint(
        (max_offset - 1) * n,
    )
    z = np.random.rand(n).cumsum()
    z /= z.max()  # normalize to sum pr to 1
    return (z * target_length).round().astype(int) + np.arange(n)


def v5_downsampled_segmentation_mask(session, n_windows):
    window_size = 2048
    # stride = 2048
    # assert window_size == stride
    # _, _, samples_per_session = session["x"].shape
    # assert samples_per_session % window_size == 0
    # n_windows = samples_per_session // window_size
    seg_mask = torch.zeros(len(session["simple_segmentations"]), 1, n_windows)
    for idx in range(seg_mask.shape[0]):
        for window in session["simple_segmentations"][idx]:
            seg_mask[
                idx,
                0,
                window["start_idx"] // window_size : window["end_idx"] // window_size,
            ] = 1
    return seg_mask


def v5_segmentation_mask(session):
    _, _, samples_per_session = session["x"].shape
    seg_mask = torch.zeros(1, 1, samples_per_session)
    for w in session["simple_segmentation"]:
        seg_mask[0, 0, w["start_idx"] : w["end_idx"]] = 1
    return seg_mask[:, None]


@cache
def get_empirical_dist(
    v5ds,
    receiver_idx,
):
    rx_spacing_str = rx_spacing_to_str(v5ds.rx_wavelength_spacing)
    empirical_radio_key = f"r{receiver_idx}" if v5ds.empirical_individual_radio else "r"
    return v5ds.empirical_data[f"{v5ds.sdr_device_type}_{rx_spacing_str}"][
        empirical_radio_key
    ]["sym" if v5ds.empirical_symmetry else "nosym"]


def get_session_idxs(
    session_idx: int,
    tiled_sessions: bool,
    snapshots_stride: int,
    snapshots_adjacent_stride: int,
    snapshots_per_session: int,
    random_adjacent_stride: bool,
):
    if tiled_sessions:
        snapshot_start_idx = session_idx * snapshots_stride
        snapshot_end_idx = (
            snapshot_start_idx + snapshots_adjacent_stride * snapshots_per_session
        )
    else:
        snapshot_start_idx = (
            session_idx * snapshots_per_session * snapshots_adjacent_stride
        )
        snapshot_end_idx = (
            (session_idx + 1) * snapshots_per_session * snapshots_adjacent_stride
        )
    if random_adjacent_stride:
        # TODO
        # can choose random n numbers 0~1 scale by target cumsum
        idxs = (
            get_idx_for_rand_session(snapshots_adjacent_stride, snapshots_per_session)
            + snapshot_start_idx
        )
        return idxs
    else:
        return np.arange(
            snapshot_start_idx, snapshot_end_idx, snapshots_adjacent_stride
        )


def data_from_precomputed(v5ds, precomputed_data, segmentation, snapshot_idxs):
    data = {}
    if "windowed_beamformer" not in v5ds.skip_fields:
        data["windowed_beamformer"] = torch.as_tensor(
            precomputed_data["windowed_beamformer"][snapshot_idxs]
        ).unsqueeze(0)
        # TODO this is hacky and not right
        if data["windowed_beamformer"].shape[2] > v5ds.windows_per_snapshot:
            data["windowed_beamformer"] = subsample_tensor(
                data["windowed_beamformer"], 2, v5ds.windows_per_snapshot
            )

    if "weighted_beamformer" not in v5ds.skip_fields:
        data["weighted_beamformer"] = torch.as_tensor(
            precomputed_data["weighted_beamformer"][snapshot_idxs]
        ).unsqueeze(0)

    # sessions x 3 x n_windows
    if "all_windows_stats" not in v5ds.skip_fields:
        data["all_windows_stats"] = torch.as_tensor(
            precomputed_data["all_windows_stats"][snapshot_idxs]
        ).unsqueeze(0)
        # TODO this is hacky and not right
        if data["all_windows_stats"].shape[3] > v5ds.windows_per_snapshot:
            data["all_windows_stats"] = subsample_tensor(
                data["all_windows_stats"], 3, v5ds.windows_per_snapshot
            )

    if "weighted_windows_stats" not in v5ds.skip_fields:
        data["weighted_windows_stats"] = torch.as_tensor(
            precomputed_data["weighted_windows_stats"][snapshot_idxs]
        ).unsqueeze(0)

    if "downsampled_segmentation_mask" not in v5ds.skip_fields:
        data["downsampled_segmentation_mask"] = (
            torch.as_tensor(
                precomputed_data["downsampled_segmentation_mask"][snapshot_idxs]
            )
            .unsqueeze(1)
            .unsqueeze(0)
        )

    if "simple_segmentations" not in v5ds.skip_fields:
        data["simple_segmentations"] = [
            segmentation[snapshot_idx]["simple_segmentation"]
            for snapshot_idx in snapshot_idxs
        ]
    return data


@contextmanager
def v5spfdataset_manager(*args, **kwds):
    if "b2:" == kwds["precompute_cache"][:3]:
        b2_cache_folder = None
        if "b2_cache_folder" in kwds:
            b2_cache_folder = kwds.pop("b2_cache_folder")

        bucket, precompute_cache_path = b2path_to_bucket_and_path(
            kwds["precompute_cache"]
        )
        normalized_name = (
            os.path.basename(kwds["prefix"]).replace(".zarr", "").replace("_nosig", "")
            + f"_segmentation_nthetas{kwds['nthetas']}"
        )

        local_yarr_fn = b2_download_folder_cache(
            f"b2://{bucket}/{precompute_cache_path}/{normalized_name}.yarr",
            b2_cache_folder=b2_cache_folder,
        )
        local_precompute_cache_path = os.path.dirname(local_yarr_fn)

        _ = b2_file_to_local_with_cache(
            f"b2://{bucket}/{precompute_cache_path}/{normalized_name}.pkl",
            b2_cache_folder=b2_cache_folder,
        )

        local_kwds = kwds.copy()
        local_kwds["precompute_cache"] = local_precompute_cache_path

        ds = v5spfdataset(*args, **local_kwds)
        try:
            yield ds
        finally:
            ds.close()

    else:
        ds = v5spfdataset(*args, **kwds)
        try:
            yield ds
        finally:
            ds.close()


def subsample_tensor(x, dim, new_size):
    old_size = x.shape[dim]
    assert old_size >= new_size
    offset = torch.randint(low=0, high=old_size - new_size, size=(1,)).item()
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(offset, offset + new_size, 1)
    return x[tuple(slices)]


class SDRDEVICE(Enum):
    UNKNOWN = 0
    SIMULATION = 1
    PLUTO = 2
    BLADERF2 = 3


def uri_to_device_type(uri):
    if "bladerf" in uri:
        return SDRDEVICE.BLADERF2
    if "simulation" in uri:
        return SDRDEVICE.SIMULATION
    if "pluto" in uri or uri.startswith("usb") or uri.startswith("ip"):
        return SDRDEVICE.PLUTO
    return SDRDEVICE.UNKNOWN


class ZarrWrapper:
    def __init__(self, zarr_array):
        self.zarr_array = zarr_array
        self.on_the_fly_dict = {}

    def __contains__(self, item):
        if item in self.zarr_array or item in self.on_the_fly_dict:
            return True
        return False

    def __getitem__(self, key):
        if key in self.zarr_array:
            return self.zarr_array[key]
        elif key in self.on_the_fly_dict:
            return self.on_the_fly_dict[key]
        print("NO KEY", key)
        raise KeyError

    def __setitem__(self, key, value):
        if key in self.zarr_array:
            raise ValueError
        self.on_the_fly_dict[key] = value


training_only_keys = [
    "ground_truth_theta",
    "ground_truth_phi",
    "craft_ground_truth_theta",
    "absolute_theta",
    "y_rad",
    "y_phi",
    "craft_y_rad",
    "y_rad_binned",
]

segmentation_based_keys = [
    "weighted_beamformer",
    # "all_windows_stats",
    "weighted_windows_stats",
    "downsampled_segmentation_mask",
    "simple_segmentations",
    "mean_phase_segmentation",
]

v5_raw_keys = v5rx_f64_keys + v5rx_2xf64_keys + ["signal_matrix"]


def data_single_radio_to_raw(d, ds):
    return {k: d[k] for k in list(set(v5_raw_keys) - set(ds.skip_fields))}


class v5inferencedataset(Dataset):
    def __init__(
        self,
        yaml_fn: str,
        nthetas: int,  # Number of theta angles for beamforming discretization
        model_config_fn: str = "",
        paired: bool = False,  # If True, return paired samples from all receivers at once
        gpu: bool = False,  # Use GPU for segmentation computation if available
        skip_fields: List[
            str
        ] = [],  # Data fields to exclude during loading to save memory
        n_parallel: int = 20,  # Number of parallel processes for segmentation
        empirical_data_fn: (
            str | None
        ) = None,  # Path to empirical distribution data file for phase-to-angle mapping
        empirical_individual_radio: bool = False,  # Use per-radio empirical distributions if True
        empirical_symmetry: bool = True,  # Use symmetric empirical distributions if True
        target_dtype=torch.float32,  # Target dtype for tensor conversion (memory optimization)
        distance_normalization: int = 1000,  # Divisor to normalize distance measurements (mm to meters)
        target_ntheta: (
            bool | None
        ) = None,  # Target number of theta bins for classification (defaults to nthetas)
        windows_per_snapshot: int = 256,  # Maximum number of windows per snapshot to use
        skip_detrend: bool = False,
        skip_segmentation: bool = True,
        vehicle_type: str = "",
        max_in_memory: int = 10,
        realtime: bool = True,
        v4: bool = False,
    ):
        # Store configuration parameters
        self.yaml_fn = yaml_fn
        self.n_parallel = n_parallel
        self.nthetas = nthetas  # Number of angles to discretize space for beamforming
        self.target_ntheta = self.nthetas if target_ntheta is None else target_ntheta

        self.realtime = realtime

        self.max_in_memory = max_in_memory
        self.min_idx = 0
        self.condition = multiprocessing.Condition()
        self.lock = multiprocessing.Lock()
        self.store = {}

        self.incoming_queue = multiprocessing.Queue()

        self.v4 = v4

        # Segmentation parameters control how raw signal is processed into windows
        # and how phase difference is computed between antenna elements
        self.skip_detrend = skip_detrend
        self.windows_per_snapshot = windows_per_snapshot

        self.distance_normalization = distance_normalization
        self.skip_fields = skip_fields
        self.skip_segmentation = skip_segmentation
        if self.skip_segmentation:
            self.skip_fields += segmentation_based_keys
        self.paired = paired
        assert self.paired
        self.gpu = gpu  # Whether to use GPU acceleration for beamforming calculations
        self.target_dtype = target_dtype
        self.precomputed_entries = 0
        self.precomputed_zarr = (
            None  # Will hold preprocessed beamforming and segmentation data
        )

        self.yaml_config = load_config(self.yaml_fn)

        if model_config_fn != "":
            self.model_config = load_config_from_fn(model_config_fn)
            self.keys_to_get = global_config_to_keys_used(self.model_config["global"])
        else:
            self.keys_to_get = global_config_to_keys_used(None)

        # Get system metadata
        self.vehicle_type = vehicle_type

        # Extract receiver properties - important for beamforming calculations
        self.wavelengths = [
            speed_of_light / receiver["f-carrier"]  # λ = c/f
            for receiver in self.yaml_config["receivers"]
        ]
        self.carrier_frequencies = [
            receiver["f-carrier"] for receiver in self.yaml_config["receivers"]
        ]
        self.rf_bandwidths = [
            receiver["bandwidth"] for receiver in self.yaml_config["receivers"]
        ]

        # Validate that all receivers have consistent configurations
        for rx_idx in range(1, 2):
            assert (
                self.yaml_config["receivers"][0]["antenna-spacing-m"]
                == self.yaml_config["receivers"][rx_idx]["antenna-spacing-m"]
            )
            assert self.wavelengths[0] == self.wavelengths[rx_idx]
            assert self.rf_bandwidths[0] == self.rf_bandwidths[rx_idx]

        # Set up receiver spacing properties - critical for beamforming
        # Spacing between antenna elements affects phase difference and angle estimation
        self.rx_spacing = self.yaml_config["receivers"][0]["antenna-spacing-m"]
        assert self.yaml_config["receivers"][1]["antenna-spacing-m"] == self.rx_spacing

        # rx_wavelength_spacing (d/λ) is a key parameter for beamforming
        # It determines how phase differences map to arrival angles
        self.rx_wavelength_spacing = self.rx_spacing / self.wavelengths[0]

        # Create receiver configs and determine device types
        self.rx_configs = [
            rx_config_from_receiver_yaml(receiver)
            for receiver in self.yaml_config["receivers"]
        ]
        self.sdr_device_types = [
            uri_to_device_type(rx_config.uri) for rx_config in self.rx_configs
        ]

        # Ensure all receivers use the same device type
        if len(self.sdr_device_types) > 1:
            for device_type in self.sdr_device_types:
                assert device_type == self.sdr_device_types[0]
        self.sdr_device_type = self.sdr_device_types[0]

        # Precompute steering vectors for beamforming
        # Steering vectors are complex weights applied to each antenna element
        # They're used to "steer" the array to look in a specific direction
        # For each possible angle (theta), calculate the appropriate phase shifts
        self.steering_vectors = [
            precompute_steering_vectors(
                receiver_positions=rx_config.rx_pos,
                carrier_frequency=rx_config.lo,
                spacing=nthetas,
            )
            for rx_config in self.rx_configs
        ]

        # Define keys to load per session
        self.keys_per_session = (
            v5rx_f64_keys + v5rx_2xf64_keys + ["rx_wavelength_spacing"]
        )
        if "signal_matrix" not in self.skip_fields:
            self.keys_per_session.append("signal_matrix")

        # Load empirical distribution data if provided
        # These are learned phase-to-angle mappings that can improve angle estimation
        if empirical_data_fn is not None:
            self.empirical_data_fn = empirical_data_fn
            self.empirical_data = pickle.load(open(empirical_data_fn, "rb"))
            self.empirical_individual_radio = empirical_individual_radio
            self.empirical_symmetry = empirical_symmetry
        else:
            self.empirical_data_fn = None
            self.empirical_data = None
        self.serving_idx = -1

    def __len__(self):
        return self.serving_idx

    def __iter__(self):
        self.serving_idx = 0
        return self

    def __next__(self):
        sample = None
        while sample is None:
            sample = self[self.serving_idx]
            if sample is None:
                logging.warning("v5inf iterator waiting long on next sample")
        self.serving_idx += 1
        return sample

    # ASSUMING EVERYTHING WILL BE REQUESTED IN SEQUENCE!!
    def __getitem__(self, idx, timeout=10.0):
        start_time = time.time()
        # with self.condition:
        print("waitinf to get get", idx, time.time() - start_time)
        while idx not in self.store or self.store[idx]["count"] != 2:
            # self.condition.wait(0.01)
            self.check_for_new_data()
            time.sleep(0.02)
            if (time.time() - start_time) > timeout:
                print("ret waitinf to get get", idx, time.time() - start_time)
                return None
        return self.store[idx]["data"]

    def check_for_new_data(self):
        with self.lock:
            while not self.incoming_queue.empty():
                # shouldnt wait since we checked above
                idx, ridx, rendered_data = self.incoming_queue.get_nowait()
                if idx not in self.store:
                    self.store[idx] = {
                        "count": 0,
                        "data": [None, None],
                    }  # entry not ready
                self.store[idx]["data"][ridx] = rendered_data
                self.store[idx]["count"] += 1

    def write_to_idx(self, idx, ridx, raw):
        # this is the heavy lifting of processing, do it on this process
        rendered_data = self.render_session(idx, ridx, raw)

        self.incoming_queue.put((idx, ridx, rendered_data))

        # with self.condition:
        #    self.condition.notify_all()

    def render_session(self, idx, ridx, data):
        snapshot_idxs = [0]  # which snapshots to get

        data["rx_wavelength_spacing"] = torch.tensor(self.rx_wavelength_spacing)

        data["gains"] = data["gains"]  # [:, None]
        data["receiver_idx"] = torch.tensor([[ridx]], dtype=torch.int)

        data["ground_truth_theta"] = torch.tensor([torch.inf])  # unknown
        data["absolute_theta"] = torch.tensor([torch.inf])
        data["y_rad"] = data["ground_truth_theta"]  # torch.inf

        data["ground_truth_phi"] = torch.tensor([torch.inf])  # unkown
        data["y_phi"] = data["ground_truth_phi"]  # torch.inf

        data["craft_ground_truth_theta"] = torch.tensor([torch.inf])  # unknown
        data["craft_y_rad"] = data["craft_ground_truth_theta"]  # torch.inf

        data["vehicle_type"] = torch.tensor(
            [encode_vehicle_type(self.vehicle_type)]
        ).reshape(1)
        data["sdr_device_type"] = torch.tensor([self.sdr_device_type.value]).reshape(1)

        if "signal_matrix" not in self.skip_fields:
            # WARNGING this does not respect flipping!
            abs_signal = data["signal_matrix"].abs().to(torch.float32)
            assert data["signal_matrix"].shape[0] == 1
            pd = torch_get_phase_diff(data["signal_matrix"][0]).to(torch.float32)
            data["abs_signal_and_phase_diff"] = torch.concatenate(
                [abs_signal, pd[None, :, None]], dim=2
            )

        data["rx_pos_mm"] = torch.vstack(
            [
                data["rx_pos_x_mm"],
                data["rx_pos_y_mm"],
            ]
        ).T

        data["tx_pos_mm"] = torch.vstack(
            [
                data["tx_pos_x_mm"],
                data["tx_pos_y_mm"],
            ]
        ).T

        data["rx_pos_xy"] = (
            data["rx_pos_mm"][snapshot_idxs].unsqueeze(0) / self.distance_normalization
        )

        data["tx_pos_xy"] = (
            data["tx_pos_mm"][snapshot_idxs].unsqueeze(0) / self.distance_normalization
        )
        breakpoint()
        segmentation = segment_session(
            data["signal_matrix"][0][0].numpy(),
            gpu=self.gpu,
            skip_beamformer=False,
            skip_detrend=self.skip_detrend,
            skip_segmentation=self.skip_segmentation,
            **{
                "steering_vectors": self.steering_vectors[ridx],
                **DEFAULT_SEGMENT_ARGS,
            },
        )

        data.update(
            data_from_precomputed(
                v5ds=self,
                precomputed_data=segmentation,
                segmentation=[segmentation],
                snapshot_idxs=[0],
            )
        )
        if not self.skip_segmentation:
            data["mean_phase_segmentation"] = torch.tensor(
                mean_phase_from_simple_segmentation([segmentation])
            ).unsqueeze(0)

            if self.empirical_data is not None:
                empirical_dist = get_empirical_dist(self, ridx)
                #  ~ 1, snapshots, ntheta(empirical_dist.shape[0])
                data["empirical"] = empirical_dist[
                    to_bin(data["mean_phase_segmentation"][0], empirical_dist.shape[0])
                ].unsqueeze(0)
                mask = data["mean_phase_segmentation"].isnan()
                data["empirical"][mask] = 1.0 / empirical_dist.shape[0]

        data["y_rad_binned"] = (
            to_bin(data["y_rad"], self.target_ntheta).unsqueeze(0).to(torch.long)
        )
        data["craft_y_rad_binned"] = (
            to_bin(data["craft_y_rad"], self.target_ntheta).unsqueeze(0).to(torch.long)
        )

        # convert to target dtype on CPU!
        for key in data:
            if isinstance(data[key], torch.Tensor) and data[key].dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
            ):
                data[key] = data[key].to(self.target_dtype)
        return data


class v5spfdataset(Dataset):
    def __init__(
        self,
        prefix: str,  # Base path for dataset files (without .zarr extension)
        nthetas: int,  # Number of theta angles for beamforming discretization
        precompute_cache: str,  # Directory to store/load precomputed segmentation data
        phi_drift_max: float = 0.2,  # Maximum allowable phase drift for quality control
        min_mean_windows: int = 10,  # Minimum average number of windows required for valid segmentation
        ignore_qc: bool = False,  # Skip quality control checks if True
        paired: bool = False,  # If True, return paired samples from all receivers at once
        gpu: bool = False,  # Use GPU for segmentation computation if available
        snapshots_per_session: int = 1,  # Number of snapshots to include in each returned session
        tiled_sessions: bool = True,  # If True, sessions overlap with stride; if False, sessions are disjoint
        snapshots_stride: int = 1,  # Step size between consecutive session starting points
        readahead: bool = False,  # Enable read-ahead for zarr storage I/O optimization
        temp_file: bool = False,  # Whether dataset is using temporary files (for in-progress recordings)
        temp_file_suffix: str = ".tmp",  # Suffix for temporary files
        skip_fields: List[
            str
        ] = [],  # Data fields to exclude during loading to save memory
        n_parallel: int = 20,  # Number of parallel processes for segmentation
        empirical_data_fn: (
            str | None
        ) = None,  # Path to empirical distribution data file for phase-to-angle mapping
        empirical_individual_radio: bool = False,  # Use per-radio empirical distributions if True
        empirical_symmetry: bool = True,  # Use symmetric empirical distributions if True
        target_dtype=torch.float32,  # Target dtype for tensor conversion (memory optimization)
        snapshots_adjacent_stride: int = 1,  # Stride between adjacent snapshots within a session
        flip: bool = False,  # Randomly flip data horizontally for data augmentation
        double_flip: bool = False,  # Apply additional flipping transformation for data augmentation
        random_adjacent_stride: bool = False,  # Use random non-uniform strides between adjacent snapshots
        distance_normalization: int = 1000,  # Divisor to normalize distance measurements (mm to meters)
        target_ntheta: (
            bool | None
        ) = None,  # Target number of theta bins for classification (defaults to nthetas)
        segmentation_version: float = SEGMENTATION_VERSION,  # Version of segmentation algorithm (3.5 by default)
        segment_if_not_exist: bool = False,  # Generate segmentation cache if missing when True
        windows_per_snapshot: int = 256,  # Maximum number of windows per snapshot to use
        skip_detrend: bool = False,
        vehicle_type: str = "",
        v4: bool = False,
        realtime: bool = False,
    ):
        logging.debug(f"loading... {prefix}")
        # Store configuration parameters
        self.n_parallel = n_parallel
        self.exclude_keys_from_cache = set(["signal_matrix"])
        self.readahead = readahead
        self.precompute_cache = precompute_cache
        self.nthetas = nthetas  # Number of angles to discretize space for beamforming
        self.target_ntheta = self.nthetas if target_ntheta is None else target_ntheta
        self.valid_entries = None
        self.temp_file = temp_file

        self.realtime = realtime

        # Segmentation parameters control how raw signal is processed into windows
        # and how phase difference is computed between antenna elements
        self.segmentation_version = (
            segmentation_version  # Controls algorithm version for signal segmentation
        )
        self.segment_if_not_exist = (
            segment_if_not_exist  # Auto-generate segmentation if missing
        )
        self.skip_detrend = skip_detrend

        self.distance_normalization = distance_normalization
        self.flip = flip
        self.double_flip = double_flip
        self.skip_fields = skip_fields
        self.paired = paired
        self.gpu = gpu  # Whether to use GPU acceleration for beamforming calculations
        self.target_dtype = target_dtype
        self.precomputed_entries = 0
        self.precomputed_zarr = (
            None  # Will hold preprocessed beamforming and segmentation data
        )

        # Configure snapshot and session settings
        self.snapshots_per_session = snapshots_per_session
        self.snapshots_adjacent_stride = snapshots_adjacent_stride
        self.random_adjacent_stride = random_adjacent_stride
        self.windows_per_snapshot = (
            windows_per_snapshot  # Max windows per snapshot for segmentation
        )
        self.tiled_sessions = tiled_sessions

        # Set up file paths
        self.prefix = prefix.replace(".zarr", "")
        self.zarr_fn = f"{self.prefix}.zarr"  # Raw signal data storage
        self.yaml_fn = f"{self.prefix}.yaml"  # Configuration file
        if temp_file:
            self.zarr_fn += temp_file_suffix
            self.yaml_fn += temp_file_suffix
            assert self.snapshots_per_session == 1

        # Load data files
        self.z = zarr_open_from_lmdb_store(
            self.zarr_fn, readahead=self.readahead, map_size=2**32
        )
        self.yaml_config = load_config(self.yaml_fn)

        self.v4 = v4  # self.yaml_config["data-version"] == 4

        # Get system metadata
        if vehicle_type != "":
            self.vehicle_type = vehicle_type
        else:
            self.vehicle_type = (
                self.get_collector_identifier()
            )  # Type of platform (wallarray/rover)
        self.emitter_type = self.get_emitter_type()
        self.n_receivers = len(self.yaml_config["receivers"])

        # Configure stride for session sampling
        if snapshots_stride < 1.0:
            snapshots_stride = max(
                int(
                    snapshots_stride
                    * self.snapshots_per_session
                    * self.snapshots_adjacent_stride
                ),
                1,
            )
        self.snapshots_stride = int(snapshots_stride)

        # Extract receiver properties - important for beamforming calculations
        self.wavelengths = [
            speed_of_light / receiver["f-carrier"]  # λ = c/f
            for receiver in self.yaml_config["receivers"]
        ]
        self.carrier_frequencies = [
            receiver["f-carrier"] for receiver in self.yaml_config["receivers"]
        ]
        self.rf_bandwidths = [
            receiver["bandwidth"] for receiver in self.yaml_config["receivers"]
        ]

        # Validate that all receivers have consistent configurations
        for rx_idx in range(1, self.n_receivers):
            assert (
                self.yaml_config["receivers"][0]["antenna-spacing-m"]
                == self.yaml_config["receivers"][rx_idx]["antenna-spacing-m"]
            ), self.zarr_fn
            assert self.wavelengths[0] == self.wavelengths[rx_idx]
            assert self.rf_bandwidths[0] == self.rf_bandwidths[rx_idx]

        # Set up receiver spacing properties - critical for beamforming
        # Spacing between antenna elements affects phase difference and angle estimation
        self.rx_spacing = self.yaml_config["receivers"][0]["antenna-spacing-m"]
        assert self.yaml_config["receivers"][1]["antenna-spacing-m"] == self.rx_spacing

        # rx_wavelength_spacing (d/λ) is a key parameter for beamforming
        # It determines how phase differences map to arrival angles
        self.rx_wavelength_spacing = self.rx_spacing / self.wavelengths[0]

        # Create receiver configs and determine device types
        self.rx_configs = [
            rx_config_from_receiver_yaml(receiver)
            for receiver in self.yaml_config["receivers"]
        ]
        self.sdr_device_types = [
            uri_to_device_type(rx_config.uri) for rx_config in self.rx_configs
        ]

        # Ensure all receivers use the same device type
        if len(self.sdr_device_types) > 1:
            for device_type in self.sdr_device_types:
                assert device_type == self.sdr_device_types[0]
        self.sdr_device_type = self.sdr_device_types[0]

        # Access receiver data
        self.receiver_data = [
            self.z.receivers[f"r{ridx}"] for ridx in range(self.n_receivers)
        ]
        self.v4_to_v5()

        # Get dimensions from the data
        self.n_snapshots, self.n_antennas_per_receiver = self.z.receivers.r0.gains.shape
        assert self.n_antennas_per_receiver == 2  # 2-element antenna array

        # Handle snapshot count settings
        if self.snapshots_per_session == -1:
            self.snapshots_per_session = self.n_snapshots

        # Calculate number of sessions based on tiling mode
        if self.tiled_sessions:
            self.n_sessions = (
                self.n_snapshots
                - self.snapshots_per_session * snapshots_adjacent_stride
            ) // self.snapshots_stride + 1
            if self.n_sessions <= 0:
                self.n_sessions = 0
        else:
            self.n_sessions = self.n_snapshots // (
                self.snapshots_per_session * snapshots_adjacent_stride
            )

        # Load signal matrices if needed (raw complex IQ samples from SDR)
        if "signal_matrix" not in self.skip_fields:
            self.signal_matrices = [
                self.z.receivers[f"r{ridx}"].signal_matrix
                for ridx in range(self.n_receivers)
            ]
            _, _, self.session_length = self.z.receivers["r0"].signal_matrix.shape

            # Verify signal matrix dimensions
            for ridx in range(self.n_receivers):
                assert self.z.receivers[f"r{ridx}"].signal_matrix.shape == (
                    self.n_snapshots,
                    self.n_antennas_per_receiver,
                    self.session_length,
                )

        # Precompute steering vectors for beamforming
        # Steering vectors are complex weights applied to each antenna element
        # They're used to "steer" the array to look in a specific direction
        # For each possible angle (theta), calculate the appropriate phase shifts
        self.steering_vectors = [
            precompute_steering_vectors(
                receiver_positions=rx_config.rx_pos,
                carrier_frequency=rx_config.lo,
                spacing=nthetas,
            )
            for rx_config in self.rx_configs
        ]

        # Define keys to load per session
        self.keys_per_session = (
            v5rx_f64_keys + v5rx_2xf64_keys + ["rx_wavelength_spacing"]
        )
        if "signal_matrix" not in self.skip_fields:
            self.keys_per_session.append("signal_matrix")

        # Load cached data
        self.refresh()

        # Validate receiver configurations
        if not self.temp_file and not self.v4:
            assert (
                self.cached_keys[0]["rx_theta_in_pis"].median()
                == self.yaml_config["receivers"][0]["theta-in-pis"]
            )
            assert (
                self.cached_keys[1]["rx_theta_in_pis"].median()
                == self.yaml_config["receivers"][1]["theta-in-pis"]
            )

        # Create expanded receiver indices for batching
        self.receiver_idxs_expanded = {}
        for idx in range(self.n_receivers):
            self.receiver_idxs_expanded[idx] = torch.tensor(
                idx, dtype=torch.int32
            ).expand(1, self.snapshots_per_session)

        # Load segmentation data for non-temporary files
        # Segmentation processes raw signal data into usable features:
        # 1. Identifies windows containing the target signal
        # 2. Calculates phase differences between antenna elements
        # 3. Computes beamforming outputs for different steering directions
        # 4. Creates masks and statistics for signal quality assessment
        if not self.temp_file:
            self.get_segmentation(
                version=self.segmentation_version,
                segment_if_not_exist=self.segment_if_not_exist,
            )

            # Calculate phase drifts and segmentation statistics
            # Phase drift measures difference between expected and measured phase
            # It's a quality metric for the dataset
            self.all_phi_drifts = self.get_all_phi_drifts()
            self.phi_drifts = torch.tensor(
                [torch.nanmean(all_phi_drift) for all_phi_drift in self.all_phi_drifts]
            )

            # Calculate statistics on segmentation quality
            if "simple_segmentations" not in self.skip_fields:
                # Average number of identified signal windows per session
                self.average_windows_in_segmentation = (
                    torch.tensor(
                        [
                            [
                                len(x["simple_segmentation"])
                                for x in self.segmentation["segmentation_by_receiver"][
                                    f"r{rx_idx}"
                                ]
                            ]
                            for rx_idx in range(self.n_receivers)
                        ]
                    )
                    .to(torch.float32)
                    .mean()
                )
                # Proportion of sessions with meaningful segmentation
                self.mean_sessions_with_maybe_valid_segmentation = (
                    torch.tensor(
                        [
                            [
                                len(x["simple_segmentation"]) > 2
                                for x in self.segmentation["segmentation_by_receiver"][
                                    f"r{rx_idx}"
                                ]
                            ]
                            for rx_idx in [0, 1]
                        ]
                    )
                    .to(torch.float32)
                    .mean()
                )
        else:
            # Initialize mean phase for temporary files
            self.mean_phase = {}
            for ridx in range(self.n_receivers):
                self.mean_phase[f"r{ridx}"] = torch.ones(len(self)) * torch.inf

        # Perform quality control checks if enabled
        # Ensures the dataset meets minimum quality standards:
        # - Phase drift within acceptable range
        # - Sufficient signal windows identified
        # - Enough sessions with valid segmentation
        if not ignore_qc:
            if "simple_segmentations" in self.skip_fields:
                raise ValueError("Cannot have QC and skip simple segmentations")
            assert not temp_file
            if abs(self.phi_drifts).max() > phi_drift_max:
                raise ValueError(
                    f"Phi-drift is too high! max acceptable is {phi_drift_max}, actual is {self.phi_drifts}"
                )
            if self.average_windows_in_segmentation < min_mean_windows:
                raise ValueError(
                    f"Min average windows ({self.average_windows_in_segmentation}) is below the min expected ({min_mean_windows})"
                )
            if self.mean_sessions_with_maybe_valid_segmentation < 0.2:
                raise ValueError(
                    "It looks like too few windows have a valid segmentation"
                )

        # Load empirical distribution data if provided
        # These are learned phase-to-angle mappings that can improve angle estimation
        if empirical_data_fn is not None:
            self.empirical_data_fn = empirical_data_fn
            self.empirical_data = pickle.load(open(empirical_data_fn, "rb"))
            self.empirical_individual_radio = empirical_individual_radio
            self.empirical_symmetry = empirical_symmetry
        else:
            self.empirical_data_fn = None
            self.empirical_data = None

    def refresh(self):
        # get how many entries are in the underlying storage
        # recompute if we need to

        self.z = zarr_open_from_lmdb_store(
            self.zarr_fn, readahead=self.readahead, map_size=2**32
        )
        valid_entries = [
            (self.z[f"receivers/r{ridx}/system_timestamp"][:] > 0).sum()
            for ridx in range(self.n_receivers)
        ]

        # TODO only copy new entries!
        if self.valid_entries is None or self.valid_entries != valid_entries:
            old_n = 0 if self.valid_entries is None else min(self.valid_entries)
            new_n = max(valid_entries)
            if old_n == 0:
                self.cached_keys = {}
            for receiver_idx in range(self.n_receivers):
                if receiver_idx not in self.cached_keys:
                    self.cached_keys[receiver_idx] = {}
                for key in self.keys_per_session:
                    if key == "rx_wavelength_spacing":
                        continue
                    if key in self.exclude_keys_from_cache:
                        continue
                    # assert key != "signal_matrix"  # its complex shouldnt get converted!
                    if old_n == 0:
                        if key in self.receiver_data[receiver_idx]:
                            self.cached_keys[receiver_idx][key] = torch.as_tensor(
                                self.receiver_data[receiver_idx][key][:].astype(
                                    np.float32
                                )  # TODO save as float32?
                            )
                        elif key in ("rx_heading_in_pis",):
                            self.cached_keys[receiver_idx][key] = (
                                torch.as_tensor(
                                    self.receiver_data[receiver_idx][
                                        "system_timestamp"
                                    ][:].astype(
                                        np.float32
                                    )  # TODO save as float32?
                                )
                                * 0
                            )
                        else:
                            raise ValueError(f"Missing key {key}")

                    else:
                        self.cached_keys[receiver_idx][key][old_n:new_n] = (
                            torch.as_tensor(
                                self.receiver_data[receiver_idx][key][
                                    old_n:new_n
                                ].astype(np.float32)
                            )
                        )
                # optimize rx / tx positions
                self.cached_keys[receiver_idx]["rx_pos_mm"] = torch.vstack(
                    [
                        self.cached_keys[receiver_idx]["rx_pos_x_mm"],
                        self.cached_keys[receiver_idx]["rx_pos_y_mm"],
                    ]
                ).T

                self.cached_keys[receiver_idx]["tx_pos_mm"] = torch.vstack(
                    [
                        self.cached_keys[receiver_idx]["tx_pos_x_mm"],
                        self.cached_keys[receiver_idx]["tx_pos_y_mm"],
                    ]
                ).T

                self.cached_keys[receiver_idx]["rx_wavelength_spacing"] = (
                    self.cached_keys[receiver_idx]["rx_spacing"]
                    / self.wavelengths[receiver_idx]
                )
            self.valid_entries = valid_entries

            self.ground_truth_thetas, self.absolute_thetas = (
                self.get_ground_truth_thetas()
            )
            self.ground_truth_phis = self.get_ground_truth_phis()
            self.craft_ground_truth_thetas = self.get_craft_ground_truth_thetas()

            rx_spacing_mismatches = (
                (self.cached_keys[0]["rx_spacing"] != self.cached_keys[1]["rx_spacing"])
                .to(float)
                .sum()
            )
            assert (
                rx_spacing_mismatches < 30
            ), f"Too many mismatches in rx_spacing {rx_spacing_mismatches}"
            return True
        return False

    def get_emitter_type(self):
        if "emitter" in self.yaml_config and "type" in self.yaml_config["emitter"]:
            return self.yaml_config["emitter"]["type"]
        return "external"

    def close(self):
        # let workers open their own
        self.z.store.close()
        self.z = None
        self.receiver_data = None
        # try and close segmentation
        self.segmentation = None
        if self.precomputed_zarr is not None:
            self.precomputed_zarr.store.close()
        self.precomputed_zarr = None

    def estimate_phi(self, data):
        x = torch.as_tensor(
            data["all_windows_stats"]
        )  # all_window_stats = trimmed_cm, trimmed_stddev, abs_signal_median
        seg_mask = torch.as_tensor(data["downsampled_segmentation_mask"])

        return torch.mul(x, seg_mask).sum(axis=2) / (seg_mask.sum(axis=2) + 0.001)

    def __len__(self):
        if self.paired:
            return self.n_sessions
        return self.n_sessions * self.n_receivers

    def v4_to_v5(self):
        if not self.v4:
            return
        # v4 to v5 upgrade
        for ridx in range(self.n_receivers):
            self.receiver_data[ridx] = ZarrWrapper(
                self.receiver_data[ridx]
            )  # wrap readonly so we can modify on the fly
            filler = self.receiver_data[ridx]["system_timestamp"][:] * 0
            self.receiver_data[ridx]["rx_heading_in_pis"] = (
                self.receiver_data[ridx]["heading"][:] / 360
            ) / 2
            for key in v5rx_f64_keys:
                if key not in self.receiver_data[ridx]:
                    self.receiver_data[ridx][key] = filler

    def reinit(self):
        if self.z is None:
            # worker_info = torch.utils.data.get_worker_info()

            self.z = zarr_open_from_lmdb_store(self.zarr_fn, readahead=self.readahead)
            self.receiver_data = [
                self.z.receivers[f"r{ridx}"] for ridx in range(self.n_receivers)
            ]
            self.v4_to_v5()

        if not self.temp_file and self.precomputed_zarr is None:
            self.get_segmentation(
                version=self.segmentation_version,
                segment_if_not_exist=self.segment_if_not_exist,
            )
            # self.precomputed_zarr = zarr_open_from_lmdb_store(
            #     self.results_fn().replace(".pkl", ".yarr"), mode="r"
            # )

    def get_spacing_identifier(self):
        rx_lo = self.cached_keys[0]["rx_lo"].median().item()
        return f"sp{self.rx_spacing:0.3f}.rxlo{rx_lo:0.4e}"

    def get_collector_identifier(self):
        if "collector" in self.yaml_config and "type" in self.yaml_config["collector"]:
            return self.yaml_config["collector"]["type"]
        if "rover" in self.zarr_fn:
            return "rover"
        return "wallarray"

    def get_wavelength_identifier(self):
        rx_lo = self.cached_keys[0]["rx_lo"].median().item()
        return f"sp{self.rx_spacing:0.3f}.rxlo{rx_lo:0.4e}.wlsp{self.rx_wavelength_spacing:0.3f}"

    def get_values_at_key(self, key, receiver_idx, idxs):
        if key == "signal_matrix":
            return torch.as_tensor(self.receiver_data[receiver_idx][key][idxs])[None]
        return self.cached_keys[receiver_idx][key][idxs]

    def get_session_idxs(self, session_idx):
        return get_session_idxs(
            session_idx,
            tiled_sessions=self.tiled_sessions,
            snapshots_stride=self.snapshots_stride,
            snapshots_adjacent_stride=self.snapshots_adjacent_stride,
            snapshots_per_session=self.snapshots_per_session,
            random_adjacent_stride=self.random_adjacent_stride,
        )

    def render_session(self, receiver_idx, session_idx, double_flip=False):
        self.reinit()

        flip_left_right = self.flip and (torch.rand(1) > 0.5).item()
        # flip_up_down = self.flip and (torch.rand(1) > 0.5).item() # MOVED TO LOSS FUNCTION

        snapshot_idxs = self.get_session_idxs(session_idx)

        data = {
            # key: r[key][snapshot_start_idx:snapshot_end_idx]
            key: self.get_values_at_key(key, receiver_idx, snapshot_idxs)
            for key in self.keys_per_session
            # 'rx_theta_in_pis', 'rx_spacing', 'rx_lo', 'rx_bandwidth', 'avg_phase_diff', 'rssis', 'gains']
        }

        data["gains"] = data["gains"][:, None]
        data["receiver_idx"] = self.receiver_idxs_expanded[receiver_idx]
        data["ground_truth_theta"] = self.ground_truth_thetas[receiver_idx][
            snapshot_idxs
        ]
        data["absolute_theta"] = self.absolute_thetas[receiver_idx][snapshot_idxs]
        data["ground_truth_phi"] = self.ground_truth_phis[receiver_idx][snapshot_idxs]
        data["craft_ground_truth_theta"] = self.craft_ground_truth_thetas[snapshot_idxs]
        data["vehicle_type"] = torch.tensor(
            [encode_vehicle_type(self.vehicle_type)]
        ).reshape(1)
        data["sdr_device_type"] = torch.tensor([self.sdr_device_type.value]).reshape(1)

        # duplicate some fields
        # no idea how to flip phi
        craft_offset = (
            data["craft_ground_truth_theta"] - data["ground_truth_theta"]
        )  # this is (rx_theta_in_pis + rx_head_in_pis ) *torch.pi
        if flip_left_right or double_flip:
            data["ground_truth_phi"] = -data["ground_truth_phi"]
            data["ground_truth_theta"] = -data["ground_truth_theta"]
        if double_flip:
            # compute the flipped gt theta
            data["ground_truth_theta"] = (
                data["ground_truth_theta"].sign() * torch.pi
                - data["ground_truth_theta"]
            )
            # phi shouldnt change
        # update craft theta
        data["craft_ground_truth_theta"] = torch_pi_norm(
            data["ground_truth_theta"] + craft_offset
        )
        # if double_flip:
        #     data["craft_ground_truth_theta"] = torch_pi_norm(
        #         data["craft_ground_truth_theta"] + torch.pi
        #     )

        data["y_rad"] = data["ground_truth_theta"]
        data["y_phi"] = data["ground_truth_phi"]
        data["craft_y_rad"] = data["craft_ground_truth_theta"]

        if "signal_matrix" not in self.skip_fields:
            # WARNGING this does not respect flipping!
            abs_signal = data["signal_matrix"].abs().to(torch.float32)
            assert data["signal_matrix"].shape[0] == 1
            pd = torch_get_phase_diff(data["signal_matrix"][0]).to(torch.float32)
            data["abs_signal_and_phase_diff"] = torch.concatenate(
                [abs_signal, pd[None, :, None]], dim=2
            )

        # find out if this is a temp file and we either need to precompute, or its not ready
        if self.temp_file and self.precomputed_entries <= session_idx:
            self.get_segmentation(
                version=self.segmentation_version,
                precompute_to_idx=session_idx,
                segment_if_not_exist=True,
            )

        data.update(
            data_from_precomputed(
                v5ds=self,
                precomputed_data=self.precomputed_zarr[f"r{receiver_idx}"],
                segmentation=(
                    self.segmentation["segmentation_by_receiver"][f"r{receiver_idx}"]
                    if not "simple_segmentations" in self.skip_fields
                    else {}
                ),
                snapshot_idxs=snapshot_idxs,
            )
        )
        # port this over in on the fly TODO

        data["mean_phase_segmentation"] = self.mean_phase[f"r{receiver_idx}"][
            snapshot_idxs
        ].unsqueeze(0)

        # TODO HANDLE NAN BETTER!!
        # data["mean_phase_segmentation"][
        #     torch.isnan(data["mean_phase_segmentation"])
        # ] = 0.0

        if flip_left_right or double_flip:
            # data["all_windows_stats"] ~ batch, snapshots, channel, window
            # channels are trimmed mean, stddev, abs_signal_median
            # need to flip mean (phase diff)
            data["all_windows_stats"][:, :, 0] = -data["all_windows_stats"][:, :, 0]
            data["mean_phase_segmentation"] = -data["mean_phase_segmentation"]

        data["rx_pos_xy"] = (
            self.cached_keys[receiver_idx]["rx_pos_mm"][snapshot_idxs].unsqueeze(0)
            / self.distance_normalization
        )

        data["tx_pos_xy"] = (
            self.cached_keys[receiver_idx]["tx_pos_mm"][snapshot_idxs].unsqueeze(0)
            / self.distance_normalization
        )

        if double_flip:
            data["rx_pos_xy"] = -data["rx_pos_xy"]
            data["tx_pos_xy"] = -data["tx_pos_xy"]

        if self.empirical_data is not None:
            empirical_dist = get_empirical_dist(self, receiver_idx)
            #  ~ 1, snapshots, ntheta(empirical_dist.shape[0])
            data["empirical"] = empirical_dist[
                to_bin(data["mean_phase_segmentation"][0], empirical_dist.shape[0])
            ].unsqueeze(0)
            mask = data["mean_phase_segmentation"].isnan()
            data["empirical"][mask] = 1.0 / empirical_dist.shape[0]

        data["y_rad_binned"] = (
            to_bin(data["y_rad"], self.target_ntheta).unsqueeze(0).to(torch.long)
        )
        data["craft_y_rad_binned"] = (
            to_bin(data["craft_y_rad"], self.target_ntheta).unsqueeze(0).to(torch.long)
        )

        # convert to target dtype on CPU!
        for key in data:
            if isinstance(data[key], torch.Tensor) and data[key].dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
            ):
                data[key] = data[key].to(self.target_dtype)

        if flip_left_right or double_flip:
            # y_rad binned depends on y_rad so we should be ok?
            # already flipped in mean phase
            # data["empirical"] = data["empirical"].flip(dims=(2,))
            data["weighted_beamformer"] = data["weighted_beamformer"].flip(dims=(2,))
            if "windowed_beamformer" in data:
                data["windowed_beamformer"] = data["windowed_beamformer"].flip(
                    dims=(3,)
                )

        return data

    def get_ground_truth_phis(self):
        ground_truth_phis = []
        for ridx in range(self.n_receivers):
            ground_truth_phis.append(
                torch_pi_norm(
                    -torch.sin(
                        self.ground_truth_thetas[
                            ridx
                        ]  # this is theta relative to our array!
                        # + self.receiver_data[ridx]["rx_theta_in_pis"][:] * np.pi
                    )  # up to negative sign, which way do we spin?
                    # or maybe this is the order of the receivers 0/1 vs 1/0 on the x-axis
                    # pretty sure this (-) is more about which receiver is closer to x+/ish
                    # a -1 here is the same as -rx_spacing!
                    * self.rx_wavelength_spacing
                    * 2
                    * torch.pi
                )
            )
        return torch.vstack(ground_truth_phis)

    def get_all_phi_drifts(self):
        all_phi_drifts = []
        for ridx in range(self.n_receivers):
            a = self.ground_truth_phis[ridx]
            b = self.mean_phase[f"r{ridx}"]
            fwd = torch_pi_norm(b - a)
            # bwd = pi_norm(a + np.pi - b)
            # mask = np.abs(fwd) < np.abs(bwd)
            # c = np.zeros(a.shape)
            # c[mask] = fwd[mask]
            # c[~mask] = bwd[~mask]
            c = fwd
            all_phi_drifts.append(c)
        return all_phi_drifts

    # ground truth theta is relative to heading of the craft!!!!
    def get_craft_ground_truth_thetas(self):
        craft_ground_truth_thetas = torch_pi_norm(
            self.ground_truth_thetas[0]
            + (
                self.cached_keys[0]["rx_theta_in_pis"][:]
                # + self.cached_keys[0]["rx_heading_in_pis"][:] # if we add this then its relative to absolute 0deg not craft!
            )
            * torch.pi
        )
        for ridx in range(1, self.n_receivers):
            _craft_ground_truth_thetas = torch_pi_norm(
                self.ground_truth_thetas[ridx]
                + (
                    self.cached_keys[ridx]["rx_theta_in_pis"][:]
                    # + self.cached_keys[ridx]["rx_heading_in_pis"][:]# if we add this then its relative to absolute 0deg not craft!
                )
                * torch.pi
            )
            error = abs(
                torch_pi_norm(craft_ground_truth_thetas - _craft_ground_truth_thetas)
            ).mean()
            if error > 0.2:
                logging.warning(
                    f"{self.zarr_fn} craft_theta off by too much! {error}"
                )  # this might not scale well to larger examples
            # i think the error gets worse
        return craft_ground_truth_thetas

    def get_ground_truth_thetas(self):
        ground_truth_thetas = []
        absolute_thetas = []
        for ridx in range(self.n_receivers):
            tx_pos = self.cached_keys[ridx]["tx_pos_mm"].T
            rx_pos = self.cached_keys[ridx]["rx_pos_mm"].T

            # compute the angle of the tx with respect to rx
            d = tx_pos - rx_pos

            rx_to_tx_theta = torch.arctan2(d[0], d[1])
            rx_theta_in_pis = self.cached_keys[ridx]["rx_theta_in_pis"]
            rx_heading_in_pis = self.cached_keys[ridx]["rx_heading_in_pis"]
            absolute_thetas.append(torch_pi_norm(rx_to_tx_theta))
            ground_truth_thetas.append(
                torch_pi_norm(
                    rx_to_tx_theta - (rx_theta_in_pis[:] + rx_heading_in_pis) * torch.pi
                )
            )
        # reduce GT thetas in case of two antennas
        # in 2D there are generally two spots that satisfy phase diff
        # lets pick the one thats infront of the craft (array)
        assert self.n_receivers == 2
        return torch.vstack(ground_truth_thetas), torch.vstack(absolute_thetas)

    def get_mean_phase(self):
        self.mean_phase = {
            f"r{r_idx}": torch.tensor(self.precomputed_zarr[f"r{r_idx}/mean_phase"][:])
            for r_idx in range(self.n_receivers)
        }

    def __getitem__(self, idx):
        if self.paired:
            assert idx < self.n_snapshots
            if self.temp_file:
                for ridx in range(self.n_receivers):
                    if idx >= self.valid_entries[ridx]:
                        return None
            double_flip = self.double_flip and (torch.rand(1) > 0.5).item()
            return [
                self.render_session(receiver_idx, idx, double_flip=double_flip)
                for receiver_idx in range(self.n_receivers)
            ]
        assert idx < self.n_snapshots * self.n_receivers
        receiver_idx = idx % self.n_receivers
        session_idx = idx // self.n_receivers
        if self.temp_file and session_idx >= self.valid_entries[receiver_idx]:
            return None
        return [self.render_session(receiver_idx, session_idx)]

    def get_estimated_thetas(self):
        estimated_thetas = {}
        for ridx in range(self.n_receivers):
            carrier_freq = self.yaml_config["receivers"][ridx]["f-carrier"]
            antenna_spacing = self.yaml_config["receivers"][ridx]["antenna-spacing-m"]
            estimated_thetas[f"r{ridx}"] = phase_diff_to_theta(
                self.mean_phase[f"r{ridx}"].numpy(),
                speed_of_light / carrier_freq,
                antenna_spacing,
                large_phase_goes_right=False,
            )
        return estimated_thetas

    def results_fn(self):
        return os.path.join(
            self.precompute_cache,
            os.path.basename(self.prefix).replace("_nosig", "")
            + f"_segmentation_nthetas{self.nthetas}.pkl",
        )

    def get_segmentation_version(self, precomputed_zarr, segmentation):
        if "version" in precomputed_zarr:
            if precomputed_zarr["version"].shape == ():
                return 3.0
            return precomputed_zarr["version"][0]
        if segmentation is None or "version" not in segmentation:
            return None
        return segmentation["version"]

    def get_segmentation(self, version, segment_if_not_exist, precompute_to_idx=-1):
        """
        Load or generate the signal segmentation and beamforming data.

        Segmentation is a critical preprocessing step that:
        1. Divides raw signal into windows (typically 2048 samples each)
        2. Identifies which windows contain the actual RF signal vs noise
        3. Computes phase differences between antenna elements
        4. Applies beamforming across multiple look directions (thetas)
        5. Extracts statistical features from the segmented signals

        Args:
            version: The segmentation algorithm version to use
            segment_if_not_exist: Whether to generate segmentation if missing
            precompute_to_idx: Index until which to precompute (-1 for all)

        Returns:
            Segmentation data dictionary
        """
        # Return cached segmentation if already loaded
        if (
            not self.temp_file
            and hasattr(self, "segmentation")
            and self.segmentation is not None
        ):
            return self.segmentation

        # Set up paths for segmentation results
        results_fn = self.results_fn()

        # Check if segmentation exists and if we should create it
        if (
            not segment_if_not_exist
            and not self.temp_file
            and not os.path.exists(results_fn)
        ):
            raise ValueError(f"Segmentation file does not exist for {results_fn}")

        # Generate segmentation if needed (file doesn't exist or is a temp file)
        if self.temp_file or not os.path.exists(results_fn):
            skip_beamformer = False
            if (
                self.temp_file
                and "windowed_beamformer" in self.skip_fields
                and "weighted_beamformer" in self.skip_fields
            ):
                skip_beamformer = True

            # Call the multiprocessing segmentation function
            # This performs the heavy computation:
            # - Window the signal data
            # - Identify signal-containing segments using statistics
            # - Compute beamforming outputs for each direction in nthetas
            # - Save results to yarr file for efficient loading
            mp_segment_zarr(
                self.zarr_fn,  # Input raw signal data
                results_fn,  # Output segmentation path
                self.steering_vectors,  # Precomputed steering vectors for beamforming
                precompute_to_idx=precompute_to_idx,  # How many samples to process
                gpu=self.gpu,  # Whether to use GPU acceleration
                n_parallel=self.n_parallel,  # Number of parallel workers
                skip_beamformer=skip_beamformer,  # Skip beamforming calculation if not needed
                temp_file=self.temp_file,
                skip_detrend=self.skip_detrend,
            )

        # Load the segmentation results
        try:
            # Load the segmentation data if needed
            if "simple_segmentations" not in self.skip_fields:
                segmentation = pickle.load(open(results_fn, "rb"))
            else:
                segmentation = {}

            # Open the precomputed data (contains beamforming outputs and statistics)
            # This yarr file contains:
            # - windowed_beamformer: Beamforming output for each window and angle
            # - all_windows_stats: Statistics for each window (phase mean, stddev, signal strength)
            # - downsampled_segmentation_mask: Binary mask indicating signal vs noise windows
            # - mean_phase: Average phase difference across all signal windows
            precomputed_zarr = zarr_open_from_lmdb_store(
                results_fn.replace(".pkl", ".yarr"),
                mode="r",
                map_size=2**32,
                readahead=False,  # Readahead disabled to avoid excessive I/O
            )

            # For temp files, determine how many entries are valid
            if self.temp_file:
                self.precomputed_entries = min(
                    [
                        (
                            np.mean(
                                precomputed_zarr[f"r{r_idx}/all_windows_stats"],
                                axis=(1, 2),
                            )
                            > 0
                        ).sum()
                        for r_idx in range(self.n_receivers)
                    ]
                )
        except pickle.UnpicklingError:
            # Handle corrupted segmentation file by removing and regenerating
            os.remove(results_fn)
            return self.get_segmentation(
                version=version,
                segment_if_not_exist=segment_if_not_exist,
                precompute_to_idx=precompute_to_idx,
            )

        # Verify segmentation version matches expected version
        current_version = self.get_segmentation_version(precomputed_zarr, segmentation)
        if not np.isclose(current_version, version):
            raise ValueError(
                f"Expected to generate a segmentation, but one already exists with different version {version} vs {current_version}"
            )

        # Store the loaded data
        self.segmentation = segmentation
        self.precomputed_zarr = precomputed_zarr

        # Extract mean phase information from the precomputed data
        # Mean phase is critical for angle-of-arrival estimation
        self.get_mean_phase()

        return self.segmentation


class SessionsDatasetSimulated(Dataset):
    def __init__(self, root_dir, snapshots_per_session=5):
        """
        Arguments:
          root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.args = load("/".join([self.root_dir, "args.pkl"]), compression="lzma")
        assert self.args.time_steps >= snapshots_per_session
        self.samples_per_session = self.args.time_steps - snapshots_per_session + 1
        self.snapshots_in_sample = snapshots_per_session
        if not self.args.live:
            print("NOT LIVE")
            self.filenames = sorted(
                filter(
                    lambda x: "args" not in x,
                    ["%s/%s" % (self.root_dir, x) for x in os.listdir(self.root_dir)],
                )
            )
            if self.args.sessions != len(
                self.filenames
            ):  # make sure its the right dataset
                print("WARNING DATASET LOOKS LIKE IT IS MISSING SOME SESSIONS!")

    def idx_to_filename_and_start_idx(self, idx):
        assert idx >= 0 and idx <= self.samples_per_session * len(self.filenames)
        return (
            self.filenames[idx // self.samples_per_session],
            idx % self.samples_per_session,
        )

    def __len__(self):
        if self.args.live:
            return self.samples_per_session * self.args.sessions
        else:
            return self.samples_per_session * len(self.filenames)

    def __getitem__(self, idx):
        session_idx = idx // self.samples_per_session
        start_idx = idx % self.samples_per_session

        if self.args.live:
            session = generate_session((self.args, session_idx))
        else:
            session = load(self.filenames[session_idx], compression="lzma")
        end_idx = start_idx + self.snapshots_in_sample
        return {k: session[k][start_idx:end_idx] for k in session.keys()}
