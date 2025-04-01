###
# Experiment 1 : wall drones , receiver center is x=135.27 cm, y=264.77cm, dist between is 6cm
###

import bisect
import logging
import os
import pickle
from contextlib import contextmanager
from enum import Enum
from functools import cache
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any, Tuple, Optional, Union, Generator
from pathlib import Path

import numpy as np
import torch
import yaml
from compress_pickle import load
from tensordict import TensorDict
from torch.utils.data import Dataset

from spf.dataset.segmentation import mp_segment_zarr
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
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto.sdr_controller import rx_config_from_receiver_yaml
from spf.utils import SEGMENTATION_VERSION, load_config, rx_spacing_to_str, to_bin


# from Stackoverflow
def yaml_as_dict(my_file: Union[str, Path]) -> Dict[str, Any]:
    my_dict: Dict[str, Any] = {}
    with open(my_file, "r") as fp:
        docs = yaml.safe_load_all(fp)
        for doc in docs:
            for key, value in doc.items():
                my_dict[key] = value
    return my_dict


def pos_to_rel(p: float, width: float) -> float:
    return 2 * (p / width - 0.5)


def rel_to_pos(r: float, width: float) -> float:
    return ((r / 2) + 0.5) * width


def encode_vehicle_type(vehicle_type: str) -> float:
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
) -> torch.Tensor:
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
def v5_collate_keys_fast(
    keys: List[str], batch: List[List[Dict[str, torch.Tensor]]]
) -> TensorDict:
    d: Dict[str, torch.Tensor] = {}
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
def v5_collate_all_fast(
    batch: List[List[Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    d: Dict[str, torch.Tensor] = {}
    for key in batch[0][0].keys():
        d[key] = torch.vstack(
            [x[key] for paired_sample in batch for x in paired_sample]
        )
    return d


def v5_collate_beamsegnet(
    batch: List[List[Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    n_windows = batch[0][0]["all_windows_stats"].shape[1]
    y_rad_list: List[torch.Tensor] = []
    y_phi_list: List[torch.Tensor] = []
    simple_segmentation_list: List[Any] = []
    all_window_stats_list: List[torch.Tensor] = []
    windowed_beamformers_list: List[torch.Tensor] = []
    downsampled_segmentation_mask_list: List[torch.Tensor] = []
    x_list: List[torch.Tensor] = []
    segmentation_mask_list: List[torch.Tensor] = []
    rx_spacing_list: List[torch.Tensor] = []
    craft_y_rad_list: List[torch.Tensor] = []
    receiver_idx_list: List[torch.Tensor] = []
    rx_pos_list: List[torch.Tensor] = []
    
    for paired_sample in batch:
        for x in paired_sample:
            y_rad_list.append(x["y_rad"])
            y_phi_list.append(x["y_phi"])
            rx_pos_list.append(x["rx_pos_xy"])
            craft_y_rad_list.append(x["craft_y_rad"])
            rx_spacing_list.append(x["rx_spacing"].reshape(-1, 1))
            all_window_stats_list.append(x["all_windows_stats"])
            windowed_beamformers_list.append(x["windowed_beamformer"])
            downsampled_segmentation_mask_list.append(x["downsampled_segmentation_mask"])
            receiver_idx_list.append(x["receiver_idx"].expand_as(x["y_rad"]))

            if "x" in batch[0][0]:
                x_list.append(x["x"])
                segmentation_mask_list.append(v5_segmentation_mask(x))

    d: Dict[str, torch.Tensor] = {
        "y_rad": torch.vstack(y_rad_list),
        "y_phi": torch.vstack(y_phi_list),
        "rx_pos_xy": torch.vstack(rx_pos_list),
        "receiver_idx": torch.vstack(receiver_idx_list),
        "craft_y_rad": torch.vstack(craft_y_rad_list),
        "rx_spacing": torch.vstack(rx_spacing_list),
        "all_windows_stats": torch.from_numpy(np.vstack(all_window_stats_list)),
        "windowed_beamformer": torch.from_numpy(np.vstack(windowed_beamformers_list)),
        "downsampled_segmentation_mask": torch.vstack(downsampled_segmentation_mask_list),
    }
    
    if "x" in batch[0][0]:
        d["x"] = torch.vstack(x_list)
        d["segmentation_mask"] = torch.vstack(segmentation_mask_list)
    return d


# using a spacing of at most max_offset and of at least 1
# get n random indexes, such that sum of the idxs is uniform
# across n,max_offset*n
def get_idx_for_rand_session(max_offset: int, n: int) -> np.ndarray:
    target_length = np.random.randint((max_offset - 1) * n)
    z = np.random.rand(n).cumsum()
    z /= z.max()  # normalize to sum pr to 1
    return (z * target_length).round().astype(int) + np.arange(n)


def v5_downsampled_segmentation_mask(
    session: Dict[str, Any], 
    n_windows: int
) -> torch.Tensor:
    window_size = 2048
    seg_mask = torch.zeros(len(session["simple_segmentations"]), 1, n_windows)
    for idx in range(seg_mask.shape[0]):
        for window in session["simple_segmentations"][idx]:
            seg_mask[
                idx,
                0,
                window["start_idx"] // window_size : window["end_idx"] // window_size,
            ] = 1
    return seg_mask


def v5_segmentation_mask(session: Dict[str, Any]) -> torch.Tensor:
    _, _, samples_per_session = session["x"].shape
    seg_mask = torch.zeros(1, 1, samples_per_session)
    for w in session["simple_segmentation"]:
        seg_mask[0, 0, w["start_idx"] : w["end_idx"]] = 1
    return seg_mask[:, None]


@contextmanager
def v5spfdataset_manager(
    *args: Any, 
    **kwds: Any
) -> Generator["v5spfdataset", None, None]:
    if "b2:" == kwds["precompute_cache"][:3]:
        b2_cache_folder: Optional[str] = None
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


def subsample_tensor(
    x: torch.Tensor, 
    dim: int, 
    new_size: int
) -> torch.Tensor:
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


def uri_to_device_type(uri: str) -> SDRDEVICE:
    if "bladerf" in uri:
        return SDRDEVICE.BLADERF2
    if "simulation" in uri:
        return SDRDEVICE.SIMULATION
    if "pluto" in uri or uri.startswith("usb") or uri.startswith("ip"):
        return SDRDEVICE.PLUTO
    return SDRDEVICE.UNKNOWN


class v5spfdataset(Dataset):
    def __init__(
        self,
        prefix: str,
        nthetas: int,
        precompute_cache: str,
        phi_drift_max: float = 0.2,
        min_mean_windows: int = 10,
        ignore_qc: bool = False,
        paired: bool = False,
        gpu: bool = False,
        snapshots_per_session: int = 1,
        tiled_sessions: bool = True,
        snapshots_stride: int = 1,
        readahead: bool = False,
        temp_file: bool = False,
        temp_file_suffix: str = ".tmp",
        skip_fields: List[str] = [],
        n_parallel: int = 20,
        empirical_data_fn: Optional[str] = None,
        empirical_individual_radio: bool = False,
        empirical_symmetry: bool = True,
        target_dtype: torch.dtype = torch.float32,
        snapshots_adjacent_stride: int = 1,
        flip: bool = False,
        double_flip: bool = False,
        random_adjacent_stride: bool = False,
        distance_normalization: int = 1000,
        target_ntheta: Optional[bool] = None,
        segmentation_version: float = SEGMENTATION_VERSION,
        segment_if_not_exist: bool = False,
        windows_per_snapshot: int = 256,
    ) -> None:
        logging.debug(f"loading... {prefix}")
        self.n_parallel = n_parallel
        self.exclude_keys_from_cache: set[str] = set(["signal_matrix"])
        self.readahead = readahead
        self.precompute_cache = precompute_cache
        self.nthetas = nthetas
        self.target_ntheta = self.nthetas if target_ntheta is None else target_ntheta
        self.valid_entries: Optional[List[int]] = None
        self.temp_file = temp_file
        self.precomputed_zarr: Optional[Any] = None  # zarr.Group type

        self.segmentation_version = segmentation_version
        self.segment_if_not_exist = segment_if_not_exist

        self.distance_normalization = distance_normalization

        self.flip = flip
        self.double_flip = double_flip
        self.skip_fields = skip_fields

        self.snapshots_per_session = snapshots_per_session
        self.snapshots_adjacent_stride = snapshots_adjacent_stride
        self.random_adjacent_stride = random_adjacent_stride
        self.windows_per_snapshot = windows_per_snapshot

        self.prefix = prefix.replace(".zarr", "")
        self.zarr_fn = f"{self.prefix}.zarr"
        self.yaml_fn = f"{self.prefix}.yaml"
        if temp_file:
            self.zarr_fn += temp_file_suffix
            self.yaml_fn += temp_file_suffix
            assert self.snapshots_per_session == 1
        self.z = zarr_open_from_lmdb_store(
            self.zarr_fn, readahead=self.readahead, map_size=2**32
        )
        self.yaml_config = load_config(self.yaml_fn)

        self.vehicle_type = self.get_collector_identifier()
        self.emitter_type = self.get_emitter_type()

        self.paired = paired
        self.n_receivers = len(self.yaml_config["receivers"])
        self.gpu = gpu
        self.tiled_sessions = tiled_sessions

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

        self.precomputed_entries = 0

        self.wavelengths: List[float] = [
            speed_of_light / receiver["f-carrier"]
            for receiver in self.yaml_config["receivers"]
        ]
        self.carrier_frequencies: List[float] = [
            receiver["f-carrier"] for receiver in self.yaml_config["receivers"]
        ]
        self.rf_bandwidths: List[float] = [
            receiver["bandwidth"] for receiver in self.yaml_config["receivers"]
        ]

        # Initialize rest of attributes
        self.rx_spacing: float
        self.rx_wavelength_spacing: float
        self.rx_configs: List[Any]  # Type from rx_config_from_receiver_yaml
        self.sdr_device_types: List[SDRDEVICE]
        self.sdr_device_type: SDRDEVICE
        self.receiver_data: List[Any]  # zarr array type
        self.n_snapshots: int
        self.n_antennas_per_receiver: int
        self.n_sessions: int
        self.signal_matrices: Optional[List[Any]] = None  # zarr array type
        self.session_length: Optional[int] = None
        self.steering_vectors: List[torch.Tensor]
        self.keys_per_session: List[str]
        self.cached_keys: Dict[int, Dict[str, torch.Tensor]] = {}
        self.receiver_idxs_expanded: Dict[int, torch.Tensor] = {}
        self.mean_phase: Dict[str, torch.Tensor] = {}
        
        # Continue with initialization...

    def refresh(self) -> bool:
        self.z = zarr_open_from_lmdb_store(
            self.zarr_fn, readahead=self.readahead, map_size=2**32
        )
        valid_entries: List[int] = [
            (self.z[f"receivers/r{ridx}/system_timestamp"][:] > 0).sum()
            for ridx in range(self.n_receivers)
        ]

        if self.valid_entries is None or self.valid_entries != valid_entries:
            old_n: int = 0 if self.valid_entries is None else min(self.valid_entries)
            new_n: int = max(valid_entries)
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
                    
                    if old_n == 0:
                        if key in self.receiver_data[receiver_idx]:
                            self.cached_keys[receiver_idx][key] = torch.as_tensor(
                                self.receiver_data[receiver_idx][key][:].astype(np.float32)
                            )
                        else:
                            if key in ("rx_heading_in_pis",):
                                self.cached_keys[receiver_idx][key] = (
                                    torch.as_tensor(
                                        self.receiver_data[receiver_idx]["system_timestamp"][:].astype(np.float32)
                                    )
                                    * 0
                                )
                            else:
                                raise ValueError(f"Missing key {key}")
                    else:
                        self.cached_keys[receiver_idx][key][old_n:new_n] = torch.as_tensor(
                            self.receiver_data[receiver_idx][key][old_n:new_n].astype(np.float32)
                        )
                    
                # optimize rx / tx positions
                self.cached_keys[receiver_idx]["rx_pos_mm"] = torch.vstack([
                    self.cached_keys[receiver_idx]["rx_pos_x_mm"],
                    self.cached_keys[receiver_idx]["rx_pos_y_mm"],
                ]).T

                self.cached_keys[receiver_idx]["tx_pos_mm"] = torch.vstack([
                    self.cached_keys[receiver_idx]["tx_pos_x_mm"],
                    self.cached_keys[receiver_idx]["tx_pos_y_mm"],
                ]).T

                self.cached_keys[receiver_idx]["rx_wavelength_spacing"] = (
                    self.cached_keys[receiver_idx]["rx_spacing"] / self.wavelengths[receiver_idx]
                )
                
            self.valid_entries = valid_entries
            self.ground_truth_thetas, self.absolute_thetas = self.get_ground_truth_thetas()
            self.ground_truth_phis = self.get_ground_truth_phis()
            self.craft_ground_truth_thetas = self.get_craft_ground_truth_thetas()

            rx_spacing_mismatches = (
                (self.cached_keys[0]["rx_spacing"] != self.cached_keys[1]["rx_spacing"])
                .to(float)
                .sum()
            )
            assert rx_spacing_mismatches < 30, f"Too many mismatches in rx_spacing {rx_spacing_mismatches}"
            return True
        return False

    def get_emitter_type(self) -> str:
        if "emitter" in self.yaml_config and "type" in self.yaml_config["emitter"]:
            return self.yaml_config["emitter"]["type"]
        return "external"

    def close(self) -> None:
        self.z.store.close()
        self.z = None
        self.receiver_data = None
        self.segmentation = None
        if self.precomputed_zarr is not None:
            self.precomputed_zarr.store.close()
        self.precomputed_zarr = None

    def estimate_phi(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.as_tensor(data["all_windows_stats"])
        seg_mask = torch.as_tensor(data["downsampled_segmentation_mask"])
        return torch.mul(x, seg_mask).sum(axis=2) / (seg_mask.sum(axis=2) + 0.001)

    def __len__(self) -> int:
        if self.paired:
            return self.n_sessions
        return self.n_sessions * self.n_receivers

    def reinit(self) -> None:
        if self.z is None:
            self.z = zarr_open_from_lmdb_store(self.zarr_fn, readahead=self.readahead)
            self.receiver_data = [
                self.z.receivers[f"r{ridx}"] for ridx in range(self.n_receivers)
            ]
        if not self.temp_file and self.precomputed_zarr is None:
            self.get_segmentation(
                version=self.segmentation_version,
                segment_if_not_exist=self.segment_if_not_exist,
            )

    def populate_from_precomputed(
        self, 
        data: Dict[str, torch.Tensor], 
        receiver_idx: int, 
        snapshot_idxs: np.ndarray
    ) -> None:
        if "windowed_beamformer" not in self.skip_fields:
            data["windowed_beamformer"] = torch.as_tensor(
                self.precomputed_zarr[f"r{receiver_idx}/windowed_beamformer"][snapshot_idxs]
            ).unsqueeze(0)
            if data["windowed_beamformer"].shape[2] > self.windows_per_snapshot:
                data["windowed_beamformer"] = subsample_tensor(
                    data["windowed_beamformer"], 2, self.windows_per_snapshot
                )

        if "weighted_beamformer" not in self.skip_fields:
            data["weighted_beamformer"] = torch.as_tensor(
                self.precomputed_zarr[f"r{receiver_idx}/weighted_beamformer"][snapshot_idxs]
            ).unsqueeze(0)

        if "all_windows_stats" not in self.skip_fields:
            data["all_windows_stats"] = torch.as_tensor(
                self.precomputed_zarr[f"r{receiver_idx}/all_windows_stats"][snapshot_idxs]
            ).unsqueeze(0)
            if data["all_windows_stats"].shape[3] > self.windows_per_snapshot:
                data["all_windows_stats"] = subsample_tensor(
                    data["all_windows_stats"], 3, self.windows_per_snapshot
                )

        if "weighted_windows_stats" not in self.skip_fields:
            data["weighted_windows_stats"] = torch.as_tensor(
                self.precomputed_zarr[f"r{receiver_idx}/weighted_windows_stats"][snapshot_idxs]
            ).unsqueeze(0)

        if "downsampled_segmentation_mask" not in self.skip_fields:
            data["downsampled_segmentation_mask"] = (
                torch.as_tensor(
                    self.precomputed_zarr[f"r{receiver_idx}/downsampled_segmentation_mask"][snapshot_idxs]
                )
                .unsqueeze(1)
                .unsqueeze(0)
            )

        if "simple_segmentations" not in self.skip_fields:
            data["simple_segmentations"] = [
                self.segmentation["segmentation_by_receiver"][f"r{receiver_idx}"][snapshot_idx]["simple_segmentation"]
                for snapshot_idx in snapshot_idxs
            ]

    def get_spacing_identifier(self) -> str:
        rx_lo = self.cached_keys[0]["rx_lo"].median().item()
        return f"sp{self.rx_spacing:0.3f}.rxlo{rx_lo:0.4e}"

    def get_collector_identifier(self) -> str:
        if "collector" in self.yaml_config and "type" in self.yaml_config["collector"]:
            return self.yaml_config["collector"]["type"]
        if "rover" in self.zarr_fn:
            return "rover"
        return "wallarray"

    def get_wavelength_identifier(self) -> str:
        rx_lo = self.cached_keys[0]["rx_lo"].median().item()
        return f"sp{self.rx_spacing:0.3f}.rxlo{rx_lo:0.4e}.wlsp{self.rx_wavelength_spacing:0.3f}"

    def get_values_at_key(
        self, 
        key: str, 
        receiver_idx: int, 
        idxs: np.ndarray
    ) -> torch.Tensor:
        if key == "signal_matrix":
            return torch.as_tensor(self.receiver_data[receiver_idx][key][idxs])[None]
        return self.cached_keys[receiver_idx][key][idxs]

    def get_session_idxs(self, session_idx: int) -> np.ndarray:
        if self.tiled_sessions:
            snapshot_start_idx = session_idx * self.snapshots_stride
            snapshot_end_idx = (
                snapshot_start_idx
                + self.snapshots_adjacent_stride * self.snapshots_per_session
            )
        else:
            snapshot_start_idx = (
                session_idx
                * self.snapshots_per_session
                * self.snapshots_adjacent_stride
            )
            snapshot_end_idx = (
                (session_idx + 1)
                * self.snapshots_per_session
                * self.snapshots_adjacent_stride
            )
        if self.random_adjacent_stride:
            idxs = (
                get_idx_for_rand_session(
                    self.snapshots_adjacent_stride, self.snapshots_per_session
                )
                + snapshot_start_idx
            )
            return idxs
        else:
            return np.arange(
                snapshot_start_idx, snapshot_end_idx, self.snapshots_adjacent_stride
            )

    @cache
    def get_empirical_dist(
        self,
        receiver_idx: int,
    ) -> torch.Tensor:
        rx_spacing_str = rx_spacing_to_str(self.rx_wavelength_spacing)
        empirical_radio_key = (
            f"r{receiver_idx}" if self.empirical_individual_radio else "r"
        )
        return self.empirical_data[f"{self.sdr_device_type}_{rx_spacing_str}"][
            empirical_radio_key
        ]["sym" if self.empirical_symmetry else "nosym"]

    def render_session(
        self, 
        receiver_idx: int, 
        session_idx: int, 
        double_flip: bool = False
    ) -> Dict[str, torch.Tensor]:
        self.reinit()

        flip_left_right = self.flip and (torch.rand(1) > 0.5).item()
        snapshot_idxs = self.get_session_idxs(session_idx)

        data: Dict[str, torch.Tensor] = {
            key: self.get_values_at_key(key, receiver_idx, snapshot_idxs)
            for key in self.keys_per_session
        }

        data["gains"] = data["gains"][:, None]
        data["receiver_idx"] = self.receiver_idxs_expanded[receiver_idx]
        data["ground_truth_theta"] = self.ground_truth_thetas[receiver_idx][snapshot_idxs]
        data["ground_truth_phi"] = self.ground_truth_phis[receiver_idx][snapshot_idxs]
        data["craft_ground_truth_theta"] = self.craft_ground_truth_thetas[snapshot_idxs]
        data["vehicle_type"] = torch.tensor(
            [encode_vehicle_type(self.vehicle_type)]
        ).reshape(1)
        data["sdr_device_type"] = torch.tensor([self.sdr_device_type.value]).reshape(1)

        craft_offset = data["craft_ground_truth_theta"] - data["ground_truth_theta"]
        
        if flip_left_right or double_flip:
            data["ground_truth_phi"] = -data["ground_truth_phi"]
            data["ground_truth_theta"] = -data["ground_truth_theta"]
        if double_flip:
            data["ground_truth_theta"] = (
                data["ground_truth_theta"].sign() * torch.pi
                - data["ground_truth_theta"]
            )

        data["craft_ground_truth_theta"] = torch_pi_norm(
            data["ground_truth_theta"] + craft_offset
        )

        data["y_rad"] = data["ground_truth_theta"]
        data["y_phi"] = data["ground_truth_phi"]
        data["craft_y_rad"] = data["craft_ground_truth_theta"]

        if "signal_matrix" not in self.skip_fields:
            abs_signal = data["signal_matrix"].abs().to(torch.float32)
            assert data["signal_matrix"].shape[0] == 1
            pd = torch_get_phase_diff(data["signal_matrix"][0]).to(torch.float32)
            data["abs_signal_and_phase_diff"] = torch.concatenate(
                [abs_signal, pd[None, :, None]], dim=2
            )

        if self.temp_file and self.precomputed_entries <= session_idx:
            self.get_segmentation(
                version=self.segmentation_version,
                precompute_to_idx=session_idx,
                segment_if_not_exist=True,
            )

        self.populate_from_precomputed(data, receiver_idx, snapshot_idxs)

        data["mean_phase_segmentation"] = self.mean_phase[f"r{receiver_idx}"][
            snapshot_idxs
        ].unsqueeze(0)

        if flip_left_right or double_flip:
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
            empirical_dist = self.get_empirical_dist(receiver_idx)
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

        for key in data:
            if isinstance(data[key], torch.Tensor) and data[key].dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
            ):
                data[key] = data[key].to(self.target_dtype)

        if flip_left_right or double_flip:
            data["weighted_beamformer"] = data["weighted_beamformer"].flip(dims=(2,))
            if "windowed_beamformer" in data:
                data["windowed_beamformer"] = data["windowed_beamformer"].flip(dims=(3,))

        return data

    def get_ground_truth_phis(self) -> torch.Tensor:
        ground_truth_phis: List[torch.Tensor] = []
        for ridx in range(self.n_receivers):
            ground_truth_phis.append(
                torch_pi_norm(
                    -torch.sin(
                        self.ground_truth_thetas[ridx]
                    )
                    * self.rx_wavelength_spacing
                    * 2
                    * torch.pi
                )
            )
        return torch.vstack(ground_truth_phis)

    def get_all_phi_drifts(self) -> List[torch.Tensor]:
        all_phi_drifts: List[torch.Tensor] = []
        for ridx in range(self.n_receivers):
            a = self.ground_truth_phis[ridx]
            b = self.mean_phase[f"r{ridx}"]
            fwd = torch_pi_norm(b - a)
            c = fwd
            all_phi_drifts.append(c)
        return all_phi_drifts

    # ground truth theta is relative to heading of the craft!!!!
    def get_craft_ground_truth_thetas(self) -> torch.Tensor:
        craft_ground_truth_thetas = torch_pi_norm(
            self.ground_truth_thetas[0]
            + (
                self.cached_keys[0]["rx_theta_in_pis"][:]
            )
            * torch.pi
        )
        for ridx in range(1, self.n_receivers):
            _craft_ground_truth_thetas = torch_pi_norm(
                self.ground_truth_thetas[ridx]
                + (
                    self.cached_keys[ridx]["rx_theta_in_pis"][:]
                )
                * torch.pi
            )
            error = abs(
                torch_pi_norm(craft_ground_truth_thetas - _craft_ground_truth_thetas)
            ).mean()
            if error > 0.2:
                logging.warning(
                    f"{self.zarr_fn} craft_theta off by too much! {error}"
                )
        return craft_ground_truth_thetas

    def get_ground_truth_thetas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ground_truth_thetas: List[torch.Tensor] = []
        absolute_thetas: List[torch.Tensor] = []
        for ridx in range(self.n_receivers):
            tx_pos = self.cached_keys[ridx]["tx_pos_mm"].T
            rx_pos = self.cached_keys[ridx]["rx_pos_mm"].T

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
        assert self.n_receivers == 2
        return torch.vstack(ground_truth_thetas), torch.vstack(absolute_thetas)

    def get_mean_phase(self) -> None:
        self.mean_phase = {
            f"r{r_idx}": torch.tensor(self.precomputed_zarr[f"r{r_idx}/mean_phase"][:])
            for r_idx in range(self.n_receivers)
        }

    def __getitem__(
        self, 
        idx: int
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
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

    def get_estimated_thetas(self) -> Dict[str, np.ndarray]:
        estimated_thetas: Dict[str, np.ndarray] = {}
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

    def results_fn(self) -> str:
        return os.path.join(
            self.precompute_cache,
            os.path.basename(self.prefix).replace("_nosig", "")
            + f"_segmentation_nthetas{self.nthetas}.pkl",
        )

    def get_segmentation_version(
        self, 
        precomputed_zarr: Any, 
        segmentation: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        if "version" in precomputed_zarr:
            if precomputed_zarr["version"].shape == ():
                return 3.0
            return precomputed_zarr["version"][0]
        if segmentation is None or "version" not in segmentation:
            return None
        return segmentation["version"]

    def get_segmentation(
        self, 
        version: float, 
        segment_if_not_exist: bool, 
        precompute_to_idx: int = -1
    ) -> Dict[str, Any]:
        if (
            not self.temp_file
            and hasattr(self, "segmentation")
            and self.segmentation is not None
        ):
            return self.segmentation

        results_fn = self.results_fn()
        if (
            not segment_if_not_exist
            and not self.temp_file
            and not os.path.exists(results_fn)
        ):
            raise ValueError(f"Segmentation file does not exist for {results_fn}")

        if self.temp_file or not os.path.exists(results_fn):
            skip_beamformer = False
            if (
                self.temp_file
                and "windowed_beamformer" in self.skip_fields
                and "weighted_beamformer" in self.skip_fields
            ):
                skip_beamformer = True
            mp_segment_zarr(
                self.zarr_fn,
                results_fn,
                self.steering_vectors,
                precompute_to_idx=precompute_to_idx,
                gpu=self.gpu,
                n_parallel=self.n_parallel,
                skip_beamformer=skip_beamformer,
            )

        try:
            if "simple_segmentations" not in self.skip_fields:
                segmentation = pickle.load(open(results_fn, "rb"))
            else:
                segmentation = {}

            precomputed_zarr = zarr_open_from_lmdb_store(
                results_fn.replace(".pkl", ".yarr"),
                mode="r",
                map_size=2**32,
                readahead=False,
            )
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
            os.remove(results_fn)
            return self.get_segmentation(
                version=version,
                segment_if_not_exist=segment_if_not_exist,
                precompute_to_idx=precompute_to_idx,
            )

        current_version = self.get_segmentation_version(precomputed_zarr, segmentation)
        if not np.isclose(current_version, version):
            raise ValueError(
                f"Expected to generate a segmentation, but one already exists with different version {version} vs {current_version}"
            )
        self.segmentation = segmentation
        self.precomputed_zarr = precomputed_zarr
        self.get_mean_phase()
        return self.segmentation


class SessionsDatasetSimulated(Dataset):
    def __init__(self, root_dir: str, snapshots_per_session: int = 5) -> None:
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
            self.filenames: List[str] = sorted(
                filter(
                    lambda x: "args" not in x,
                    ["%s/%s" % (self.root_dir, x) for x in os.listdir(self.root_dir)],
                )
            )
            if self.args.sessions != len(
                self.filenames
            ):  # make sure its the right dataset
                print("WARNING DATASET LOOKS LIKE IT IS MISSING SOME SESSIONS!")

    def idx_to_filename_and_start_idx(self, idx: int) -> Tuple[str, int]:
        assert idx >= 0 and idx <= self.samples_per_session * len(self.filenames)
        return (
            self.filenames[idx // self.samples_per_session],
            idx % self.samples_per_session,
        )

    def __len__(self) -> int:
        if self.args.live:
            return self.samples_per_session * self.args.sessions
        else:
            return self.samples_per_session * len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        session_idx = idx // self.samples_per_session
        start_idx = idx % self.samples_per_session

        if self.args.live:
            session = generate_session((self.args, session_idx))
        else:
            session = load(self.filenames[session_idx], compression="lzma")
        end_idx = start_idx + self.snapshots_in_sample
        return {k: session[k][start_idx:end_idx] for k in session.keys()}
