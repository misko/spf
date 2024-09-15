###
# Experiment 1 : wall drones , receiver center is x=135.27 cm, y=264.77cm, dist between is 6cm
###

import bisect
import os
import pickle
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import numpy as np
import torch
import tqdm
import yaml
from compress_pickle import load
from deepdiff import DeepDiff
from tensordict import TensorDict
from torch.utils.data import Dataset

from spf.dataset.rover_idxs import (  # v3rx_column_names,
    v3rx_avg_phase_diff_idxs,
    v3rx_beamformer_start_idx,
    v3rx_column_names,
    v3rx_gain_idxs,
    v3rx_rssi_idxs,
    v3rx_rx_pos_idxs,
    v3rx_rx_theta_idx,
    v3rx_time_idxs,
)
from spf.dataset.spf_generate import generate_session
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys
from spf.dataset.wall_array_v1_idxs import (
    v1_beamformer_start_idx,
    v1_column_names,
    v1_time_idx,
    v1_tx_pos_idxs,
)
from spf.dataset.wall_array_v2_idxs import (  # v3rx_column_names,
    v2_avg_phase_diff_idxs,
    v2_beamformer_start_idx,
    v2_column_names,
    v2_gain_idxs,
    v2_rssi_idxs,
    v2_rx_pos_idxs,
    v2_rx_theta_idx,
    v2_time_idx,
    v2_tx_pos_idxs,
)
from spf.plot.image_utils import (
    detector_positions_to_theta_grid,
    labels_to_source_images,
    radio_to_image,
)
from spf.rf import (
    ULADetector,
    phase_diff_to_theta,
    pi_norm,
    precompute_steering_vectors,
    segment_session,
    segment_session_star,
    speed_of_light,
    torch_get_phase_diff,
    torch_pi_norm,
)
from spf.scripts.create_empirical_p_dist import rx_spacing_to_str
from spf.sdrpluto.sdr_controller import rx_config_from_receiver_yaml
from spf.utils import (
    SEGMENTATION_VERSION,
    new_yarr_dataset,
    to_bin,
    zarr_open_from_lmdb_store,
    zarr_shrink,
)


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


# from stackoverflow
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


default_segment_args = {
    "window_size": 2048,
    "stride": 2048,
    "trim": 20.0,
    "mean_diff_threshold": 0.2,
    "max_stddev_threshold": 0.5,
    "drop_less_than_size": 3000,
    "min_abs_signal": 40,
}


def segment_single_session(
    zarr_fn, steering_vectors_for_receiver, session_idx, receiver_idx, gpu=False
):
    args = {
        "zarr_fn": zarr_fn,
        "receiver": receiver_idx,
        "session_idx": session_idx,
        "steering_vectors": steering_vectors_for_receiver,
        "gpu": gpu,
        **default_segment_args,
    }
    return segment_session(**args)


def mp_segment_zarr(
    zarr_fn,
    results_fn,
    steering_vectors_for_all_receivers,
    precompute_to_idx=-1,
    gpu=False,
    n_parallel=20,
    skip_beamformer=False,
):

    z = zarr_open_from_lmdb_store(zarr_fn)
    yarr_fn = results_fn.replace(".pkl", ".yarr")
    already_computed = 0
    # print(z.tree())
    previous_simple_segmentation = {
        f"r{r_idx}": [] for r_idx in range(len(z["receivers"]))
    }
    precomputed_zarr = None
    if os.path.exists(results_fn) and os.path.exists(yarr_fn):
        previous_simple_segmentation = pickle.load(open(results_fn, "rb"))[
            "segmentation_by_receiver"
        ]
        precomputed_zarr = zarr_open_from_lmdb_store(yarr_fn, mode="rw", map_size=2**32)
        already_computed = min(
            [
                (
                    np.sum(precomputed_zarr[f"r{r_idx}/all_windows_stats"], axis=(1, 2))
                    > 0
                ).sum()
                for r_idx in range(len(z["receivers"]))
            ]
        )

    if precompute_to_idx < 0:
        precompute_to_idx = min(
            [
                (z[f"receivers/r{ridx}/system_timestamp"][:] > 0).sum()
                for ridx in range(len(z["receivers"]))
            ]
        )
    else:
        precompute_to_idx += 1

    # we dont need to segment anything
    if precompute_to_idx <= already_computed:
        return already_computed

    assert len(z["receivers"]) == 2

    n_sessions = z.receivers["r0"].system_timestamp.shape[0]

    results_by_receiver = {}
    for r_idx in [0, 1]:
        r_name = f"r{r_idx}"
        inputs = [
            {
                "zarr_fn": zarr_fn,
                "receiver": r_name,
                "session_idx": idx,
                "steering_vectors": steering_vectors_for_all_receivers[r_idx],
                "gpu": gpu,
                "skip_beamformer": skip_beamformer,
                **default_segment_args,
            }
            for idx in range(already_computed, precompute_to_idx)
        ]

        # n_parallel = 0
        if n_parallel > 0:
            with Pool(
                min(cpu_count(), n_parallel)
            ) as pool:  # cpu_count())  # cpu_count() // 4)
                results_by_receiver[r_name] = list(
                    tqdm.tqdm(
                        pool.imap(segment_session_star, inputs), total=len(inputs)
                    )
                )
        else:
            results_by_receiver[r_name] = list(
                tqdm.tqdm(map(segment_session_star, inputs), total=len(inputs))
            )

    segmentation_zarr_fn = results_fn.replace(".pkl", ".yarr")
    all_windows_stats_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "all_windows_stats"
    ].shape
    windowed_beamformer_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "windowed_beamformer"
    ].shape
    weighted_beamformer_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "windowed_beamformer"
    ].shape[1:]
    downsampled_segmentation_mask_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "downsampled_segmentation_mask"
    ].shape

    if precomputed_zarr is None:
        precomputed_zarr = new_yarr_dataset(
            filename=segmentation_zarr_fn,
            n_receivers=2,
            all_windows_stats_shape=all_windows_stats_shape,
            windowed_beamformer_shape=windowed_beamformer_shape,
            weighted_beamformer_shape=weighted_beamformer_shape,
            downsampled_segmentation_mask_shape=downsampled_segmentation_mask_shape,
            mean_phase_shape=(n_sessions,),
        )

    for r_idx in [0, 1]:
        # collect all windows stats
        precomputed_zarr[f"r{r_idx}/all_windows_stats"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["all_windows_stats"][None] for x in results_by_receiver[f"r{r_idx}"]]
        )
        # collect windowed beamformer
        precomputed_zarr[f"r{r_idx}/windowed_beamformer"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["windowed_beamformer"][None] for x in results_by_receiver[f"r{r_idx}"]]
        )
        # collect weighted beamformer
        precomputed_zarr[f"r{r_idx}/weighted_beamformer"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["weighted_beamformer"][None] for x in results_by_receiver[f"r{r_idx}"]]
        )
        # collect downsampled segmentation mask
        precomputed_zarr[f"r{r_idx}/downsampled_segmentation_mask"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [
                x["downsampled_segmentation_mask"][None]
                for x in results_by_receiver[f"r{r_idx}"]
            ]
        )

        # precompute mean phase and remove NaNs for 0s
        mean_phase = np.hstack(
            [
                torch.tensor([x["mean"] for x in result["simple_segmentation"]]).mean()
                for result in results_by_receiver[f"r{r_idx}"]
            ]
        )
        mean_phase[~np.isfinite(mean_phase)] = 0

        assert np.isfinite(mean_phase).all()
        precomputed_zarr[f"r{r_idx}/mean_phase"][
            already_computed:precompute_to_idx
        ] = mean_phase

    # print(previous_simple_segmentation)
    # print("WINDOWED BEAMFORMER 7", precomputed_zarr["r0/windowed_beamformer"][7])
    simple_segmentation = {}
    for r_idx in [0, 1]:
        simple_segmentation[f"r{r_idx}"] = (
            previous_simple_segmentation[f"r{r_idx}"][:already_computed]
            + [
                {"simple_segmentation": x["simple_segmentation"]}
                for x in results_by_receiver[f"r{r_idx}"]
            ]
            + [
                {"simple_segmentation": []}
                for _ in range(n_sessions - precompute_to_idx)
            ]
        )

    precomputed_zarr.store.close()
    precomputed_zarr = None
    zarr_shrink(segmentation_zarr_fn)
    pickle.dump(
        {
            "version": SEGMENTATION_VERSION,
            "segmentation_by_receiver": simple_segmentation,
        },
        open(results_fn, "wb"),
    )
    return precompute_to_idx


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


def v5_collate_keys_fast(keys: List[str], batch: Dict[str, torch.Tensor]):
    d = {}
    for key in keys:
        d[key] = torch.vstack(
            [x[key] for paired_sample in batch for x in paired_sample]
        )
        if key == "windowed_beamformer" or key == "all_windows_stats":
            d[key] = d[key].to(torch.float32)

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


class v5spfdataset(Dataset):
    def __init__(
        self,
        prefix,
        nthetas,
        precompute_cache,
        phi_drift_max=0.2,
        min_mean_windows=10,
        ignore_qc=False,
        paired=False,
        gpu=False,
        snapshots_per_session=1,
        tiled_sessions=True,  # session 0 overlaps heavily with session 1 except last snapshot
        snapshots_stride=1,
        readahead=False,
        temp_file=False,
        temp_file_suffix=".tmp",
        skip_fields=[],
        n_parallel=20,
        empirical_data_fn=None,
        empirical_individual_radio=False,
        empirical_symmetry=True,
        target_dtype=torch.float32,
    ):
        self.n_parallel = n_parallel
        self.exclude_keys_from_cache = set(["signal_matrix"])
        self.readahead = readahead
        self.precompute_cache = precompute_cache
        self.nthetas = nthetas
        self.valid_entries = None
        # self.prefix = prefix
        # print("OPEN", prefix)
        self.temp_file = temp_file

        self.skip_fields = skip_fields

        self.snapshots_per_session = snapshots_per_session

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
        self.yaml_config = yaml.safe_load(open(self.yaml_fn, "r"))
        self.paired = paired
        self.n_receivers = len(self.yaml_config["receivers"])
        self.gpu = gpu
        self.tiled_sessions = tiled_sessions

        if snapshots_stride < 1.0:
            snapshots_stride = max(
                int(snapshots_stride * self.snapshots_per_session), 1
            )
        self.snapshots_stride = int(snapshots_stride)

        self.precomputed_entries = 0

        self.wavelengths = [
            speed_of_light / receiver["f-carrier"]
            for receiver in self.yaml_config["receivers"]
        ]

        self.rx_configs = [
            rx_config_from_receiver_yaml(receiver)
            for receiver in self.yaml_config["receivers"]
        ]

        self.receiver_data = [
            self.z.receivers[f"r{ridx}"] for ridx in range(self.n_receivers)
        ]

        self.n_snapshots, self.n_antennas_per_receiver = self.z.receivers.r0.gains.shape
        assert self.n_antennas_per_receiver == 2

        if self.snapshots_per_session == -1:
            self.snapshots_per_session = self.n_snapshots

        if self.tiled_sessions:
            self.n_sessions = (
                self.n_snapshots - self.snapshots_per_session
            ) // self.snapshots_stride + 1
            if self.n_sessions <= 0:
                self.n_sessions = 0
        else:
            self.n_sessions = self.n_snapshots // self.snapshots_per_session

        if "signal_matrix" not in self.skip_fields:
            self.signal_matrices = [
                self.z.receivers[f"r{ridx}"].signal_matrix
                for ridx in range(self.n_receivers)
            ]
            _, _, self.session_length = self.z.receivers["r0"].signal_matrix.shape

            for ridx in range(self.n_receivers):
                assert self.z.receivers[f"r{ridx}"].signal_matrix.shape == (
                    self.n_snapshots,
                    self.n_antennas_per_receiver,
                    self.session_length,
                )

        self.steering_vectors = [
            precompute_steering_vectors(
                receiver_positions=rx_config.rx_pos,
                carrier_frequency=rx_config.lo,
                spacing=nthetas,
            )
            for rx_config in self.rx_configs
        ]

        self.keys_per_session = v5rx_f64_keys + v5rx_2xf64_keys
        if "signal_matrix" not in self.skip_fields:
            self.keys_per_session.append("signal_matrix")

        self.refresh()

        self.receiver_idxs_expanded = {}
        for idx in range(self.n_receivers):
            self.receiver_idxs_expanded[idx] = (
                torch.tensor(idx, dtype=torch.int32)
                .expand(1, self.snapshots_per_session)
                .share_memory_()
            )

        if not self.temp_file:
            self.get_segmentation()

            self.all_phi_drifts = self.get_all_phi_drifts()
            self.phi_drifts = torch.tensor(
                [torch.nanmean(all_phi_drift) for all_phi_drift in self.all_phi_drifts]
            )
            if "simple_segmentations" not in self.skip_fields:
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
            # if we are a temp verison
            self.mean_phase = {}
            for ridx in range(self.n_receivers):
                self.mean_phase[f"r{ridx}"] = torch.ones(len(self)) * torch.inf

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

        self.target_dtype = target_dtype

        if empirical_data_fn is not None:
            self.empirical_data = pickle.load(open(empirical_data_fn, "rb"))
            self.empirical_individual_radio = empirical_individual_radio
            self.empirical_symmetry = empirical_symmetry
        else:
            self.empirical_data = None
        # self.close()

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
                    if key in self.exclude_keys_from_cache:
                        continue
                    # assert key != "signal_matrix"  # its complex shouldnt get converted!
                    # print(key, self.receiver_data[receiver_idx][key][:].dtype)
                    # print(key, self.exclude_keys_from_cache)
                    if old_n == 0:
                        self.cached_keys[receiver_idx][key] = torch.as_tensor(
                            self.receiver_data[receiver_idx][key][:].astype(
                                np.float32
                            )  # TODO save as float32?
                        ).share_memory_()
                    else:
                        self.cached_keys[receiver_idx][key][old_n:new_n] = (
                            torch.as_tensor(
                                self.receiver_data[receiver_idx][key][
                                    old_n:new_n
                                ].astype(np.float32)
                            )
                        )
            self.valid_entries = valid_entries
            self.ground_truth_thetas = self.get_ground_truth_thetas().share_memory_()
            self.ground_truth_phis = self.get_ground_truth_phis().share_memory_()
            self.craft_ground_truth_thetas = (
                self.get_craft_ground_truth_thetas().share_memory_()
            )

            return True
        return False

    # def get_mean_phase(self, ridx, idx):
    #     v = self.mean_phase[f"r{ridx}"][idx]
    #     if torch.isfinite(v):
    #         return v
    #     else:
    #         print("NOT VALID")

    def close(self):
        # let workers open their own
        self.z.store.close()
        self.z = None
        self.receiver_data = None
        # try and close segmentation
        self.segmentation = None
        self.precomputed_zarr = None

    def estimate_phi(self, data):
        x = torch.as_tensor(data["all_windows_stats"])
        seg_mask = torch.as_tensor(data["downsampled_segmentation_mask"])

        return torch.mul(x, seg_mask).sum(axis=2) / (seg_mask.sum(axis=2) + 0.001)

    def __len__(self):
        if self.paired:
            return self.n_sessions
        return self.n_sessions * self.n_receivers

    def reinit(self):
        if self.z is None:
            # worker_info = torch.utils.data.get_worker_info()

            self.z = zarr_open_from_lmdb_store(self.zarr_fn, readahead=self.readahead)
            self.receiver_data = [
                self.z.receivers[f"r{ridx}"] for ridx in range(self.n_receivers)
            ]
        if not self.temp_file and self.precomputed_zarr is None:
            self.get_segmentation()
            # self.precomputed_zarr = zarr_open_from_lmdb_store(
            #     self.results_fn().replace(".pkl", ".yarr"), mode="r"
            # )

    def populate_from_precomputed(
        self, data, receiver_idx, snapshot_start_idx, snapshot_end_idx
    ):
        if "windowed_beamformer" not in self.skip_fields:
            data["windowed_beamformer"] = (
                torch.as_tensor(
                    self.precomputed_zarr[f"r{receiver_idx}/windowed_beamformer"][
                        snapshot_start_idx:snapshot_end_idx
                    ]
                )
                .unsqueeze(0)
                .share_memory_()
            )

        if "weighted_beamformer" not in self.skip_fields:
            data["weighted_beamformer"] = (
                torch.as_tensor(
                    self.precomputed_zarr[f"r{receiver_idx}/weighted_beamformer"][
                        snapshot_start_idx:snapshot_end_idx
                    ]
                )
                .unsqueeze(0)
                .share_memory_()
            )

        # sessions x 3 x n_windows
        if "all_windows_stats" not in self.skip_fields:
            data["all_windows_stats"] = (
                torch.as_tensor(
                    self.precomputed_zarr[f"r{receiver_idx}/all_windows_stats"][
                        snapshot_start_idx:snapshot_end_idx
                    ]
                )
                .unsqueeze(0)
                .share_memory_()
            )

        if "downsampled_segmentation_mask" not in self.skip_fields:
            data["downsampled_segmentation_mask"] = (
                torch.as_tensor(
                    self.precomputed_zarr[f"r{receiver_idx}"][
                        "downsampled_segmentation_mask"
                    ][snapshot_start_idx:snapshot_end_idx]
                )
                .unsqueeze(1)
                .unsqueeze(0)
                .share_memory_()
            )

        if "simple_segmentations" not in self.skip_fields:
            data["simple_segmentations"] = [
                d["simple_segmentation"]
                for d in self.segmentation["segmentation_by_receiver"][
                    f"r{receiver_idx}"
                ][snapshot_start_idx:snapshot_end_idx]
            ]

    def get_values_at_key(self, key, receiver_idx, start_idx, end_idx):
        if key == "signal_matrix":
            return torch.as_tensor(
                self.receiver_data[receiver_idx][key][start_idx:end_idx]
            )
        return self.cached_keys[receiver_idx][key][start_idx:end_idx]

    def render_session(self, receiver_idx, session_idx):
        self.reinit()
        if self.tiled_sessions:
            snapshot_start_idx = session_idx * self.snapshots_stride
            snapshot_end_idx = snapshot_start_idx + self.snapshots_per_session

        else:
            snapshot_start_idx = session_idx * self.snapshots_per_session
            snapshot_end_idx = (session_idx + 1) * self.snapshots_per_session

        data = {
            # key: r[key][snapshot_start_idx:snapshot_end_idx]
            key: self.get_values_at_key(
                key, receiver_idx, snapshot_start_idx, snapshot_end_idx
            )
            for key in self.keys_per_session  # 'rx_theta_in_pis', 'rx_spacing', 'rx_lo', 'rx_bandwidth', 'avg_phase_diff', 'rssis', 'gains']
        }
        data["receiver_idx"] = self.receiver_idxs_expanded[receiver_idx]
        data["ground_truth_theta"] = self.ground_truth_thetas[receiver_idx][
            snapshot_start_idx:snapshot_end_idx
        ]
        data["ground_truth_phi"] = self.ground_truth_phis[receiver_idx][
            snapshot_start_idx:snapshot_end_idx
        ]
        data["craft_ground_truth_theta"] = self.craft_ground_truth_thetas[
            snapshot_start_idx:snapshot_end_idx
        ]

        # duplicate some fields
        data["y_rad"] = data["ground_truth_theta"]
        data["y_phi"] = data["ground_truth_phi"]
        data["craft_y_rad"] = data["craft_ground_truth_theta"]

        if "signal_matrix" not in self.skip_fields:
            abs_signal = data["signal_matrix"].abs().to(torch.float32)
            pd = torch_get_phase_diff(data["signal_matrix"]).to(torch.float32)
            data["abs_signal_and_phase_diff"] = torch.concatenate(
                [abs_signal[:, [0]], abs_signal[:, [1]], pd[:, None]], dim=1
            )

        # find out if this is a temp file and we either need to precompute, or its not ready
        if self.temp_file and self.precomputed_entries <= session_idx:
            self.get_segmentation(precompute_to_idx=session_idx)

        self.populate_from_precomputed(
            data, receiver_idx, snapshot_start_idx, snapshot_end_idx
        )
        # port this over in on the fly TODO
        data["mean_phase_segmentation"] = self.mean_phase[f"r{receiver_idx}"][
            snapshot_start_idx:snapshot_end_idx
        ].unsqueeze(0)

        data["rx_pos_xy"] = torch.vstack(
            [
                self.cached_keys[receiver_idx]["rx_pos_x_mm"][
                    snapshot_start_idx:snapshot_end_idx
                ],
                self.cached_keys[receiver_idx]["rx_pos_y_mm"][
                    snapshot_start_idx:snapshot_end_idx
                ],
            ]
        ).T.unsqueeze(0)
        data["tx_pos_xy"] = torch.vstack(
            [
                self.cached_keys[receiver_idx]["tx_pos_x_mm"][
                    snapshot_start_idx:snapshot_end_idx
                ],
                self.cached_keys[receiver_idx]["tx_pos_y_mm"][
                    snapshot_start_idx:snapshot_end_idx
                ],
            ]
        ).T.unsqueeze(0)

        # if torch.rand(1).item() < self.snapshots_per_session * 0.00000005:
        #     print("COLLECT")
        #     gc.collect()
        # self.close()
        if self.empirical_data is not None:
            rx_spacing_str = rx_spacing_to_str(data["rx_spacing"][0].item())
            empirical_radio_key = (
                f"r{receiver_idx}" if self.empirical_individual_radio else "r"
            )
            empirical_dist = self.empirical_data[rx_spacing_str][empirical_radio_key][
                "sym" if self.empirical_symmetry else "nosym"
            ]
            data["empirical"] = empirical_dist[
                to_bin(data["mean_phase_segmentation"][0], empirical_dist.shape[0])
            ].unsqueeze(0)

        data["y_rad_binned"] = (
            to_bin(data["y_rad"], self.nthetas).unsqueeze(0).to(torch.long)
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
                    * self.cached_keys[ridx]["rx_spacing"]
                    * 2
                    * torch.pi
                    / self.wavelengths[ridx]
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

    def get_craft_ground_truth_thetas(self):
        craft_ground_truth_thetas = torch_pi_norm(
            self.ground_truth_thetas[0]
            + self.cached_keys[0]["rx_theta_in_pis"][:] * torch.pi
        )
        for ridx in range(1, self.n_receivers):
            _craft_ground_truth_thetas = torch_pi_norm(
                self.ground_truth_thetas[ridx]
                + self.cached_keys[ridx]["rx_theta_in_pis"][:] * torch.pi
            )
            assert (
                abs(
                    torch_pi_norm(
                        craft_ground_truth_thetas - _craft_ground_truth_thetas
                    )
                ).mean()
                < 0.01
            )  # this might not scale well to larger examples
            # i think the error gets worse
        return craft_ground_truth_thetas

    def get_ground_truth_thetas(self):
        ground_truth_thetas = []
        for ridx in range(self.n_receivers):
            tx_pos = torch.vstack(
                [
                    self.cached_keys[ridx]["tx_pos_x_mm"],
                    self.cached_keys[ridx]["tx_pos_y_mm"],
                ]
            )
            rx_pos = torch.vstack(
                [
                    self.cached_keys[ridx]["rx_pos_x_mm"],
                    self.cached_keys[ridx]["rx_pos_y_mm"],
                ]
            )

            # compute the angle of the tx with respect to rx
            d = tx_pos - rx_pos

            rx_to_tx_theta = torch.arctan2(d[0], d[1])
            rx_theta_in_pis = self.cached_keys[ridx]["rx_theta_in_pis"]
            ground_truth_thetas.append(
                torch_pi_norm(rx_to_tx_theta - rx_theta_in_pis[:] * torch.pi)
            )
        # reduce GT thetas in case of two antennas
        # in 2D there are generally two spots that satisfy phase diff
        # lets pick the one thats infront of the craft (array)
        assert self.n_receivers == 2
        return torch.vstack(ground_truth_thetas)

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
            return [
                self.render_session(receiver_idx, idx)
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
                self.mean_phase[f"r{ridx}"],
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

    def get_segmentation(self, precompute_to_idx=-1):
        if (
            not self.temp_file
            and hasattr(self, "segmentation")
            and self.segmentation is not None
        ):
            return self.segmentation

        # otherwise its the first time loading for non temp
        # or this is a temp file
        results_fn = self.results_fn()

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
                readahead=True,
            )
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
            return self.get_segmentation(precompute_to_idx=precompute_to_idx)

        current_version = self.get_segmentation_version(precomputed_zarr, segmentation)
        if current_version != SEGMENTATION_VERSION:
            os.remove(results_fn)
            return self.get_segmentation(precompute_to_idx=precompute_to_idx)
        self.segmentation = segmentation
        self.precomputed_zarr = precomputed_zarr
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


class SessionsDatasetReal(Dataset):
    def check_file(self, filename):
        try:
            m = self.get_m(filename, bypass=True)
        except Exception as e:
            print(
                "SessionDatasetReal: Dropping file from loading because memmap failed",
                e,
                filename,
            )
            return False
        if self.check_files:
            status = not (np.abs(m).mean(axis=1) == 0).any()
            if not status:
                print(
                    "SessionDatasetReal: Dropping file from loading because it looks like all zeros",
                    filename,
                )
                return False
        return True

    def idx_to_fileidx_and_startidx(self, idx):
        file_idx = bisect.bisect_right(self.cumsum_sessions_per_file, idx) - 1
        if file_idx >= len(self.sessions_per_file):
            return None
        start_idx = (
            idx - self.cumsum_sessions_per_file[file_idx]
        )  # *(self.snapshots_in_sample*self.step_size)
        return file_idx, start_idx

    def __len__(self):
        return self.len

    def get_m(self, filename, bypass=False):
        if bypass:
            return np.memmap(
                filename,
                dtype="float32",
                mode="r",
                shape=self.file_shape(),
            )

        if filename not in self.m_cache:
            self.m_cache[filename] = np.memmap(
                filename,
                dtype="float32",
                mode="r",
                shape=self.file_shape(),
            )
        return self.m_cache[filename]

    def get_all_files_with_extension(self, extension):
        return filter(
            lambda x: f".{extension}" in x,
            ["%s/%s" % (self.root_dir, x) for x in os.listdir(self.root_dir)],
        )

    def get_all_valid_files(self):
        return sorted(
            filter(
                self.check_file,
                self.get_all_files_with_extension("npy"),
            )
        )


class SessionsDatasetRealV1(SessionsDatasetReal):
    def file_shape(self):
        return (self.snapshots_in_file, len(self.column_names))

    def __init__(
        self,
        root_dir,
        snapshots_in_file=400000,
        nthetas=65,
        snapshots_in_session=128,
        nsources=1,
        width=3000,
        receiver_pos_x=1352.7,
        receiver_pos_y=2647.7,
        receiver_spacing=60.0,
        step_size=1,
        seed=1337,
    ):
        # time_step,x,y,mean_angle,_mean_angle #0,1,2,3,4
        # m = np.memmap(filename, dtype='float32', mode='r', shape=(,70))
        """
        Arguments:
          root_dir (string): Directory with all the images.
        """
        assert nsources == 1  # TODO implement more
        self.check_files = True
        self.root_dir = root_dir
        self.nthetas = nthetas
        self.thetas = np.linspace(-np.pi, np.pi, self.nthetas)

        self.column_names = v1_column_names(nthetas=self.nthetas)
        self.args = dotdict(
            {
                "width": width,
            }
        )
        self.m_cache = {}
        self.receiver_pos = np.array(
            [
                [receiver_pos_x - receiver_spacing / 2, receiver_pos_y],
                [receiver_pos_x + receiver_spacing / 2, receiver_pos_y],
            ]
        )
        self.detector_position = np.array([[receiver_pos_x, receiver_pos_y]])
        self.snapshots_in_file = snapshots_in_file
        self.snapshots_in_session = snapshots_in_session
        self.step_size = step_size
        self.filenames = self.get_all_valid_files()
        if len(self.filenames) == 0:
            print("SessionsDatasetReal: No valid files to load from")
            raise ValueError
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.filenames)
        # self.datas=[
        #    np.memmap(
        #        filename,
        #        dtype='float32',
        #        mode='r',
        #        shape=(self.snapshots_in_file,self.nthetas+5)) for filename in self.filenames
        # ]
        self.sessions_per_file = [
            self.get_m(filename, bypass=True).shape[0]
            - (self.snapshots_in_session * self.step_size)
            for filename in self.filenames
        ]
        self.cumsum_sessions_per_file = np.cumsum([0] + self.sessions_per_file)
        self.len = sum(self.sessions_per_file)
        self.zeros = np.zeros((self.snapshots_in_session, 5))
        self.ones = np.ones((self.snapshots_in_session, 5))
        self.widths = (
            np.ones((self.snapshots_in_session, 1), dtype=np.int32) * self.args.width
        )
        self.halfpis = -np.ones((self.snapshots_in_session, 1)) * np.pi / 2
        # print("WARNING BY DEFAULT FLIPPING RADIO FEATURE SINCE COORDS WERE WRONG IN PI COLLECT!")

    def __getitem__(self, idx):
        fileidx, startidx = self.idx_to_fileidx_and_startidx(idx)
        m = self.get_m(self.filenames[fileidx])[
            startidx : startidx
            + self.snapshots_in_session * self.step_size : self.step_size
        ]

        detector_position_at_t = np.broadcast_to(
            self.detector_position, (m.shape[0], 2)
        )
        detector_orientation_at_t = self.zeros[:, [0]]
        source_positions_at_t = m[:, v1_tx_pos_idxs()][:, None]
        diffs = (
            source_positions_at_t - detector_position_at_t[:, None]
        )  # broadcast over nsource dimension
        # diffs=(batchsize, nsources, 2)
        source_theta_at_t = np.arctan2(
            diffs[:, 0, [0]], diffs[:, 0, [1]]
        )  # rotation to the right around x=0, y+
        return {
            "broadcasting_positions_at_t": self.ones[:, [0]][
                :, None
            ],  # TODO multi source
            "source_positions_at_t": source_positions_at_t,
            "source_velocities_at_t": self.zeros[:, :2][:, None],  # TODO calc velocity,
            "receiver_positions_at_t": np.broadcast_to(
                self.receiver_pos[None], (m.shape[0], 2, 2)
            ),
            "beam_former_outputs_at_t": m[:, v1_beamformer_start_idx() :],
            "thetas_at_t": np.broadcast_to(
                self.thetas[None], (m.shape[0], self.thetas.shape[0])
            ),
            "time_stamps": m[:, [v1_time_idx()]],
            "width_at_t": self.widths,
            "detector_orientation_at_t": detector_orientation_at_t,  # self.halfpis*0,#np.arctan2(1,0)=np.pi/2
            "detector_position_at_t": detector_position_at_t,
            "source_theta_at_t": source_theta_at_t,
            "source_distance_at_t": self.zeros[:, [0]][:, None],
        }


class SessionsDatasetRealExtension(SessionsDatasetReal):
    def file_shape(self):
        return (2, self.snapshots_in_file, len(self.column_names))

    def get_yaml_config(self):
        yaml_config = None
        for yaml_fn in self.get_all_files_with_extension("yaml"):
            if yaml_config is None:
                yaml_config = yaml.safe_load(open(yaml_fn, "r"))
            else:
                yaml_config_b = yaml.safe_load(open(yaml_fn, "r"))
                ddiff = DeepDiff(yaml_config, yaml_config_b, ignore_order=True)
                if self.check_files and len(ddiff):
                    raise ValueError("YAML configs do not match")
        return yaml_config

    def __init__(
        self,
        root_dir: str,
        column_names: List[str],
        snapshots_in_session: int = 128,  # how many points we consider a session
        nsources: int = 1,
        step_size: int = 1,  # how far apart snapshots_in_session are spaced out
        seed: int = 1337,
        check_files: bool = True,
        filenames=None,
    ):
        # time_step,x,y,mean_angle,_mean_angle #0,1,2,3,4
        # m = np.memmap(filename, dtype='float32', mode='r', shape=(,70))
        """
        Arguments:
          root_dir (string): Directory with all the images.
        """
        self.check_files = check_files
        self.root_dir = root_dir
        self.step_size = step_size
        yaml_config = self.get_yaml_config()
        self.snapshots_in_file = yaml_config["n-records-per-receiver"]
        if snapshots_in_session == -1:
            self.snapshots_in_session = self.snapshots_in_file
        else:
            assert snapshots_in_session > 0
            self.snapshots_in_session = snapshots_in_session

        assert self.snapshots_in_file % step_size == 0
        assert (self.snapshots_in_file / step_size) % self.snapshots_in_session == 0

        assert nsources == 1  # TODO implement more

        self.nthetas = yaml_config["n-thetas"]
        self.thetas = np.linspace(-np.pi, np.pi, self.nthetas)

        # self.column_names = v2_column_names(nthetas=self.nthetas)
        self.column_names = column_names

        if filenames is not None:
            assert len(filenames) > 0
            self.filenames = filenames
        else:
            self.filenames = self.get_all_valid_files()
        self.args = dotdict(
            {
                "width": yaml_config["width"],
            }
        )
        self.m_cache = {}

        # in case we need them later generate the offsets from the
        # RX center for antenna 0 and antenna 1
        self.n_receivers = len(yaml_config["receivers"])
        self.rx_antenna_offsets = []
        for receiver in yaml_config["receivers"]:
            self.rx_antenna_offsets.append(
                ULADetector(
                    sampling_frequency=None,
                    n_elements=2,
                    spacing=receiver["antenna-spacing-m"] * 1000,
                    orientation=receiver["theta-in-pis"] * np.pi,
                ).all_receiver_pos()
            )
        self.rx_antenna_offsets = np.array(self.rx_antenna_offsets)

        if len(self.filenames) == 0:
            print("SessionsDatasetReal: No valid files to load from")
            raise ValueError
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.filenames)

        self.sessions_per_file = [
            (
                int(
                    self.get_m(filename, bypass=True).shape[1]
                    / (
                        self.snapshots_in_session * self.step_size
                    )  # sessions per receiver
                )
            )
            * 2  # TODO should use n-receivers
            for filename in self.filenames
        ]
        self.cumsum_sessions_per_file = np.cumsum([0] + self.sessions_per_file)
        self.len = sum(self.sessions_per_file)

        # optimizations : cached constants values
        self.zeros = np.zeros((self.snapshots_in_session, 5))
        self.ones = np.ones((self.snapshots_in_session, 5))
        self.widths = (
            np.ones((self.snapshots_in_session, 1), dtype=np.int32) * self.args.width
        )
        self.halfpis = -np.ones((self.snapshots_in_session, 1)) * np.pi / 2


class SessionsDatasetRealV2(SessionsDatasetRealExtension):
    def __init__(self, *args, **kwargs):
        super(SessionsDatasetRealV2, self).__init__(
            column_names=v2_column_names(), *args, **kwargs
        )

    def __getitem__(self, idx):
        if idx is None:
            return None
        if type(idx) is tuple:
            # need to figure out which receiver A/B we are using here
            fileidx, unadjusted_startidx = self.idx_to_fileidx_and_startidx(idx[1])
            sessions_per_receiver = self.sessions_per_file[fileidx] // 2
            if idx[1] < 0 or idx[1] >= sessions_per_receiver:
                raise IndexError
            return self[idx[0] * sessions_per_receiver + idx[1]]
        if idx < 0 or idx >= self.len:
            raise IndexError

        # need to figure out which receiver A/B we are using here
        fileidx, unadjusted_startidx = self.idx_to_fileidx_and_startidx(idx)
        sessions_per_receiver = self.sessions_per_file[fileidx] // 2

        receiver_idx = 0  # assume A
        startidx = unadjusted_startidx
        if unadjusted_startidx >= sessions_per_receiver:  # use B
            receiver_idx = 1
            startidx = unadjusted_startidx - sessions_per_receiver
        # receiver_idx = 1 - receiver_idx
        m = self.get_m(self.filenames[fileidx])[
            receiver_idx,
            startidx : startidx
            + self.snapshots_in_session * self.step_size : self.step_size,
        ]

        rx_position_at_t = m[:, v2_rx_pos_idxs()]
        rx_orientation_at_t = m[:, v2_rx_theta_idx()]
        rx_antenna_positions_at_t = (
            rx_position_at_t[:, None] + self.rx_antenna_offsets[receiver_idx]
        )

        tx_positions_at_t = m[:, v2_tx_pos_idxs()][:, None]
        diffs = (
            tx_positions_at_t - rx_position_at_t[:, None]
        )  # broadcast over nsource dimension
        # diffs=(batchsize, nsources, 2)
        source_theta_at_t = pi_norm(
            np.arctan2(diffs[:, 0, [0]], diffs[:, 0, [1]]) - rx_orientation_at_t
        )  # rotation to the right around x=0, y+

        return {
            "broadcasting_positions_at_t": self.ones[:, [0]][
                :, None
            ],  # TODO multi source
            "source_positions_at_t": tx_positions_at_t,
            "source_velocities_at_t": self.zeros[:, :2][:, None],  # TODO calc velocity,
            "receiver_positions_at_t": rx_antenna_positions_at_t,
            "beam_former_outputs_at_t": m[:, v2_beamformer_start_idx() :],
            "thetas_at_t": np.broadcast_to(
                self.thetas[None], (m.shape[0], self.thetas.shape[0])
            ),
            "time_stamps": m[:, [v2_time_idx()]],
            "width_at_t": self.widths,
            "detector_orientation_at_t": rx_orientation_at_t,  # self.halfpis*0,#np.arctan2(1,0)=np.pi/2
            "detector_position_at_t": rx_position_at_t,
            "source_theta_at_t": source_theta_at_t,  # theta already accounting for orientation of detector
            "source_distance_at_t": self.zeros[:, [0]][:, None],
            "avg_phase_diffs": m[:, v2_avg_phase_diff_idxs()],
            "gain": m[:, [v2_gain_idxs()[receiver_idx]]],
            "rssi": m[:, [v2_rssi_idxs()[receiver_idx]]],
            "other_gain": m[:, [v2_gain_idxs()[1 - receiver_idx]]],
            "other_rssi": m[:, [v2_rssi_idxs()[1 - receiver_idx]]],
        }


class SessionsDatasetRealV3Rx(SessionsDatasetRealExtension):
    def __init__(self, *args, **kwargs):
        super(SessionsDatasetRealV3Rx, self).__init__(
            column_names=v3rx_column_names(), *args, **kwargs
        )

    def times2x32_to_64(self, m):
        return np.frombuffer(m[:, v3rx_time_idxs()].copy(), dtype=np.float64)

    def __getitem__(self, idx):
        if idx is None:
            return None
        if type(idx) is tuple:
            # need to figure out which receiver A/B we are using here
            fileidx, unadjusted_startidx = self.idx_to_fileidx_and_startidx(idx[1])
            sessions_per_receiver = self.sessions_per_file[fileidx] // 2
            if idx[1] < 0 or idx[1] >= sessions_per_receiver:
                raise IndexError
            return self[idx[0] * sessions_per_receiver + idx[1]]
        if idx < 0 or idx >= self.len:
            raise IndexError

        # need to figure out which receiver A/B we are using here
        fileidx, unadjusted_startidx = self.idx_to_fileidx_and_startidx(idx)
        sessions_per_receiver = self.sessions_per_file[fileidx] // 2

        receiver_idx = 0  # assume A
        startidx = unadjusted_startidx
        if unadjusted_startidx >= sessions_per_receiver:  # use B
            receiver_idx = 1
            startidx = unadjusted_startidx - sessions_per_receiver
        # receiver_idx = 1 - receiver_idx
        m = self.get_m(self.filenames[fileidx])[
            receiver_idx,
            startidx : startidx
            + self.snapshots_in_session * self.step_size : self.step_size,
        ]

        rx_position_at_t = m[:, v3rx_rx_pos_idxs()]
        rx_orientation_at_t = m[:, v3rx_rx_theta_idx()]
        rx_antenna_positions_at_t = (
            rx_position_at_t[:, None] + self.rx_antenna_offsets[receiver_idx]
        )
        # _z = struct.unpack("d", struct.pack("ff", a, b))[0]
        return {
            "broadcasting_positions_at_t": self.ones[:, [0]][
                :, None
            ],  # TODO multi source
            "source_velocities_at_t": self.zeros[:, :2][:, None],  # TODO calc velocity,
            "receiver_positions_at_t": rx_antenna_positions_at_t,
            "beam_former_outputs_at_t": m[:, v3rx_beamformer_start_idx() :],
            "thetas_at_t": np.broadcast_to(
                self.thetas[None], (m.shape[0], self.thetas.shape[0])
            ),
            "time_stamps": self.times2x32_to_64(m),
            "width_at_t": self.widths,
            "detector_orientation_at_t": rx_orientation_at_t,  # self.halfpis*0,#np.arctan2(1,0)=np.pi/2
            "detector_position_at_t": rx_position_at_t,
            "source_distance_at_t": self.zeros[:, [0]][:, None],
            "avg_phase_diffs": m[:, v3rx_avg_phase_diff_idxs()],
            "gain": m[:, [v3rx_gain_idxs()[receiver_idx]]],
            "rssi": m[:, [v3rx_rssi_idxs()[receiver_idx]]],
            "other_gain": m[:, [v3rx_gain_idxs()[1 - receiver_idx]]],
            "other_rssi": m[:, [v3rx_rssi_idxs()[1 - receiver_idx]]],
        }


class SessionsDatasetRealTask2(SessionsDatasetRealV1):
    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        # normalize these before heading out
        d["source_positions_at_t_normalized_centered"] = 2 * (
            d["source_positions_at_t"] / self.args.width - 0.5
        )
        d["source_velocities_at_t_normalized"] = (
            d["source_velocities_at_t"] / self.args.width
        )
        d["detector_position_at_t_normalized_centered"] = 2 * (
            d["detector_position_at_t"] / self.args.width - 0.5
        )
        d["source_distance_at_t_normalized"] = d["source_distance_at_t"] / (
            self.args.width / 2
        )
        return d  # ,d['source_positions_at_t']


class SessionsDatasetTask2(SessionsDatasetSimulated):
    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        # normalize these before heading out
        d["source_positions_at_t_normalized_centered"] = 2 * (
            d["source_positions_at_t"] / self.args.width - 0.5
        )
        d["source_velocities_at_t_normalized"] = (
            d["source_velocities_at_t"] / self.args.width
        )
        d["detector_position_at_t_normalized_centered"] = 2 * (
            d["detector_position_at_t"] / self.args.width - 0.5
        )
        d["source_distance_at_t_normalized"] = d["source_distance_at_t"] / (
            self.args.width / 2
        )
        return d  # ,d['source_positions_at_t']


class SessionsDatasetTask2WithImages(SessionsDatasetSimulated):
    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        # normalize these before heading out
        d["source_positions_at_t_normalized"] = 2 * (
            d["source_positions_at_t"] / self.args.width - 0.5
        )
        d["source_velocities_at_t_normalized"] = (
            d["source_velocities_at_t"] / self.args.width
        )
        d["detector_position_at_t_normalized"] = 2 * (
            d["detector_position_at_t"] / self.args.width - 0.5
        )
        d["source_distance_at_t_normalized"] = d["source_distance_at_t"].mean(
            axis=2
        ) / (self.args.width / 2)

        d["source_image_at_t"] = labels_to_source_images(
            d["source_positions_at_t"][None], self.args.width
        )[0]
        d["detector_theta_image_at_t"] = detector_positions_to_theta_grid(
            d["detector_position_at_t"][None], self.args.width
        )[0]
        d["radio_image_at_t"] = radio_to_image(
            d["beam_former_outputs_at_t"][None],
            d["detector_theta_image_at_t"][None],
            d["detector_orientation_at_t"][None],
        )[0]

        return d  # ,d['source_positions_at_t']


def collate_fn_beamformer(_in):
    d = {k: torch.from_numpy(np.stack([x[k] for x in _in])) for k in _in[0]}
    b, s, n_sources, _ = d["source_positions_at_t"].shape

    # times = d["time_stamps"] / (0.00001 + d["time_stamps"].max(axis=2, keepdim=True)[0])

    source_theta = d["source_theta_at_t"].mean(axis=2)
    # distances = d["source_distance_at_t_normalized"].mean(axis=2, keepdims=True)
    _, _, beam_former_bins = d["beam_former_outputs_at_t"].shape
    perfect_labels = torch.zeros((b, s, beam_former_bins))

    idxs = (beam_former_bins * (d["source_theta_at_t"] + np.pi) / (2 * np.pi)).int()
    smooth_bins = int(beam_former_bins * 0.25 * 0.5)
    for _b in torch.arange(b):
        for _s in torch.arange(s):
            for smooth in range(-smooth_bins, smooth_bins + 1):
                perfect_labels[_b, _s, (idxs[_b, _s] + smooth) % beam_former_bins] = (
                    1 / (1 + smooth**2)
                )
            perfect_labels[_b, _s] /= perfect_labels[_b, _s].sum() + 1e-9
    r = {
        "input": torch.concatenate(
            [
                # d['signal_matrixs_at_t'].reshape(b,s,-1),
                (
                    d["signal_matrixs_at_t"]
                    / d["signal_matrixs_at_t"].abs().mean(axis=[2, 3], keepdims=True)
                ).reshape(
                    b, s, -1
                ),  # normalize the data
                d["signal_matrixs_at_t"]
                .abs()
                .mean(axis=[2, 3], keepdims=True)
                .reshape(b, s, -1),  #
                d["detector_orientation_at_t"].to(torch.complex64),
            ],
            axis=2,
        ),
        "beamformer": d["beam_former_outputs_at_t"],
        "labels": perfect_labels,
        "thetas": source_theta,
    }
    return r


def collate_fn_transformer_filter(_in):
    d = {k: torch.from_numpy(np.stack([x[k] for x in _in])) for k in _in[0]}
    b, s, n_sources, _ = d["source_positions_at_t"].shape

    # normalized_01_times = d["time_stamps"] / (
    #     0.0000001 + d["time_stamps"].max(axis=1, keepdim=True)[0]
    # )
    normalized_times = (
        d["time_stamps"] - d["time_stamps"].max(axis=1, keepdims=True)[0]
    ) / 100

    normalized_pirads_detector_theta = d["detector_orientation_at_t"] / np.pi

    space_diffs = (
        d["detector_position_at_t_normalized_centered"][:, :-1]
        - d["detector_position_at_t_normalized_centered"][:, 1:]
    )
    space_delta = torch.cat(
        [
            torch.zeros(b, 1, 2),
            space_diffs,
        ],
        axis=1,
    )

    normalized_pirads_space_theta = torch.cat(
        [
            torch.zeros(b, 1, 1),
            (torch.atan2(space_diffs[..., 1], space_diffs[..., 0]))[:, :, None] / np.pi,
        ],
        axis=1,
    )

    space_dist = torch.cat(
        [
            torch.zeros(b, 1, 1),
            torch.sqrt(torch.pow(space_diffs, 2).sum(axis=2, keepdim=True)),
        ],
        axis=1,
    )
    return {
        "drone_state": torch.cat(
            [
                d["detector_position_at_t_normalized_centered"],  # 2: 2
                # normalized_01_times, #-times.max(axis=2,keepdim=True)[0], # 1: 3
                # normalized_times, #-times.max(axis=2,keepdim=True)[0], # 1: 3
                space_delta,  # 2: 4
                normalized_pirads_space_theta,  # 1: 5
                space_dist,  # 1: 6
                normalized_pirads_detector_theta,  # 1: 7
            ],
            dim=2,
        ).float(),
        "times": normalized_times,
        "emitter_position_and_velocity": torch.cat(
            [
                d["source_positions_at_t_normalized_centered"],
                d["source_velocities_at_t_normalized"],
            ],
            dim=3,
        ),
        "emitters_broadcasting": d["broadcasting_positions_at_t"],
        "emitters_n_broadcasts": d["broadcasting_positions_at_t"].cumsum(axis=1),
        "radio_feature": torch.cat(
            [
                torch.log(d["beam_former_outputs_at_t"].mean(axis=2, keepdim=True))
                / 20,
                d["beam_former_outputs_at_t"]
                / d["beam_former_outputs_at_t"].mean(
                    axis=2, keepdim=True
                ),  # maybe pass in log values?
            ],
            dim=2,
        ).float(),
    }


def collate_fn(_in):
    d = {k: torch.from_numpy(np.stack([x[k] for x in _in])) for k in _in[0]}
    b, s, n_sources, _ = d["source_positions_at_t"].shape

    # times=d['time_stamps']/(0.00001+d['time_stamps'].max(axis=2,keepdim=True)[0])
    times = d["time_stamps"] / (
        0.0000001 + d["time_stamps"].max(axis=1, keepdim=True)[0]
    )

    # deal with source positions
    source_position = (
        d["source_positions_at_t_normalized"][
            torch.where(d["broadcasting_positions_at_t"] == 1)[:-1]
        ]
        .reshape(b, s, 2)
        .float()
    )
    source_velocity = (
        d["source_velocities_at_t_normalized"][
            torch.where(d["broadcasting_positions_at_t"] == 1)[:-1]
        ]
        .reshape(b, s, 2)
        .float()
    )

    # deal with detector position features
    # diffs=source_positions-d['detector_position_at_t_normalized']
    # source_thetab=(torch.atan2(diffs[...,1],diffs[...,0]))[:,:,None]/np.pi # batch, snapshot,1, x ,y
    source_theta = d["source_theta_at_t"].mean(axis=2) / np.pi
    detector_theta = d["detector_orientation_at_t"] / np.pi
    # distancesb=torch.sqrt(torch.pow(diffs, 2).sum(axis=2,keepdim=True))
    distances = d["source_distance_at_t_normalized"].mean(axis=2, keepdims=True)
    space_diffs = (
        d["detector_position_at_t_normalized"][:, :-1]
        - d["detector_position_at_t_normalized"][:, 1:]
    )
    space_delta = torch.cat(
        [
            torch.zeros(b, 1, 2),
            space_diffs,
        ],
        axis=1,
    )

    space_theta = torch.cat(
        [
            torch.zeros(b, 1, 1),
            (torch.atan2(space_diffs[..., 1], space_diffs[..., 0]))[:, :, None] / np.pi,
        ],
        axis=1,
    )

    space_dist = torch.cat(
        [
            torch.zeros(b, 1, 1),
            torch.sqrt(torch.pow(space_diffs, 2).sum(axis=2, keepdim=True)),
        ],
        axis=1,
    )

    # create the labels
    labels = torch.cat(
        [
            source_position,  # zero center the positions
            (source_theta + detector_theta + 1) % 2.0
            - 1,  # initialy in units of np.pi?
            distances,  # try to zero center?
            space_delta,
            space_theta,
            space_dist,
            source_velocity,
        ],
        axis=2,
    ).float()  # .to(device)

    # create the features
    inputs = {
        "drone_state": torch.cat(
            [
                d["detector_position_at_t_normalized"],
                times,  # -times.max(axis=2,keepdim=True)[0],
                space_delta,
                space_theta,
                space_dist,
                detector_theta,
            ],
            dim=2,
        ).float(),
        "radio_feature": torch.cat(
            [
                torch.log(d["beam_former_outputs_at_t"].mean(axis=2, keepdim=True))
                / 20,
                d["beam_former_outputs_at_t"]
                / d["beam_former_outputs_at_t"].mean(
                    axis=2, keepdim=True
                ),  # maybe pass in log values?
            ],
            dim=2,
        ).float(),
    }
    if "radio_image_at_t" in d:
        radio_images = d["radio_image_at_t"].float()
        label_images = d["source_image_at_t"].float()
        return {
            "inputs": inputs,
            "input_images": radio_images,
            "labels": labels,
            "label_images": label_images,
        }
    return {"inputs": inputs, "labels": labels}
