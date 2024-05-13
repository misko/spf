###
# Experiment 1 : wall drones , receiver center is x=135.27 cm, y=264.77cm, dist between is 6cm
###

import bisect
import os
import pickle
import time
from multiprocessing import Pool
from typing import List

import numpy as np
import torch
import tqdm
import yaml
from compress_pickle import load
from deepdiff import DeepDiff
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
    SEGMENTATION_VERSION,
    ULADetector,
    phase_diff_to_theta,
    pi_norm,
    precompute_steering_vectors,
    segment_session_star,
    speed_of_light,
    torch_get_phase_diff,
)
from spf.sdrpluto.sdr_controller import rx_config_from_receiver_yaml
from spf.utils import zarr_open_from_lmdb_store


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


def mp_segment_zarr(zarr_fn, results_fn):
    z = zarr_open_from_lmdb_store(zarr_fn)
    n_sessions, _, _ = z.receivers["r0"].signal_matrix.shape

    results_by_receiver = {}
    for r_idx in [0, 1]:
        r_name = f"r{r_idx}"
        inputs = [
            {
                "zarr_fn": zarr_fn,
                "receiver": r_name,
                "session_idx": idx,
                "window_size": 2048,
                "stride": 2048,
                "trim": 20.0,
                "mean_diff_threshold": 0.2,
                "max_stddev_threshold": 0.5,
                "drop_less_than_size": 3000,
                "min_abs_signal": 40,
            }
            for idx in range(n_sessions)
        ]
        with Pool(8) as pool:
            results_by_receiver[r_name] = list(
                tqdm.tqdm(pool.imap(segment_session_star, inputs), total=len(inputs))
            )
    pickle.dump(
        {
            "version": SEGMENTATION_VERSION,
            "segmentation_by_receiver": results_by_receiver,
        },
        open(results_fn, "wb"),
    )


def v5_prepare_session(session):  # session -> x,y
    abs_signal = session["signal_matrix"].abs().to(torch.float32)
    pd = torch_get_phase_diff(session["signal_matrix"]).to(torch.float32)
    gt_theta = session["ground_truth_theta"]
    if abs(gt_theta.item()) > torch.pi / 2:
        return (
            torch.vstack([abs_signal[1], abs_signal[0], -pd])[None],
            gt_theta.sign().item() * torch.pi - gt_theta[None],
        )
    return (
        torch.vstack([abs_signal[0], abs_signal[1], pd])[None],
        gt_theta[None],
    )


# target_thetas (N,1)
def v5_thetas_to_targets(target_thetas, nthetas):
    if target_thetas.ndim == 1:
        target_thetas = target_thetas.reshape(-1, 1)
    p = torch.exp(
        -(
            (
                target_thetas
                - torch.linspace(
                    -torch.pi / 2, torch.pi / 2, nthetas, device=target_thetas.device
                ).reshape(1, -1)
            )
            ** 2
        )
    )
    return torch.nn.functional.normalize(p, p=1, dim=1)


def v5_collate_beamsegnet(batch):
    return (
        torch.vstack([x["x"] for x in batch]),
        # torch.vstack([x["y_discrete"] for x in batch]),
        torch.vstack([x["y_rad"] for x in batch]),
        [x["simple_segementation"] for x in batch],
    )


class v5spfdataset(Dataset):
    def __init__(self, prefix, nthetas):
        prefix = prefix.replace(".zarr", "")
        self.nthetas = nthetas
        self.prefix = prefix
        self.zarr_fn = f"{prefix}.zarr"
        self.yaml_fn = f"{prefix}.yaml"
        self.z = zarr_open_from_lmdb_store(self.zarr_fn)
        self.yaml_config = yaml.safe_load(open(self.yaml_fn, "r"))

        self.n_receivers = len(self.yaml_config["receivers"])

        self.rx_configs = [
            rx_config_from_receiver_yaml(receiver)
            for receiver in self.yaml_config["receivers"]
        ]

        self.receiver_data = [
            self.z.receivers[f"r{ridx}"] for ridx in range(self.n_receivers)
        ]

        self.signal_matrices = [
            self.z.receivers[f"r{ridx}"].signal_matrix
            for ridx in range(self.n_receivers)
        ]

        self.n_sessions, self.n_antennas_per_receiver, self.session_length = (
            self.signal_matrices[0].shape
        )
        assert self.n_antennas_per_receiver == 2
        for ridx in range(self.n_receivers):
            assert self.signal_matrices[ridx].shape == (
                self.n_sessions,
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

        self.keys_per_session = v5rx_f64_keys + v5rx_2xf64_keys + ["signal_matrix"]

        self.ground_truth_thetas = self.get_ground_truth_thetas()
        self.get_segmentation()

    def __len__(self):
        return self.n_sessions * self.n_receivers

    def render_session(self, receiver_idx, session_idx):
        r = self.receiver_data[receiver_idx]
        data = {key: r[key][session_idx] for key in self.keys_per_session}

        data["ground_truth_theta"] = self.ground_truth_thetas[receiver_idx][session_idx]
        data = {
            k: (
                torch.from_numpy(v)
                if type(v) not in (np.float64, float)
                else torch.Tensor([v])
            )
            for k, v in data.items()
        }
        data["x"], data["y_rad"] = v5_prepare_session(data)
        # data["y_discrete"] = v5_thetas_to_targets(data["y_rad"], self.nthetas)
        data["simple_segementation"] = self.segmentation["segmentation_by_receiver"][
            f"r{receiver_idx}"
        ][session_idx]
        return data

    def get_ground_truth_thetas(self):
        ground_truth_thetas = []
        for ridx in range(self.n_receivers):
            rx_theta_in_pis = self.receiver_data[ridx]["rx_theta_in_pis"]
            tx_pos = np.array(
                [
                    self.receiver_data[ridx]["tx_pos_x_mm"],
                    self.receiver_data[ridx]["tx_pos_y_mm"],
                ]
            )
            rx_pos = np.array(
                [
                    self.receiver_data[ridx]["rx_pos_x_mm"],
                    self.receiver_data[ridx]["rx_pos_y_mm"],
                ]
            )

            # compute the angle of the tx with respect to rx
            d = tx_pos - rx_pos

            rx_to_tx_theta = np.arctan2(d[0], d[1])
            ground_truth_thetas.append(
                pi_norm(rx_to_tx_theta - rx_theta_in_pis[:] * np.pi)
            )
        return ground_truth_thetas

    def __getitem__(self, idx):
        assert idx < self.n_sessions * self.n_receivers
        receiver_idx = idx % self.n_receivers
        return self.render_session(receiver_idx, idx // self.n_receivers)

    def get_segmentation_mean_phase(self):
        segmentation = self.get_segmentation()
        mean_phase_results = {}
        for receiver, results in segmentation['segmentation_by_receiver'].items():
            mean_phase_results[receiver] = np.array(
                [
                    np.array([x["mean"] for x in result["simple_segmentation"]]).mean()
                    for result in results
                ]
            )
        return mean_phase_results

    def get_estimated_thetas(self):
        mean_phase = self.get_segmentation_mean_phase()
        estimated_thetas = {}
        for ridx in range(self.n_receivers):
            carrier_freq = self.yaml_config["receivers"][ridx]["f-carrier"]
            antenna_spacing = self.yaml_config["receivers"][ridx]["antenna-spacing-m"]
            estimated_thetas[f"r{ridx}"] = phase_diff_to_theta(
                mean_phase[f"r{ridx}"],
                speed_of_light / carrier_freq,
                antenna_spacing,
                large_phase_goes_right=False,
            )
        return estimated_thetas

    def get_segmentation(self):
        if not hasattr(self, "segmentation"):
            results_fn = self.prefix + "_segmentation.pkl"
            if not os.path.exists(results_fn):
                mp_segment_zarr(self.zarr_fn, results_fn)
            segmentation = pickle.load(open(results_fn, "rb"))
            if (
                "version" not in segmentation
                or segmentation["version"] != SEGMENTATION_VERSION
            ):
                os.remove(results_fn)
                return self.get_segmentation()
            self.segmentation = segmentation
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
    # breakpoint()
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
