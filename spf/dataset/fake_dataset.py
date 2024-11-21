import argparse
import os
import random
import shutil

import numpy as np
import torch
import yaml
import zarr

from spf.data_collector import rx_config_from_receiver_yaml

# V5 data format
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys, v5rx_new_dataset
from spf.rf import speed_of_light, torch_get_avg_phase_notrim, torch_pi_norm_pi
from spf.scripts.create_empirical_p_dist import (
    create_empirical_p_dist,
    get_empirical_p_dist_parser,
)
from spf.sdrpluto.sdr_controller import rx_config_from_receiver_yaml
from spf.utils import torch_random_signal_matrix, zarr_open_from_lmdb_store, zarr_shrink


@torch.jit.script
def phi_to_signal_matrix(
    phi: torch.Tensor,
    buffer_size: int,
    noise: float,
    phi_drift: float,
    generator: torch.Generator,
):
    big_phi = phi.repeat(buffer_size).reshape(1, -1)
    big_phi_with_noise = (
        big_phi + torch.randn((1, buffer_size), generator=generator) * noise
    )
    offsets = torch.zeros(big_phi.shape, dtype=torch.complex64)
    return (
        torch.vstack(
            [
                torch.exp(1j * (offsets + phi_drift)),
                torch.exp(1j * (offsets + big_phi_with_noise)),
            ]
        )
        * 200
    )


"""
theta is the angle from array normal to incident
phi is phase difference

delta_distance  = d*sin(theta)
phi = delta_distance * 2pi / lambda = sin(theta)*d*2pi/lambda
theta = arcsin(lambda * phi / (d*2pi))
"""

fake_yaml = """
# The ip of the emitter
# When the emitter is brought online it is verified
# by a receiver that it actually is broadcasting
emitter:
  type: esp32
  motor_channel: 1

# Two receivers each with two antennas
# When a receiver is brought online it performs
# phase calibration using an emitter equidistant from
# both receiver antenna
# The orientation of the receiver is described in 
# multiples of pi
receivers:
  - receiver-uri: fake
    theta-in-pis: -0.25
    antenna-spacing-m: 0.05075 # 50.75 mm 
    nelements: 2
    array-type: linear
    rx-gain-mode: fast_attack
    rx-buffers: 2
    rx-gain: -3
    buffer-size: 524288
    f-intermediate: 100000 #1.0e5
    f-carrier: 2412000000 #2.5e9
    f-sampling: 16000000 # 16.0e6
    bandwidth: 300000 #3.0e5
    motor_channel: 0
  - receiver-uri: fake
    theta-in-pis: 1.25
    antenna-spacing-m: 0.05075 # 50.75 mm 
    nelements: 2
    array-type: linear
    rx-gain-mode: fast_attack
    rx-buffers: 2
    rx-gain: -3
    buffer-size: 524288
    f-intermediate: 100000 #1.0e5
    f-carrier: 2412000000 #2.5e9
    f-sampling: 16000000 # 16.0e6
    bandwidth: 300000 #3.0e5
    motor_channel: 0


n-thetas: 65
n-records-per-receiver: 5
width: 4000
calibration-frames: 10
routine: null
skip_phase_calibration: true
  

data-version: 5
seconds-per-sample: 5.0
"""


def create_empirical_dist_for_datasets(datasets, precompute_cache, nthetas):

    parser = get_empirical_p_dist_parser()

    empirical_pkl_fn = precompute_cache + "/full.pkl"

    args = parser.parse_args(
        [
            "--out",
            empirical_pkl_fn,
            "--nbins",
            f"{nthetas}",
            "--nthetas",
            f"{nthetas}",
            "--precompute-cache",
            precompute_cache,
            "--device",
            "cpu",
            "-d",
        ]
        + datasets
    )
    create_empirical_p_dist(args)
    return empirical_pkl_fn


def create_fake_dataset(
    yaml_config_str,
    filename,
    orbits=1,
    n=50,
    noise=0.01,
    phi_drift=0.0,
    radius=10000,
    seed=0,
):
    random.seed(seed)
    np.random.seed(seed)

    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)
    yaml_fn = f"{filename}.yaml"
    zarr_fn = f"{filename}.zar"
    seg_fn = f"{filename}_segmentation.pkl"
    for fn in [yaml_fn, zarr_fn, seg_fn]:
        if os.path.exists(fn):
            os.remove(fn)
    yaml_config = yaml.safe_load(yaml_config_str)
    yaml_config["n-records-per-receiver"] = n

    with open(f"{filename}.yaml", "w") as outfile:
        yaml.dump(yaml_config, outfile, default_flow_style=False)
    rx_config = rx_config_from_receiver_yaml(yaml_config["receivers"][0])

    _lambda = speed_of_light / rx_config.lo

    m = v5rx_new_dataset(
        filename=f"{filename}.zarr",
        timesteps=yaml_config["n-records-per-receiver"],
        buffer_size=rx_config.buffer_size,
        n_receivers=len(yaml_config["receivers"]),
        chunk_size=512,
        compressor=None,
        config=yaml_config,
    )

    thetas = torch_pi_norm_pi(
        torch.linspace(0, 2 * torch.pi * orbits, yaml_config["n-records-per-receiver"])
    )

    def theta_to_phi(theta, antenna_spacing_m, _lambda):
        return torch.sin(theta) * antenna_spacing_m * 2 * torch.pi / _lambda

    def phi_to_theta(phi, antenna_spacing_m, _lambda, limit=False):
        sin_arg = _lambda * phi / (antenna_spacing_m * 2 * torch.pi)
        # assert sin_arg.min()>-1
        # assert sin_arg.max()<1
        if limit:
            edge = 1 - 1e-8
            sin_arg = torch.clip(sin_arg, min=-edge, max=edge)
        v = torch.arcsin(_lambda * phi / (antenna_spacing_m * 2 * torch.pi))
        return v, torch.pi - v

    rnd_noise = torch.randn(thetas.shape[0], generator=torch_generator)

    # signal_matrix = np.vstack([np.exp(1j * phis), np.ones(phis.shape)])

    for receiver_idx in range(2):
        receiver_thetas = (
            thetas - yaml_config["receivers"][receiver_idx]["theta-in-pis"] * np.pi
        )
        phis_nonoise = theta_to_phi(receiver_thetas, rx_config.rx_spacing, _lambda)
        phis = torch_pi_norm_pi(phis_nonoise + rnd_noise * noise)
        # _thetas = phi_to_theta(phis, rx_config.rx_spacing, _lambda, limit=True)

        for record_idx in range(yaml_config["n-records-per-receiver"]):
            signal_matrix = phi_to_signal_matrix(
                phis[[record_idx]],
                rx_config.buffer_size,
                noise,
                phi_drift * torch.pi * (1 if receiver_idx == 0 else -1),
                generator=torch_generator,
            )

            noise_matrix = torch_random_signal_matrix(
                signal_matrix.reshape(-1).shape[0], generator=torch_generator
            ).reshape(signal_matrix.shape)
            # add stripes
            window_size = 2048 * 4
            for x in range(0, rx_config.buffer_size, window_size):
                if (x // window_size) % 3 == 0:
                    signal_matrix[:, x : x + window_size] = noise_matrix[
                        :, x : x + window_size
                    ]

            data = {
                "rx_pos_x_mm": 0,
                "rx_pos_y_mm": 0,
                "tx_pos_x_mm": torch.sin(thetas[record_idx]) * radius,
                "tx_pos_y_mm": torch.cos(thetas[record_idx]) * radius,
                "system_timestamp": 1.0 + record_idx * 5.0,
                "rx_theta_in_pis": yaml_config["receivers"][receiver_idx][
                    "theta-in-pis"
                ],
                "rx_spacing": rx_config.rx_spacing,
                "rx_lo": rx_config.lo,
                "rx_bandwidth": rx_config.rf_bandwidth,
                "avg_phase_diff": torch_get_avg_phase_notrim(signal_matrix),  # , 0.0),
                "rssis": [0, 0],
                "gains": [0, 0],
                "rx_heading": 0,
            }

            z = m[f"receivers/r{receiver_idx}"]
            z.signal_matrix[record_idx] = signal_matrix.numpy()
            for k in v5rx_f64_keys + v5rx_2xf64_keys:
                z[k][record_idx] = data[k]
            # nthetas = 64 + 1

            # steering_vectors = precompute_steering_vectors(
            #     receiver_positions=rx_config.rx_pos,
            #     carrier_frequency=rx_config.lo,
            #     spacing=nthetas,
            # )
            # beam_sds = beamformer_given_steering_nomean(
            #     steering_vectors=steering_vectors,
            #     signal_matrix=signal_matrix,
            # )


def compare_and_copy_n(prefix, src, dst, n):
    if isinstance(src, zarr.hierarchy.Group):
        for key in src.keys():
            compare_and_copy_n(prefix + "/" + key, src[key], dst[key], n)
    else:
        if prefix == "/config":
            if src.shape != ():
                dst[:] = src[:]
        else:
            for x in range(n):
                dst[x] = src[x]


def compare_and_copy_nth(prefix, src, dst, n):
    if isinstance(src, zarr.hierarchy.Group):
        for key in src.keys():
            compare_and_copy_nth(prefix + "/" + key, src[key], dst[key], n)
    else:
        if prefix == "/config":
            if src.shape != ():
                dst[:] = src[:]
        else:
            dst[n] = src[n]


def partial_dataset(input_fn, output_fn, n):
    input_fn.replace(".zarr", "")
    z = zarr_open_from_lmdb_store(input_fn + ".zarr")
    timesteps, _, buffer_size = z["receivers/r0/signal_matrix"].shape
    input_yaml_fn = input_fn + ".yaml"
    output_yaml_fn = output_fn + ".yaml"
    yaml_config = yaml.safe_load(open(input_yaml_fn, "r"))
    shutil.copyfile(input_yaml_fn, output_yaml_fn)
    new_z = v5rx_new_dataset(
        filename=output_fn + ".zarr",
        timesteps=timesteps,
        buffer_size=buffer_size,
        n_receivers=len(yaml_config["receivers"]),
        chunk_size=512,
        compressor=None,
        config=yaml_config,
        remove_if_exists=False,
    )
    compare_and_copy_n("", z, new_z, n)
    new_z.store.close()
    new_z = None
    zarr_shrink(output_fn)


class PartialDatasetController:
    def __init__(self, input_fn, output_fn):
        input_fn.replace(".zarr", "")
        self.z = zarr_open_from_lmdb_store(input_fn + ".zarr")
        timesteps, _, buffer_size = self.z["receivers/r0/signal_matrix"].shape
        input_yaml_fn = input_fn + ".yaml"
        output_yaml_fn = output_fn + ".yaml"
        yaml_config = yaml.safe_load(open(input_yaml_fn, "r"))
        shutil.copyfile(input_yaml_fn, output_yaml_fn)
        self.new_z = v5rx_new_dataset(
            filename=output_fn + ".zarr",
            timesteps=timesteps,
            buffer_size=buffer_size,
            n_receivers=len(yaml_config["receivers"]),
            chunk_size=512,
            compressor=None,
            config=yaml_config,
            remove_if_exists=False,
        )

    def copy_nth(self, n):
        compare_and_copy_nth("", self.z, self.new_z, n)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        required=False,
        default="fake_dataset",
    )
    parser.add_argument(
        "--orbits",
        type=int,
        required=False,
        default="2",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=False,
        default="1024",
    )
    parser.add_argument(
        "--noise",
        type=float,
        required=False,
        default="0.3",
    )
    parser.add_argument(
        "--phi-drift",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    create_fake_dataset(
        fake_yaml,
        args.filename,
        orbits=args.orbits,
        n=args.n,
        noise=args.noise,
        phi_drift=args.phi_drift,
        radius=10000,
        seed=args.seed,
    )
