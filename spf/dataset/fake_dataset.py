import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
import random
from spf.data_collector import rx_config_from_receiver_yaml
from spf.dataset.spf_dataset import pi_norm

# V5 data format
from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys, v5rx_new_dataset
from spf.rf import (
    beamformer_given_steering_nomean,
    get_avg_phase,
    pi_norm,
    precompute_steering_vectors,
    speed_of_light,
)
from spf.sdrpluto.sdr_controller import rx_config_from_receiver_yaml
from spf.utils import random_signal_matrix

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

    thetas = pi_norm(
        np.linspace(0, 2 * np.pi * orbits, yaml_config["n-records-per-receiver"])
    )

    def theta_to_phi(theta, antenna_spacing_m, _lambda):
        return np.sin(theta) * antenna_spacing_m * 2 * np.pi / _lambda

    def phi_to_theta(phi, antenna_spacing_m, _lambda, limit=False):
        sin_arg = _lambda * phi / (antenna_spacing_m * 2 * np.pi)
        # assert sin_arg.min()>-1
        # assert sin_arg.max()<1
        if limit:
            edge = 1 - 1e-8
            sin_arg = np.clip(sin_arg, a_min=-edge, a_max=edge)
        v = np.arcsin(_lambda * phi / (antenna_spacing_m * 2 * np.pi))
        return v, np.pi - v

    rnd_noise = np.random.randn(thetas.shape[0])

    # signal_matrix = np.vstack([np.exp(1j * phis), np.ones(phis.shape)])

    for receiver_idx in range(2):
        receiver_thetas = (
            thetas - yaml_config["receivers"][receiver_idx]["theta-in-pis"] * np.pi
        )
        phis_nonoise = theta_to_phi(receiver_thetas, rx_config.rx_spacing, _lambda)
        phis = pi_norm(phis_nonoise + rnd_noise * noise)
        _thetas = phi_to_theta(phis, rx_config.rx_spacing, _lambda, limit=True)

        for record_idx in range(yaml_config["n-records-per-receiver"]):
            big_phi = phis[[record_idx], None].repeat(rx_config.buffer_size, axis=1)
            big_phi_with_noise = big_phi + np.random.randn(*big_phi.shape) * noise
            offsets = np.random.uniform(-np.pi, np.pi, big_phi.shape) * 0
            signal_matrix = (
                np.vstack(
                    [
                        np.exp(
                            1j
                            * (
                                offsets
                                + phi_drift * np.pi * (1 if receiver_idx == 0 else -1)
                            )
                        ),
                        np.exp(1j * (offsets + big_phi_with_noise)),
                    ]
                )
                * 200
            )
            noise_matrix = random_signal_matrix(
                signal_matrix.reshape(-1).shape[0]
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
                "tx_pos_x_mm": np.sin(thetas[record_idx]) * radius,
                "tx_pos_y_mm": np.cos(thetas[record_idx]) * radius,
                "system_timestamp": record_idx * 5.0,
                "rx_theta_in_pis": yaml_config["receivers"][receiver_idx][
                    "theta-in-pis"
                ],
                "rx_spacing": rx_config.rx_spacing,
                "rx_lo": rx_config.lo,
                "rx_bandwidth": rx_config.rf_bandwidth,
                "avg_phase_diff": get_avg_phase(signal_matrix),
                "rssis": [0, 0],
                "gains": [0, 0],
            }

            z = m[f"receivers/r{receiver_idx}"]
            z.signal_matrix[record_idx] = signal_matrix
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
