import numpy as np


def get_column_names_v2(nthetas=65):
    thetas = np.linspace(-np.pi, np.pi, nthetas)
    return [
        "timestamp",
        "tx_pos_x",
        "tx_pos_y",
        "rx_pos_x",
        "rx_pos_y",
        "rx_theta_in_pis",
        "rx_spacing",
        "avg_phase_diff_1",
        "avg_phase_diff_2",
    ] + ["beamformer_angle_%0.4f" % theta for theta in thetas]
