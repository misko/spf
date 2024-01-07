from functools import cache

import numpy as np


def v2_column_names(nthetas=65):
    thetas = np.linspace(-np.pi, np.pi, nthetas)
    return [
        "timestamp",
        "tx_pos_x",
        "tx_pos_y",
        "rx_pos_x",
        "rx_pos_y",
        "rx_theta",
        "rx_spacing",
        "avg_phase_diff_1",
        "avg_phase_diff_2",
    ] + ["beamformer_angle_%0.4f" % theta for theta in thetas]


@cache
def v2_rx_pos_idxs():
    return [
        v2_column_names().index("rx_pos_x"),
        v2_column_names().index("rx_pos_y"),
    ]


@cache
def v2_rx_theta_idxs():
    return [
        v2_column_names().index("rx_theta"),
    ]


@cache
def v2_tx_pos_idxs():
    return [
        v2_column_names().index("tx_pos_x"),
        v2_column_names().index("tx_pos_y"),
    ]
