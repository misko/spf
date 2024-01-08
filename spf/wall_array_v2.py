from functools import cache

import numpy as np


def v2_column_names(nthetas=65):
    thetas = np.linspace(-np.pi, np.pi, nthetas)
    return [
        "timestamp",
        "tx_pos_x_mm",
        "tx_pos_y_mm",
        "rx_pos_x_mm",
        "rx_pos_y_mm",
        "rx_theta",
        "rx_spacing_m",
        "avg_phase_diff_1",
        "avg_phase_diff_2",
    ] + ["beamformer_angle_%0.4f" % theta for theta in thetas]


@cache
def v2_time_idx():
    return v2_column_names().index("timestamp")


@cache
def v2_rx_pos_idxs():
    return [
        v2_column_names().index("rx_pos_x_mm"),
        v2_column_names().index("rx_pos_y_mm"),
    ]


@cache
def v2_avg_phase_diff_idxs():
    return [
        v2_column_names().index("avg_phase_diff_1"),
        v2_column_names().index("avg_phase_diff_2"),
    ]


@cache
def v2_rx_theta_idx():
    return [
        v2_column_names().index("rx_theta"),
    ]


@cache
def v2_tx_pos_idxs():
    return [
        v2_column_names().index("tx_pos_x_mm"),
        v2_column_names().index("tx_pos_y_mm"),
    ]


@cache
def v2_beamformer_start_idx():
    for idx, column_name in enumerate(v2_column_names()):
        if "beamformer" in column_name:
            return idx
    return None
