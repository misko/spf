from functools import cache

import numpy as np


def v1_column_names(nthetas=65):
    thetas = np.linspace(-np.pi, np.pi, nthetas)
    return [
        "timestamp",
        "tx_pos_x_mm",
        "tx_pos_y_mm",
        "avg_phase_diff_1",
        "avg_phase_diff_2",
    ] + ["beamformer_angle_%0.4f" % theta for theta in thetas]


@cache
def v1_time_idx():
    return v1_column_names().index("timestamp")


@cache
def v1_tx_pos_idxs():
    return [
        v1_column_names().index("tx_pos_x_mm"),
        v1_column_names().index("tx_pos_y_mm"),
    ]


@cache
def v1_beamformer_start_idx():
    for idx, column_name in enumerate(v1_column_names()):
        if "beamformer" in column_name:
            return idx
    return None
