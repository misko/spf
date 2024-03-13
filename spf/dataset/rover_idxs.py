# v3

from functools import cache

import numpy as np


def v3rx_column_names(nthetas=65):
    thetas = np.linspace(-np.pi, np.pi, nthetas)
    return [
        "system_timestamp",
        "gps_timestamp_1",
        "gps_timestamp_2",
        "lat",
        "long",
        "heading",
        "rx_theta",
        "rx_spacing_m",
        "avg_phase_diff_1",
        "avg_phase_diff_2",
        "rssi0",
        "rssi1",
        "gain0",
        "gain1",
    ] + ["beamformer_angle_%0.4f" % theta for theta in thetas]


@cache
def v3rx_time_idx():
    return v3rx_column_names().index("gps_timestamp")


@cache
def v3rx_rx_pos_idxs():
    return [
        v3rx_column_names().index("lat"),
        v3rx_column_names().index("long"),
    ]


@cache
def v3rx_rssi_idxs():
    return [
        v3rx_column_names().index("rssi0"),
        v3rx_column_names().index("rssi0"),
    ]


@cache
def v3rx_gain_idxs():
    return [
        v3rx_column_names().index("gain0"),
        v3rx_column_names().index("gain1"),
    ]


@cache
def v3rx_avg_phase_diff_idxs():
    return [
        v3rx_column_names().index("avg_phase_diff_1"),
        v3rx_column_names().index("avg_phase_diff_2"),
    ]


@cache
def v3rx_rx_theta_idx():
    return [
        v3rx_column_names().index("rx_theta"),
    ]


@cache
def v3rx_beamformer_start_idx():
    for idx, column_name in enumerate(v3rx_column_names()):
        if "beamformer" in column_name:
            return idx
    return None
