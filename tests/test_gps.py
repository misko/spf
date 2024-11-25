import numpy as np
from haversine import inverse_haversine

from spf.dataset.v4_tx_rx_to_v5 import lat_lon_to_xy
from spf.gps.boundaries import franklin_safe
from spf.gps.gps_utils import calc_bearing, swap_lat_long
from spf.rf import pi_norm


# create a circle in GPS long/lat , then calculate the bearing
# and confirm its as expected (0north, + to east, - to west)
def test_gps_circle():
    N = 256
    orbits = 2
    start_point = franklin_safe.mean(axis=0)

    gt_theta = np.linspace(0, orbits * 2 * np.pi, N)
    long_lat_circle = np.array(
        [
            swap_lat_long(inverse_haversine(swap_lat_long(start_point), 0.05, dir))
            for dir in gt_theta
        ]
    )

    for idx in range(N):
        a = np.deg2rad(calc_bearing(start_point.reshape(1, -1), long_lat_circle[[idx]]))
        b = gt_theta[idx]
        assert np.isclose(pi_norm(a - b), 0, atol=0.01)


def test_gps_to_xy():
    gps_center_long, gps_center_lat = franklin_safe.mean(axis=0)
    eps = 0.0001

    tol = 1e-4

    # exact center
    x, y = lat_lon_to_xy(
        lat=gps_center_lat,
        lon=gps_center_long,
        center_lat=gps_center_lat,
        center_lon=gps_center_long,
    )
    assert abs(x) < tol and abs(y) < tol

    # move right
    x, y = lat_lon_to_xy(
        lat=gps_center_lat,
        lon=gps_center_long + eps,
        center_lat=gps_center_lat,
        center_lon=gps_center_long,
    )
    assert x > 8 and abs(y) < tol

    # move left
    x, y = lat_lon_to_xy(
        lat=gps_center_lat,
        lon=gps_center_long - eps,
        center_lat=gps_center_lat,
        center_lon=gps_center_long,
    )
    assert x < -8 and abs(y) < tol

    # move up
    x, y = lat_lon_to_xy(
        lat=gps_center_lat + eps,
        lon=gps_center_long,
        center_lat=gps_center_lat,
        center_lon=gps_center_long,
    )
    assert abs(x) < tol and y > 8

    # move down
    x, y = lat_lon_to_xy(
        lat=gps_center_lat - eps,
        lon=gps_center_long,
        center_lat=gps_center_lat,
        center_lon=gps_center_long,
    )
    assert abs(x) < tol and y < -8
