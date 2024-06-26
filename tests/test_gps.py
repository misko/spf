from spf.gps.gps_utils import calc_bearing, swap_lat_long
from spf.rf import pi_norm
import numpy as np
from spf.gps.boundaries import franklin_safe
from haversine import inverse_haversine


# create a circle in GPS long/lat , then calculate the bearing
# and confirm its as expected (0north, + to east, - to west)
def test_gps_circle():
    N = 256
    orbits = 2
    start_point = franklin_safe.mean(axis=0)

    gt_theta = np.linspace(0, orbits * 2 * np.pi, N)
    long_lat_circle = [
        swap_lat_long(inverse_haversine(swap_lat_long(start_point), 0.05, dir))
        for dir in gt_theta
    ]

    for idx in range(N):
        a = np.deg2rad(calc_bearing(start_point, long_lat_circle[idx]))
        b = gt_theta[idx]
        assert np.isclose(pi_norm(a - b), 0, atol=0.01)
