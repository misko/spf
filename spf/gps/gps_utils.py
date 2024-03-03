import numpy as np

# https://en.wikipedia.org/wiki/Haversine_formula


def swap_lat_long(x):
    return (x[1], x[0])


def degnorm(x):
    return ((x + 180.0) % 360.0) - 180.0


def pinorm(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


# https://mapscaping.com/how-to-calculate-bearing-between-two-coordinates/
# p0 = [ long , lat ] # in radians
# p1 = [ long , lat ] # in radians
# return degrees from north that angle from p0->p1 makes
# dead north is 0deg, to the right is + and to the left is -
def calc_bearing(p0, p1):
    # Calculate the bearing
    p0_rad = np.deg2rad(p0)
    p1_rad = np.deg2rad(p1)
    bearing_rad = np.arctan2(
        np.sin(p1_rad[0] - p0_rad[0]) * np.cos(p1_rad[1]),
        np.cos(p0_rad[1]) * np.sin(p1_rad[1])
        - np.sin(p0_rad[1]) * np.cos(p1_rad[1]) * np.cos(p1_rad[0] - p0_rad[0]),
    )

    return np.rad2deg(pinorm(bearing_rad))  # np.rad2deg(bearing_rad)


"""
Craft is facing p0 from p1 and would like to go to p2
"""


def calc_relative_bearing(p_facing, p_current, p_desired):
    return (
        degnorm(calc_bearing(p_current, p_desired) - calc_bearing(p_current, p_facing)),
        calc_bearing(p_current, p_facing),
        calc_bearing(p_current, p_desired),
    )
