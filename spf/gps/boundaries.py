import numpy as np
from haversine import Unit, haversine

from spf.gps.gps_utils import swap_lat_long

#  long, long
# san francisco ~ 37 lat, -122 long

crissy_boundary_convex = np.array(
    [
        (-122.4612172, 37.8031271),  # 1
        (-122.4606378, 37.8041697),  # 2
        (-122.4665438, 37.8046402),  # 3
        (-122.4671598, 37.8038337),  # 4
    ]
)

crissy_boundary = np.array(
    [
        (-122.4611099, 37.8030423),
        (-122.4602301, 37.8038306),
        (-122.4610348, 37.804619),
        (-122.4635024, 37.8045681),
        (-122.4659164, 37.8047122),
        (-122.4674828, 37.8052887),
        (-122.4680933, 37.8046305),
        (-122.4673315, 37.8037744),
        # (-122.4647659, 37.8025219),
        # (-122.4640363, 37.8025049),
        (-122.4640363, 37.8033103),
        # (-122.46399, 37.80177),
    ]
)

franklin_safe = np.array(
    [
        (-122.4098606, 37.7652836),
        (-122.409815, 37.7648647),
        (-122.40895, 37.7649104),
        (-122.4089795, 37.7653207),
    ]
)

franklin_boundary = np.array(
    [
        (-122.4099585, 37.7653281),
        (-122.409929, 37.7648128),
        (-122.4088186, 37.7648722),
        (-122.4088588, 37.7654108),
    ]
)


def boundary_to_diamond(boundary):
    return np.array(
        [
            (boundary[0] + boundary[3]) / 2,
            (boundary[0] + boundary[1]) / 2,
            (boundary[1] + boundary[2]) / 2,
            (boundary[2] + boundary[3]) / 2,
        ]
    )


fort_baker_boundary = np.array(
    [
        (-122.47849870754949, 37.8346129400644),
        (-122.47781031352912, 37.83575589657535),
        (-122.4785852164001, 37.83604915163242),
        (-122.47933261070422, 37.83508052068589),
    ]
)


boundaries = {
    "franklin_safe": franklin_safe,
    "fort_baker_boundary": fort_baker_boundary,
}


# gps = (long,lat) as input to this function
def find_closest_boundary_with_distance(gps):
    try:
        distances = sorted(
            [
                (
                    haversine(
                        swap_lat_long(gps),
                        swap_lat_long(boundary_points.mean(axis=0)),
                        unit=Unit.METERS,
                    ),
                    boundary_name,
                )
                for boundary_name, boundary_points in boundaries.items()
            ]
        )
        return distances[0]
    except ValueError:
        return np.inf, None


def find_closest_boundary(gps, cutoff=np.inf):
    distance, boundary = find_closest_boundary_with_distance(gps)
    if distance > cutoff:
        return None
    return boundary
