import numpy as np
import pytest

from spf.gps.boundaries import boundaries
from spf.mavlink.mavlink_controller import (
    drone_get_planner,
    meters_to_degrees,
    rest_offset_to_degrees,
)

EARTH_RADIUS_M = 6378137.0
ROUTINES = ("bounce", "circle", "center", "diamond")


def _distance_m(a, b, latitude_deg):
    """Local ENU distance in metres between two (long, lat) points."""
    m_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    m_per_deg_long = m_per_deg_lat * np.cos(np.radians(latitude_deg))
    return float(
        np.hypot((a[0] - b[0]) * m_per_deg_long, (a[1] - b[1]) * m_per_deg_lat)
    )


@pytest.mark.parametrize("routine", ROUTINES)
@pytest.mark.parametrize("boundary_name", sorted(boundaries))
def test_no_offset_is_bit_identical_to_centroid(routine, boundary_name):
    """An unconfigured rover must behave exactly as it did before rest offsets."""
    boundary = boundaries[boundary_name]
    centroid = boundary.mean(axis=0)
    planner = drone_get_planner(routine, boundary)
    assert planner.home_point is None
    assert np.array_equal(planner.get_home_point(), centroid)
    assert np.array_equal(planner.start_point, centroid)


@pytest.mark.parametrize("routine", ROUTINES)
def test_zero_offset_is_treated_as_unset(routine):
    boundary = boundaries["franklin_safe"]
    planner = drone_get_planner(routine, boundary, rest_offset_m=[0.0, 0.0])
    assert planner.home_point is None
    assert np.array_equal(planner.get_home_point(), boundary.mean(axis=0))


@pytest.mark.parametrize(
    "offset", [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
)
@pytest.mark.parametrize("boundary_name", sorted(boundaries))
def test_offset_lands_at_the_requested_metres(offset, boundary_name):
    """The (east, north) metre request must survive the degree conversion."""
    boundary = boundaries[boundary_name]
    centroid = boundary.mean(axis=0)
    planner = drone_get_planner("bounce", boundary, rest_offset_m=offset)
    home = planner.get_home_point()

    m_per_deg_lat = (np.pi / 180.0) * EARTH_RADIUS_M
    m_per_deg_long = m_per_deg_lat * np.cos(np.radians(centroid[1]))
    east_m = (home[0] - centroid[0]) * m_per_deg_long
    north_m = (home[1] - centroid[1]) * m_per_deg_lat

    assert east_m == pytest.approx(offset[0], abs=1e-6)
    assert north_m == pytest.approx(offset[1], abs=1e-6)


def test_longitude_is_scaled_by_cos_latitude():
    """East and North are NOT interchangeable in degrees (~26% at these sites)."""
    dlong, dlat = meters_to_degrees(1.0, 1.0, 37.8)
    assert dlong > dlat
    assert dlong / dlat == pytest.approx(1.0 / np.cos(np.radians(37.8)), rel=1e-9)


def test_the_four_rover_offsets_are_mutually_separated():
    boundary = boundaries["franklin_safe"]
    centroid = boundary.mean(axis=0)
    spec = {1: (1.0, 1.0), 2: (1.0, -1.0), 3: (-1.0, 1.0), 4: (-1.0, -1.0)}
    homes = {
        rover: drone_get_planner("bounce", boundary, rest_offset_m=off).get_home_point()
        for rover, off in spec.items()
    }
    for a in spec:
        for b in spec:
            if a < b:
                assert _distance_m(homes[a], homes[b], centroid[1]) >= 1.99


@pytest.mark.parametrize("offset", [(1.0, -1.0), (-5.0, 5.0)])
def test_circle_pattern_geometry_is_never_offset(offset):
    """circle_center must stay fence-centred — CirclePlanner has no bounds check,
    so shifting the ring would silently drive outside the geofence."""
    boundary = boundaries["franklin_safe"]
    plain = drone_get_planner("circle", boundary)
    shifted = drone_get_planner("circle", boundary, rest_offset_m=offset)
    assert np.array_equal(
        np.asarray(plain.circle_center), np.asarray(shifted.circle_center)
    )
    assert plain.circle_radius == shifted.circle_radius
    # ...but the resting position did move
    assert not np.array_equal(shifted.get_home_point(), boundary.mean(axis=0))


def test_center_routine_parks_at_the_offset_point():
    boundary = boundaries["franklin_safe"]
    planner = drone_get_planner("center", boundary, rest_offset_m=(-1.0, -1.0))
    assert np.array_equal(planner.stationary_point, planner.get_home_point())


@pytest.mark.parametrize(
    "bad", [[1.0, 2.0, 3.0], [1.0], float("nan"), [float("nan"), 1.0], [1.0, float("inf")]]
)
def test_malformed_offsets_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        rest_offset_to_degrees(bad, boundaries["franklin_safe"])
