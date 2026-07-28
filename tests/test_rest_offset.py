import numpy as np
import pytest

from spf.gps.boundaries import boundaries
from spf.mavlink.mavlink_controller import (
    drone_get_planner,
    meters_to_degrees,
    rest_offset_to_degrees,
)

EARTH_RADIUS_M = 6371008.8  # match haversine, as the code does
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


# --- integration: the config -> collector -> planner wiring -------------------
# The unit tests above call drone_get_planner() directly, so they would all pass
# even if the YAML key were misspelled or the collector never read it. These
# tests exercise the real production configs and the real key name.

import pathlib

import yaml as _yaml

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / (
    "data_collection/rover/rover_v3.1/capture_configs"
)
COLLECTOR = pathlib.Path(__file__).resolve().parents[1] / "spf/mavlink_radio_collection.py"
REST_OFFSET_KEY = "rest-offset-m"

EXPECTED_OFFSETS = {1: [1.0, 1.0], 2: [1.0, -1.0], 3: [-1.0, 1.0]}


def test_collector_reads_the_same_key_the_configs_write():
    """Guards against key-name drift between the YAML and the collector."""
    assert REST_OFFSET_KEY in COLLECTOR.read_text()


@pytest.mark.parametrize("rover_id", sorted(EXPECTED_OFFSETS))
def test_production_configs_declare_the_intended_offset(rover_id):
    config = _yaml.safe_load(
        (CONFIG_DIR / f"rover{rover_id}_production_v7.yaml").read_text()
    )
    assert config[REST_OFFSET_KEY] == EXPECTED_OFFSETS[rover_id]


@pytest.mark.parametrize("rover_id", sorted(EXPECTED_OFFSETS))
def test_production_config_drives_the_planner_end_to_end(rover_id):
    """Mirror what the collector does: read the YAML, build the planner."""
    config = _yaml.safe_load(
        (CONFIG_DIR / f"rover{rover_id}_production_v7.yaml").read_text()
    )
    boundary = boundaries["franklin_safe"]
    centroid = boundary.mean(axis=0)
    planner = drone_get_planner(
        config["routine"],
        boundary=boundary,
        rest_offset_m=config.get(REST_OFFSET_KEY),  # exactly the collector's expression
    )
    home = planner.get_home_point()
    assert not np.array_equal(home, centroid)
    assert _distance_m(home, centroid, centroid[1]) == pytest.approx(
        float(np.hypot(*EXPECTED_OFFSETS[rover_id])), abs=1e-6
    )


def test_the_three_production_rovers_rest_apart():
    boundary = boundaries["franklin_safe"]
    centroid = boundary.mean(axis=0)
    homes = {}
    for rover_id in EXPECTED_OFFSETS:
        config = _yaml.safe_load(
            (CONFIG_DIR / f"rover{rover_id}_production_v7.yaml").read_text()
        )
        homes[rover_id] = drone_get_planner(
            config["routine"], boundary=boundary, rest_offset_m=config.get(REST_OFFSET_KEY)
        ).get_home_point()
    for a in homes:
        for b in homes:
            if a < b:
                assert _distance_m(homes[a], homes[b], centroid[1]) >= 1.99


@pytest.mark.parametrize(
    "legacy_name",
    [
        "rover_receiver_config_pi_3mhz_35mm.yaml",
        "rover_receiver_config_pi_3mhz_43mm.yaml",
        "rover_single_receiver_config_pi_3mhz.yaml",
    ],
)
def test_configs_without_the_key_are_unaffected(legacy_name):
    """Legacy/bench configs must still resolve to the plain centroid.

    These carry `routine: null` — the routine came from the collector's -r flag
    in the legacy flow — so the routine is supplied here the same way.
    """
    boundary = boundaries["franklin_safe"]
    centroid = boundary.mean(axis=0)
    legacy = _yaml.safe_load((CONFIG_DIR / legacy_name).read_text())
    assert REST_OFFSET_KEY not in legacy
    planner = drone_get_planner(
        "bounce",  # legacy flow: routine comes from -r, not the YAML
        boundary=boundary,
        rest_offset_m=legacy.get(REST_OFFSET_KEY),
    )
    assert planner.home_point is None
    assert np.array_equal(planner.get_home_point(), centroid)


# --- direct equivalence: [0,0] must be indistinguishable from omitting the key ---
# The two tests above assert the same properties separately, which only IMPLIES
# equivalence. These compare the two planners against each other directly.

ZERO_FORMS = [
    [0.0, 0.0],
    (0.0, 0.0),
    [0, 0],
    (0.0, -0.0),  # IEEE negative zero
    np.zeros(2),
]


def _offset_relevant_state(planner, routine):
    """Every attribute a rest offset could possibly influence.

    Excludes CirclePlanner.direction and the diamond point order, which are
    randomised per construction and unrelated to the offset.
    """
    state = {
        "home_point": planner.home_point,
        "start_point": np.asarray(planner.start_point),
        "get_home_point": planner.get_home_point(),
    }
    if routine == "center":
        state["stationary_point"] = np.asarray(planner.stationary_point)
    if routine == "circle":
        state["circle_center"] = np.asarray(planner.circle_center)
        state["circle_radius"] = planner.circle_radius
    return state


@pytest.mark.parametrize("zero", ZERO_FORMS)
@pytest.mark.parametrize("routine", ROUTINES)
@pytest.mark.parametrize("boundary_name", sorted(boundaries))
def test_zero_offset_is_indistinguishable_from_omitting_the_key(
    zero, routine, boundary_name
):
    boundary = boundaries[boundary_name]
    omitted = drone_get_planner(routine, boundary)
    explicit_zero = drone_get_planner(routine, boundary, rest_offset_m=zero)

    a = _offset_relevant_state(omitted, routine)
    b = _offset_relevant_state(explicit_zero, routine)
    assert a.keys() == b.keys()
    for key in a:
        if a[key] is None or b[key] is None:
            assert a[key] is None and b[key] is None, key
        elif isinstance(a[key], np.ndarray):
            assert np.array_equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("zero", ZERO_FORMS)
def test_zero_offset_short_circuits_before_any_arithmetic(zero):
    """A zero offset must return None, not a zero-valued degree offset.

    Returning array([0.0, 0.0]) would still be numerically correct, but it would
    set home_point and take the non-default code path — this asserts the
    original path is preserved exactly.
    """
    assert rest_offset_to_degrees(zero, boundaries["franklin_safe"]) is None


# --- magnitude, measured with the SAME metric the rover uses ------------------
# The tests above re-implement the code's own flat-earth conversion, so they are
# partly circular. These assert the resulting displacement using `haversine` —
# the exact function Drone.distance_to_target uses to decide "arrived" — so an
# error in the conversion cannot hide behind a matching error in the test.

from haversine import Unit as _Unit
from haversine import haversine as _haversine

from spf.gps.gps_utils import swap_lat_long as _swap


def _haversine_m(a, b):
    """Distance in metres between two (long, lat) points, as the rover measures it."""
    return _haversine(_swap(a), _swap(b), unit=_Unit.METERS)


@pytest.mark.parametrize(
    "offset,expected_m",
    [
        ((2.0, 2.0), np.sqrt(8.0)),  # the case asked about
        ((1.0, 1.0), np.sqrt(2.0)),
        ((-2.0, -2.0), np.sqrt(8.0)),
        ((2.0, -2.0), np.sqrt(8.0)),
        ((3.0, 4.0), 5.0),  # 3-4-5, catches an axis swap
        ((0.0, 5.0), 5.0),  # pure north
        ((5.0, 0.0), 5.0),  # pure east
        ((10.0, 10.0), np.sqrt(200.0)),
    ],
)
@pytest.mark.parametrize("boundary_name", sorted(boundaries))
def test_offset_magnitude_by_haversine(offset, expected_m, boundary_name):
    boundary = boundaries[boundary_name]
    centroid = boundary.mean(axis=0)
    home = drone_get_planner("bounce", boundary, rest_offset_m=offset).get_home_point()
    assert _haversine_m(centroid, home) == pytest.approx(expected_m, rel=1e-4)


@pytest.mark.parametrize("boundary_name", sorted(boundaries))
def test_pure_east_and_north_land_on_their_own_axis(boundary_name):
    """A pure-East offset must not move latitude, and vice versa.

    This is the assertion that would fail if (long, lat) were ever swapped.
    """
    boundary = boundaries[boundary_name]
    centroid = boundary.mean(axis=0)

    east = drone_get_planner("bounce", boundary, rest_offset_m=(5.0, 0.0)).get_home_point()
    assert east[1] == centroid[1]  # latitude untouched
    assert east[0] > centroid[0]  # longitude increased (east is +long)
    assert _haversine_m(centroid, east) == pytest.approx(5.0, rel=1e-4)

    north = drone_get_planner("bounce", boundary, rest_offset_m=(0.0, 5.0)).get_home_point()
    assert north[0] == centroid[0]  # longitude untouched
    assert north[1] > centroid[1]  # latitude increased
    assert _haversine_m(centroid, north) == pytest.approx(5.0, rel=1e-4)


def test_equal_east_and_north_metres_give_UNEQUAL_degree_offsets():
    """The cos(latitude) correction is real: (2,2) m is not a square in degrees."""
    boundary = boundaries["franklin_safe"]
    centroid = boundary.mean(axis=0)
    home = drone_get_planner("bounce", boundary, rest_offset_m=(2.0, 2.0)).get_home_point()
    dlong = home[0] - centroid[0]
    dlat = home[1] - centroid[1]
    assert dlong > dlat  # a degree of longitude is shorter here
    assert dlong / dlat == pytest.approx(1.0 / np.cos(np.radians(centroid[1])), rel=1e-6)
    # ...yet both legs are 2 m on the ground
    assert _haversine_m(centroid, (home[0], centroid[1])) == pytest.approx(2.0, rel=1e-4)
    assert _haversine_m(centroid, (centroid[0], home[1])) == pytest.approx(2.0, rel=1e-4)
