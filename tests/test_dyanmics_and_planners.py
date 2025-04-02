import numpy as np

from spf.gps.boundaries import franklin_boundary
from spf.grbl.grbl_interactive import (
    GRBL_STEP_SIZE,
    BouncePlanner,
    Dynamics,
    home_bounding_box,
)


def test_grbl_bounce():
    # GRBL

    boundary = home_bounding_box
    bp = BouncePlanner(
        dynamics=Dynamics(bounding_box=boundary),
        start_point=boundary.mean(axis=0),
        step_size=GRBL_STEP_SIZE,
    )
    n = 20000
    p = bp.yield_points()
    z = np.array([next(p) for x in range(n)])

    assert (np.unique(z).size / n) > 0.9


def test_gps_bounce():
    # GPS

    boundary = franklin_boundary
    boundary -= franklin_boundary.mean(axis=0)
    bp = BouncePlanner(
        dynamics=Dynamics(
            bounding_box=boundary,
            bounds_radius=0.000000001,
        ),
        start_point=boundary.mean(axis=0),
        epsilon=0.0000001,
        step_size=0.00001,
    )

    n = 20000
    p = bp.yield_points()
    z = np.array([next(p) for x in range(n)])

    assert (np.unique(z).size / n) > 0.9
