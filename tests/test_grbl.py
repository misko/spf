import os
import subprocess
import sys
import tempfile
import time

import matplotlib.path as pltpath
import numpy as np
from shapely import geometry

import spf
from spf.grbl.grbl_interactive import (
    GRBL_STEP_SIZE,
    BouncePlanner,
    GRBLDynamics,
    home_bounding_box,
    home_calibration_point,
    home_pA,
    home_pB,
)
from spf.grbl_radio_collection import grbl_radio_main, grbl_radio_parser

root_dir = os.path.dirname(os.path.dirname(spf.__file__))


def get_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    return env


def test_steps_and_steps_inverse():
    dynamics = GRBLDynamics(
        calibration_point=np.array([0, 0]),
        pA=home_pA,
        pB=home_pB,
        bounding_box=[],
    )

    n = 10
    for x in np.linspace(0, 2000, n + 1):
        for y in np.linspace(0, 4000, n + 1):
            p = np.array([x, y])
            steps = dynamics.to_steps(np.array(p))
            back = dynamics.from_steps(steps)
            assert np.isclose(p, back).all()


def test_xaxis():
    dynamics = GRBLDynamics(
        calibration_point=np.array([0, 0]),
        pA=home_pA,
        pB=home_pB,
        bounding_box=[],
    )

    n = 10
    y = 0
    for x in np.linspace(0, 2000, n + 1):
        p = np.array([x, y])
        steps = dynamics.to_steps(np.array(p))
        back = dynamics.from_steps(steps)
        print("p", p, "steps", steps, "back", back)
        if x > 0:
            assert (
                steps[0] == -steps[1]
            )  # if we are sliding across the x axis we should have a constant length ,
            # so decrease in one is increase in the other
        else:
            assert steps[0] == steps[1]  # not physically possible
        assert steps[0] == -x
        assert np.isclose(p, back).all()


def check_in_range(x, a, b):
    return x >= a and x <= b


def test_polygon_contains():
    n = 5000
    stretch = 3
    points = (np.random.rand(n, 2) - 0.5) * stretch

    cube_points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    plt_cube_path = pltpath.Path(cube_points)
    plt_start_time = time.time()
    plt_inside = plt_cube_path.contains_points(points)
    plt_end_time = time.time()

    shapely_polygon = geometry.Polygon(cube_points)
    shapely_start_time = time.time()
    shapely_inside = np.array(
        [shapely_polygon.contains(geometry.Point(point)) for point in points]
    )
    shapely_end_time = time.time()

    print(plt_end_time - plt_start_time, shapely_end_time - shapely_start_time)

    # check classic way
    inside_classic = [
        check_in_range(x[0], 0, 1) and check_in_range(x[1], 0, 1) for x in points
    ]

    assert (plt_inside == inside_classic).all()
    assert (shapely_inside == inside_classic).all()


def test_binary_search_edge():
    dynamics = GRBLDynamics(
        calibration_point=home_calibration_point,
        pA=home_pA,
        pB=home_pB,
        bounding_box=home_bounding_box,
    )
    planner = BouncePlanner(dynamics, start_point=[0, 0], step_size=GRBL_STEP_SIZE)

    direction = np.array([3, 1])
    p = np.array([1500, 900])

    length = dynamics.binary_search_edge(
        left=0, right=10000, xy=p, direction=direction, epsilon=0.001
    )
    last_point = length * direction + p

    dynamics.get_boundary_vector_near_point(last_point)
    lp, nd = planner.get_bounce_pos_and_new_direction(p, direction)
    lp, nd = planner.get_bounce_pos_and_new_direction(lp, nd)
    planner.single_bounce(direction, p)
    [x for x in planner.bounce(p, 10)]


def test_grbl_simple_move(script_runner):
    subprocess.check_output(
        f"cat {root_dir}/tests/grbl_test_simple_move | python3 {root_dir}/spf/grbl/grbl_interactive.py none",
        timeout=180,
        shell=True,
        env=get_env(),
        stderr=subprocess.STDOUT,
    ).decode()


def test_grbl_bounce(script_runner):
    subprocess.check_output(
        f"cat {root_dir}/tests/grbl_test_bounce | python3 {root_dir}/spf/grbl/grbl_interactive.py none",
        timeout=180,
        shell=True,
        env=get_env(),
        stderr=subprocess.STDOUT,
    ).decode()


def test_grbl_radio_collection_bounce(script_runner):
    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.check_output(
            f"python3 {root_dir}/spf/grbl_radio_collection.py -c {root_dir}/tests/wall_array_v2_external_test.yaml"
            + f" -r bounce -o {tmpdirname}",
            timeout=180,
            shell=True,
            env=get_env(),
            stderr=subprocess.STDOUT,
        ).decode()


def test_grbl_radio_collection_bounce_internal():
    parser = grbl_radio_parser()
    with tempfile.TemporaryDirectory() as tmpdirname:
        args = parser.parse_args(
            args=[
                "-c",
                f"{root_dir}/tests/wall_array_v2_external_test_secondspersample.yaml",
                "-r",
                "bounce",
                "-o",
                f"{tmpdirname}",
            ]
        )
        grbl_radio_main(args)
