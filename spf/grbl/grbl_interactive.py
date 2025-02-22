import argparse
import logging
import queue
import random
import select
import sys
import threading
import time
from abc import ABC, abstractmethod

import matplotlib.path as pltpath
import numpy as np
import serial
from scipy.spatial import ConvexHull

home_pA = np.array([3568, 0])
home_pB = np.array([0, 0])
home_bounding_box = np.array(
    [
        [500, 400],
        [3000, 450],
        [3000, 2700],
        [1900, 2700],
        [900, 1900],
        [500, 500],
    ]
)
# rx_calibration_point = np.array([1930, 2770])
rx_calibration_point = np.array([1930, 2600])
tx_calibration_point = np.array([550, 450])
circle_center = np.array([2000, 1500])
max_circle_diameter = 1900
min_circle_diameter = 900

run_grbl = True


def chord_length_to_angle(chord_length, radius):
    return 2 * np.arcsin(chord_length / (2 * radius))


def stop_grbl():
    logging.info("STOP GRBL")
    global run_grbl
    run_grbl = False


GRBL_STEP_SIZE = 8

"""
MotorMountA                         MotorMountB
        .                           .
            .                   .
                .           .
                    .   .
                    Origin
                 (Offset from MotorMountB)


We cant go higher than origin, because there wouldn't
be any grip for the GT2 cable, we need to maintain some
minimal angle

There are two reference frames,
    A) One where motor mount B is the center
    B) One where origin is the center (mountB+offset)

The second frame is the one we can calibrate too, since its
impossible to get the payload to MotorMountB , because there
would be no tension on the GT2 belts

"""
home_calibration_point = np.array([500, 400])


def a_to_b_in_stepsize_np(a, b, step_size):
    if np.isclose(a, b).all():
        return [b]
    # move by step_size from where we are now to the target position
    distance = np.linalg.norm(b - a)
    steps = np.arange(1, np.ceil(distance / step_size) + 1) * step_size

    direction = (b - a) / np.linalg.norm(b - a)
    points = a.reshape(1, 2) + direction.reshape(1, 2) * steps.reshape(len(steps), 1)
    points[-1] = b
    return points


def a_to_b_in_stepsize(a, b, step_size):
    return a_to_b_in_stepsize_np(a, b, step_size)
    if np.isclose(a, b).all():
        return [b]
    # move by step_size from where we are now to the target position
    points = []
    direction = (b - a) / np.linalg.norm(b - a)
    distance = np.linalg.norm(b - a)
    _l = step_size
    while _l < distance:
        points.append(_l * direction + a)
        _l += step_size
    points.append(b)
    return points


class BoundingBoxIsNonConvexException(Exception):
    pass


class PointOutOfBoundsException(Exception):
    pass


class Dynamics:
    def __init__(self, bounding_box, unsafe=False, bounds_radius=0.00001):
        self.bounds_radius = bounds_radius
        self.unsafe = unsafe
        self.bounding_box = bounding_box
        if len(bounding_box) >= 3:
            hull = ConvexHull(bounding_box)
            if len(np.unique(hull.simplices)) != len(bounding_box):
                logging.error(
                    "Points do not form a simple hull, most likely non convex"
                )
                logging.error(
                    "Points in the hull are, "
                    + ",".join(
                        map(str, [bounding_box[x] for x in np.unique(hull.simplices)])
                    )
                )
                raise BoundingBoxIsNonConvexException()
            self.polygon = pltpath.Path(bounding_box)
        else:
            self.polygon = None

    def to_steps(self, p):
        if (
            (not self.unsafe)
            and (self.polygon is not None)
            and not self.polygon.contains_point(p, radius=self.bounds_radius)
        ):  # todo a bit hacky but works
            raise PointOutOfBoundsException(
                "Point we want to move to will be out of bounds"
            )

    def from_steps(self, state):
        return state

    def binary_search_edge(self, left, right, xy, direction, epsilon):
        if (right - left) < epsilon:
            return left
        midpoint = (right + left) / 2
        p = midpoint * direction + xy
        try:
            steps = self.to_steps(p)  # noqa
            # actual = self.from_steps(*steps)
            return self.binary_search_edge(midpoint, right, xy, direction, epsilon)
        except PointOutOfBoundsException:
            return self.binary_search_edge(left, midpoint, xy, direction, epsilon)

    def distance_from_point_to_segment(self, p, v0, v1):
        # rint("DISTANCE SEGMENT", p, v0, v1)
        # center to v0
        v1 = v1 - v0
        p = p - v0
        # lets get v1 -> y axis
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = np.array([-v1_norm[1], v1_norm[0]])

        inv_m = np.vstack([v1_norm, v2_norm]).T
        m = np.linalg.inv(inv_m)

        _v1 = m @ v1
        _p = m @ p
        assert np.isclose(_v1[1], 0)
        if _p[0] > _v1[0]:
            return np.linalg.norm(_p - _v1)
        elif _p[0] < 0:
            return np.linalg.norm(_p)
        return np.abs(_p[1])

    def get_boundary_vector_near_point(self, p):
        if self.polygon is None:
            raise ValueError

        bvec = None
        max_score = np.inf
        nverts = len(self.polygon.vertices)
        for i in range(nverts):
            v0 = self.polygon.vertices[i % nverts]
            v1 = self.polygon.vertices[(i + 1) % nverts]

            score = max(
                np.dot(p - v0, v1 - v0)
                / (np.linalg.norm(v0 - v1) * np.linalg.norm(p - v0) + 0.00001),
                np.dot(p - v1, v1 - v1)
                / (np.linalg.norm(v0 - v1) * np.linalg.norm(p - v1) + 0.00001),
            )
            score = self.distance_from_point_to_segment(p, v0, v1)
            if score < max_score:
                max_score = score
                bvec = (v1 - v0) / np.linalg.norm(v1 - v0)
        return bvec


class GRBLDynamics(Dynamics):
    def __init__(self, calibration_point, pA, pB, bounding_box, unsafe=False):
        super().__init__(bounding_box=bounding_box, unsafe=unsafe)
        self.calibration_point = calibration_point
        self.pA = pA
        self.pB = pB

    # a is the left motor on the wall
    # b is the right motor on the wall
    # (a-b)_x > 0
    # x is positive from b towards a
    # y is positive up to ceiling and negative down to floor

    def from_steps(self, state):
        a_motor_steps, b_motor_steps = state
        """
        The motor steps are relative to calibration point
        (0,0) motor steps means we are at the calibration point and have
        r1 /r2 corresponding to calibration point lengths
        """
        r1 = np.linalg.norm(self.pA - self.calibration_point) + a_motor_steps
        r2 = np.linalg.norm(self.pB - self.calibration_point) + b_motor_steps
        d = np.linalg.norm(self.pA - self.pB)

        """
        x^2 = y^2 + r_2^2
        (x-d)^2 = x^2 - 2xd + d^2 = y^2 + r_1^2

        x = (r_2^2 - d^2 - r_1^2 ) / (2*d)

        """
        x = (r2 * r2 - d * d - r1 * r1) / (2 * d)  # on different axis
        x_dir = (self.pA - self.pB) / d  # unit vector from pB to pA

        y = np.sqrt(r1 * r1 - x * x)
        y_dir = np.array([-x_dir[1], x_dir[0]])  # x_dir.T # orthogonal to x

        position_relative_to_calibration_point = np.round(
            self.pA + y_dir * y + x_dir * x, 1
        )
        return position_relative_to_calibration_point

    def to_steps(self, p):
        if (
            (not self.unsafe)
            and (self.polygon is not None)
            and not self.polygon.contains_point(p, radius=0.00001)
        ):  # todo a bit hacky but works
            raise PointOutOfBoundsException(
                "Point we want to move to will be out of bounds"
            )
        # motor_steps = ( distance_between_pivot and point ) - (distance between pivot and calibration point)
        a_motor_steps = np.linalg.norm(self.pA - p) - np.linalg.norm(
            self.pA - self.calibration_point
        )
        b_motor_steps = np.linalg.norm(self.pB - p) - np.linalg.norm(
            self.pB - self.calibration_point
        )
        # check the point again
        new_point = self.from_steps([a_motor_steps, b_motor_steps])
        if (
            (not self.unsafe)
            and (self.polygon is not None)
            and not self.polygon.contains_point(new_point, radius=0.00001)
        ):  # todo a bit hacky but works
            raise PointOutOfBoundsException("Inverted point is out of bounds")

        return a_motor_steps, b_motor_steps


class Planner(ABC):
    def __init__(
        self, dynamics, start_point, step_size=GRBL_STEP_SIZE, epsilon=1, seed=None
    ):
        self.dynamics = dynamics
        self.current_direction = None
        self.epsilon = epsilon  # original was 0.001
        self.start_point = start_point
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

    def get_bounce_pos_and_new_direction(self, p, direction):
        distance_to_bounce = self.dynamics.binary_search_edge(
            0, 10000, p, direction, self.epsilon
        )
        last_point_before_bounce = distance_to_bounce * direction + p

        # parallel component stays the same
        # negatate the perpendicular component
        bvec = self.dynamics.get_boundary_vector_near_point(last_point_before_bounce)
        bvec_perp = np.array([bvec[1], -bvec[0]])
        new_direction = (
            np.dot(direction, bvec) * bvec - np.dot(direction, bvec_perp) * bvec_perp
        )
        new_direction /= np.linalg.norm(new_direction)
        return last_point_before_bounce, new_direction

    def random_direction(self):
        theta = self.rng.uniform(0, 2 * np.pi)
        return np.array([np.sin(theta), np.cos(theta)])

    @abstractmethod
    def yield_points(self):
        pass

    def get_planner_start_position(self):
        return None


class BouncePlanner(Planner):
    def single_bounce(self, direction, p):
        bounce_point, new_direction = self.get_bounce_pos_and_new_direction(
            p, direction
        )

        to_points = a_to_b_in_stepsize(p, bounce_point, step_size=self.step_size)

        # add some noise to the new direction
        theta = self.rng.uniform(0, 2 * np.pi)
        percent_random = 0.05
        new_direction = (
            1 - percent_random
        ) * new_direction + percent_random * np.array([np.sin(theta), np.cos(theta)])
        return to_points, new_direction

    def yield_points(self):
        yield from self.bounce(self.start_point)

    def bounce(self, current_p, total_bounces=-1):
        global run_grbl
        # if no previous direciton lets initialize one
        if self.current_direction is None:
            self.current_direction = self.random_direction()
        # add some random noise
        percent_random = 0.05
        self.current_direction = (
            1 - percent_random
        ) * self.current_direction + percent_random * self.random_direction()
        self.current_direction /= np.linalg.norm(self.current_direction)

        n_bounce = 0
        stall = 0
        last_point = None
        while total_bounces < 0 or n_bounce < total_bounces:
            print("BOUNCING")
            if not run_grbl:
                logging.info("Exiting bounce early")
                break
            to_points, new_direction = self.single_bounce(
                self.current_direction, current_p
            )

            logging.info(
                f"{str(self)},{str(to_points[0])},{str(to_points[-1])},{str(new_direction)},{len(to_points)}"
            )
            assert len(to_points) > 0
            yield from to_points
            current_p = to_points[-1]
            if len(to_points) == 1:
                if last_point is None:
                    last_point = to_points[0]
                elif (last_point == to_points[0]).all():
                    stall += 1
                    if stall > 3:
                        new_direction = self.random_direction()
                        logging.info("Stalled and picking random direction")
                else:
                    last_point = None
                    stall = 0
            else:
                last_point = None
                stall = 0
            # else:
            self.current_direction = new_direction
            n_bounce += 1
        logging.info("Exiting bounce")


class StationaryPlanner(Planner):
    def __init__(
        self, dynamics, start_point, stationary_point, step_size=GRBL_STEP_SIZE
    ):
        super().__init__(
            dynamics=dynamics, start_point=start_point, step_size=step_size
        )
        self.stationary_point = stationary_point

    def yield_points(
        self,
    ):
        # move to the stationary point
        yield from a_to_b_in_stepsize(
            self.start_point, self.stationary_point, step_size=self.step_size
        )
        # stay still
        while True:
            yield self.stationary_point

    def get_planner_start_position(self):
        return self.stationary_point


class PointCycle(Planner):
    def __init__(
        self,
        dynamics,
        start_point,
        points,
        step_size=GRBL_STEP_SIZE,
    ):
        super().__init__(
            dynamics=dynamics, start_point=start_point, step_size=step_size
        )
        self.points = points

    def get_planner_start_position(self):
        return self.points[0]

    def yield_points(self):
        # start at the top of the circle
        current_p = self.start_point

        # start at the top
        while True:
            for point in self.points:
                next_p = point
                yield from a_to_b_in_stepsize(
                    current_p,
                    next_p,
                    step_size=self.step_size,
                )
                current_p = next_p


class CirclePlanner(Planner):
    def __init__(
        self,
        dynamics,
        start_point,
        step_size=GRBL_STEP_SIZE,
        circle_diameter=max_circle_diameter,
        circle_center=[0, 0],
    ):
        super().__init__(
            dynamics=dynamics, start_point=start_point, step_size=step_size
        )
        self.direction = ((np.random.rand() > 0.5) - 0.5) * 2
        self.circle_center = circle_center
        self.circle_radius = circle_diameter / 2
        self.angle_increment = chord_length_to_angle(self.step_size, self.circle_radius)

    def get_planner_start_position(self):
        return self.angle_to_pos(0)

    def angle_to_pos(self, angle):
        return self.circle_center + self.circle_radius * np.array(
            [-np.sin(angle), np.cos(angle)]
        )

    def yield_points(self):
        # start at the top of the circle
        current_p = self.start_point

        # start at the top
        current_angle = 0
        while current_angle < 360:
            next_p = self.angle_to_pos(current_angle)
            yield from a_to_b_in_stepsize(
                current_p,
                next_p,
                step_size=self.step_size,
            )
            current_p = next_p
            current_angle += self.direction * self.angle_increment


class CalibrationV1Planner(Planner):
    def __init__(self, dynamics, start_point, step_size=GRBL_STEP_SIZE, y_bump=150):
        super().__init__(
            dynamics=dynamics, start_point=start_point, step_size=step_size
        )
        self.y_bump = y_bump

    def get_planner_start_position(self):
        return tx_calibration_point

    def yield_points(self):
        start_p = np.array(tx_calibration_point)
        max_y = np.max([x[1] for x in home_bounding_box])
        direction_left = np.array([1, 0])  # ride the x dont change y
        direction_right = np.array([-1, 0])  # ride the x dont change y

        current_p = self.start_point

        while True:
            # get to the starting position
            yield from a_to_b_in_stepsize(current_p, start_p, step_size=self.step_size)
            current_p = start_p

            while current_p[1] + self.y_bump < max_y:
                # move to far wall
                next_p, _ = self.get_bounce_pos_and_new_direction(
                    current_p, direction_left
                )
                yield from a_to_b_in_stepsize(
                    current_p, next_p, step_size=self.step_size
                )
                current_p = next_p

                # move a tiny bit down
                next_p = current_p + np.array([0, self.y_bump])
                yield from a_to_b_in_stepsize(
                    current_p, next_p, step_size=self.step_size
                )
                current_p = next_p

                # move back
                next_p, _ = self.get_bounce_pos_and_new_direction(
                    current_p, direction_right
                )
                yield from a_to_b_in_stepsize(
                    current_p, next_p, step_size=self.step_size
                )
                current_p = next_p


class FakeStream:
    def __init__(self, cq, rq):
        self.cq = cq
        self.rq = rq

    def write(self, x):
        self.cq.put(x)

    def readline(self):
        return self.rq.get()

    def close(self):
        pass


class GRBLController:
    def __init__(self, serial_fn, dynamics, channel_to_motor_map):
        self.dynamics = dynamics
        # Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
        self.position = {"time": time.time(), "xy": np.zeros((2, 2))}
        self.serial_fn = serial_fn
        if serial_fn is None or "none" in serial_fn:

            #
            self._commands_queue = queue.Queue()
            self._move_queue = queue.Queue()
            self._response_queue = queue.Queue()
            self._max_speed = 4000  # mm/s
            self._timestep = 0.1
            self._current_position = {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0}
            self._targets = self._current_position.copy()

            self._fake_grbl_thread = threading.Thread(
                target=self._run_fake_grbl, daemon=True
            )
            self._fake_grbl_thread.start()

            self._move_thread = threading.Thread(
                target=self._move_fake_grbl, daemon=True
            )
            self._move_thread.start()

            self.s = FakeStream(self._commands_queue, self._response_queue)
            print("STARTED FAKE GRBL")
        elif serial_fn is not None:
            self.s = serial.Serial(
                serial_fn, 115200, timeout=3, write_timeout=3
            )  # GRBL operates at 115200 baud. Leave that part alone.
            self.s.write("?".encode())
            grbl_out = self.s.readline()  # get the response
            logging.info(f"GRBL ONLINE {grbl_out}")

        self.update_status()
        time.sleep(0.05)
        self.channel_to_motor_map = channel_to_motor_map

    def _should_move(self):
        for k in self._targets:
            if self._current_position[k] != self._targets[k]:
                return True
        return False

    def _move_fake_grbl(self):
        step_size = self._timestep * self._max_speed
        while True:
            if self._should_move():  # lets move
                updated_position = self._current_position.copy()
                for k in self._targets:
                    max_diff = 0.0
                    if self._current_position[k] != self._targets[k]:
                        max_diff = max(
                            abs(self._current_position[k] - self._targets[k]), max_diff
                        )
                        if self._current_position[k] > self._targets[k]:
                            # need to decrease
                            updated_position[k] = max(
                                self._targets[k], self._current_position[k] - step_size
                            )
                        else:
                            # need to increase
                            updated_position[k] = min(
                                self._targets[k], self._current_position[k] + step_size
                            )
                self._current_position = updated_position
                # time.sleep(min(max_diff / self._max_speed, self._timestep))
            else:
                command = self._move_queue.get().decode().strip()
                targets = self._current_position.copy()
                # G0 X4.54 Y-1.64 Z4.54 A-1.64
                for move_str in command.split()[1:]:
                    channel = move_str[0]
                    targets[channel] = float(move_str[1:])
                self._targets = targets
                # breakpoint()

    def _run_fake_grbl(self):
        while True:
            command = self._commands_queue.get().decode().strip()
            response = "OK"
            if command.strip() == "?":
                response = (
                    f"<Idle|WPos:{self._current_position['X']},{self._current_position['Y']},"
                    + f"{self._current_position['Z']},{self._current_position['A']}|FS:0,0|Ov:100,100,100>"
                )
            elif command.split()[0] == "G0":
                self._move_queue.put(command.encode())
            else:
                print("FAKE GRBL WEIRD COMMAND", command)

            self._response_queue.put(response.encode())

    def push_reset(self):
        self.s.write(b"\x18")
        return self.s.readline().decode().strip()

    def update_status(self, skip_write=False):
        if not skip_write:
            time.sleep(0.01)
            try:
                self.s.write("?".encode())
            except Exception as e:
                logging.error("FAILED UPDATE STATUS!!!" + str(e))
                raise ValueError
        time.sleep(0.01)

        response = self.s.readline().decode().strip()
        time.sleep(0.01)
        try:
            motor_position_str = response.split("|")[1]
        except Exception as e:
            if response.strip() == "ok" or response.strip() == "":
                return self.update_status(skip_write=not skip_write)
            logging.warning(f"Failed to parse grbl output, {response}, {e}")
            return self.update_status(skip_write=not skip_write)
        b0_motor_steps, a0_motor_steps, b1_motor_steps, a1_motor_steps = map(
            float, motor_position_str[len("MPos:") :].split(",")
        )

        xy0 = self.dynamics.from_steps([a0_motor_steps, b0_motor_steps])
        xy1 = self.dynamics.from_steps([a1_motor_steps, b1_motor_steps])

        is_moving = (self.position["xy"][0] != xy0).any() or (
            self.position["xy"][1] != xy1
        ).any()
        self.position = {
            "a0_motor_steps": a0_motor_steps,
            "b0_motor_steps": b0_motor_steps,
            "a1_motor_steps": a1_motor_steps,
            "b1_motor_steps": b1_motor_steps,
            "xy": [xy0, xy1],
            "is_moving": is_moving,
            "time": time.time(),
        }
        return self.position

    def wait_while_moving(self):
        global run_grbl
        while run_grbl:
            old_pos = self.update_status()
            time.sleep(0.05)
            new_pos = self.update_status()
            if (
                old_pos["a_motor_steps"] == new_pos["a_motor_steps"]
                and old_pos["b_motor_steps"] == new_pos["b_motor_steps"]
            ):
                return
            time.sleep(0.01)

    def move_to(self, points):  # takes in a list of points equal to length of map
        global run_grbl
        gcode_move = ["G0"]
        for c in points:
            if not run_grbl:
                return
            motors = self.channel_to_motor_map[c]
            a_motor_steps, b_motor_steps = self.dynamics.to_steps(points[c])
            gcode_move += [
                "%s%0.2f %s%0.2f" % (motors[0], b_motor_steps, motors[1], a_motor_steps)
            ]
        cmd = " ".join(gcode_move)
        time.sleep(0.01)
        try:
            self.s.write((cmd + "\n").encode())  # Send g-code block to grbl
        except Exception as e:
            logging.error("FAIL MOVE TO" + str(e))
            raise ValueError
        time.sleep(0.01)
        grbl_out = (  # noqa
            self.s.readline()
        )  # Wait for grbl response with carriage return
        time.sleep(0.01)

    def set_current_position(self, motor_channel, steps):
        a_motor_steps, b_motor_steps = steps
        motors = self.channel_to_motor_map[motor_channel]
        cmd = "G92 %s%0.2f %s%0.2f" % (
            motors[0],
            b_motor_steps,
            motors[1],
            a_motor_steps,
        )
        time.sleep(0.01)
        self.s.write((cmd + "\n").encode())  # Send g-code block to grbl
        time.sleep(0.01)
        self.update_status()

    def distance_to_targets(self, target_points):
        current_position = self.update_status()
        return {
            c: np.linalg.norm(current_position["xy"][c] - target_points[c])
            for c in target_points
        }

    def targets_far_out(self, target_points, tolerance=50):
        dists = self.distance_to_targets(target_points)
        for c in dists:
            if dists[c] >= tolerance:
                return True
        return False

    def move_to_iter(self, points_by_channel, return_once_stopped=False):
        global run_grbl
        previous_points = None
        while run_grbl:
            next_points = get_next_points(points_by_channel)
            if len(next_points) == 0:
                break
            self.move_to(next_points)
            while self.targets_far_out(next_points):
                time.sleep(0.1)
            if previous_points is not None and return_once_stopped:
                if np.all([next_points[x] == previous_points[x] for x in next_points]):
                    return
            previous_points = next_points

    def close(self):
        self.s.close()


def get_next_points(channel_iterators):
    ret = {}
    for c in channel_iterators:
        try:
            ret[c] = next(channel_iterators[c])
        except StopIteration:
            pass
    return ret


class GRBLManager:
    def __init__(self, controller, routine):
        self.controller = controller
        self.channels = list(self.controller.channel_to_motor_map.keys())
        self.planners = [None for x in self.channels]
        self.routines = {
            "v1_calibrate": self.v1_calibrate,
            "v4_calibrate": self.v4_calibrate,
            "rx_circle": self.rx_circle,
            "rx_random_circle": self.rx_random_circle,
            "tx_circle": self.tx_circle,
            "bounce": self.bounce,
        }
        self.ready = threading.Lock()
        self.ready.acquire()
        self.routine = routine
        self.planner_started_moving = False
        if self.routine is not None:
            self.planner_thread = threading.Thread(target=self.run_planner, daemon=True)

    def has_planner_started_moving(self):
        return self.planner_started_moving

    def run_planner(self):
        logging.info("GRBL thread runner")
        self.routines[self.routine]()  # setup run
        logging.info("Waiting for GRBL to get into position")
        self.get_ready()  # move into position
        if self.routine is None:
            logging.info("No routine to run, just spining")
            while True:
                time.sleep(0.5)
        else:
            try:
                if self.routine in self.routines:
                    logging.info(f"RUNNING ROUTINE {self.routine}")
                    self.routines[self.routine]()
                    self.get_ready()
                    self.planner_started_moving = True
                    if self.controller.serial_fn is not None:
                        self.run()
                else:
                    raise ValueError(f"Unknown grbl routine f{self.routine}")
            except Exception as e:
                logging.error(e)
        logging.info("GRBL thread runner loop")
        time.sleep(10)  # cool off the motor

    def start(self):
        self.planner_thread.start()

    def get_ready(self):
        # default start points are the current start points
        self.requested_start_points = self.controller.position["xy"][:]  # make a copy
        for c in self.channels:
            requested_point = self.planners[c].get_planner_start_position()
            if requested_point is not None:
                self.requested_start_points[c] = requested_point

        logging.info("Moving into position")
        # planners to move into place
        move_into_place_planners = {
            c_idx: StationaryPlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][c_idx],
                stationary_point=self.requested_start_points[c_idx],
            ).yield_points()
            for c_idx in range(len(self.planners))
        }
        self.controller.move_to_iter(move_into_place_planners, return_once_stopped=True)
        # wait for finish moving
        while run_grbl and self.controller.position["is_moving"]:
            self.controller.update_status()
            time.sleep(0.1)

    def run(self):
        # planners to run routine
        logging.info("Running routine")
        for c_idx in range(len(self.planners)):
            self.planners[c_idx].start_point = self.requested_start_points[c_idx]
        points_by_channel = {c: self.planners[c].yield_points() for c in self.channels}
        self.controller.move_to_iter(points_by_channel)

    def bounce(self):
        self.planners = [
            BouncePlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][c_idx],
            )
            for c_idx in range(len(self.planners))
        ]

    def tx_circle(self):
        self.planners = [
            StationaryPlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][0],
                stationary_point=circle_center,
            ),
            CirclePlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][1],
                circle_diameter=max_circle_diameter,
                circle_center=circle_center,
            ),
        ]

    def rx_circle(self):
        self.planners = [
            CirclePlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][0],
                circle_diameter=max_circle_diameter,
                circle_center=circle_center,
            ),
            StationaryPlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][1],
                stationary_point=circle_center,
            ),
        ]

    def v4_calibrate(self):
        self.planners = [
            BouncePlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][0],
            ),
            StationaryPlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][0],
                stationary_point=rx_calibration_point,
            ),
        ]

    def rx_random_circle(self):
        self.planners = [
            CirclePlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][0],
                circle_diameter=random.randint(
                    min_circle_diameter, max_circle_diameter
                ),
                circle_center=circle_center,
            ),
            StationaryPlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][1],
                stationary_point=circle_center,
            ),
        ]

    def v1_calibrate(self):
        self.planners = [
            StationaryPlanner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][0],
                stationary_point=rx_calibration_point,
            ),
            CalibrationV1Planner(
                self.controller.dynamics,
                start_point=self.controller.position["xy"][1],
            ),
        ]

    def close(self):
        self.controller.close()


def get_default_dynamics(unsafe=False):
    return GRBLDynamics(
        calibration_point=home_calibration_point,
        pA=home_pA,
        pB=home_pB,
        bounding_box=home_bounding_box,
        unsafe=unsafe,
    )


def get_default_gm(serial_fn, routine, unsafe=False):
    dynamics = get_default_dynamics(unsafe)

    # planners = {0: Planner(dynamics), 1: Planner(dynamics)}

    controller = GRBLController(
        serial_fn, dynamics, channel_to_motor_map={0: "XY", 1: "ZA"}
    )

    return GRBLManager(controller, routine=routine)


def exit_fine():
    global run_grbl
    run_grbl = False
    print("EXIT!!")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("serial", help="serial device to use")
    parser.add_argument("--unsafe", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    logging.basicConfig(
        handlers=handlers,
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )

    gm = get_default_gm(args.serial, unsafe=args.unsafe, routine=None)

    if gm.controller.position["is_moving"]:
        print("Waiting for grbl to stop moving before starting...")
        while gm.controller.position["is_moving"]:
            time.sleep(0.1)
            gm.controller.update_status()
    print(
        """q = quit
        s = status
        X,Y = move to X,Y

        Routines:"""
    )
    for k in gm.routines:
        print("        ", k)
    while run_grbl:
        if select.select(
            [
                sys.stdin,
            ],
            [],
            [],
            0.1,
        )[0]:
            line = sys.stdin.readline()
            line = line.strip()
            if line in gm.routines:
                gm.routines[line]()
                gm.get_ready()
                gm.run()
            elif line.split()[0] == "timer":
                t = threading.Timer(float(line.split()[1]), exit_fine)
                t.start()
            elif len(line) == 0:
                continue
            elif line == "q":
                sys.exit(0)
            elif line == "s":
                p = gm.controller.update_status()
                print(p)
            elif len(line.split()) == 3:
                target_channel = int(line.split()[0])
                points_iter = {
                    target_channel: iter(
                        a_to_b_in_stepsize(
                            gm.controller.update_status()["xy"][target_channel],
                            np.array([float(x) for x in line.split()[1:]]),
                            5,
                        )
                    )
                }
                gm.controller.move_to_iter(points_iter)
            else:
                points_iter = {
                    c: iter(
                        a_to_b_in_stepsize(
                            gm.controller.update_status()["xy"][c],
                            np.array([float(x) for x in line.split()]),
                            5,
                        )
                    )
                    for c in [0, 1]
                }
                gm.controller.move_to_iter(points_iter)
            time.sleep(0.01)

    gm.close()
