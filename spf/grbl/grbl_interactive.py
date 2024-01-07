import logging
import sys
import time

import matplotlib.path as pltpath
import numpy as np
import serial
from scipy.spatial import ConvexHull

home_pA = np.array([3568, 0])
home_pB = np.array([0, 0])
home_bounding_box = [
    [300, 400],
    [3100, 400],
    [3100, 2850],
    # [300,1500],
    [800, 1000],
]

run_grbl = True


def stop_grbl():
    logging.info("STOP GRBL")
    global run_grbl
    run_grbl = False


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
home_calibration_point = np.array([300, 400])


def a_to_b_in_stepsize(a, b, step_size):
    if np.isclose(a, b).all():
        return [b]
    # move by step_size from where we are now to the target position
    points = [a]
    direction = (b - a) / np.linalg.norm(b - a)
    distance = np.linalg.norm(b - a)
    _l = 0
    while _l < distance:
        points.append(_l * direction + a)
        _l = _l + step_size
    points.append(b)
    return points


class Dynamics:
    def __init__(self, calibration_point, pA, pB, bounding_box):
        self.calibration_point = calibration_point
        self.pA = pA
        self.pB = pB
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
                raise ValueError
            self.polygon = pltpath.Path(bounding_box)
        else:
            self.polygon = None

    # a is the left motor on the wall
    # b is the right motor on the wall
    # (a-b)_x > 0
    # x is positive from b towards a
    # y is positive up to ceiling and negative down to floor

    def from_steps(self, a_motor_steps, b_motor_steps):
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
        if (self.polygon is not None) and not self.polygon.contains_point(
            p, radius=0.01
        ):  # todo a bit hacky but works
            raise ValueError
        # motor_steps = ( distance_between_pivot and point ) - (distance between pivot and calibration point)
        a_motor_steps = np.linalg.norm(self.pA - p) - np.linalg.norm(
            self.pA - self.calibration_point
        )
        b_motor_steps = np.linalg.norm(self.pB - p) - np.linalg.norm(
            self.pB - self.calibration_point
        )
        return a_motor_steps, b_motor_steps

    def binary_search_edge(self, left, right, xy, direction, epsilon):
        if (right - left) < epsilon:
            return left
        midpoint = (right + left) / 2
        p = midpoint * direction + xy
        try:
            steps = self.to_steps(p)  # noqa
            # actual = self.from_steps(*steps)
            return self.binary_search_edge(midpoint, right, xy, direction, epsilon)
        except ValueError:
            return self.binary_search_edge(left, midpoint, xy, direction, epsilon)

    def get_boundary_vector_near_point(self, p):
        if self.polygon is None:
            raise ValueError

        bvec = None
        max_score = 0
        nverts = len(self.polygon.vertices)
        for i in range(nverts):
            v0 = self.polygon.vertices[i % nverts]
            v1 = self.polygon.vertices[(i + 1) % nverts]
            score = max(
                np.dot(p - v0, v1 - v0)
                / (np.linalg.norm(v0 - v1) * np.linalg.norm(p - v0) + 0.01),
                np.dot(p - v1, v1 - v1)
                / (np.linalg.norm(v0 - v1) * np.linalg.norm(p - v1) + 0.01),
            )
            if score > max_score:
                max_score = score
                bvec = (v1 - v0) / np.linalg.norm(v1 - v0)
        return bvec


class Planner:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.current_direction = None
        self.epsilon = 1  # original was 0.001

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

    def single_bounce(self, direction, p, step_size=5):
        bounce_point, new_direction = self.get_bounce_pos_and_new_direction(
            p, direction
        )

        to_points = a_to_b_in_stepsize(p, bounce_point, step_size=step_size)

        # add some noise to the new direction
        theta = np.random.uniform(2 * np.pi)
        percent_random = 0.05
        new_direction = (
            1 - percent_random
        ) * new_direction + percent_random * np.array([np.sin(theta), np.cos(theta)])

        return to_points, new_direction

    def random_direction(self):
        theta = np.random.uniform(2 * np.pi)
        return np.array([np.sin(theta), np.cos(theta)])

    def bounce(self, start_p, n_bounces):
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
        while n_bounce < n_bounces or n_bounces == -1:
            if not run_grbl:
                logging.info("Exiting bounce early")
                break
            to_points, new_direction = self.single_bounce(
                self.current_direction, start_p
            )
            assert len(to_points) > 0
            yield from to_points
            start_p = to_points[-1]
            self.current_direction = new_direction
            n_bounce += 1
        logging.info("Exiting bounce")


class GRBLController:
    def __init__(self, serial_fn, dynamics, channel_to_motor_map):
        self.dynamics = dynamics
        # Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
        self.s = serial.Serial(
            serial_fn, 115200, timeout=0.3, write_timeout=0.3
        )  # GRBL operates at 115200 baud. Leave that part alone.
        self.s.write("?".encode())
        grbl_out = self.s.readline()  # get the response
        logging.info(f"GRBL ONLINE {grbl_out}")
        self.position = {"time": time.time(), "xy": np.zeros(2)}
        self.update_status()
        time.sleep(0.05)
        self.channel_to_motor_map = channel_to_motor_map

    def push_reset(self):
        self.s.write(b"\x18")
        return self.s.readline().decode().strip()

    def update_status(self, skip_write=False):
        if not skip_write:
            time.sleep(0.01)
            self.s.write("?".encode())
        time.sleep(0.01)

        response = self.s.readline().decode().strip()
        time.sleep(0.01)
        # print("STATUS",response)
        # <Idle|MPos:-3589.880,79.560,0.000,0.000|FS:0,0>
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

        xy0 = self.dynamics.from_steps(a0_motor_steps, b0_motor_steps)
        xy1 = self.dynamics.from_steps(a1_motor_steps, b1_motor_steps)

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
        self.s.write((cmd + "\n").encode())  # Send g-code block to grbl
        time.sleep(0.01)
        grbl_out = (  # noqa
            self.s.readline()
        )  # Wait for grbl response with carriage return
        time.sleep(0.01)
        # print("MOVE TO RESPONSE", grbl_out)

    def distance_to_targets(self, target_points):
        current_position = self.update_status()
        return {
            c: np.linalg.norm(current_position["xy"][c] - target_points[c])
            for c in target_points
        }

    def targets_far_out(self, target_points, tolerance=80):
        dists = self.distance_to_targets(target_points)
        for c in dists:
            if dists[c] >= tolerance:
                return True
        return False

    def move_to_iter(self, points_by_channel):
        global run_grbl
        while run_grbl:
            next_points = get_next_points(points_by_channel)
            if len(next_points) == 0:
                break
            self.move_to(next_points)
            while self.targets_far_out(next_points):
                pass

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
    def __init__(self, controller, planners):
        self.controller = controller
        self.channels = list(self.controller.channel_to_motor_map.keys())
        self.planners = planners

    def bounce(self, n_bounces, direction=None):
        start_positions = self.controller.update_status()["xy"]
        points_by_channel = {
            c: self.planners[c].bounce(start_positions[c], n_bounces)
            for c in self.channels
        }
        self.controller.move_to_iter(points_by_channel)


def get_default_gm(serial_fn):
    dynamics = Dynamics(
        calibration_point=home_calibration_point,
        pA=home_pA,
        pB=home_pB,
        bounding_box=home_bounding_box,
    )

    planners = {0: Planner(dynamics), 1: Planner(dynamics)}

    controller = GRBLController(
        serial_fn, dynamics, channel_to_motor_map={0: "XY", 1: "ZA"}
    )

    return GRBLManager(controller, planners)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("grblman: %s device" % sys.argv[0])
        sys.exit(1)

    serial_fn = sys.argv[1]

    gm = get_default_gm()

    print(
        """
        q = quit
        bounce = bounce
        s = status
        X,Y = move to X,Y
    """
    )
    for line in sys.stdin:
        line = line.strip()
        if line == "q":
            sys.exit(1)
        elif line == "bounce":
            # gm.bounce(20000)
            gm.bounce(40)
        elif line == "s":
            p = gm.controller.update_status()
            print(p)
        else:
            current_positions = gm.controller.update_status()["xy"]
            p_main = np.array([float(x) for x in line.split()])
            if True:
                points_iter = {
                    c: iter(a_to_b_in_stepsize(current_positions[c], p_main, 5))
                    for c in [0, 1]
                }
                gm.controller.move_to_iter(points_iter)
            # except ValueError:
            #    print("Position is out of bounds!")
        time.sleep(0.01)

    gm.close()
