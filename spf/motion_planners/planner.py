import logging
from abc import ABC, abstractmethod

import numpy as np

from .dynamics import a_to_b_in_stepsize, chord_length_to_angle


class Planner(ABC):
    def __init__(self, dynamics, start_point, step_size, epsilon=1, seed=None):
        self.dynamics = dynamics
        self.current_direction = None
        self.epsilon = epsilon  # original was 0.001
        self.start_point = start_point
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)
        self.running = True

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

    def stop(self):
        self.running = False


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
            if not self.running:
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
    def __init__(self, dynamics, start_point, stationary_point, step_size):
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
        while self.running:
            yield self.stationary_point

    def get_planner_start_position(self):
        return self.stationary_point


class PointCycle(Planner):
    def __init__(
        self,
        dynamics,
        start_point,
        points,
        step_size,
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
                if not self.running:
                    return
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
        step_size,
        circle_diameter,
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
            if not self.running:
                return
            next_p = self.angle_to_pos(current_angle)
            yield from a_to_b_in_stepsize(
                current_p,
                next_p,
                step_size=self.step_size,
            )
            current_p = next_p
            current_angle += self.direction * self.angle_increment
