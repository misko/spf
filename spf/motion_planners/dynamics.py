import numpy as np
import matplotlib.path as pltpath
from scipy.spatial import ConvexHull
import logging


class BoundingBoxIsNonConvexException(Exception):
    pass


class PointOutOfBoundsException(Exception):
    pass

def chord_length_to_angle(chord_length, radius):
    return 2 * np.arcsin(chord_length / (2 * radius))


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

