"""Angular reference frames, and the rule that you never pool across them.

A theta prediction is only meaningful against a stated frame. SPF has three, and
they are not interchangeable:

``craft_relative``
    Bearing to the emitter measured from the craft's own heading. This is what
    ``v5spfdataset.craft_ground_truth_thetas`` holds and what the dual-radio
    filters estimate by default.
``absolute_north``
    Bearing in the world frame, heading removed. ``PFSingleThetaDualRadioNN``
    switches to this when constructed with ``absolute=True``.
``radio_folded``
    A single receiver's array-relative bearing after the front/back fold
    (``spf.rf.reduce_theta_to_positive_y``). A 2-element array cannot resolve
    that ambiguity alone, so single-radio filters are scored here.

Why this module exists: ``PFSingleThetaDualRadioNN.metrics`` reports
``absolute=True`` and ``absolute=False`` runs under the same ``mse_craft_theta``
key, while scoring them against *different* ground truths. On one rover capture
that reads as 0.33 versus 1.89 -- an apparent 5.7x win that is really two
different questions. Averaging or ranking across that boundary is always a bug,
so every aggregation here is keyed on the frame and refuses to merge two.
"""

CRAFT_RELATIVE = "craft_relative"
ABSOLUTE_NORTH = "absolute_north"
RADIO_FOLDED = "radio_folded"

FRAMES = frozenset({CRAFT_RELATIVE, ABSOLUTE_NORTH, RADIO_FOLDED})


class FrameMismatch(ValueError):
    """Raised when values from different angular frames would be combined."""


def check_frame(frame):
    """Validate and return a frame name."""
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; expected one of {sorted(FRAMES)}")
    return frame


def require_same_frame(frames):
    """Return the single frame shared by ``frames``, or raise FrameMismatch."""
    distinct = {check_frame(f) for f in frames}
    if len(distinct) == 0:
        raise ValueError("no frames given")
    if len(distinct) > 1:
        raise FrameMismatch(
            "refusing to combine angular frames "
            f"{sorted(distinct)}: they are scored against different ground truth"
        )
    return distinct.pop()
