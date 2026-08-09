"""Turn a raw filter result into the labels a report groups by.

Two labels, both of which the reporters previously got wrong or omitted:

**movement** -- vehicle plus capture routine. Inferring it from the filename by
substring is unsafe for merged v7 rover data, whose stores are named
``<RX capture>.<TX capture>``. A bounce-RX / circle-TX pair contains *both*
words, and the old reporter tested ``"circle" in name`` first, so every merged
rover store was labelled ``circle`` when its RX routine was ``bounce``. The
capture yaml records the routine; use it, and fall back to the filename only
when it is absent.

**frame** -- which angular frame the result was scored in. This is not recorded
on the result, but it is fully determined by the filter type and the ``absolute``
flag, so it can be recovered rather than having to change every filter's metrics
dict. See ``spf.evaluation.frames`` for why pooling across frames is always a
bug.
"""

from spf.evaluation.frames import ABSOLUTE_NORTH, CRAFT_RELATIVE, RADIO_FOLDED

# Order matters: the more specific routine names must be tested first, or
# "random_circle" is swallowed by "circle".
_ROUTINE_TOKENS = (
    "random_circle",
    "bounce",
    "center",
    "diamond",
    "circle",
    "calibrate",
)


def ds_fn_to_movement(ds_fn, routine=None):
    """``<vehicle>_<routine>``, e.g. ``rover_bounce`` or ``2dwallarray_circle``."""
    name = str(ds_fn).lower()
    movement = ""
    if routine is not None and str(routine) != "unknown":
        movement = str(routine)
    else:
        for token in _ROUTINE_TOKENS:
            if token in name:
                movement = token
                break
    if movement == "":
        raise ValueError(f"cannot determine movement for {ds_fn}")
    vehicle = "rover" if "rover" in name else "2dwallarray"
    return f"{vehicle}_{movement}"


def result_to_frame(result):
    """Recover the angular frame a result was scored in.

    Single-radio filters score against the front/back-folded per-radio bearing.
    Dual-radio filters score against the craft-relative bearing, except the NN
    dual-radio filter with ``absolute=True``, which scores against the circular
    mean of the north-referenced bearings -- a different ground truth reported
    under the same ``mse_craft_theta`` key.
    """
    if result.get("absolute", False):
        return ABSOLUTE_NORTH
    _type = str(result.get("type", "")).lower()
    if "single_radio" in _type:
        return RADIO_FOLDED
    return CRAFT_RELATIVE
