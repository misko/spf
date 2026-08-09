"""Group sweep results and average their metrics -- never across frames.

The existing reporter builds its grouping key from every non-metric field and
averages whatever lands together. That is fine until two runs differ in a field
that changes what "ground truth" means, at which point it averages answers to
two different questions. ``frame`` is always part of the key here, and a missing
frame is an error rather than a default.
"""

from collections import OrderedDict

import numpy as np

from spf.evaluation.frames import check_frame


def _key(result, group_keys):
    return tuple((k, result.get(k)) for k in group_keys)


def group_results(results, group_keys):
    """Bucket result dicts by ``group_keys``, with ``frame`` always included.

    Each result must carry a ``frame`` and a ``metrics`` dict. Returns an
    ordered mapping from key tuple -> list of results.
    """
    keys = list(group_keys)
    if "frame" not in keys:
        keys = ["frame"] + keys

    grouped = OrderedDict()
    for i, result in enumerate(results):
        if "frame" not in result:
            raise ValueError(
                f"result {i} has no 'frame'; every prediction must declare the "
                "angular frame it was scored in"
            )
        check_frame(result["frame"])
        if "metrics" not in result:
            raise ValueError(f"result {i} has no 'metrics'")
        grouped.setdefault(_key(result, keys), []).append(result)
    return grouped


def summarize_group(group):
    """Mean/std/n for every metric shared by all members of one group.

    Metrics present on some members but not others are dropped rather than
    averaged over a varying denominator, and the dropped names are reported so
    the omission is visible instead of silent.
    """
    if not group:
        raise ValueError("empty group")
    metric_sets = [set(r["metrics"].keys()) for r in group]
    common = set.intersection(*metric_sets)
    dropped = sorted(set.union(*metric_sets) - common)

    out = {"n_runs": len(group)}
    for name in sorted(common):
        values = np.array([float(r["metrics"][name]) for r in group], dtype=np.float64)
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_std"] = float(values.std())
    if dropped:
        out["dropped_metrics"] = dropped
    return out


def aggregate(results, group_keys):
    """Group then summarize. Returns a list of flat row dicts, ready for CSV/JSON."""
    rows = []
    for key, group in group_results(results, group_keys).items():
        row = dict(key)
        row.update(summarize_group(group))
        rows.append(row)
    return rows


def rank(rows, metric="mse_mean", ascending=True, min_runs=1):
    """Order aggregated rows by a metric, within each frame separately.

    Ranking pools nothing: rows keep their frame, and the caller gets one ranked
    list per frame. A single global ranking across frames would be meaningless
    for exactly the reason `frames` documents.
    """
    by_frame = OrderedDict()
    for row in rows:
        if row.get("n_runs", 0) < min_runs:
            continue
        if metric not in row:
            continue
        by_frame.setdefault(row["frame"], []).append(row)
    for frame in by_frame:
        by_frame[frame].sort(key=lambda r: r[metric], reverse=not ascending)
    return by_frame
