"""Turn a local sweep's results into artifacts you can actually commit.

``run_filters_report.py`` writes CSV, and ``.gitignore`` matches ``**/*.csv``
(with only the JLC fab files excepted), so a sweep result in that format can
never be checked in. This writes ``results.json`` plus a ``LEADERBOARD.md``
table, both of which are tracked, matching the append-only report convention
already used under ``spf/calibrations/*/reports/<name>_<date>_v1/``. The
hand-written ``REPORT.md`` in the same directory is never touched.

It also groups on the angular **frame**, so a craft-relative run and an
absolute-north run of the same hyperparameters never land in one average.
"""

import argparse
import json
import logging
import os
import pickle

from spf.evaluation.aggregate import aggregate
from spf.filters.labels import ds_fn_to_movement, result_to_frame

# Fields that describe the run rather than measure it; everything else that is
# not a metric becomes part of the grouping key.
_NON_GROUPING = {"metrics", "ds_fn", "precompute_cache", "full_name", "runtime"}


def load_results(workdir):
    """Every result dict written by LocalResultsStore under ``workdir``."""
    out = []
    for dirpath, _, filenames in os.walk(workdir):
        for name in sorted(filenames):
            if not name.endswith(".pkl"):
                continue
            with open(os.path.join(dirpath, name), "rb") as f:
                out.extend(pickle.load(f))
    return out


def annotate(results):
    """Attach ``frame`` and ``movement`` so the results can be grouped."""
    annotated = []
    for result in results:
        result = dict(result)
        result["frame"] = result_to_frame(result)
        result["movement"] = ds_fn_to_movement(
            result.get("ds_fn", ""), result.get("routine")
        )
        annotated.append(result)
    return annotated


def grouping_keys(results):
    """Every non-metric field shared by all results, in stable order.

    Call this on ONE filter family at a time. Across families the intersection
    contains no hyperparameter at all -- an EKF result carries phi_std/p, a PF
    result carries N/theta_err/seed -- so a whole-sweep intersection silently
    averages every configuration together and the leaderboard ranks nothing.
    """
    if not results:
        return []
    common = set(results[0])
    for result in results[1:]:
        common &= set(result)
    return sorted(common - _NON_GROUPING)


def build_report(workdir):
    results = annotate(load_results(workdir))
    if not results:
        raise ValueError(f"no results found under {workdir}")

    # Group within each filter family. Families have disjoint hyperparameters and
    # are never ranked against each other anyway -- they get separate tables.
    by_type = {}
    for result in results:
        by_type.setdefault(result.get("type", "unknown"), []).append(result)

    rows = []
    for _type in sorted(by_type):
        family = by_type[_type]
        rows.extend(aggregate(family, grouping_keys(family)))

    return {
        "workdir": workdir,
        "n_results": len(results),
        "n_groups": len(rows),
        "families": {t: len(v) for t, v in sorted(by_type.items())},
        "rows": rows,
    }


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


# Each filter family names its error metric differently, so the column to rank on
# cannot be hard-coded: dual-radio filters report mse_craft_theta, single-radio
# ones mse_single_radio_theta, XY adds mse_xy. Preference order, best first.
_SORT_PREFERENCE = (
    "mse_craft_theta_mean",
    "mse_single_radio_theta_mean",
    "mse_mean",
    "mse_xy_mean",
)


def default_sort_metric(rows):
    """The metric to rank these rows by, discovered from what they carry."""
    present = set().union(*(set(r) for r in rows)) if rows else set()
    for name in _SORT_PREFERENCE:
        if name in present:
            return name
    candidates = sorted(
        c for c in present if c.startswith("mse") and c.endswith("_mean")
    )
    return candidates[0] if candidates else None


def report_to_markdown(report, sort_by=None, top=25):
    """A per-frame leaderboard. Frames get their own table -- never one ranking."""
    lines = [
        "# Filter sweep report",
        "",
        f"- results: **{report['n_results']}** in **{report['n_groups']}** groups",
        f"- work dir: `{report['workdir']}`",
        "",
        "Rows are grouped by every non-metric field **and by angular frame**. A"
        " craft-relative and an absolute-north run of the same hyperparameters"
        " answer different questions and are never averaged or ranked together.",
        "",
    ]
    by_frame = {}
    for row in report["rows"]:
        by_frame.setdefault(row["frame"], []).append(row)

    for frame in sorted(by_frame):
        rows = by_frame[frame]
        # resolve per frame: single-radio and dual-radio rows carry different
        # metric names, and they never share a frame anyway
        metric = sort_by or default_sort_metric(rows)
        if metric is not None and all(metric in r for r in rows):
            rows = sorted(rows, key=lambda r: r[metric])
            heading = f"## frame: `{frame}` (ranked by `{metric}`)"
        else:
            heading = f"## frame: `{frame}` (unranked: no shared error metric)"
        lines += [heading, ""]
        shown = rows[:top]
        columns = [c for c in sorted(shown[0]) if c != "frame"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in shown:
            lines.append(
                "| " + " | ".join(_fmt(row.get(c, "")) for c in columns) + " |"
            )
        if len(rows) > top:
            lines.append("")
            lines.append(f"_{len(rows) - top} further rows omitted; see results.json._")
        lines.append("")
    return "\n".join(lines)


def write_report(workdir, output_dir, sort_by=None, top=25):
    report = build_report(workdir)
    os.makedirs(output_dir, exist_ok=True)
    json_fn = os.path.join(output_dir, "results.json")
    with open(json_fn, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    # LEADERBOARD.md, not REPORT.md: the human-written narrative for a run lives
    # at REPORT.md by repo convention (see spf/calibrations/*/reports/), and a
    # rerun of this script must never overwrite it.
    md_fn = os.path.join(output_dir, "LEADERBOARD.md")
    with open(md_fn, "w") as f:
        f.write(report_to_markdown(report, sort_by=sort_by, top=top))
    return json_fn, md_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sort-by", type=str, default=None)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    json_fn, md_fn = write_report(
        args.work_dir, args.output_dir, sort_by=args.sort_by, top=args.top
    )
    logging.info(f"wrote {json_fn} and {md_fn}")
