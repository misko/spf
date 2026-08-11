"""Hyperparameter heatmaps: MSE over two axes, one panel per filter family.

Reproduces the view used in the March 2025 "Droning on 2 (SPF)" deck --
``theta_dot_err`` against ``N`` for the particle filters, ``phi_std`` against
``noise_std`` for the EKFs -- because that view answers a question a leaderboard
cannot: **is the optimum inside the grid, or did the grid stop first?**

Each cell is the BEST corpus-mean MSE achievable at that (x, y), minimising over
every other hyperparameter and averaging across seeds. That is the decision-
relevant statistic for "should I extend this axis": if the minimum sits on an
edge and the surface is still descending into it, the grid is truncating.

Cells are annotated, the optimum is boxed, and any optimum on an edge is called
out in the title -- the failure mode this exists to catch is a grid whose winner
is a wall, which is invisible in a ranked table.

Usage::

    python spf/filters/plot_hyperparam_heatmap.py \\
        --results <report>/results.json --output-dir <report>/figures
"""

import argparse
import collections
import json
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from spf.evaluation import metrics  # noqa: E402
from spf.filters.plot_sweep_summary import mse_of, per_config_across_seeds  # noqa: E402

# Preferred (x, y) per family, matching the 2025 deck. Falls back to the two
# axes with the most distinct values when a family carries neither pair.
PREFERRED = [("N", "theta_dot_err"), ("noise_std", "phi_std")]

# Everything that is a hyperparameter rather than a label.
CANDIDATE_AXES = (
    "N", "theta_err", "theta_dot_err", "phi_std", "p", "noise_std", "dynamic_R",
)


def family_configs(results_rows):
    """{(type, frame): {config_dict_tuple: seed-mean corpus MSE}}."""
    per = per_config_across_seeds(results_rows)
    out = collections.defaultdict(dict)
    for (fam, frame, cfg), per_seed in per.items():
        out[(fam, frame)][cfg] = float(np.mean(per_seed))
    return out


def choose_axes(cfgs):
    """The two axes to plot: the deck's pair if present, else the widest two."""
    counts = {}
    for name in CANDIDATE_AXES:
        vals = {dict(c).get(name) for c in cfgs}
        vals.discard(None)
        if len(vals) > 1:
            counts[name] = len(vals)
    for x, y in PREFERRED:
        if x in counts and y in counts:
            return x, y
    if len(counts) < 2:
        return None
    ranked = sorted(counts, key=lambda k: -counts[k])
    return ranked[0], ranked[1]


def grid_for(cfgs, xname, yname):
    """(xs, ys, Z) where Z[j, i] is the best MSE at (xs[i], ys[j])."""
    xs = sorted({dict(c)[xname] for c in cfgs if dict(c).get(xname) is not None})
    ys = sorted({dict(c)[yname] for c in cfgs if dict(c).get(yname) is not None})
    Z = np.full((len(ys), len(xs)), np.nan)
    for c, m in cfgs.items():
        d = dict(c)
        if d.get(xname) is None or d.get(yname) is None:
            continue
        i, j = xs.index(d[xname]), ys.index(d[yname])
        if np.isnan(Z[j, i]) or m < Z[j, i]:
            Z[j, i] = m
    return xs, ys, Z


def edge_warning(xs, ys, Z, xname, yname, plateau=0.05):
    """Axes whose optimum is on a boundary AND genuinely still descending.

    Naively flagging "the argmin is on an edge" is wrong when the optimum is a
    broad plateau: the survey's single-radio NN surface is 0.23 across the WHOLE
    theta_dot_err=0.2 row, from N=128 to N=32768, so which cell wins is noise and
    the argmin lands on an edge by luck. Extending the grid there would buy
    nothing.

    So an axis is only reported as truncating when NO near-optimal cell (within
    ``plateau`` of the best) is interior on that axis. That is the condition that
    actually means "the grid stopped before the surface did".
    """
    best = np.nanmin(Z)
    near = np.argwhere(np.isfinite(Z) & (Z <= best * (1.0 + plateau)))
    notes = []
    for name, axis, vals in ((xname, 1, xs), (yname, 0, ys)):
        if len(vals) < 3:
            continue
        idxs = {int(c[axis]) for c in near}
        if any(0 < i < len(vals) - 1 for i in idxs):
            continue  # a near-optimal cell is interior -> the grid contains it
        edge = "low" if max(idxs) == 0 else "high"
        line = np.nanmin(Z, axis=1 - axis)
        i = 0 if edge == "low" else len(vals) - 1
        nb = line[1] if edge == "low" else line[-2]
        if not np.isfinite(nb) or line[i] <= 0:
            continue
        notes.append(
            f"{name} optimum on the {edge} edge ({nb / line[i] - 1:+.0%} to neighbour, "
            f"no near-optimal cell interior)"
        )
    return notes


def fig_family(fam, frame, cfgs, out_dir):
    axes = choose_axes(cfgs)
    if axes is None:
        return None
    xname, yname = axes
    xs, ys, Z = grid_for(cfgs, xname, yname)
    if Z.size == 0 or np.all(np.isnan(Z)):
        return None

    # floor the size: a 2x3 grid still needs room for the title, and the
    # coarse sweep produces exactly that
    fig, ax = plt.subplots(
        figsize=(max(1.15 * len(xs) + 4.0, 9.5), max(0.52 * len(ys) + 3.4, 4.6))
    )
    im = ax.imshow(Z, cmap="YlGnBu", origin="upper", aspect="auto",
                   vmin=np.nanmin(Z), vmax=min(np.nanmax(Z), metrics.UNIFORM_RANDOM_MSE))
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=8)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{v:g}" for v in ys], fontsize=8)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    jb, ib = np.unravel_index(np.nanargmin(Z), Z.shape)
    for j in range(len(ys)):
        for i in range(len(xs)):
            if np.isnan(Z[j, i]):
                continue
            # white text on the dark (bad) end, black on the light end
            frac = (Z[j, i] - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-9)
            ax.text(i, j, f"{Z[j, i]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if frac > 0.55 else "black")
    ax.add_patch(plt.Rectangle((ib - 0.5, jb - 0.5), 1, 1, fill=False,
                               edgecolor="tab:red", lw=2.5))

    notes = edge_warning(xs, ys, Z, xname, yname)
    title = (f"{fam.replace('_single_theta', '').replace('_', ' ')}  [{frame}]\n"
             f"best {Z[jb, ib]:.3f} rad² at {xname}={xs[ib]:g}, {yname}={ys[jb]:g}"
             f"   (uniform random {metrics.UNIFORM_RANDOM_MSE:.2f})")
    if notes:
        # one warning per line -- joined they run past the axes on a narrow grid
        title += "\n⚠ " + "\n⚠ ".join(notes)
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("best corpus-mean MSE (rad²), minimised over other axes")
    fig.tight_layout()

    safe = f"{fam}__{frame}".replace("/", "_")
    fn = os.path.join(out_dir, f"heatmap_{safe}.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn, notes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    rows = json.load(open(a.results))["rows"]
    fams = family_configs(rows)
    truncated = []
    for (fam, frame), cfgs in sorted(fams.items()):
        got = fig_family(fam, frame, cfgs, a.output_dir)
        if got is None:
            print(f"skipped {fam} [{frame}]: fewer than two swept axes")
            continue
        fn, notes = got
        print("wrote", fn)
        for n in notes:
            truncated.append(f"{fam} [{frame}]: {n}")
    if truncated:
        print("\nGRID TRUNCATION -- these optima sit on a boundary:")
        for t in truncated:
            print("  ", t)
