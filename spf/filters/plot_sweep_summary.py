"""Summary figures for a filter sweep report.

Four questions a reader of a sweep report has, none of which a leaderboard table
answers well:

1. how do the families rank, and by how much against the random floor
2. how much does the corpus mean move with the RNG seed alone
3. does array geometry (d/lambda) explain performance
4. how sensitive is each family to its hyperparameters

Every MSE panel is drawn against the **uniform-random floor** (pi^2/3 = 3.29
rad^2). Without it a bar chart of MSEs has no scale: 2.6 looks like a number
rather than "barely better than guessing".

Usage::

    python spf/filters/plot_sweep_summary.py \\
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

UNIFORM_RANDOM_MSE = np.pi**2 / 3.0

# measured frame yield per d/lambda -- H4 is confounded with this
YIELD = {0.67317: 64.8, 0.68181: 64.8, 0.82703: 75.8,
         0.83765: 75.8, 0.90397: 36.9, 0.91557: 71.3}


def mse_of(row):
    return row.get("mse_craft_theta_mean", row.get("mse_single_radio_theta_mean"))


def config_of(row):
    keys = ("N", "theta_err", "theta_dot_err", "absolute", "phi_std", "p",
            "noise_std", "dynamic_R", "rx_idx")
    return tuple((k, row.get(k)) for k in keys)


def per_config_across_seeds(rows):
    """{(family, frame, config): [corpus-mean MSE per seed]}.

    Two levels: within a seed, average the per-spacing rows weighted by dataset
    count to get one corpus mean; then collect those across seeds.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        m = mse_of(r)
        if m is None:
            continue
        acc[(r["type"], r["frame"], config_of(r))][r.get("seed")].append(
            (m, r["n_runs"])
        )
    out = {}
    for key, byseed in acc.items():
        out[key] = [
            sum(m * n for m, n in v) / sum(n for _, n in v) for v in byseed.values()
        ]
    return out


def short(name):
    return name.replace("_single_theta", "").replace("_", " ")


def fig_family_ranking(per_cfg, out_dir):
    """Best configuration per family, against the random floor."""
    best = {}
    for (fam, frame, _cfg), per_seed in per_cfg.items():
        m = float(np.mean(per_seed))
        k = (fam, frame)
        if k not in best or m < best[k][0]:
            best[k] = (m, float(np.std(per_seed)))
    items = sorted(best.items(), key=lambda kv: kv[1][0])
    labels = [f"{short(f)}\n[{fr}]" for (f, fr), _ in items]
    vals = [v[0] for _, v in items]
    errs = [v[1] for _, v in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["tab:green" if "NN" in f else "tab:blue" for (f, _), _ in items]
    ax.barh(range(len(items)), vals, xerr=errs, color=colors, alpha=0.85,
            error_kw={"ecolor": "black", "capsize": 3})
    ax.axvline(UNIFORM_RANDOM_MSE, color="tab:red", ls="--", lw=1.6)
    # inside the axes and at the top -- below the last bar it falls off the figure
    ax.text(UNIFORM_RANDOM_MSE, len(items) - 0.4, "  uniform random (π²/3)",
            color="tab:red", fontsize=9, va="top", ha="left")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("corpus-mean MSE (rad²), best configuration, ±1σ across 5 seeds")
    ax.set_title("Best configuration per filter family (green = NN likelihood)")
    ax.grid(axis="x", alpha=0.25)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + max(errs) + 0.05, i, f"{v:.3f}  ({1 - v / UNIFORM_RANDOM_MSE:+.0%})",
                va="center", fontsize=8)
    ax.set_xlim(0, UNIFORM_RANDOM_MSE * 1.25)
    fig.tight_layout()
    fn = os.path.join(out_dir, "family_ranking.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def fig_seed_spread(per_cfg, out_dir, n_datasets=None):
    """How much the corpus mean moves with the seed alone."""
    fams = collections.defaultdict(list)
    for (fam, frame, _c), per_seed in per_cfg.items():
        if len(per_seed) < 2:
            continue
        m = float(np.mean(per_seed))
        if m > 0:
            fams[f"{short(fam)}\n[{frame}]"].append(float(np.std(per_seed)) / m)
    if not fams:
        return None
    order = sorted(fams, key=lambda k: np.median(fams[k]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot([fams[k] for k in order], labels=order, showfliers=False)
    ax.set_ylabel("seed-to-seed std / mean, per configuration")
    # The dataset count must come from the data: this figure is generated for
    # both the 16-store tuning sweep and the 48-store confirmation, and a
    # hardcoded "16" is simply false on the latter.
    n = f"{n_datasets} datasets" if n_datasets else "the corpus"
    ax.set_title(
        "Corpus-mean stability across 5 seeds\n"
        f"per-DATASET spread was measured at 42–106%; averaging {n} is what "
        "brings it here"
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fn = os.path.join(out_dir, "seed_spread.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def best_per_spacing(rows):
    """{spacing: {(family, frame): seed-mean MSE of the best CONFIGURATION}}.

    The statistic has to match ``fig_family_ranking``: average across seeds
    first, then minimise over configurations. Minimising over the raw rows
    instead picks the single luckiest (configuration, seed, dataset) draw at
    that spacing -- with seed-to-seed spread measured at 42-106% per dataset
    that is mostly a picture of the RNG, and it reads 3-5x better than anything
    reproducible.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        m = mse_of(r)
        if m is None:
            continue
        s = round(r["rx_wavelength_spacing"], 5)
        key = (r["type"], r["frame"], config_of(r))
        acc[s][key].append((m, r["n_runs"], r.get("seed")))

    out = {}
    for s, by_cfg in acc.items():
        best = {}
        for (fam, frame, _cfg), obs in by_cfg.items():
            per_seed = collections.defaultdict(list)
            for m, n, seed in obs:
                per_seed[seed].append((m, n))
            seed_means = [
                sum(m * n for m, n in v) / sum(n for _, n in v)
                for v in per_seed.values()
            ]
            mean = float(np.mean(seed_means))
            if (fam, frame) not in best or mean < best[(fam, frame)]:
                best[(fam, frame)] = mean
        out[s] = best
    return out


def datasets_per_spacing(rows):
    """How many captures back each spacing. ``n_runs`` is that count already."""
    out = {}
    for r in rows:
        s = round(r["rx_wavelength_spacing"], 5)
        out[s] = max(out.get(s, 0), r["n_runs"])
    return out


def fig_by_spacing(rows, out_dir):
    """H4: is a wider (more aliased) array worse -- and is there enough data to say?"""
    per = best_per_spacing(rows)
    n_datasets = datasets_per_spacing(rows)
    spacings = sorted(per)
    fams = sorted({f for v in per.values() for f in v})

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for fam in fams:
        ys = [per[s].get(fam, np.nan) for s in spacings]
        ax[0].plot(spacings, ys, marker="o", lw=1.2, ms=5,
                   label=f"{short(fam[0])} [{fam[1]}]")
    ax[0].axhline(UNIFORM_RANDOM_MSE, color="tab:red", ls="--", lw=1.4)
    ax[0].text(spacings[-1], UNIFORM_RANDOM_MSE, "uniform random ", color="tab:red",
               fontsize=8, va="bottom", ha="right")
    ax[0].set_xlabel("d / λ")
    ax[0].set_ylabel("best configuration's seed-mean MSE (rad²)")
    ax[0].set_title("H4: performance vs array spacing\n(all spacings are past the λ/2 limit)")
    # below the curves: the random line is the top of the axis, so the legend
    # cannot sit up there without covering it
    ax[0].legend(fontsize=7, loc="center left", framealpha=0.9)
    ax[0].set_ylim(0, UNIFORM_RANDOM_MSE * 1.08)
    ax[0].grid(alpha=0.25)

    # How much evidence sits under each point on the left. Half these spacings
    # come from ONE capture, which is the real limit on reading H4 off the left
    # panel -- more than any effect of geometry.
    ax2 = ax[1]
    counts = [n_datasets.get(s, 0) for s in spacings]
    x = np.arange(len(spacings))
    bars = ax2.bar(x, counts, color=["tab:red" if c < 2 else "tab:blue" for c in counts],
                   alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s:.5f}" for s in spacings], fontsize=8)
    ax2.set_xlabel("d / λ")
    ax2.set_ylabel("datasets contributing (red = a single capture)")
    ax2.set_title(
        "What backs each point on the left\n"
        "3 of 6 spacings are one capture, so their curve is one dataset's luck"
    )
    for xi, (c, s) in enumerate(zip(counts, spacings)):
        ax2.text(xi, c + 0.08, f"{c}\nyield {YIELD.get(s, float('nan')):.0f}%",
                 ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylim(0, max(counts) + 1.6)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fn = os.path.join(out_dir, "by_spacing.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def fig_hyperparam_sensitivity(per_cfg, out_dir):
    """Spread of achievable MSE within each family -- does tuning matter?"""
    fams = collections.defaultdict(list)
    for (fam, frame, _c), per_seed in per_cfg.items():
        fams[f"{short(fam)}\n[{frame}]"].append(float(np.mean(per_seed)))
    order = sorted(fams, key=lambda k: np.min(fams[k]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot([fams[k] for k in order], labels=order, showfliers=True,
               flierprops={"markersize": 2, "alpha": 0.4})
    ax.axhline(UNIFORM_RANDOM_MSE, color="tab:red", ls="--", lw=1.4)
    ax.text(0.6, UNIFORM_RANDOM_MSE, " uniform random", color="tab:red", fontsize=8,
            va="bottom")
    ax.set_ylabel("corpus-mean MSE (rad²) over every configuration")
    ax.set_title(
        "Hyperparameter sensitivity: the spread a family can reach\n"
        "a box straddling the red line means a bad configuration is no better than guessing"
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fn = os.path.join(out_dir, "hyperparam_sensitivity.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    rows = json.load(open(a.results))["rows"]
    per_cfg = per_config_across_seeds(rows)
    n_datasets = sum(datasets_per_spacing(rows).values())
    for fn in (
        fig_family_ranking(per_cfg, a.output_dir),
        fig_seed_spread(per_cfg, a.output_dir, n_datasets=n_datasets),
        fig_by_spacing(rows, a.output_dir),
        fig_hyperparam_sensitivity(per_cfg, a.output_dir),
    ):
        if fn:
            print("wrote", fn)
