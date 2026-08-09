"""Summary figures for an empirical-table rebuild report.

The generator can dump two raw heatmaps per key, which is ~100 images for a full
rebuild -- too many to review and far too many to commit. These four answer the
questions a reader of the report actually has:

1. what the newly-added tables look like
2. which (spacing, carrier) combinations the table covers, and how well
3. how the rebuilt legacy keys compare to the table they replace
4. how much usable signal each key was built from

Usage::

    python spf/calibrations/empirical_p_dist/make_report_figures.py \\
        --table empirical_dists/full_20260809_v1.pkl \\
        --baseline empirical_dists/full.pkl \\
        --new-keys SDRDEVICE.PLUTO_0.68181 ... \\
        --output-dir <report>/figures
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pickle  # noqa: E402
import torch  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

PROVENANCE_KEY = "__provenance__"


def load(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


def spacing_keys(table):
    return sorted(k for k in table if k != PROVENANCE_KEY)


def arr(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def fig_new_tables(table, new_keys, out_dir):
    """The actual new content: P(theta|phi) for each newly-covered key."""
    keys = [k for k in new_keys if k in table]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        m = arr(table[k]["r"]["sym"])
        ax.imshow(
            m,
            extent=[-np.pi, np.pi, -np.pi, np.pi],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        n = (
            table[PROVENANCE_KEY]["keys"][k]["n_datasets"]
            if PROVENANCE_KEY in table
            else "?"
        )
        ax.set_title(f"{k.replace('SDRDEVICE.','')}\n{n} datasets", fontsize=9)
        ax.set_xlabel("theta (ground truth)")
        ax.set_ylabel("phi (measured)")
    fig.suptitle("Newly covered keys: P(theta | phi), pooled radios, symmetrised")
    fig.tight_layout()
    fn = os.path.join(out_dir, "new_keys_tables.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def fig_coverage(table, new_keys, out_dir):
    """Which (d/lambda, device) the table covers and how many datasets back each."""
    prov = table.get(PROVENANCE_KEY, {}).get("keys", {})
    rows = []
    for k in spacing_keys(table):
        device, dl = k.rsplit("_", 1)
        rows.append(
            (
                float(dl),
                prov.get(k, {}).get("n_datasets", 0),
                device.replace("SDRDEVICE.", ""),
                k in new_keys,
            )
        )
    rows.sort()
    fig, ax = plt.subplots(figsize=(12, 5))
    for device, marker in (("PLUTO", "o"), ("BLADERF2", "s")):
        xs = [r[0] for r in rows if r[2] == device and not r[3]]
        ys = [r[1] for r in rows if r[2] == device and not r[3]]
        ax.scatter(
            xs, ys, marker=marker, s=42, label=f"{device} (existing)", alpha=0.75
        )
    xs = [r[0] for r in rows if r[3]]
    ys = [r[1] for r in rows if r[3]]
    ax.scatter(
        xs,
        ys,
        marker="*",
        s=260,
        color="tab:red",
        label="new in this rebuild",
        zorder=5,
    )
    ax.axvline(0.5, ls=":", c="grey")
    ax.text(
        0.505,
        ax.get_ylim()[1] * 0.92,
        "d/lambda = 0.5\n(spatially aliased above)",
        fontsize=8,
        color="grey",
    )
    ax.set_yscale("symlog")
    ax.set_xlabel("d / lambda")
    ax.set_ylabel("contributing datasets (symlog)")
    ax.set_title("Table coverage: every key, how much data backs it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fn = os.path.join(out_dir, "coverage.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def fig_vs_baseline(table, baseline, out_dir):
    """How far the rebuilt legacy keys moved from the table they replace."""
    shared = sorted(set(spacing_keys(table)) & set(spacing_keys(baseline)))
    if not shared:
        return None
    prov = table.get(PROVENANCE_KEY, {}).get("keys", {})
    corrs, maxd, ns, labels = [], [], [], []
    for k in shared:
        a, b = arr(baseline[k]["r"]["sym"]).ravel(), arr(table[k]["r"]["sym"]).ravel()
        corrs.append(float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else np.nan)
        maxd.append(float(np.abs(a - b).max()))
        ns.append(prov.get(k, {}).get("n_datasets", 0))
        labels.append(k.replace("SDRDEVICE.", ""))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(ns, corrs, s=40, alpha=0.8)
    for n, c, lab in zip(ns, corrs, labels):
        if c < 0.99:
            ax[0].annotate(
                lab, (n, c), fontsize=7, textcoords="offset points", xytext=(5, -3)
            )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("contributing datasets (log)")
    ax[0].set_ylabel("corr(rebuilt, baseline)")
    ax[0].set_title("Agreement with the previous table\nvs how much data backs the key")
    ax[0].grid(alpha=0.25)
    ax[1].hist(corrs, bins=25)
    ax[1].set_xlabel("corr(rebuilt, baseline)")
    ax[1].set_ylabel("keys")
    ax[1].set_title(
        f"{sum(c > 0.999 for c in corrs)} of {len(corrs)} keys at corr > 0.999"
    )
    fig.tight_layout()
    fn = os.path.join(out_dir, "vs_baseline.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def fig_key_mass(table, new_keys, out_dir):
    """Total histogram mass per key -- the usable (theta, phi) pairs behind it."""
    keys = spacing_keys(table)
    prov = table.get(PROVENANCE_KEY, {}).get("keys", {})
    order = sorted(keys, key=lambda k: prov.get(k, {}).get("n_datasets", 0))
    ns = [prov.get(k, {}).get("n_datasets", 0) for k in order]
    colors = ["tab:red" if k in new_keys else "tab:blue" for k in order]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(order)), ns, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([k.replace("SDRDEVICE.", "") for k in order], fontsize=7)
    ax.set_xscale("symlog")
    ax.set_xlabel("contributing datasets (symlog)")
    ax.set_title(
        "Datasets per key (red = new in this rebuild)\n"
        "note existing keys backed by as few as 1"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fn = os.path.join(out_dir, "datasets_per_key.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--new-keys", nargs="*", default=[])
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    table = load(a.table)
    new_keys = set(a.new_keys)

    written = [
        fig_new_tables(table, sorted(new_keys), a.output_dir),
        fig_coverage(table, new_keys, a.output_dir),
        fig_key_mass(table, new_keys, a.output_dir),
    ]
    if a.baseline:
        written.append(fig_vs_baseline(table, load(a.baseline), a.output_dir))
    for fn in written:
        if fn:
            print("wrote", fn)
