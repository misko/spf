"""Per-dataset trajectory figures, replayed from dumped tracks.

Bearing against time for one capture: ground truth as a black line, each
approach as a coloured line, its +-1 sigma as a matching fill, and a metrics
table carrying both random-prediction floors.

``plot_trajectory_comparison.py`` makes the same figure by RUNNING the filters,
which costs a dataset open plus seven filter runs per capture. This replays the
``.npz`` tracks written by ``dump_tracks.py`` instead, so the whole corpus is
seconds rather than an hour, and every figure is exactly the run that produced
the committed numbers rather than a fresh draw.

Usage::

    python spf/filters/plot_tracks.py \\
        --tracks-dir <tracks>/ --output-dir <figs>/ [--datasets NAME ...]
"""

import argparse
import collections
import glob
import logging
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from spf.filters.plot_trajectory_comparison import plot_comparison  # noqa: E402


def label_for(_type, rx_idx):
    """Match the live plotter's labels so the two figures are comparable."""
    label = _type.replace("_single_theta", "").replace("_", " ")
    if _type.startswith("PF_single_theta_single_radio") or _type.startswith(
        "EKF_single_theta_single_radio"
    ):
        label += f" rx{rx_idx}"
    return label


def load_by_dataset(tracks_dir):
    """{ds_name: [(label, theta, sigma, gt, frame), ...]} from the npz tracks."""
    out = collections.defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(tracks_dir, "*.npz"))):
        with np.load(fn) as z:
            ds_name = str(z["ds_fn"])
            _type, frame = str(z["type"]), str(z["frame"])
            label = label_for(_type, int(z["rx_idx"]))
            if str(z["seed"]) != "0":
                label += f" seed{z['seed']}"
            out[ds_name].append(
                (label, z["theta"], z["sigma"], z["gt"], frame)
            )
    return out


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help="dataset names (basenames) to plot; default every dataset present",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="plot at most this many datasets")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    by_ds = load_by_dataset(args.tracks_dir)
    if not by_ds:
        raise SystemExit(f"no .npz tracks under {args.tracks_dir}")
    names = sorted(by_ds)
    if args.datasets:
        wanted = set(args.datasets)
        names = [n for n in names if n in wanted or os.path.basename(n) in wanted]
        missing = wanted - set(names) - {os.path.basename(n) for n in names}
        if missing:
            raise SystemExit(f"no tracks for: {sorted(missing)}")
    if args.limit:
        names = names[: args.limit]

    written = []
    for name in names:
        runs = by_ds[name]
        n_steps = len(runs[0][3])
        fig = plot_comparison(
            runs,
            f"Filter comparison — {name}\n"
            f"{n_steps} timesteps · {len(runs)} approaches · replayed from tracks",
            max_steps=args.max_steps,
        )
        out = os.path.join(args.output_dir, f"{name}__comparison.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)
        logging.info(f"wrote {out}")

    print(f"wrote {len(written)} figures to {args.output_dir}")
