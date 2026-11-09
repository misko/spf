import argparse
import datetime
import json
import logging
import math
import os
import pickle
import platform
import socket
import subprocess
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from spf.dataset.spf_dataset import v5spfdataset
from spf.dataset.phase_corrected_dataset import PhaseCorrectedDataset
from spf.utils import SEGMENTATION_VERSION, rx_spacing_to_str

# Reserved top-level entry in the pickled table. Double underscores keep it
# unmistakable next to the "<DEVICE>_<d/lambda>" spacing keys, and nothing
# iterates the table -- get_empirical_dist (spf_dataset.py) is the sole reader
# and indexes by exact key -- so adding it cannot disturb a consumer.
PROVENANCE_KEY = "__provenance__"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git_info():
    """Commit, branch and dirty flag of the tree that generated a table."""

    def run(*cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    porcelain = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # a dirty tree means the committed code is NOT what produced this file
        "dirty": None if porcelain is None else bool(porcelain),
    }


def _file_md5(fn):
    import hashlib

    try:
        with open(fn, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def _dataset_fingerprint(prefix):
    """Cheap identity for a source dataset.

    Path plus size and mtime of the zarr store, not a content hash: the sources
    run to hundreds of GB and hashing them would dominate the build.
    """
    zarr_fn = str(prefix) if str(prefix).endswith(".zarr") else f"{prefix}.zarr"
    data_mdb = os.path.join(zarr_fn, "data.mdb")
    try:
        st = os.stat(data_mdb if os.path.exists(data_mdb) else zarr_fn)
        return {"path": zarr_fn, "bytes": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"path": zarr_fn, "bytes": None, "mtime": None}


def resolve_precompute_cache(prefix, caches, nthetas):
    """First cache in ``caches`` holding this dataset's segmentation, else None.

    A single rebuild spans corpora living on different volumes -- the historical
    captures under md2, the 2026 rover merges under qnap01 -- and v5spfdataset
    takes exactly one cache path, so the right one has to be chosen per dataset.
    """
    base = os.path.basename(str(prefix)).replace(".zarr", "")
    for cache in caches:
        yarr = os.path.join(cache, f"{base}_segmentation_nthetas{nthetas}.yarr")
        if os.path.isdir(yarr):
            return cache
    return None


def get_heatmap_for_radio(dss, radio_idx, bins):
    ground_truth_thetas = np.hstack([ds.ground_truth_thetas[radio_idx] for ds in dss])
    mean_phase = np.hstack([ds.mean_phase[f"r{radio_idx}"] for ds in dss])
    mask = np.isfinite(mean_phase)
    return np.histogram2d(
        ground_truth_thetas[mask], mean_phase[mask], bins=bins
    )  # heatmap, xedges, yedges


def get_heatmap(dss, bins=50):
    heatmaps = []
    for ridx in [0, 1]:
        heatmaps.append(get_heatmap_for_radio(dss, ridx, bins=bins)[0])
    return (heatmaps[0].copy() + heatmaps[1].copy()) / 2


def create_heatmaps_and_plot(dss, bins, save_fig_to_prefix=None):
    # theta norm is where if you sum over all phi for a specific theta
    # you get back 1.0
    fig_theta_norm, axs_theta_norm = plt.subplots(2, 3, figsize=(15, 10))
    fig_phi_norm, axs_phi_norm = plt.subplots(2, 3, figsize=(15, 10))
    row_idx = 0
    heatmaps = {"r0": {}, "r1": {}, "r": {}}
    eps = 1e-10
    for symmetry in [False, True]:
        r0, _, _ = get_heatmap_for_radio(dss, 0, bins)
        r1, _, _ = get_heatmap_for_radio(dss, 1, bins)
        r = (r0 + r1) / 2
        if symmetry:
            r0 = apply_symmetry_rules_to_heatmap(r0)
            r1 = apply_symmetry_rules_to_heatmap(r1)
            r = apply_symmetry_rules_to_heatmap(r)
        extent = [-torch.pi, torch.pi, -torch.pi, torch.pi]
        # r0,r1,r are matricies of format m[theta][phi]
        # normalizing by dividing by sum of axis=0 (theta)
        # results in r[:,0].sum()==1
        # then taking transpose so r[0].sum()==1 and r[phi][theta]
        r0_phi_norm = (r0 / (r0.sum(axis=0, keepdims=True) + eps)).T
        r1_phi_norm = (r1 / (r1.sum(axis=0, keepdims=True) + eps)).T
        r_phi_norm = (r / (r.sum(axis=0, keepdims=True) + eps)).T

        heatmaps["r0"]["sym" if symmetry else "nosym"] = torch.tensor(r0_phi_norm)
        heatmaps["r1"]["sym" if symmetry else "nosym"] = torch.tensor(r1_phi_norm)
        heatmaps["r"]["sym" if symmetry else "nosym"] = torch.tensor(r_phi_norm)

        # write maps in map[phi][theta] = pr(theta | phi)
        axs_phi_norm[row_idx, 0].imshow(r0_phi_norm, extent=extent)
        axs_phi_norm[row_idx, 0].set_title(f"Radio0,sym={symmetry}")
        axs_phi_norm[row_idx, 1].imshow(r1_phi_norm, extent=extent)
        axs_phi_norm[row_idx, 1].set_title(f"Radio1,sym={symmetry}")
        axs_phi_norm[row_idx, 2].imshow(r_phi_norm, extent=extent)
        axs_phi_norm[row_idx, 2].set_title(f"Radio0+1,sym={symmetry}")
        for _x in range(3):
            axs_phi_norm[row_idx, _x].set_xlabel("Theta (gt)")
            axs_phi_norm[row_idx, _x].set_ylabel("Phase diff (obs)")

        # r0_theta_norm is such that
        # r[0].sum()==1 and r[theta][phi]
        r0_theta_norm = r0 / (r0.sum(axis=1, keepdims=True) + eps)
        r1_theta_norm = r1 / (r1.sum(axis=1, keepdims=True) + eps)
        r_theta_norm = r / (r.sum(axis=1, keepdims=True) + eps)

        # write maps in map[phi][theta] = pr(theta | phi)
        axs_theta_norm[row_idx, 0].imshow(r0_theta_norm.T, extent=extent)
        axs_theta_norm[row_idx, 0].set_title(f"Radio0,sym={symmetry}")
        axs_theta_norm[row_idx, 1].imshow(r1_theta_norm.T, extent=extent)
        axs_theta_norm[row_idx, 1].set_title(f"Radio1,sym={symmetry}")
        axs_theta_norm[row_idx, 2].imshow(r_theta_norm.T, extent=extent)
        axs_theta_norm[row_idx, 2].set_title(f"Radio0+1,sym={symmetry}")
        for _x in range(3):
            axs_theta_norm[row_idx, _x].set_xlabel("Theta (gt)")
            axs_theta_norm[row_idx, _x].set_ylabel("Phase diff (obs)")

        row_idx += 1
    fig_phi_norm.suptitle(f"theta conditional on phi")
    fig_theta_norm.suptitle(f"phi conditional on theta")
    if save_fig_to_prefix is not None:
        fig_phi_norm.savefig(f"{save_fig_to_prefix}_phi_norm.png")
        fig_theta_norm.savefig(f"{save_fig_to_prefix}_theta_norm.png")
    # Close unconditionally: the figures are built either way, so skipping this
    # when not saving leaks two per key -- 88 on a full rebuild, well past the
    # point matplotlib starts warning about open figures.
    plt.close(fig_phi_norm)
    plt.close(fig_theta_norm)
    return heatmaps


def apply_symmetry_rules_to_heatmap(h):
    bins = h.shape[0]
    # h[theta][phi]
    # half is restricting to positive y_rad, -theta -> theta
    # positive theta , phi is same as negative theta, - phi
    half = h[: math.ceil(bins / 2)] + np.flip(h[math.floor(bins // 2) :])
    # pi/2+epsilon is same as pi/2-epsilon
    half = half + np.flip(half, axis=0)
    full = np.vstack([half[:-1], np.flip(half)])
    return full  # / full.sum(axis=1, keepdims=True)


def get_empirical_p_dist_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        help="dataset prefixes",
        nargs="+",
        required=False,
        default=[],
    )
    parser.add_argument(
        "--datasets-from-file",
        type=str,
        default=None,
        help="file with one dataset prefix per line; merged with --datasets. "
        "A full rebuild is ~2500 paths, which is unwieldy on argv.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="empirical-dist.pkl",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite --out if it exists. Off by default so a rebuild cannot "
        "silently replace a committed table such as empirical_dists/full.pkl.",
    )
    parser.add_argument(
        "--max-load-failures",
        type=int,
        default=None,
        help="abort if more than this many datasets fail to load. Unset keeps "
        "the historical behaviour of skipping them, but they are always "
        "recorded in the table's provenance rather than only logged.",
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        help="print the provenance of an existing table and exit",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--nthetas",
        type=int,
        default=65,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--precompute-cache",
        type=str,
        nargs="+",
        required=True,
        help="one or more segmentation caches, searched in order per dataset. "
        "Several are needed when a rebuild spans corpora on different volumes.",
    )
    parser.add_argument(
        "--phase-correction",
        type=str,
        default="none",
        choices=["none", "constant", "arm_lut", "shuffled"],
        help="apply a gain-phase correction to mean_phase while building the table. "
        "A table built with a correction is ONLY valid for inference that applies the "
        "same one; the runner asserts this from __provenance__.",
    )
    parser.add_argument("--phase-model-fn", type=str, required=False, default=None)
    parser.add_argument("--output-fig-prefix", type=str, required=False, default=None)
    return parser


def create_empirical_p_dist(args):
    if args.output_fig_prefix is not None and os.path.dirname(args.output_fig_prefix):
        os.makedirs(os.path.dirname(args.output_fig_prefix), exist_ok=True)

    if getattr(args, "force", False) is False and os.path.exists(args.out):
        raise FileExistsError(
            f"{args.out} exists; refusing to overwrite. Pass --force, or choose a "
            "new name. Empirical tables are referenced by every model config and "
            "by past results, so replacing one in place silently changes history."
        )

    caches = args.precompute_cache
    if isinstance(caches, str):  # tolerate a bare string from programmatic callers
        caches = [caches]

    prefixes = list(args.datasets)
    if getattr(args, "datasets_from_file", None):
        with open(args.datasets_from_file) as f:
            prefixes += [line.strip() for line in f if line.strip()]
    if not prefixes:
        raise ValueError("no datasets given; pass --datasets or --datasets-from-file")

    datasets = []
    loaded_records = []
    failures = []

    for prefix in tqdm(prefixes, total=len(prefixes)):
        cache = resolve_precompute_cache(prefix, caches, args.nthetas)
        if cache is None:
            # Distinguish "no segmentation anywhere" from a load error: it is the
            # difference between missing precompute and a corrupt dataset, and the
            # original code reported both as the same opaque failure.
            failures.append(
                {
                    "prefix": str(prefix),
                    "reason": "no segmentation found in any --precompute-cache",
                }
            )
            logging.error(f"No segmentation for {prefix} in any of {caches}")
            continue
        try:
            ds = v5spfdataset(
                prefix,
                precompute_cache=cache,
                nthetas=args.nthetas,
                skip_fields=set(["signal_matrix"]),
                paired=False,
                ignore_qc=True,
                gpu=args.device == "cuda",
            )
            if args.phase_correction != "none":
                # fails closed outside the model's support, so this touches only
                # the captures the correction actually covers
                ds = PhaseCorrectedDataset(
                    ds, args.phase_correction, args.phase_model_fn
                )
            datasets.append(ds)
            record = _dataset_fingerprint(prefix)
            record["precompute_cache"] = cache
            loaded_records.append(record)
        except Exception as e:
            failures.append(
                {"prefix": str(prefix), "reason": f"{type(e).__name__}: {e}"}
            )
            logging.error(f"Failed to load {prefix} with error {str(e)}")

    if args.max_load_failures is not None and len(failures) > args.max_load_failures:
        raise RuntimeError(
            f"{len(failures)} datasets failed to load, over the "
            f"--max-load-failures limit of {args.max_load_failures}. A table built "
            "from a silently truncated corpus looks identical to a complete one."
        )
    if not datasets:
        raise RuntimeError("no datasets loaded; nothing to build")

    datasets_by_devicetype_and_spacing = {}

    counts = {}

    for dataset in datasets:
        check0 = (
            (
                dataset.cached_keys[0]["rx_wavelength_spacing"]
                == dataset.cached_keys[0]["rx_wavelength_spacing"].median()
            )
            .to(torch.float)
            .mean()
        )
        check1 = (
            (
                dataset.cached_keys[1]["rx_wavelength_spacing"]
                == dataset.cached_keys[1]["rx_wavelength_spacing"].median()
            )
            .to(torch.float)
            .mean()
        )
        if check0 != 1.0 or check1 != 1.0:
            # breakpoint()
            logging.warning(
                f"{dataset.zarr_fn} Failed consistentcy check for rx spacing! {check0} {check1}"
            )

        rx_spacing_str = rx_spacing_to_str(
            dataset.cached_keys[0]["rx_wavelength_spacing"].median()
        )
        assert rx_spacing_str == rx_spacing_to_str(
            dataset.cached_keys[1]["rx_wavelength_spacing"].median()
        ), f'Failed rx_spacing_str check {rx_spacing_str} vs  {rx_spacing_to_str( dataset.cached_keys[1]["rx_wavelength_spacing"].median())}'
        if "0.0000" in rx_spacing_str:
            print(dataset.zarr_fn, rx_spacing_str)
        rx_devicetype_and_spacing_str = (
            f"{dataset.sdr_device_types[0]}_{rx_spacing_str}"
        )
        # assert "0.00000" not in rx_spacing_str
        # breakpoint()
        if rx_devicetype_and_spacing_str not in counts:
            counts[rx_devicetype_and_spacing_str] = {}
        rx_lo_and_spacing = dataset.get_spacing_identifier()
        if rx_lo_and_spacing not in counts[rx_devicetype_and_spacing_str]:
            counts[rx_devicetype_and_spacing_str][rx_lo_and_spacing] = 0
        counts[rx_devicetype_and_spacing_str][rx_lo_and_spacing] += 1

        if rx_devicetype_and_spacing_str not in datasets_by_devicetype_and_spacing:
            datasets_by_devicetype_and_spacing[rx_devicetype_and_spacing_str] = []
        datasets_by_devicetype_and_spacing[rx_devicetype_and_spacing_str].append(
            dataset
        )

    print("Found spacings:", datasets_by_devicetype_and_spacing.keys())
    for rx_devicetype_and_spacing_str in counts:
        print(rx_devicetype_and_spacing_str)
        # breakpoint()
        for rx_lo_and_spacing, count in counts[rx_devicetype_and_spacing_str].items():
            print("\t", rx_lo_and_spacing, count)

    heatmaps = {}
    for (
        rx_devicetype_and_spacing_str,
        _datasets,
    ) in datasets_by_devicetype_and_spacing.items():
        # Only write figures when a prefix was asked for. Previously the f-string
        # was built unconditionally, so an unset --output-fig-prefix still saved
        # ~2 files per key named "None_*.png" into the current directory.
        fig_prefix = None
        if args.output_fig_prefix is not None:
            fig_prefix = (
                f"{args.output_fig_prefix}_rxwavelengthspacing"
                f"{rx_devicetype_and_spacing_str}_nbins{args.nbins}"
            )
        heatmaps[rx_devicetype_and_spacing_str] = create_heatmaps_and_plot(
            _datasets,
            args.nbins,
            save_fig_to_prefix=fig_prefix,
        )

    assert (
        PROVENANCE_KEY not in heatmaps
    ), f"a spacing key collided with the reserved {PROVENANCE_KEY} entry"
    heatmaps[PROVENANCE_KEY] = build_provenance(
        args, caches, prefixes, loaded_records, failures, counts, sorted(heatmaps)
    )

    with open(args.out, "wb") as f:
        pickle.dump(heatmaps, f)
    logging.info(
        f"wrote {args.out}: {len(heatmaps) - 1} keys from {len(datasets)} datasets "
        f"({len(failures)} unusable)"
    )
    return heatmaps


def build_provenance(args, caches, requested, loaded, failures, counts, keys):
    """Everything needed to explain, audit, or regenerate a table.

    Embedded in the table itself rather than kept alongside it: a pickle that
    travels without its README is a pickle nobody can safely reuse.
    """
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git": _git_info(),
        # The commit alone does not pin the code when the tree is dirty -- and it
        # often is, on a machine several sessions share. Hash the generator too,
        # so the exact source that produced a table is identifiable regardless.
        "generator_md5": _file_md5(os.path.abspath(__file__)),
        "segmentation_version": SEGMENTATION_VERSION,
        "params": {
            "nbins": args.nbins,
            "nthetas": args.nthetas,
            "device": args.device,
            "precompute_caches": list(caches),
            "out": args.out,
            "phase_correction": args.phase_correction,
            "phase_model_fn": args.phase_model_fn,
        },
        "datasets": {
            "requested": len(requested),
            "loaded": len(loaded),
            "failed": len(failures),
            "records": loaded,
            "failures": failures,
        },
        "keys": {
            key: {
                "n_datasets": sum(counts.get(key, {}).values()),
                "by_lo_and_spacing": dict(counts.get(key, {})),
            }
            for key in keys
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "platform": platform.platform(),
        },
    }


def load_provenance(pkl_fn):
    """Provenance of an existing table, or None for one built before this existed."""
    with open(pkl_fn, "rb") as f:
        return pickle.load(f).get(PROVENANCE_KEY)


if __name__ == "__main__":

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    parser = get_empirical_p_dist_parser()
    args = parser.parse_args()
    if args.show is not None:
        prov = load_provenance(args.show)
        if prov is None:
            print(
                f"{args.show} carries no {PROVENANCE_KEY}; it predates provenance "
                "recording, so its inputs are not recoverable from the file."
            )
        else:
            print(json.dumps(prov, indent=2, default=str))
        sys.exit(0)
    create_empirical_p_dist(args)
