"""What produced each merged dataset -- resolved from the corpus, not assumed.

Every merged store is named ``<rx_name>.<tx_name>.zarr``, so its sources are
recoverable by name even when the store itself records nothing about them. This
walks a set of campaign directories, indexes every source capture, resolves both
sources for every merged dataset, and reports what the merged store cannot say
about itself: which physical Pluto was r0, and whether the capture behind it
finalised.

That second question is not hypothetical. The merged filename drops a source's
``.tmp`` suffix, so a dataset built from an unfinalised capture is indistinguishable
by name from one built from a clean one. Running this over july+august 2026 found
five of 24 in exactly that state -- three ``in_progress``, two ``incomplete``.

Reads attrs only; no signal_matrix is touched, so it runs in seconds over a
184 GB corpus.

Usage:
  python -m spf.scripts.merged_provenance_inventory \\
      --campaigns /mnt/qnap01/mouse9911/rovers_july_2026 \\
                  /mnt/qnap01/mouse9911/rovers_august_2026 \\
      [--json out.json] [--fit-projection]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from spf.dataset.provenance import is_finalized, store_suffix


def index_sources(campaign_dirs):
    """basename (normalised to .zarr) -> path, for every non-merged capture."""
    index = {}
    for root in campaign_dirs:
        for pattern in ("**/*.zarr", "**/*.zarr.tmp"):
            for path in glob.glob(os.path.join(root, pattern), recursive=True):
                base = os.path.basename(path.rstrip("/"))
                if ".rover_" in base:
                    continue  # that is a merged store, not a source
                index.setdefault(base.replace(".zarr.tmp", ".zarr"), path)
    return index


def split_merged_name(basename):
    """``<rx>.zarr.<tx>.zarr`` -> ``(rx.zarr, tx.zarr)``, or ``None``."""
    parts = basename.rstrip("/").split(".rover_")
    if len(parts) != 2:
        return None
    return parts[0] + ".zarr", "rover_" + parts[1]


def _fit_projection_center(z):
    """Recover the aeqd centre from the RX's raw and projected positions.

    The store holds gps_lat/gps_long AND rx_pos_x_mm/rx_pos_y_mm, which
    over-determines the projection -- so the centre a merge derived from the mean
    TX GPS is recoverable even when it was never recorded. Verified to 0.0000 mm
    RMS on a real dataset.
    """
    import numpy as np
    from pyproj import Proj
    from scipy.optimize import least_squares

    lat = z["receivers/r0/gps_lat"][:]
    lon = z["receivers/r0/gps_long"][:]
    px = z["receivers/r0/rx_pos_x_mm"][:]
    py = z["receivers/r0/rx_pos_y_mm"][:]
    ok = (lat != 0) & (lon != 0) & np.isfinite(px) & np.isfinite(py)
    if ok.sum() < 8:
        return None
    lat, lon, px, py = lat[ok][:400], lon[ok][:400], px[ok][:400], py[ok][:400]

    def resid(c):
        proj = Proj(proj="aeqd", lat_0=c[0], lon_0=c[1], units="m")
        x, y = proj(lon, lat)
        return np.concatenate([x * 1000 - px, y * 1000 - py])

    sol = least_squares(resid, x0=[lat.mean(), lon.mean()], xtol=1e-14, ftol=1e-14)
    return {
        "lat_0": float(sol.x[0]),
        "lon_0": float(sol.x[1]),
        "residual_rms_mm": float((sol.fun**2).mean() ** 0.5),
    }


def build(campaign_dirs, merged_dirs, fit_projection=False):
    from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

    index = index_sources(campaign_dirs)
    rows = []
    for mdir in merged_dirs:
        for store in sorted(glob.glob(os.path.join(mdir, "*.zarr"))):
            base = os.path.basename(store)
            names = split_merged_name(base)
            row = {
                "merged": base,
                "merged_dir": mdir,
                "rx": None,
                "tx": None,
                "unresolved": [],
            }
            if names is None:
                row["unresolved"].append("could not split merged name")
                rows.append(row)
                continue
            for role, name in zip(("rx", "tx"), names):
                path = index.get(name)
                if path is None:
                    row["unresolved"].append(f"{role}: {name}")
                    continue
                entry = {
                    "store": name,
                    "path": path,
                    "suffix": store_suffix(path),
                    "finalized": is_finalized(path),
                }
                try:
                    zs = zarr_open_from_lmdb_store(path, mode="r")
                    root = dict(zs.attrs)
                    entry["capture_status"] = root.get("capture_status")
                    entry["root_attrs"] = root
                    if role == "rx":
                        entry["receivers"] = {
                            r: dict(zs[f"receivers/{r}"].attrs)
                            for r in ("r0", "r1")
                            if f"receivers/{r}" in zs
                        }
                        entry["serials"] = {
                            r: a.get("sdr_serial")
                            for r, a in entry["receivers"].items()
                        }
                except Exception as error:  # noqa: BLE001 - report, do not abort
                    entry["read_error"] = f"{type(error).__name__}: {error}"
                row[role] = entry
            if fit_projection:
                try:
                    row["projection_fit"] = _fit_projection_center(
                        zarr_open_from_lmdb_store(store, mode="r")
                    )
                except Exception as error:  # noqa: BLE001
                    row["projection_fit"] = {"error": str(error)}
            rows.append(row)
    return rows


def render(rows):
    print(f"\n  {len(rows)} merged dataset(s)\n")
    header = f"  {'merged (RX part)':<48}{'RX status':<13}{'final':<9}{'r0 serial':<13}{'TX final'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    suspect, unresolved = [], []
    for row in rows:
        rx, tx = row.get("rx") or {}, row.get("tx") or {}
        name = (rx.get("store") or row["merged"])[6:50]
        status = str(rx.get("capture_status", "?"))
        fin = "yes" if rx.get("finalized") else "NO .tmp"
        serial = str((rx.get("serials") or {}).get("r0") or "-")[-10:]
        txfin = "yes" if tx.get("finalized") else ("NO .tmp" if tx else "?")
        if row["unresolved"]:
            unresolved.append(row)
        if not rx.get("finalized", True) or status not in ("complete", "?"):
            suspect.append((name, status, rx.get("suffix")))
        print(f"  {name:<48}{status:<13}{fin:<9}{serial:<13}{txfin}")

    print()
    if unresolved:
        print(f"  {len(unresolved)} dataset(s) with unresolved sources:")
        for row in unresolved:
            for item in row["unresolved"]:
                print(f"    {row['merged'][:70]}  ->  {item}")
    if suspect:
        print(f"  WARN: {len(suspect)} dataset(s) built from a capture that never "
              "finalised.")
        print("    The merged filename drops the .tmp, so these are indistinguishable")
        print("    by name from clean ones:")
        for name, status, suffix in suspect:
            print(f"      {name}  capture_status={status}  source{suffix}")
    else:
        print("  PASS: every merged dataset came from a finalised capture.")
    return 1 if (unresolved or suspect) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaigns", nargs="+", required=True,
                        help="campaign roots to index sources from")
    parser.add_argument("--merged-dirs", nargs="+", default=None,
                        help="merged dirs to inventory (default: */merged* under campaigns)")
    parser.add_argument("--json", default=None, help="write the full record here")
    parser.add_argument("--fit-projection", action="store_true",
                        help="recover each store's aeqd centre (slower)")
    args = parser.parse_args(argv)

    merged_dirs = args.merged_dirs
    if merged_dirs is None:
        merged_dirs = sorted(
            d
            for root in args.campaigns
            for d in glob.glob(os.path.join(root, "merged*"))
            if os.path.isdir(d)
        )
    rows = build(args.campaigns, merged_dirs, fit_projection=args.fit_projection)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True, default=str)
        print(f"  wrote {args.json}")
    return render(rows)


if __name__ == "__main__":
    sys.exit(main())
