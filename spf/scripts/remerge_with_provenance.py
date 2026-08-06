"""Recreate merged datasets so they carry their own provenance.

Datasets merged before 2026-08-06 record nothing about the radios that made
them: ``sdr_serial`` was dropped with the other 30 receiver attrs, and
``capture_status`` with two of the three root attrs. Their provenance can be
reconstructed after the fact -- ``merged_provenance_inventory.py`` does exactly
that, by splitting ``<rx>.<tx>.zarr`` and looking the sources back up -- but that
is archaeology. It works only while the sources still sit beside the output, and
it asks every future reader to redo it.

This re-runs the merges with the writer in place, so the answer is in the store.

TWO WAYS TO CHOOSE PAIRS
------------------------
``--from-inventory`` reproduces the pairings of an existing merged directory,
reading them out of the inventory JSON. The inventory is the recipe for the
re-merge and is not needed afterwards.

``--from-overlap`` derives pairings for a day that was never merged, from GPS
time overlap. Never from filenames: on 2026-08-05 four of twelve captures were
named from a clock up to 4h28m slow, because a Pi has no battery-backed RTC and
a rover booted in the field restores the time recorded at its last shutdown.
``rover_..._18_32_23..._RO1`` reads as an afternoon solo run with no emitter; it
is really the 23:00 session with RO2 transmitting throughout. ``gps_timestamp``
comes from the GPS receiver and is true UTC regardless of what the Pi believed.

Each RX is assigned to the TX it overlaps MOST, not to every TX it overlaps at
all. Sessions abut closely enough that overlap-based pairing alone also emits
partial cross-session merges -- near-duplicates, which are harder to notice than
missing ones.

FILENAMES ARE PRESERVED, DELIBERATELY
-------------------------------------
The precompute cache is keyed by the merged dataset's BASENAME
(``<name>_segmentation_nthetas65.yarr``), not its path. Re-merging changes no
array -- only attrs -- so a cached segmentation stays valid for a recreated store
of the same name. Preserving names is what keeps ~3 GB of precompute from having
to be regenerated.

Resumable: an output that already exists is skipped, so a run interrupted after
six hours continues rather than restarting.

Usage:
  python -m spf.scripts.remerge_with_provenance --from-inventory inv.json \\
      --output-suffix _prov1 [--dry-run]
  python -m spf.scripts.remerge_with_provenance --from-overlap /path/aug5 \\
      --output /path/merged_aug5 [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

MIN_OVERLAP_S = 300  # ~500 timesteps at the ~1.7 Hz these captures run at


def _base(path):
    return os.path.basename(path.rstrip("/")).replace(".zarr.tmp", "").replace(".zarr", "")


def output_for(out_dir, rx_path, tx_path):
    return os.path.join(out_dir, f"{_base(rx_path)}.{_base(tx_path)}.zarr")


# ------------------------------------------------------------- pair sources ---


def pairs_from_inventory(inventory_path, output_suffix):
    """Reproduce the pairings of already-merged directories."""
    with open(inventory_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    jobs = []
    for row in rows:
        rx, tx = row.get("rx"), row.get("tx")
        if not rx or not tx or row.get("unresolved"):
            print(f"  SKIP (unresolved sources): {row['merged'][:70]}")
            continue
        out_dir = row["merged_dir"].rstrip("/") + output_suffix
        jobs.append({
            "rx": rx["path"], "tx": tx["path"], "out_dir": out_dir,
            "note": f"rx {rx['suffix']} status={rx.get('capture_status')}",
        })
    return jobs


def _gps_window(path):
    from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

    z = zarr_open_from_lmdb_store(path, mode="r")
    t = z["receivers/r0/gps_timestamp"][:]
    t = t[t > 0]
    if len(t) == 0:
        return None
    return float(t.min()), float(t.max())


def pairs_from_overlap(capture_dir, out_dir, tx_tag="RO2"):
    """Derive TX/RX pairings from GPS time overlap -- never from filenames."""
    stores = sorted(glob.glob(os.path.join(capture_dir, "*.zarr")))
    txs, rxs = [], []
    for path in stores:
        window = _gps_window(path)
        if window is None:
            print(f"  SKIP (no GPS): {os.path.basename(path)[:70]}")
            continue
        (txs if f"tag_{tx_tag}" in path else rxs).append((path, window))
    print(f"  {len(txs)} TX, {len(rxs)} RX with usable GPS")

    jobs = []
    for rx_path, (ra, rb) in rxs:
        best, best_overlap = None, 0.0
        for tx_path, (ta, tb) in txs:
            overlap = min(tb, rb) - max(ta, ra)
            if overlap > best_overlap:
                best, best_overlap = tx_path, overlap
        if best is None or best_overlap < MIN_OVERLAP_S:
            print(f"  SKIP (no TX overlaps >= {MIN_OVERLAP_S}s): "
                  f"{os.path.basename(rx_path)[:60]}")
            continue
        jobs.append({
            "rx": rx_path, "tx": best, "out_dir": out_dir,
            "note": f"overlap {best_overlap/60:.1f} min",
        })
    return jobs


# ------------------------------------------------------------------- runner ---


def run(jobs, dry_run=False, extra_args=()):
    total = len(jobs)
    done = skipped = failed = 0
    for i, job in enumerate(jobs, 1):
        out = output_for(job["out_dir"], job["rx"], job["tx"])
        label = os.path.basename(out)[:72]
        if os.path.exists(out):
            print(f"  [{i}/{total}] SKIP exists  {label}")
            skipped += 1
            continue
        print(f"  [{i}/{total}] MERGE        {label}")
        print(f"              rx={os.path.basename(job['rx'])[:60]}")
        print(f"              tx={os.path.basename(job['tx'])[:60]}  ({job['note']})")
        if dry_run:
            continue
        os.makedirs(job["out_dir"], exist_ok=True)
        started = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "spf.scripts.v7_tx_rx_merge",
             "--txs", job["tx"], "--rxs", job["rx"],
             "--output", job["out_dir"], *extra_args],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(out):
            failed += 1
            print(f"              FAILED rc={result.returncode}")
            print("              " + (result.stderr or "")[-600:].replace("\n", "\n              "))
        else:
            done += 1
            print(f"              ok in {(time.time()-started)/60:.1f} min")
    print(f"\n  merged={done}  skipped={skipped}  failed={failed}  of {total}")
    return 1 if failed else 0


def verify(jobs):
    """Every recreated store must answer the questions it was recreated for."""
    from spf.dataset.provenance import load_provenance, radio_identity

    bad = 0
    for job in jobs:
        out = output_for(job["out_dir"], job["rx"], job["tx"])
        if not os.path.exists(out):
            continue
        prov = load_provenance(out)
        identity = radio_identity(prov)
        root = (prov or {}).get("root") or {}
        ok = bool(identity) and "projection" in root and "rx_source" in root
        if not ok:
            bad += 1
        print(f"  {'ok ' if ok else 'BAD'} {os.path.basename(out)[:60]:<62}"
              f"r0={str(identity.get('r0'))[-10:]}  "
              f"rx_final={root.get('rx_source', {}).get('finalized')}")
    print(f"\n  {'PASS' if not bad else f'FAIL: {bad} store(s) without provenance'}")
    return 1 if bad else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-inventory", help="inventory JSON to reproduce pairings from")
    source.add_argument("--from-overlap", help="capture dir to pair by GPS overlap")
    parser.add_argument("--output-suffix", default="_prov1",
                        help="appended to each source merged dir (--from-inventory)")
    parser.add_argument("--output", help="output dir (--from-overlap)")
    parser.add_argument("--tx-tag", default="RO2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args, extra = parser.parse_known_args(argv)

    if args.from_inventory:
        jobs = pairs_from_inventory(args.from_inventory, args.output_suffix)
    else:
        if not args.output:
            parser.error("--from-overlap requires --output")
        jobs = pairs_from_overlap(args.from_overlap, args.output, args.tx_tag)

    print(f"\n  {len(jobs)} merge job(s)\n")
    if args.verify_only:
        return verify(jobs)
    status = run(jobs, dry_run=args.dry_run, extra_args=extra)
    if not args.dry_run:
        print("\n  verifying provenance:")
        status |= verify(jobs)
    return status


if __name__ == "__main__":
    sys.exit(main())
