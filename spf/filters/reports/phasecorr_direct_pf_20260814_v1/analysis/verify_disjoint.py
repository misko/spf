"""Verify defect (a): are the merged stores sharing an RX prefix disjoint in time?

Read-only. Opens each merged zarr with the standard read-only LMDB path and reads
only ``receivers/r*/system_timestamp``. Writes nothing anywhere.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store  # noqa: E402


def rx_capture_id(store_path: str) -> str:
    """The physical RX recording a merged store ``<rx>.<tx>.zarr`` was cut from."""
    return store_path.split("/")[-1].split(".")[0]


def timestamps(store_path, ridx):
    z = zarr_open_from_lmdb_store(store_path, mode="r")
    try:
        t = np.asarray(z[f"receivers/r{ridx}/system_timestamp"][:], dtype=float)
    finally:
        z.store.close()
    return t[np.isfinite(t) & (t > 0)]


def main(list_fn):
    fns = [line.strip() for line in open(list_fn) if line.strip()]
    groups = defaultdict(list)
    for f in fns:
        groups[rx_capture_id(f)].append(f)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"{len(fns)} merged stores -> {len(groups)} physical RX captures; "
          f"{len(multi)} captures appear in more than one store\n")

    total_extra = 0
    for k, v in sorted(multi.items()):
        print(k)
        for ridx in (0, 1):
            ts = [timestamps(f, ridx) for f in v]
            spans = [f"[{t.min():.1f},{t.max():.1f}]" for t in ts]
            sets = [set(np.round(t, 6).tolist()) for t in ts]
            inter = set.intersection(*sets)
            union = set.union(*sets)
            first = len(sets[0])
            print(f"  r{ridx}: n={[len(t) for t in ts]} spans={spans} "
                  f"exact-ts overlap={len(inter)} union={len(union)} "
                  f"gained_over_keep_first={len(union) - first}")
            if ridx == 0:
                total_extra += len(union) - first
        # time-range overlap in seconds between the two intervals
        for ridx in (0,):
            ts = [timestamps(f, ridx) for f in v]
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    lo = max(ts[i].min(), ts[j].min())
                    hi = min(ts[i].max(), ts[j].max())
                    print(f"      pair({i},{j}) interval overlap = "
                          f"{max(0.0, hi - lo):.1f} s")
        print()
    print(f"raw rows gained on r0 by unioning instead of keep-first: {total_extra}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/e_inf1_filter_sweep/stage3_rover_all_n48.txt")
