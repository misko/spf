"""Cross-check the within-buffer gain-stability statistic under every available
definition. Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

MERGED = Path("/mnt/qnap01/mouse9911/rovers_2026/merged")
by_rx = {}
for p in sorted(MERGED.glob("*.zarr")):
    i = p.name.index(".rover_")
    by_rx.setdefault(p.name[:i], []).append(p)

acc = {}
for rxcap, paths in sorted(by_rx.items()):
    z = zarr_open_from_lmdb_store(str(sorted(paths)[0]), mode="r")
    for r in sorted(z["receivers"].keys()):
        g = z["receivers"][r]
        lo = np.asarray(g["rx_lo"][:])
        gee = np.asarray(g["gain_endpoints_equal"][:])
        fgc = np.asarray(g["first_gain_change_sample"][:])
        flags = np.asarray(g["gain_metadata_flags"][:])
        for k, v in (("lo", lo), ("gee", gee), ("fgc", fgc), ("flags", flags)):
            acc.setdefault(k, []).append(v)
LO = np.concatenate(acc["lo"])
GEE = np.concatenate(acc["gee"])
FGC = np.concatenate(acc["fgc"])
FL = np.concatenate(acc["flags"])

out = {}
for label, m in (("all", np.ones_like(LO, bool)),
                 ("5766", LO == 5766000000.0),
                 ("5840", LO == 5840000000.0)):
    gee, fgc, fl = GEE[m], FGC[m], FL[m]
    out[label] = dict(
        n=int(m.sum()),
        endpoints_unequal_both_arms=float(1 - gee.all(axis=1).mean()),
        endpoints_unequal_elementwise=float((~gee).mean()),
        endpoints_unequal_arm0=float(1 - gee[:, 0].mean()),
        endpoints_unequal_arm1=float(1 - gee[:, 1].mean()),
        any_gain_change_observed_either_arm=float((fgc >= 0).any(axis=1).mean()),
        any_gain_change_observed_elementwise=float((fgc >= 0).mean()),
        gain_change_arm0=float((fgc[:, 0] >= 0).mean()),
        gain_change_arm1=float((fgc[:, 1] >= 0).mean()),
        flag_histogram={str(int(k)): int(v)
                        for k, v in Counter(fl.tolist()).most_common(10)},
    )
print(json.dumps(out, indent=1))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
