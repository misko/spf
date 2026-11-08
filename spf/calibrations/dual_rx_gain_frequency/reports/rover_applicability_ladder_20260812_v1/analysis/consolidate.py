"""Merge every measurement into the report's analysis.json, and add the
per-arm hardware-state demand shares. Read-only over /mnt."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from spf.calibrations.gain_state_phase_model_v1.gain_tables import (
    GainTables, band_for_lo)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

SCRATCH = Path(sys.argv[1])
REPORT = Path(sys.argv[2])
gt = GainTables()

out = json.loads((SCRATCH / "out" / "analysis.json").read_text())
fold_real = json.loads((SCRATCH / "fold_real.json").read_text())
gee = json.loads((SCRATCH / "gee_check.json").read_text())

# ---- per-arm level demand
MERGED = Path("/mnt/qnap01/mouse9911/rovers_2026/merged")
by_rx = {}
for p in sorted(MERGED.glob("*.zarr")):
    by_rx.setdefault(p.name[: p.name.index(".rover_")], []).append(p)
G, LO = [], []
for _rx, paths in sorted(by_rx.items()):
    z = zarr_open_from_lmdb_store(str(sorted(paths)[0]), mode="r")
    for r in sorted(z["receivers"].keys()):
        G.append(np.asarray(z["receivers"][r]["gains"][:]))
        LO.append(np.asarray(z["receivers"][r]["rx_lo"][:]))
G = np.concatenate(G)
LO = np.concatenate(LO)

for lo_s, entry in out["rover_corpus"]["per_lo"].items():
    lo = float(lo_s)
    band = band_for_lo(lo)
    g = G[LO == lo]
    arms = np.concatenate([g[:, 0], g[:, 1]])
    dec = {gi: gt.state(band, gi) for gi in {int(round(x)) for x in np.unique(arms)}}
    for field in ("mixer", "lna", "tia", "lpf"):
        c = Counter(getattr(dec[int(round(x))], field) for x in arms)
        entry[f"{field}_arm_share"] = {
            str(k): v / len(arms) for k, v in sorted(c.items())
        }
    entry["gain_db_arm_share"] = {
        str(int(k)): v / len(arms)
        for k, v in sorted(Counter(int(round(x)) for x in arms).items())
    }

# ---- E-GSC7 per-run steps
a = json.loads(subprocess.run(
    ["git", "-C", "/home/mouse9911/gits/spf", "show",
     "main:spf/calibrations/dual_rx_gain_frequency/reports/"
     "e_gsc7_iio_20260812_v1/analysis.json"],
    capture_output=True, text=True, check=True).stdout)
out["e_gsc7"]["steps_by_run"] = {
    f"{e['radio']}_{e['transport']}": e["steps_deg"] for e in a["results_5766"]
}
out["e_gsc7"]["h2_targets"] = {
    f"{e['radio']}_{e['transport']}": dict(step_sum_deg=e["step_sum_deg"],
                                           h2_error_deg=e["h2_error_deg"])
    for e in a["results_5766"]
}

out["segmentation_fold_measured"] = fold_real
out["rover_corpus"]["gain_stability_definitions"] = gee
out["ladder"] = json.loads((SCRATCH / "ladder_rebuilt.json").read_text())

REPORT.mkdir(parents=True, exist_ok=True)
(REPORT / "analysis.json").write_text(json.dumps(out, indent=1, sort_keys=True))
(REPORT / "ladder_rebuilt.json").write_text(
    (SCRATCH / "ladder_rebuilt.json").read_text())
print("wrote", REPORT / "analysis.json")
for lo, e in out["rover_corpus"]["per_lo"].items():
    print(lo, "mixer arm share:",
          {k: round(v, 4) for k, v in e["mixer_arm_share"].items()})
    print("    lna arm share:",
          {k: round(v, 4) for k, v in e["lna_arm_share"].items()})
