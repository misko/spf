"""The blocker that outranks coefficients: is there a usable equal-gain anchor?

Also a full LO census, because a share quoted without its denominator is not a
measurement. READ-ONLY throughout.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SPF_REPO",
                           Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(REPO))

from spf.calibrations.gain_state_phase_model_v1.gain_tables import (  # noqa: E402
    default_tables,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store  # noqa: E402

MERGED = Path("/mnt/qnap01/mouse9911/rovers_2026/merged")


def main(out_path: str):
    tab = default_tables()
    lo_census_dedup: Counter = Counter()
    lo_census_raw: Counter = Counter()
    # (dedup?) -> carrier -> per-stream equal-gain frame counts
    eq_per_stream: dict[str, dict[int, list]] = {
        "dedup": defaultdict(list), "raw": defaultdict(list)}
    eq_total = {"dedup": Counter(), "raw": Counter()}
    n_total = {"dedup": Counter(), "raw": Counter()}
    eq_gain_hist: dict[int, Counter] = defaultdict(Counter)
    caps_with_eq = {"dedup": set(), "raw": set()}
    caps_all = {"dedup": set(), "raw": set()}
    # rule-5: RF words equal on both arms
    rf_equal = Counter()
    seen: set[str] = set()

    for zp in sorted(MERGED.glob("*.zarr")):
        rx_prefix = zp.name.split(".")[0]
        modes = ["raw"] + ([] if rx_prefix in seen else ["dedup"])
        seen.add(rx_prefix)
        z = zarr_open_from_lmdb_store(str(zp), mode="r")
        for r in list(z["receivers"].keys()):
            rr = z[f"receivers/{r}"]
            lo = np.round(np.asarray(rr["rx_lo"][:], dtype=float)).astype(np.int64)
            gains = np.asarray(rr["gains"][:], dtype=float)
            ok = np.isfinite(gains).all(axis=1)
            gi = np.round(gains).astype(np.int64)
            for mode in modes:
                for c, n in Counter(lo.tolist()).items():
                    (lo_census_dedup if mode == "dedup" else lo_census_raw)[c] += n
                for carrier in np.unique(lo):
                    m = (lo == carrier) & ok
                    if not m.any():
                        continue
                    gg = gi[m]
                    eq = gg[:, 0] == gg[:, 1]
                    n_total[mode][int(carrier)] += int(m.sum())
                    eq_total[mode][int(carrier)] += int(eq.sum())
                    eq_per_stream[mode][int(carrier)].append(int(eq.sum()))
                    caps_all[mode].add(zp.name)
                    if eq.any():
                        caps_with_eq[mode].add(zp.name)
                    if mode == "dedup":
                        for g in gg[eq][:, 0]:
                            eq_gain_hist[int(carrier)][int(g)] += 1
                        band = "high"
                        s1 = [tab.state(band, int(a)) for a in gg[:, 0]]
                        s2 = [tab.state(band, int(b)) for b in gg[:, 1]]
                        rf_equal[int(carrier)] += sum(
                            1 for a, b in zip(s1, s2)
                            if a is not None and b is not None
                            and a.rf_words == b.rf_words
                        )

    out = {
        "lo_census_dedup": {str(k): int(v) for k, v in
                            sorted(lo_census_dedup.items())},
        "lo_census_raw": {str(k): int(v) for k, v in sorted(lo_census_raw.items())},
        "carrier_share_dedup": {
            str(k): v / sum(lo_census_dedup.values())
            for k, v in sorted(lo_census_dedup.items())
        },
        "equal_gain": {},
        "captures_with_any_equal_gain": {
            "dedup": f"{len(caps_with_eq['dedup'])} of {len(caps_all['dedup'])}",
            "raw": f"{len(caps_with_eq['raw'])} of {len(caps_all['raw'])}",
        },
        "rf_words_equal_fraction_dedup": {
            str(c): rf_equal[c] / n_total["dedup"][c]
            for c in sorted(n_total["dedup"]) if n_total["dedup"][c]
        },
    }
    for mode in ("dedup", "raw"):
        out["equal_gain"][mode] = {}
        for c in sorted(n_total[mode]):
            per = np.array(eq_per_stream[mode][c])
            out["equal_gain"][mode][str(c)] = {
                "n_pair_observations": int(n_total[mode][c]),
                "n_exact_equal": int(eq_total[mode][c]),
                "fraction": eq_total[mode][c] / n_total[mode][c],
                "n_streams": int(len(per)),
                "streams_with_any": int((per > 0).sum()),
                "median_equal_frames_per_stream": float(np.median(per)),
                "max_equal_frames_per_stream": int(per.max()) if len(per) else 0,
            }
        tot_n = sum(n_total[mode].values())
        tot_eq = sum(eq_total[mode].values())
        out["equal_gain"][mode]["corpus_wide_fraction"] = tot_eq / tot_n
    out["equal_gain_gain_histogram_dedup"] = {
        str(c): dict(sorted(h.items())) for c, h in eq_gain_hist.items()
    }
    Path(out_path).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k != "equal_gain_gain_histogram_dedup"}, indent=1)[:4000])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
