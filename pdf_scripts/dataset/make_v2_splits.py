"""Generate v2scan split manifests from the dataset-quality scan (scan_v=2).

Additive only: never modifies existing split files. Outputs new txt manifests +
MANIFEST.json (sha256 + provenance) into --out-dir (default /mnt/md2/splits/v2scan).

Rules (see claude_docs/03_datasets/data_quality_plan.md and
claude_docs/04_training_inference/val_expansion_plan.md):

Val subsets (must be subsets of the frozen val list, never edited):
  val_clean_v2    = val ∩ status==OK
  val_degraded_v2 = val ∩ (status==QUARANTINE ∪ noisy-FLAG)
  val_915_v2      = val ∩ rx_lo ∈ [0.90, 0.95] GHz

Train regimes (from the frozen train list):
  train_r1_labelclean = train
                        − status==ERROR − not-in-scan (ghosts)
                        − duplicate lines (kept-first; dupes may be intentional
                          5x-style upweighting — logged, see lab log Mar 12 2025)
                        − train∩val leak
                        − frozen_tail (stale-position tails)
                        − |mount_diff| >= MOUNT_THRESH (miswired/rotated array)
                        − rover 2025_04_05 bounce (interference session; postdates
                          bounce_rover.txt exclusion list)
  train_r2_nodegraded = r1 − nan>20% QUAR − noisy-FLAG
"""

import argparse
import hashlib
import json
import os
from collections import OrderedDict

import pandas as pd

MOUNT_THRESH = 0.5
# "915" stratum = sub-GHz ISM (868 MHz + 915 MHz captures)
BAND_915 = (0.5e9, 1.0e9)


def norm(p):
    return os.path.basename(str(p).strip()).replace(".zarr", "")


def sha256_file(fn):
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(fn):
    with open(fn) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        default="data_quality_reports/scan_2026_07_12_v2/metrics.csv",
    )
    ap.add_argument(
        "--train-txt",
        default="/mnt/md2/splits/apr17_train_nosig_noroverbounce_noblade.txt",
    )
    ap.add_argument(
        "--val-txt", default="/mnt/md2/splits/apr17_val_nosig_noroverbounce.txt"
    )
    ap.add_argument("--out-dir", default="/mnt/md2/splits/v2scan")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df["name"] = df["dataset"].map(norm)
    df = df.set_index("name")
    reasons = df["reasons"].fillna("")

    def tagged(tag):
        return set(df.index[reasons.str.contains(tag, regex=False)])

    status = df["status"]
    errors = set(df.index[status == "ERROR"])
    quarantine = set(df.index[status == "QUARANTINE"])
    ok = set(df.index[status == "OK"])
    noisy = tagged("FLAG:r0_noisy") | tagged("FLAG:r1_noisy")
    nan20 = tagged("QUAR:nan>20%")
    frozen = set(df.index[df["frozen_tail"].fillna(False).astype(bool)])
    mount = set(
        df.index[df["mount_diff"].abs().fillna(0) >= MOUNT_THRESH]
    )
    band915 = set(
        df.index[(df["rx_lo"] >= BAND_915[0]) & (df["rx_lo"] <= BAND_915[1])]
    )

    train_lines = read_lines(args.train_txt)
    val_lines = read_lines(args.val_txt)
    train_names = [norm(x) for x in train_lines]
    val_names = [norm(x) for x in val_lines]
    val_set = set(val_names)

    # --- val subsets (preserve original val line text/order) ---
    val_line_by_name = OrderedDict()
    for line, name in zip(val_lines, val_names):
        val_line_by_name.setdefault(name, line)

    def val_subset(names):
        return [val_line_by_name[n] for n in val_line_by_name if n in names]

    subsets = {
        "val_clean_v2.txt": val_subset(ok),
        "val_degraded_v2.txt": val_subset(quarantine | noisy),
        "val_915_v2.txt": val_subset(band915),
    }

    # --- train regimes (preserve original train line text/order) ---
    seen, dupes, r1, dropped = set(), [], [], []

    def is_rover_apr5_bounce(name):
        return name.startswith("rover_2025_04_05") and "bounce" in name

    for line, name in zip(train_lines, train_names):
        if name in seen:
            dupes.append(name)
            continue
        seen.add(name)
        reason = None
        if name in errors:
            reason = "error"
        elif name not in df.index:
            reason = "ghost_not_in_scan"
        elif name in val_set:
            reason = "train_val_leak"
        elif name in frozen:
            reason = "frozen_tail"
        elif name in mount:
            reason = "mount_anomaly"
        elif is_rover_apr5_bounce(name):
            reason = "rover_apr5_bounce"
        if reason:
            dropped.append((name, reason))
        else:
            r1.append((line, name))

    r2 = [(line, name) for line, name in r1 if name not in nan20 and name not in noisy]

    outputs = dict(subsets)
    outputs["train_r1_labelclean.txt"] = [line for line, _ in r1]
    outputs["train_r2_nodegraded.txt"] = [line for line, _ in r2]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {
        "generator": "spf/scripts/make_v2_splits.py",
        "inputs": {
            args.metrics: sha256_file(args.metrics),
            args.train_txt: sha256_file(args.train_txt),
            args.val_txt: sha256_file(args.val_txt),
        },
        "rules": {
            "mount_thresh": MOUNT_THRESH,
            "band_915_hz": BAND_915,
            "noisy": "reasons contains FLAG:r0_noisy or FLAG:r1_noisy",
            "nan20": "reasons contains QUAR:nan>20%",
        },
        "duplicate_train_lines_skipped": sorted(set(dupes)),
        "r1_drop_counts": {},
        "files": {},
    }
    from collections import Counter

    manifest["r1_drop_counts"] = dict(Counter(reason for _, reason in dropped))
    manifest["r1_dropped"] = sorted(f"{n}:{r}" for n, r in dropped)

    for fn, lines in outputs.items():
        path = os.path.join(args.out_dir, fn)
        assert not os.path.exists(path), f"refusing to overwrite {path}"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        manifest["files"][fn] = {"n": len(lines), "sha256": sha256_file(path)}
        print(f"{fn}: {len(lines)}")

    print(f"train dupes skipped: {len(dupes)}; r1 drops: {manifest['r1_drop_counts']}")
    mpath = os.path.join(args.out_dir, "MANIFEST.json")
    assert not os.path.exists(mpath), f"refusing to overwrite {mpath}"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
