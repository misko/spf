"""Pick a small, balanced, reproducible dataset sample for a sweep's tuning stage.

The coarse stage of a filter sweep runs ~348 configurations x 5 seeds, which is
affordable on ~16 datasets and not on 565. Which 16 decides what the tuning stage
can see, so it cannot be "the first 16" or "16 at random":

* **The first N** is ordered by capture time, so it is one or two sessions --
  one antenna spacing, one weather, one set of rover quirks.
* **Uniform random** follows the corpus's own imbalance. In the 2026 rover
  corpus that means d/lambda 0.82703 (19 stores) crowds out 0.91557 (3), and the
  stage-2 winner is then tuned for the majority spacing.

So sample by **strata** -- (vehicle, routine, d/lambda, capture day) by default --
round-robin, so every stratum contributes one dataset before any contributes two.
That maximises coverage of the axes a filter's hyperparameters actually respond
to. Selection is deterministic given ``--seed``, and the chosen list plus the
full strata census are written out so a report can state exactly what was
sampled and what it was sampled from.

Reads only the capture yaml per dataset -- no zarr or segmentation is opened, so
this runs in a second over thousands of datasets.

Usage::

    python spf/filters/stratified_sample.py \\
        --datasets-from-glob '/mnt/qnap01/mouse9911/rovers_2026/merged/*.zarr' \\
        -n 16 --seed 0 \\
        --out <report>/stage2_sample.txt \\
        --manifest <report>/stage2_sample.json
"""

import argparse
import glob as globlib
import json
import os
import random
from collections import Counter, defaultdict

import yaml

C_LIGHT = 299792458.0

# Available strata axes. Defaults cover what the filters are known to respond to:
# the array geometry (d/lambda), the motion pattern, and the capture session.
AXES = ("vehicle", "routine", "d_lambda", "day", "carrier", "spacing_m")
DEFAULT_AXES = ("vehicle", "routine", "d_lambda", "day")


def describe(prefix):
    """Strata attributes for one dataset, from its capture yaml alone."""
    yaml_fn = str(prefix).replace(".zarr", "") + ".yaml"
    with open(yaml_fn) as f:
        cfg = yaml.safe_load(f)
    rx = cfg["receivers"][0]
    spacing = float(rx["antenna-spacing-m"])
    carrier = float(rx["f-carrier"])
    name = os.path.basename(str(prefix))
    # Prefer the recorded routine; a merged v7 store is <RX>.<TX> and its name
    # contains BOTH rovers' routines, so the filename cannot disambiguate.
    routine = str(cfg.get("routine", "unknown"))
    return {
        "prefix": str(prefix),
        "vehicle": "rover" if "rover" in name.lower() else "2dwallarray",
        "routine": routine,
        "spacing_m": spacing,
        "carrier": carrier,
        "d_lambda": round(spacing / (C_LIGHT / carrier), 5),
        "day": _day_from_name(name),
    }


def _day_from_name(name):
    """YYYY_MM_DD from a capture name, else 'unknown'.

    Only used to spread the sample across sessions; a capture whose clock was
    restored on boot carries a wrong date, which costs spread but never
    correctness -- so this never gates anything.
    """
    parts = name.split("_")
    for i in range(len(parts) - 3):
        y, m, d = parts[i + 1 : i + 4]
        if len(y) == 4 and y.isdigit() and len(m) == 2 and len(d) == 2:
            return f"{y}_{m}_{d}"
    return "unknown"


def stratum_key(record, axes):
    return tuple(record[a] for a in axes)


def stratified_sample(records, n, axes=DEFAULT_AXES, seed=0):
    """``n`` records spread as evenly as possible over the strata.

    Round-robin over strata: every stratum yields its first pick before any
    yields a second. Deterministic given ``seed``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    strata = defaultdict(list)
    for r in records:
        strata[stratum_key(r, axes)].append(r)

    rng = random.Random(seed)
    # Shuffle within each stratum, and shuffle stratum order, so which member is
    # taken (and which strata get the extras when n is not a multiple) does not
    # depend on filesystem ordering.
    #
    # Iterate strata in SORTED key order, not dict order. dict order is insertion
    # order, which follows the input list, so shuffling in that order consumes the
    # RNG differently for the same corpus listed differently -- and the sample
    # would silently depend on how the caller globbed its files.
    for key in sorted(strata):
        strata[key].sort(key=lambda r: r["prefix"])
        rng.shuffle(strata[key])
    order = sorted(strata)
    rng.shuffle(order)

    chosen, depth = [], 0
    while len(chosen) < n:
        progressed = False
        for key in order:
            if depth < len(strata[key]):
                chosen.append(strata[key][depth])
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:  # every stratum exhausted
            break
        depth += 1
    return chosen, strata


def build_manifest(records, chosen, strata, axes, n, seed):
    chosen_keys = Counter(stratum_key(r, axes) for r in chosen)
    return {
        "axes": list(axes),
        "seed": seed,
        "requested": n,
        "selected": len(chosen),
        "population": {"datasets": len(records), "strata": len(strata)},
        "strata": [
            {
                "key": dict(zip(axes, key)),
                "in_population": len(members),
                "in_sample": chosen_keys.get(key, 0),
            }
            for key, members in sorted(strata.items(), key=lambda kv: str(kv[0]))
        ],
        "selected_datasets": [r["prefix"] for r in chosen],
    }


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=[], help="dataset prefixes")
    p.add_argument("--datasets-from-file", default=None)
    p.add_argument("--datasets-from-glob", default=None)
    p.add_argument("-n", "--num", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES), choices=AXES)
    p.add_argument("--out", required=True, help="one selected prefix per line")
    p.add_argument("--manifest", default=None, help="JSON strata census")
    return p


def collect(args):
    prefixes = list(args.datasets)
    if args.datasets_from_file:
        with open(args.datasets_from_file) as f:
            prefixes += [x.strip() for x in f if x.strip()]
    if args.datasets_from_glob:
        prefixes += sorted(globlib.glob(args.datasets_from_glob))
    if not prefixes:
        raise ValueError("no datasets given")
    return sorted(set(prefixes))


if __name__ == "__main__":
    args = get_parser().parse_args()
    prefixes = collect(args)

    records, skipped = [], []
    for p in prefixes:
        try:
            records.append(describe(p))
        except Exception as e:
            skipped.append({"prefix": p, "reason": f"{type(e).__name__}: {e}"})
    if not records:
        raise RuntimeError("no datasets could be described")

    chosen, strata = stratified_sample(records, args.num, tuple(args.axes), args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in chosen:
            f.write(r["prefix"] + "\n")

    manifest = build_manifest(
        records, chosen, strata, tuple(args.axes), args.num, args.seed
    )
    manifest["skipped"] = skipped
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    print(
        f"population: {len(records)} datasets in {len(strata)} strata "
        f"(axes {list(args.axes)})"
    )
    if skipped:
        print(f"skipped {len(skipped)} undescribable datasets")
    print(f"selected: {len(chosen)} -> {args.out}")
    if len(chosen) < args.num:
        print(f"NOTE only {len(chosen)} available, fewer than the {args.num} requested")
    for row in manifest["strata"]:
        if row["in_sample"]:
            print(
                f"  {row['in_sample']}/{row['in_population']:3d}  "
                + "  ".join(f"{k}={v}" for k, v in row["key"].items())
            )
