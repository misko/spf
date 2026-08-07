"""Pre-register a conditioning-chosen sparse comb (E-GSP7).

E-GSC3 showed E-CAL3's ten-LO refit failed because its uniform 600 MHz spacing
aliased the two ripple delays onto each other -- condition number 17.92 against
a 2.35 median over random 10-LO combs, worse than 1,999 of 2,000 of them. The
open row in the E-GSC decision ledger is whether a comb chosen *by conditioning*
works prospectively.

This picks that comb BEFORE any data exists. The objective is the committed
`ripple_conditioning` from
reports/gain_state_computational_20260807_v1/analysis/gsc2b_extras.py, copied
verbatim so the selection cannot drift if that file changes.

Constraint: >= 3 LOs in each AD9361 gain-table band. L10 is explicit that the
model interpolates within a measured span and must not extrapolate across a
band, so starving a band would test a known-bad configuration.
"""

from __future__ import annotations

import json

import numpy as np

TAU_FLEET = (2.56e-9, 0.92e-9)          # gsc_common.TAU_FLEET
LOW_EDGE_HZ = 1_300_000_000             # gain_tables.py band edges
HIGH_EDGE_HZ = 4_000_000_000
CANDIDATES = np.arange(400_000_000, 5_900_000_001, 50_000_000, dtype=np.int64)
SEED = 20260807


def ripple_conditioning(f_hz, taus):
    """Verbatim from gsc2b_extras.py."""
    cols = []
    for t in taus:
        cols.append(np.cos(2 * np.pi * f_hz * t))
        cols.append(np.sin(2 * np.pi * f_hz * t))
    M = np.column_stack(cols)
    M = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))


BASIS = np.column_stack(
    [f(2 * np.pi * CANDIDATES * t) for t in TAU_FLEET for f in (np.cos, np.sin)]
)
BAND_ID = np.where(
    CANDIDATES < LOW_EDGE_HZ, 0, np.where(CANDIDATES < HIGH_EDGE_HZ, 1, 2)
)


def cond_rows(rows):
    M = BASIS[rows]
    M = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))


def feasible(rows, min_per_band=3):
    return np.bincount(BAND_ID[rows], minlength=3).min() >= min_per_band


def optimize(n, restarts=40, iters=1200, rng=None):
    rng = rng or np.random.default_rng(SEED)
    best_rows, best_cost = None, np.inf
    for _ in range(restarts):
        while True:
            rows = rng.choice(len(CANDIDATES), size=n, replace=False)
            if feasible(rows):
                break
        cost = cond_rows(rows)
        for _ in range(iters):
            trial = rows.copy()
            trial[rng.integers(0, n)] = rng.integers(0, len(CANDIDATES))
            if len(np.unique(trial)) != n or not feasible(trial):
                continue
            trial_cost = cond_rows(trial)
            if trial_cost < cost:
                rows, cost = trial, trial_cost
        if cost < best_cost:
            best_rows, best_cost = rows, cost
    return np.sort(CANDIDATES[best_rows]), best_cost


def random_reference(n, draws=2000, rng=None):
    rng = rng or np.random.default_rng(SEED + 1)
    return np.asarray(
        [
            cond_rows(rng.choice(len(CANDIDATES), size=n, replace=False))
            for _ in range(draws)
        ]
    )


def uniform_comb(n):
    idx = np.round(np.linspace(0, len(CANDIDATES) - 1, n)).astype(int)
    return CANDIDATES[idx]


def main():
    result = {
        "tau_fleet_s": list(TAU_FLEET),
        "candidate_pool": {
            "start_hz": int(CANDIDATES[0]),
            "stop_hz": int(CANDIDATES[-1]),
            "step_hz": 50_000_000,
            "count": len(CANDIDATES),
        },
        "objective": "ripple_conditioning, verbatim from gsc2b_extras.py",
        "constraint": ">=3 LOs per AD9361 gain-table band",
        "seed": SEED,
    }
    dense = ripple_conditioning(CANDIDATES, TAU_FLEET)
    print(f"pool: {len(CANDIDATES)} LOs @ 50 MHz; dense cond = {dense:.4f}\n")
    for n in (10, 16):
        comb, cost = optimize(n)
        rand = random_reference(n)
        unif = uniform_comb(n)
        unif_cost = ripple_conditioning(unif, TAU_FLEET)
        beat = float((rand < cost).mean() * 100)
        result[f"n{n}"] = {
            "comb_hz": [int(v) for v in comb],
            "comb_mhz": [int(v // 1_000_000) for v in comb],
            "condition_number": cost,
            "random_median": float(np.median(rand)),
            "random_best_of_2000": float(rand.min()),
            "percent_of_random_better": beat,
            "uniform_comb_mhz": [int(v // 1_000_000) for v in unif],
            "uniform_condition_number": unif_cost,
            "dense_condition_number": dense,
            "band_counts": {
                "low": int((comb < LOW_EDGE_HZ).sum()),
                "middle": int(((comb >= LOW_EDGE_HZ) & (comb < HIGH_EDGE_HZ)).sum()),
                "high": int((comb >= HIGH_EDGE_HZ).sum()),
            },
        }
        r = result[f"n{n}"]
        print(f"=== N = {n} ===")
        print(f"  chosen   cond {cost:8.4f}   bands {r['band_counts']}")
        print(f"  uniform  cond {unif_cost:8.4f}   <- the E-CAL3 failure mode")
        print(f"  random   median {np.median(rand):.4f}, best-of-2000 {rand.min():.4f}")
        print(f"  {beat:.2f}% of random combs beat the chosen one")
        print(f"  LOs (MHz): {r['comb_mhz']}\n")
    with open("comb_selection.json", "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
