# P0 Issue #4 — `swap_lat_long` inconsistent array vs tuple branch

File: `spf/gps/gps_utils.py:6-10`

```python
def swap_lat_long(x):
    if isinstance(x, np.ndarray) and x.ndim == 2:
        assert x.shape[1] == 2
        return np.hstack([x[:, [0]], x[:, [1]]])   # cols 0,1 — NO SWAP
    return (x[1], x[0])                              # swaps
```

## 1. CONFIRM — array branch is a genuine no-op; tuple branch swaps

YES, confirmed both by reading and empirically.

- Array branch (`gps_utils.py:9`): `np.hstack([x[:, [0]], x[:, [1]]])` re-stacks
  column 0 then column 1 — **identical column order**, a pure no-op (it does not
  even copy-with-swap; output == input).
- Tuple branch (`gps_utils.py:10`): `(x[1], x[0])` — **swaps** the two elements.

Empirical check:
```
2D array in : [[10,20],[30,40]]  out: [[10,20],[30,40]]   # NO swap
1D array in : [10,20]            out: (20.0, 10.0)        # swap
tuple in (10,20)                 out: (20.0, 10.0)         # swap
```

Note the branch condition is `ndim == 2`. A **1-D** numpy array (`ndim == 1`)
falls through to the tuple branch via `x[1], x[0]` and DOES swap. So the no-op
applies **only** to a 2-D `(N,2)` numpy array.

## 2. REACHABILITY & WHICH BRANCH — every caller

There are exactly 3 non-definition call sites. **None of them passes a 2-D array.**

| Caller | Argument | Type / shape | Branch | Swaps? |
|---|---|---|---|---|
| `mavlink_controller.py:412` | `self.gps` | `np.array([self.long, self.lat])` shape `(2,)`, 1-D (`:806`) | tuple branch | YES |
| `mavlink_controller.py:412` | `target_point` (= `point`, indexed `point[0]/[1]`, `:434`) | 1-D `(2,)` or 2-tuple | tuple branch | YES |
| `boundaries.py:136` | `gps` | 1-D `(2,)` array or 2-list (see callers below) | tuple branch | YES |
| `boundaries.py:137` | `boundary_points.mean(axis=0)` | boundary is `(N,2)` → `.mean(axis=0)` is `(2,)`, 1-D | tuple branch | YES |
| `test_gps.py:20` | `start_point` (`franklin_safe.mean(axis=0)`) and `inverse_haversine(...)` result | both 1-D `(2,)` / tuple | tuple branch | YES |

Callers of `find_closest_boundary*` (feed `gps` to `boundaries.py:136`):
- `mavlink_radio_collection.py:250`: `find_closest_boundary(drone.gps)` — `drone.gps` is 1-D `(2,)`.
- `v4_tx_rx_to_v5.py:266,271`: passes `tx_gps_long_lat` / `rx_gps_long_lats[ridx]`,
  which are **2-tuples of scalars** (`get_non_zero_mean(...), get_non_zero_mean(...)`, `:248,255`).
- `gps_check.py:32`: passes `[longs[idx], lats[idx]]` — a 2-element **list**.
- `position_check.py`: does not call `swap_lat_long` (uses xy mm only).

**Conclusion: the buggy no-op (2-D array) branch is NOT exercised by any caller in
the repo.** Every live path passes a 1-D array, tuple, or list, all of which take
the swapping tuple branch. The no-op branch is dead code under current usage.

## 3. DATA-FLOW IMPACT

The output feeds `haversine(...)` (from the `haversine` package), which expects
points as **(lat, long)**. Internally the rover stores GPS as **(long, lat)**
(`self.gps = np.array([self.long, self.lat])`, `:806`; boundaries are authored
`(long, lat)`, see `boundaries.py:6` comment and the `(-122.x, 37.x)` arrays).
So `swap_lat_long` exists precisely to convert stored `(long,lat)` → haversine's
`(lat,long)`. The swap is **required for correctness**.

If the no-op branch ran on real `(long,lat)` data, haversine would receive
`(long, lat)` = roughly `(-122, 37)` interpreted as `(lat=-122, long=37)`. lat
−122° is non-physical; haversine would still compute a number (no crash) but the
distance would be **grossly wrong** → wrong boundary selection, wrong
`distance_to_target`, broken RTL/tolerance logic. This is the silently-wrong
failure mode the issue warns about.

**Compensation check:** There is NO compensating second swap. Callers pass raw
`(long,lat)` directly; the single `swap_lat_long` call is the only transform.
Because all live callers happen to hit the *correct* (swapping) tuple branch, the
math is currently correct — not by a cancelling double-swap, but because the
broken branch is simply never reached.

## 4. TRIGGER — when would the wrong branch run on real GPS?

It would trigger only if a caller passed a **2-D `(N,2)` numpy array** of GPS
points to `swap_lat_long` (e.g. batching a whole trajectory of fixes, or passing
a full boundary array instead of its `.mean(axis=0)`). No current rover routine
does this:
- live rover nav (`distance_to_target`/`move_to_point`) → 1-D `self.gps` + 1-D/tuple point.
- boundary safety (`find_closest_boundary` from `mavlink_radio_collection.py`) →
  1-D `drone.gps`, and `boundary.mean(axis=0)` is 1-D.

So on a live rover **the wrong branch does not run**; the geometry is correct.
The risk is latent: the 2-D branch was clearly *written to* swap (asserts
`shape[1]==2`) but implements a no-op, so any future vectorized caller would be
silently wrong.

## 5. REAL-WORLD / TESTS

- `tests/test_gps.py:20` uses `swap_lat_long` but only on 1-D/tuple inputs
  (`test_gps_circle`). It exercises the **swapping** branch and would pass; it
  does **not** test the 2-D array branch at all. No test pins the array-branch
  behavior, so the no-op would not be caught by the suite.
- No assertion or test would fail today. The bug is masked because callers
  consistently pass 1-D/tuple data (which swaps correctly), never a 2-D array.

## 6. SEVERITY VERDICT — DOWNGRADE (latent bug, not active P0)

Justification:
- The inconsistency is **real and confirmed**: the 2-D array branch is a no-op
  while it is plainly intended to swap (it asserts `shape[1]==2` and is named
  `swap_lat_long`).
- However, **no caller in the repository reaches that branch.** All live GPS
  distance/boundary math passes 1-D arrays / tuples / lists, which take the
  correct swapping path. Current on-rover behavior is correct; distances,
  boundary enforcement, and RTL/tolerance are not affected today.
- Therefore this is a **latent / dead-code defect**, not an active wrong-result
  bug. It is a P0-shaped trap (a future vectorized caller would silently get
  wrong distances with no crash and no test failure), but it is not currently
  producing wrong results. Verdict: **DOWNGRADE from active P0 to latent bug /
  P2**, with a "fix before any vectorized use" flag.

## 7. FIX SKETCH (+ compensation warning)

Minimal fix — make the 2-D branch actually swap:

```python
def swap_lat_long(x):
    if isinstance(x, np.ndarray) and x.ndim == 2:
        assert x.shape[1] == 2
        return np.hstack([x[:, [1]], x[:, [0]]])   # was [:,[0]],[:,[1]] — swap cols
    return (x[1], x[0])
```

(Index 1 then 0, mirroring the tuple branch.)

**Compensation / re-check warning before applying:**
- Confirm NO current caller secretly *depends* on the 2-D branch being a no-op.
  Grep result shows none pass 2-D arrays, so this fix changes no live behavior —
  but verify there is no out-of-tree / notebook / script caller batching
  trajectories. If one exists and was authored to "work" against the no-op, it
  would itself contain a compensating manual swap that this fix would break.
- The fix is behavior-preserving for all in-repo callers (they never hit this
  branch), so it is safe but also does NOT fix any current wrong result — its
  value is purely defensive/future-proofing.
- Do NOT also "fix" callers to pass 2-D arrays as part of this change; keep the
  change scoped to the helper only, to avoid introducing a real double-swap.
```
