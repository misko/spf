# GRBL + Motion-Planner Stack — Reference

Deep-dive reference (Passes 1–3 per [`DOCUMENTATION_GUIDE.md`](../../DOCUMENTATION_GUIDE.md)) for the
cable-robot motion stack:

- `spf/grbl/grbl_interactive.py` — `GRBLDynamics`, `GRBLController`, `GRBLManager`, fake-GRBL, REPL.
- `spf/motion_planners/planner.py` — `Planner` base + `BouncePlanner`, `CirclePlanner`,
  `StationaryPlanner`, `PointCycle`, `CalibrationV1Planner`.
- `spf/motion_planners/dynamics.py` — `Dynamics`, `chord_length_to_angle`, `a_to_b_in_stepsize`,
  boundary geometry.

**Pinned commit:** `574c454` (line numbers verified against this revision).

**Units / frames.** All GRBL positions are 2-D `[x, y]` in **motor steps ≈ mm** in the cable-robot
plane: `pA` (left/"A" motor) at `[3568,0]`, `pB` (right/"B" motor) at `[0,0]`; `x` positive from B
toward A, `y` positive *up* toward the ceiling (workspace is entirely `y>0`, below the motor line).
"Motor steps" returned by `to_steps` are **deltas relative to the calibration point**, not absolute
belt lengths. Planner angles are **radians** unless explicitly noted. GPS planners (via
`mavlink_controller.drone_get_planner`) reuse the same planner classes over a `[long,lat]`-degree
boundary; see [`00_concepts/conventions.md`](../00_concepts/conventions.md).

Status legend: ✅ Active · ⚠️ legacy/concern · 🔴 Broken (verified) · 🧪 experimental/unused.

---

## Architecture / control flow (Pass 2)

```
grbl_radio_collection.grbl_radio_main(args)            (also: mavlink uses planners directly)
  └─ get_default_gm(serial_fn, routine)
       ├─ GRBLDynamics(calibration_point,pA,pB,bounding_box)   # kinematics + bounds polygon
       └─ GRBLController(serial_fn, dynamics, {0:"XY",1:"ZA"})  # serial I/O + position cache
  └─ GRBLManager(controller, routine).start()           # spawns run_planner thread
        run_planner():
          routines[routine]()        # build self.planners (one per channel)  [setup]
          get_ready()                # StationaryPlanner each channel → move into start pos
          routines[routine]()        # build planners AGAIN  (see Pass-3 note)
          get_ready(); run()
            run(): move_to_iter({c: planner.yield_points()})
              loop: get_next_points → controller.move_to(pts) → poll targets_far_out → update_status
```

`GRBLController.move_to` is the only producer of G-code motion; `update_status` is the only reader
of position. `data_collector` (`GrblDataCollectorRaw`) consumes
`position_controller.controller.position["xy"]` (data_collector.py:594-649) to label each capture —
so a wrong `from_steps`/`update_status` value silently mislabels the dataset, not just the motion.

**Live routines** (built by `GRBLManager`): `bounce`, `rx_circle`, `tx_circle`,
`rx_random_circle`, `v4_calibrate` (all wired to real collection). `v1_calibrate` /
`CalibrationV1Planner` are 🧪 legacy. Tests drive `bounce` end-to-end (test_grbl.py:153-178).

---

## `GRBLDynamics.to_steps(p)`  ·  `grbl_interactive.py:173`  ·  ✅ Active

**Summary.** Inverse kinematics: an `[x,y]` target → `(a_motor_steps, b_motor_steps)` deltas from
the calibration point. Each "step" value is `‖motor−p‖ − ‖motor−calibration‖`, i.e. how much the
A/B belt must pay out relative to the calibrated home length. Bounds-checks the target *and* the
re-inverted point before returning.

**Signature**
- Inputs: `p` (np 2-vec, x/y motor-steps frame).
- Returns: `(a_motor_steps, b_motor_steps)` floats (belt deltas).
- Side effects: raises `PointOutOfBoundsException` (twice-checked) unless `self.unsafe`; calls
  `from_steps` for the re-validation. No mutation.

**Called from** — `GRBLController.move_to` (369-379, every move), `Dynamics.binary_search_edge`
(dynamics.py:87, via the boundary bisection used by every bounce). Fan-in: high (every motion).
Tests: `test_grbl.py::test_steps_and_steps_inverse`, `::test_xaxis`.

**Contract**
- *Pre:* `p` inside the bounding-box polygon (within `radius=1e-5`) unless `unsafe`. Workspace is
  `y>0`.
- *Post:* `from_steps(to_steps(p)) ≈ p` (round-trip verified numerically for in-bounds points; see
  Analysis). Raises rather than returning an out-of-bounds inverse.
- *Invariant:* on the x-axis (`y=0`) `a_steps == −b_steps == −x` (test_xaxis 64-72).

**Analysis.** *Correctness:* round-trip confirmed for `[1930,2600]`,`[2000,1500]`,`[550,450]`,
`[1000,1000]` (re-derived the math without the `serial` import). The double bounds-check (raw `p`
then inverted `new_point`) guards against the `from_steps` sign ambiguity below.
*Issues:* none new. Note the **hard-coded `radius=0.00001`** here (178,194) shadows the
configurable `self.bounds_radius` that the *base* `Dynamics.to_steps` uses (dynamics.py:72) — GPS
planners that pass `bounds_radius=1e-9` get that finer radius only through the base class, never
through `GRBLDynamics`. Benign for GRBL (always uses 1e-5) but an inconsistency. 🔍

---

## `GRBLDynamics.from_steps(state)`  ·  `grbl_interactive.py:144`  ·  ✅ Active  ·  ⚠️ silently-wrong below the motor line

**Summary.** Forward kinematics: `(a_steps,b_steps)` → `[x,y]`. Reconstructs belt lengths
`r1,r2`, solves the two-circle intersection for `x`, then `y=√(r1²−x²)`, and always picks the
**positive-y** branch (`+y_dir*y`, 169).

**Signature**
- Inputs: `state=(a_motor_steps,b_motor_steps)` belt deltas.
- Returns: np 2-vec `[x,y]` rounded to 1 dp.
- Side effects: none (pure). `np.sqrt` can yield `NaN`.

**Called from** — `to_steps` (re-validation), `GRBLController.update_status` (339-340, parses both
channels every poll). Fan-in: high.

**Contract**
- *Pre:* `r1² ≥ x²` (point reachable). *Post:* a point on the **y>0** side of the A–B axis.

**Analysis.** *Issues:*
1. **Positive-y branch only** (165-169) — `y=√(...)` discards sign, so any target *below* the motor
   line (`y<0`) is silently reflected to `+y`. ⚠️ **no-crash / silently-wrong**, but **not reached**
   in practice: the real `home_bounding_box` and all GPS boundaries lie strictly on one side, and
   `to_steps`'s double-check rejects out-of-workspace points first. Reachability: dead for shipped
   configs. Confidence ✅ (read the branch).
2. **`np.sqrt(r1²−x²)` unguarded → NaN** (165) if a caller feeds inconsistent steps (e.g. a corrupt
   `MPos` line in `update_status`). No NaN guard anywhere downstream; a NaN `xy` would silently
   poison `position["xy"]`, `is_moving`, and the dataset label. Reachability: only on malformed
   serial input. 🔍 analysis, ⚠️ no-crash.

---

## `GRBLController.update_status(skip_write=False)`  ·  `grbl_interactive.py:316`  ·  ⚠️ recursion concern

**Summary.** Sends `?` to GRBL, parses the `<...|MPos:b0,a0,b1,a1|...>` status line into both
channels' `xy`, caches into `self.position`, returns it. On an unparseable / `ok` / empty line it
**recurses** (toggling `skip_write`) to read again.

**Signature**
- Inputs: `skip_write` (bool) — skip the `?` write, just read (used by the recursion).
- Returns: `self.position` dict: `a0/b0/a1/b1_motor_steps`, `xy=[xy0,xy1]`, `is_moving`, `time`.
- Side effects: serial write+read; three `time.sleep(0.01)`; **recursion**; logging; mutates
  `self.position`.

**Called from** — `move_to_iter` poll (429 via `targets_far_out`→`distance_to_targets`),
`GRBLManager.get_ready` (523), REPL `s`/move commands, `set_current_position`. Fan-in: high. No
direct test (only exercised through the fake-GRBL subprocess tests).

**Contract**
- *Pre:* serial open; GRBL configured to report `MPos:` (machine, not work, position).
- *Post:* `position["xy"]` is two valid 2-vecs **unless** parsing diverged (see issue).
- *Invariant:* parse order `b0,a0,b1,a1` ↔ motor map `{0:"XY",1:"ZA"}` ↔ `move_to` write order
  (X←b0, Y←a0, Z←b1, A←a1). **Verified consistent** (Pass 3).

**Analysis.** *Correctness:* the b/a parse order matches the move_to write order — the "A/B swap"
the Pass-1 inventory flagged is a false alarm (both sides use b-then-a per channel). *Issues:*
1. **Unbounded recursion on a chatty/stuck device** (331-334) — every `ok`/blank/unparseable line
   recurses with no depth cap; `skip_write` merely alternates write/no-write each level. A device
   that streams non-status chatter (or a permanently malformed `MPos`) blows the Python recursion
   limit → `RecursionError`. Reachability: real serial only (the fake GRBL only ever emits a clean
   status or `OK`, and `OK` is consumed by `_run_fake_grbl`'s response queue, so the test path
   doesn't trigger it). 🔍 analysis (did not reproduce on hardware); **confirms KNOWN_ISSUES
   recursion concern**.
2. **No `try` around the `from_steps`/`np.sqrt`** — a malformed `MPos` that still has 4
   comma-separated floats parses fine but can drive `from_steps` to NaN (see `from_steps` issue 2).

---

## `GRBLController.move_to(points)`  ·  `grbl_interactive.py:369`  ·  ✅ Active

**Summary.** Builds one `G0` block moving every channel to its target `xy`, via `to_steps`, and
writes it to serial. The only motion producer.

**Signature**
- Inputs: `points` (dict `channel → xy`).
- Returns: None. Side effects: `to_steps` (**may raise `PointOutOfBoundsException`**); serial
  write+read; sleeps; reads global `run_grbl`; logging.

**Called from** — `move_to_iter` (428). Fan-in: all motion. Tested via fake-GRBL subprocess tests.

**Analysis.** *Issues:*
1. **`to_steps` raises inside the move loop** (376) — if any channel's next planned point lands
   just outside the polygon (planner bounce/circle math can produce a point on the wrong side of
   the `1e-5` radius), `move_to` raises, which propagates up through `move_to_iter` → `run` →
   `run_planner`'s bare `except Exception` (493-494) that merely **logs and stops the routine
   thread**. The collection keeps running but the platform **silently stops moving**. Reachability:
   live (any planner that grazes the boundary). 🔍 — depends on planner output; ⚠️ no-crash at the
   process level (swallowed). This is the most consequential reachable failure mode in the stack.

---

## `GRBLController.wait_while_moving()`  ·  `grbl_interactive.py:356`  ·  🔴 Broken (dead)

**Summary.** Intends to poll `update_status` twice and return once the motor steps stop changing.

**Analysis — confirms KNOWN_ISSUES #29.** Lines 363-364 index `old_pos["a_motor_steps"]` /
`["b_motor_steps"]`, but `update_status` returns keys `a0_motor_steps,b0_motor_steps,a1_motor_steps,
b1_motor_steps` (345-353). Those keys **do not exist** → guaranteed `KeyError` on first iteration.
**Reachability: dead** — repo-wide grep finds **zero callers** (only the definition). Severity:
low (broken × unreached). Confidence ✅ verified (read both key sets + grep). The live "wait for
motion to stop" path is instead `move_to_iter`'s `targets_far_out` poll (429) and `get_ready`'s
`position["is_moving"]` loop (522), which use the *correct* keys — so the bug is latent, not active.

---

## `GRBLManager.run_planner` / `get_ready` / `run`  ·  `grbl_interactive.py:473 / 501 / 526`  ·  ✅ Active

**Summary.** Thread body that (1) builds planners via the routine fn, (2) `get_ready` moves every
channel to its planner's start position using a throwaway `StationaryPlanner`, (3) re-builds
planners and `run`s their `yield_points` streams through `move_to_iter`.

**Analysis.** *Correctness:* the routine fn is **called twice** (475 setup, 486 run) — the first
call's planners exist only so `get_ready` (501) can read `get_planner_start_position`; `run` (529)
overwrites their `start_point` with `requested_start_points`. Functionally fine but means any
randomness in routine construction (`rx_random_circle`'s `random.randint`, `CirclePlanner`'s
unseeded `np.random.rand` direction) is **sampled twice and discarded once** — the start-position
plan and the actual run use *different* random draws. ⚠️ benign for circles (same geometry) but the
travel-into-position and the run can disagree on direction. 🔍

*Issue:* `run_planner`'s bare `except Exception` (493) swallows everything (incl. the `move_to`
out-of-bounds path above) and only logs — failures stop the platform silently while data collection
believes it is still moving. ⚠️ no-crash. Reachability: live.

---

## `Planner.get_bounce_pos_and_new_direction(p, direction)`  ·  `planner.py:19`  ·  ✅ Active

**Summary.** Core reflection: bisect along `direction` to the boundary (`binary_search_edge`), get
the nearest edge's unit vector `bvec`, and reflect `direction` — keep the along-edge component,
negate the perpendicular one (specular bounce).

**Signature**
- Inputs: `p` (2-vec), `direction` (2-vec, need not be unit).
- Returns: `(last_point_before_bounce, new_direction)` — `new_direction` is renormalized to unit.
- Side effects: recursion via `binary_search_edge`; pure otherwise.

**Called from** — `BouncePlanner.single_bounce` (52), `CalibrationV1Planner.yield_points`,
test_grbl.py:127-128. Fan-in: every bounce routine + GPS bounce.

**Analysis — reflection geometry verified.** `bvec_perp=[bvec[1],-bvec[0]]` is orthogonal to
`bvec`; since `{bvec,bvec_perp}` is orthonormal, `direction = (d·bvec)bvec + (d·bvec_perp)bvec_perp`,
and `new_direction = (d·bvec)bvec − (d·bvec_perp)bvec_perp` negates exactly the perpendicular
component — **correct specular reflection** (planner.py:27-32). ✅ verified algebraically.
*Issues:* if `|direction|` from the bounce is anti-parallel to the edge such that the reflected
vector points back out of bounds, the renormalize (32) can't fix direction; recovery relies on the
`single_bounce` 5 % noise (planner.py:59-63) and the stall counter. The empirical bounce tests
(`>90 %` unique points over 20 000 steps) confirm it doesn't get permanently stuck in practice.

---

## `CirclePlanner.yield_points()`  ·  `planner.py:199`  ·  🔴 deg/rad bug (silently-wrong)

**Summary.** Walks a circle by accumulating `angle_increment` and emitting `angle_to_pos(angle)`.

**Analysis — confirms KNOWN_ISSUES #5.** `angle_increment = chord_length_to_angle(step,radius)` is
in **radians** (dynamics.py:14, `2·arcsin(...)`), and `angle_to_pos` consumes it as radians
(`np.sin(angle)`, 196). But the loop guard is `while current_angle < 360` (205). A radian
accumulator compared to the literal `360` runs **360/2π ≈ 57.3 full revolutions** before stopping,
not one.
- GRBL `rx_circle` (radius 950, step 8): increment 0.00842 rad → loop ends at 42 750 steps =
  **~57 revolutions** (one revolution is 746 steps). Re-derived numerically.
- mavlink `circle` (radius 1.5e-4, step 1e-4): increment 0.680 rad → ~530 steps = **~84
  revolutions**.

⚠️ **no-crash / silently-wrong** — `angle_to_pos` wraps cleanly modulo 2π, so every yielded point is
*on the circle*; the platform just re-traces the same circle ~57–84× before the planner ends and the
routine restarts. No bad geometry, but the "one circle" intent is wrong: collection captures many
redundant revolutions per "circle". Reachability: **live** — every circle routine
(`rx_circle`/`tx_circle`/`rx_random_circle` and mavlink `circle`). Confidence ✅ verified. The fix
(`< 2*np.pi`) is a one-line change but **must be coordinated with downstream** — anything counting
"circles" or sizing buffers off the point count has been compensating for 57× the points.

*Second issue (low):* `CirclePlanner.__init__` (186) picks direction with **unseeded
`np.random.rand()`**, bypassing the `seed`/`self.rng` plumbing — circle direction is not
reproducible even with a fixed seed. ⚠️ minor.

---

## `BouncePlanner.bounce(current_p, total_bounces=-1)`  ·  `planner.py:69`  ·  ✅ Active

**Summary.** Infinite (default) bounce generator with stall recovery: each iteration computes one
bounce segment, yields its points, and if it keeps getting single-point segments at the same place
>3× picks a fresh random direction.

**Analysis.** *Correctness:* `assert len(to_points) > 0` (95) holds because `a_to_b_in_stepsize_np`
always returns ≥1 point (even when `a≈b` it returns `[b]`). Stall logic (98-111) is sound:
`last_point == to_points[0]` uses ndarray `.all()`, resets on progress. *Issues:*
1. **`print("BOUNCING")` every iteration** (84) — log/stdout spam, unconditional. ⚠️ noise, not
   correctness. Reachability: live (every bounce step).
2. **Stall recovery sets `new_direction` but loses it on the next line** — when stalled it sets
   `new_direction = self.random_direction()` (104) then `self.current_direction = new_direction`
   (113) and re-enters; the random direction *is* applied. ✅ no bug (verified the assignment order).

---

## Smaller / leaf functions (Pass 3 summary)

- **`chord_length_to_angle(chord,radius)`** (dynamics.py:14) — `2·arcsin(chord/2r)`, radians.
  Unguarded `arcsin` → NaN if `chord > 2·radius` (step bigger than diameter). Not reached for
  shipped configs (step≪diameter). ✅ pure. The **radian** output is the root of the CirclePlanner
  bug — the function is correct; its consumer mis-compares to 360.
- **`a_to_b_in_stepsize` / `_np`** (dynamics.py:18,31) — vectorized stepper; forces last point
  exactly to `b`, no off-by-one (verified: 0→10 step 8 → `[[8,0],[10,0]]`). The non-`_np` wrapper
  body (33-44) is **dead code** after the unconditional `return` (32). Type quirk: returns a
  1-element **list** `[b]` when `a≈b`, an ndarray otherwise — both indexable, callers tolerate it.
- **`Dynamics.binary_search_edge`** (dynamics.py:81) — recursive bisection; depth bounded by
  `log2((right−left)/epsilon)` (~24 for 10000/0.001) so the recursion is safe (unlike
  `update_status`). Catches `PointOutOfBoundsException` to bisect inward. ✅.
- **`Dynamics.get_boundary_vector_near_point`** (dynamics.py:114) — returns nearest edge unit
  vector. The cosine `score` (125-130) is **immediately overwritten** by
  `distance_from_point_to_segment` (131) → the dot-product block (incl. the `v1-v1` zero-vector) is
  **dead**. Result is correct (uses distance), but the dead block is confusing. ⚠️ dead code.
- **`StationaryPlanner.yield_points`** (planner.py:125) / **`PointCycle.yield_points`** (156) —
  move-to-then-hold / cycle-forever; both honor `self.running`. ✅.
- **`get_default_dynamics` / `get_default_gm`** (grbl_interactive.py:631,641) — wire home constants
  + `channel_to_motor_map={0:"XY",1:"ZA"}`. ✅.
- **Fake GRBL** (`_run_fake_grbl`,`_move_fake_grbl`, 263-310) — 🧪 test-only. `max_diff` (269-273)
  is reset per-key and only feeds a **commented-out** sleep (285) → harmless dead variable.

---

## Test inventory & coverage gaps (Pass 3)

| Test | Covers | Gap |
|---|---|---|
| `test_grbl.py::test_steps_and_steps_inverse` | `to_steps`/`from_steps` round-trip on a 11×11 grid | only in-bounds y>0; never exercises the negative-y `from_steps` reflection or NaN path |
| `test_grbl.py::test_xaxis` | x-axis step symmetry `a=−b=−x` | single line only |
| `test_grbl.py::test_binary_search_edge` | bisection + `get_bounce_pos_and_new_direction` + `single_bounce` + 10 bounces | doesn't assert reflection *correctness*, only that it runs |
| `test_grbl.py::test_grbl_simple_move`/`_bounce` | REPL + fake-GRBL end-to-end via subprocess | fake GRBL never returns chatty/`ok` lines → **`update_status` recursion untested** |
| `test_grbl.py::test_grbl_radio_collection_bounce[_internal]` | full `grbl_radio_main` bounce collection | only `bounce`; **no circle test** → CirclePlanner 57× bug uncaught |
| `test_dyanmics_and_planners.py::test_grbl_bounce`/`test_gps_bounce` | bounce yields >90 % unique pts over 20 000 steps (GRBL + GPS) | asserts spread, not geometry; no circle/stationary/pointcycle |

**Biggest gaps:** (1) **no `CirclePlanner` test at all** — the deg/rad over-revolution ships
untested; (2) **`wait_while_moving` is never called** by any test or prod path (dead); (3)
`update_status`'s recursion / NaN paths are unreachable from the fake GRBL, so they're untested.

---

## Cluster summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `GRBLDynamics.to_steps` | ✅ | high | ✅ round-trip | hard-coded `radius=1e-5` shadows `bounds_radius` |
| `GRBLDynamics.from_steps` | ✅ ⚠️ | high | ✅ (in-bounds) | positive-y-only branch (silently-wrong below motor line; unreached); `sqrt`→NaN unguarded |
| `update_status` | ⚠️ | high | ❌ (indirect) | unbounded recursion on chatty serial; b/a parse order *consistent* with `move_to` |
| `move_to` | ✅ | all motion | ❌ (indirect) | `to_steps` raise → swallowed by `run_planner` → silent stop |
| `wait_while_moving` | 🔴 dead | 0 | ❌ | KeyError on `a_motor_steps` (KNOWN #29) — **dead, no callers** |
| `run_planner`/`run` | ✅ | entry | ✅ (bounce) | routine fn called twice (random draws differ); bare `except` swallows |
| `get_bounce_pos_and_new_direction` | ✅ | every bounce | ✅ (runs) | reflection math **verified correct** |
| `CirclePlanner.yield_points` | 🔴 | live (all circles) | ❌ | **deg/rad: 360 rad ≈ 57 revolutions** (KNOWN #5); unseeded direction |
| `BouncePlanner.bounce` | ✅ | live | ✅ (>90 % unique) | `print("BOUNCING")` spam; stall logic OK |
| `chord_length_to_angle` | ✅ | circles | ❌ | radians output (correct); arcsin NaN if step>diameter |
| `a_to_b_in_stepsize` | ✅ | everywhere | ✅ (indirect) | dead loop body after return; list-vs-ndarray return type |
| `get_boundary_vector_near_point` | ⚠️ | every bounce | ✅ (runs) | dead cosine block (overwritten) |

**Cross-cutting themes.**
1. **Silently-wrong geometry, not crashes, is the dominant risk.** The two reachable correctness
   defects (`CirclePlanner` 57× over-revolution; `from_steps` positive-y branch) and the NaN path
   produce *wrong values* that flow straight into `position["xy"]` and thus the **dataset labels**
   in `data_collector`, never raising. Any fix must check whether downstream code already
   compensates (e.g. circle point counts).
2. **Failures are swallowed.** `run_planner`'s bare `except` (493) turns a `move_to`
   out-of-bounds raise into a silent stop while collection keeps running — the platform freezes but
   the run looks healthy.
3. **The b/a motor ordering is actually consistent** (write `move_to` ↔ parse `update_status` ↔
   round-trip) — the Pass-1 "A/B swap" concern is refuted.
4. **Recursion safety is split:** `binary_search_edge` is depth-bounded and safe; `update_status`
   is not (no cap) — the genuine recursion hazard.
