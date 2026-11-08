# P0 #5 Investigation: CirclePlanner.yield_points radians-vs-degrees loop bound

**File:** `spf/motion_planners/planner.py:199-215` (CirclePlanner.yield_points)
**Verdict: DOWNGRADE — confirmed code defect, but NOT a silently-wrong-motion bug. The motion stays a correct circle; the only effect is the generator yields ~57x more points before exhausting, which is masked by an outer record-count stop condition.**

---

## 1. CONFIRM — increment is radians, bound is 360

`chord_length_to_angle` returns **radians** (`spf/motion_planners/dynamics.py:14-15`):

```python
def chord_length_to_angle(chord_length, radius):
    return 2 * np.arcsin(chord_length / (2 * radius))
```

`arcsin` returns radians, so `angle_increment` is in radians (`planner.py:189`):

```python
self.angle_increment = chord_length_to_angle(self.step_size, self.circle_radius)
```

The loop bound is the literal `360` (`planner.py:205`):

```python
current_angle = 0
while current_angle < 360:
    ...
    current_angle += self.direction * self.angle_increment   # planner.py:215
```

**The bound `360` is in degrees-magnitude but `current_angle` accumulates in radians.** A full circle is `2*pi ≈ 6.283` radians, so the loop runs until `current_angle` reaches 360 radians.

**Overshoot factor = 360 / (2*pi) ≈ 57.30.**

Computed for the real GRBL config (`GRBL_STEP_SIZE=8`, `max_circle_diameter=1900`, radius=950):
- `angle_increment ≈ 0.008421` rad
- one true revolution = `2*pi / 0.008421 ≈ 746` steps
- loop runs `360 / 0.008421 ≈ 42,750` steps total
- = **~57.3 full revolutions** of the circle before the generator raises StopIteration.

**Important nuance:** `angle_to_pos` (`planner.py:194-197`) uses `np.sin`/`np.cos`, which interpret their argument as radians and are periodic mod 2*pi:

```python
def angle_to_pos(self, angle):
    return self.circle_center + self.circle_radius * np.array([-np.sin(angle), np.cos(angle)])
```

So the emitted positions are **periodic** — every 2*pi of accumulated angle retraces the *same* circle. This is NOT a spiral, NOT out-of-bounds, NOT a geometrically wrong path. Each revolution is a correct circle of the requested diameter centred at `circle_center`. The defect is purely "the circle is traced 57x instead of once before the generator ends."

---

## 2. REACHABILITY — is the circle routine actually used?

`CirclePlanner` is instantiated in three places:

- **GRBL** (`spf/grbl/grbl_interactive.py`): methods `tx_circle` (line 552), `rx_circle` (line 563), `rx_random_circle` (line 595). These are registered in `GRBLManager.routines` (lines 455-462): `"rx_circle"`, `"rx_random_circle"`, `"tx_circle"`.
- **Mavlink/drone** (`spf/mavlink/mavlink_controller.py:178`): `drone_get_planner(routine="circle")`.

Routine selection in data collection: `grbl_radio_collection.py:86` sets `yaml_config["routine"]` from the `-r/--routine` CLI arg; `get_default_gm(..., routine=...)` (line 160) wires it into `GRBLManager`, which dispatches via `self.routines[self.routine]()` (`grbl_interactive.py:475,486`).

So the circle routines **are live and selectable** — a collection launched with `--routine rx_circle` (or `tx_circle` / `rx_random_circle`, or drone `circle`) hits this exact code path. They are not dead code; they coexist with `bounce`/`center`. Whether any *current* campaign uses them depends on the launch command, but the path is fully reachable.

No checked-in YAML/config hard-codes `routine: rx_circle` (sampled configs show `routine: null`, `bounce`, `diamond`); `run_filters_report*.py` reference a `"circle"` movement label, implying circle datasets have existed historically.

---

## 3. ACTUAL EFFECT — does the bound control how far the circle goes, or does an outer consumer stop early?

**An outer stop condition dominates.** `yield_points` is a generator consumed in a planner thread (`grbl_interactive.py:531-532` via `move_to_iter`), while data collection is governed independently by a **fixed record count**:

- `spf/data_collector.py:455`: `for record_index in tqdm(range(self.yaml_config["n-records-per-receiver"])):` — collection ends after N records regardless of planner state.
- The planner generator and the collector run asynchronously; the collector samples whatever XY the platform currently holds.

`move_to_iter` (`grbl_interactive.py:421-434`) only terminates the planner when the generator raises StopIteration (`get_next_points`, lines 440-447) or `run_grbl` goes false. With the bug, StopIteration happens after ~57 revolutions instead of 1.

**Consequence in practice:**
- If `n-records-per-receiver` is reached before ~57 revolutions complete (the normal case — collections are time/record-bounded), the 360-radian bound is **never reached** and is irrelevant. The platform keeps tracing the correct circle; data is correct.
- The bug would only manifest if a run were intended to stop *when one revolution finishes* (relying on generator exhaustion to end the circle). No such reliance exists: collection length is set by record count, not by circle completion. So even at the boundary, the worst case is "the circle keeps going for more revolutions than a naive reader of the code expects" — still tracing the same correct circle.

So the overshoot does **not** produce wrong motion. It produces *more of the same correct motion* before the generator would self-terminate.

---

## 4. TRIGGER — what real run would drive a "wrong" circle?

No run drives geometrically wrong circle motion. The closest observable effect:

- A GRBL collection launched `--routine rx_circle` / `tx_circle` / `rx_random_circle`, or a drone run with `routine: circle`, where the operator *expected the circle to auto-stop after one lap* via generator exhaustion. It will instead keep looping (~57 laps) until the record count stops it. Since record count is the actual terminator, even this is benign.

There is **no platform/config combination** where the bug yields off-circle, out-of-bounds, or spiral motion, because position is periodic in radians.

---

## 5. REAL-WORLD TEST COVERAGE

`tests/test_dyanmics_and_planners.py` covers only `BouncePlanner` (`test_grbl_bounce`, `test_gps_bounce`; lines 12, 28). **There is NO test for CirclePlanner geometry** (grep for `Circle`/`circle` in that file returns nothing). A test would not catch this — and even a geometry test asserting "points lie on the circle" would *pass*, because every yielded point is on the correct circle. Only a test asserting "generator yields exactly one revolution / N≈746 points" would expose it.

---

## 6. SEVERITY VERDICT: **DOWNGRADE (confirmed defect, low impact)**

Justification:
- The radians-vs-360 mismatch is a **real bug** in the loop bound (confirmed, ~57.3x).
- BUT it does **not** cause wrong motion: `angle_to_pos` is periodic mod 2*pi, so every revolution is the correct circle — no spiral, no overshoot of geometry, no out-of-bounds.
- The 360 bound is **overridden in practice** by the fixed `n-records-per-receiver` collection stop (`data_collector.py:455`); the generator's termination point is irrelevant to data correctness.
- This is the documented downgrade criterion: "a bound that's overridden by an outer stop condition" → not P0.

It is not a silently-wrong-result data bug. Recorded XY positions are correct circle positions; downstream sees a valid circle trajectory (just possibly more laps than the author intended). Reclassify as a **low-severity code-clarity/correctness cleanup**, not a P0.

Residual minor risk: the `direction` randomization (`planner.py:186`) plus a near-zero start may mean the generator runs much longer than expected, very slightly increasing the chance a record-bounded run never "wraps back" cleanly — but this does not corrupt any sample.

---

## 7. FIX SKETCH

Make the bound match the radian accumulator (one full revolution = `2*pi`):

```python
# planner.py:205
while current_angle < 2 * np.pi:        # was: while current_angle < 360
```

(Equivalently, accumulate in degrees and convert in `angle_to_pos`, but the one-line bound fix is minimal and correct.)

**Downstream re-check warning before applying:** changing the bound from 360 to `2*pi` changes the generator from ~57 revolutions to exactly 1 revolution. If any operational workflow currently relies on the circle running "indefinitely" (until record count) by accident of this bug, capping it at one revolution would make the platform **stop moving after one lap** while collection continues — yielding many records at a single stationary point. Before merging, confirm that circle collections are configured so `n-records-per-receiver` corresponds to <= one revolution (~746 records at step=8, diam=1900), OR wrap the angle (`current_angle %= 2*pi`) inside a `while self.running:` loop so the circle repeats indefinitely like `PointCycle.yield_points` does. The latter is the safer behavior-preserving fix:

```python
current_angle = 0
while self.running:
    next_p = self.angle_to_pos(current_angle)
    yield from a_to_b_in_stepsize(current_p, next_p, step_size=self.step_size)
    current_p = next_p
    current_angle += self.direction * self.angle_increment   # no upper bound; runs until stop()/record count
```

Recommend the indefinite-loop form to preserve current observed behavior while removing the meaningless `360` constant.
