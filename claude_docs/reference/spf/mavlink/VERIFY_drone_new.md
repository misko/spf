# Adversarial Verification — new consequential claims (`mavlink_controller.py`)

**Source:** `/home/mouse9911/gits/spf/spf/mavlink/mavlink_controller.py`
**Doc under test:** `claude_docs/reference/spf/mavlink/drone_controller.md`
**Mode:** READ-ONLY. Skeptical. Exact lines quoted.
**Context:** `Drone` drives a real rover over MAVLink. Three threads touch shared state:
main, message-loop (`process_messages:779`, runs all `handle_*`), planner (`run_planner:466`).
None of the motion-state fields are lock-protected (the only synchronization primitives are
`message_condition` / `single_condition`, used to toggle `message_loop`/`single_operation`,
not to guard field reads). This is the backdrop for several claims below.

---

## MC-4 — Readiness EKF gate ORs `EKF_POS_HORIZ_REL` twice and omits `EKF_POS_HORIZ_ABS`

**VERDICT: CONFIRMED.  Severity: HIGH (safety-relevant).  Confidence: HIGH.
Does it genuinely weaken the arm/launch gate: YES.**

Exact construction, lines 264-268:

```python
self.healthy_ekf_flag = (
    EKF_STATUS_STRING_TO_DEC["EKF_ATTITUDE"]
    | EKF_STATUS_STRING_TO_DEC["EKF_POS_HORIZ_REL"]
    | EKF_STATUS_STRING_TO_DEC["EKF_POS_HORIZ_REL"]
)
```

The bit table (36-48):

```
1:  EKF_ATTITUDE
8:  EKF_POS_HORIZ_REL
16: EKF_POS_HORIZ_ABS
```

- `EKF_POS_HORIZ_REL` (bit 8) is ORed **twice** (lines 266 AND 267). The second OR is a
  no-op (`8 | 8 == 8`).
- `EKF_POS_HORIZ_ABS` (bit 16) is **never referenced** anywhere in the file
  (`grep` confirms only one occurrence, at the table definition line 41).
- Resulting mask: `1 | 8 | 8 = 9` (attitude + relative horizontal position only).

**How it feeds the gate.** `handle_EKF_STATUS_REPORT` (813-817):

```python
if msg.flags & self.healthy_ekf_flag == self.healthy_ekf_flag:
    self.ekf_healthy = True
else:
    self.ekf_healthy = False
```

`ekf_healthy` is required for `drone_ready` in `handle_HEARTBEAT` (863-871):

```python
if (
    mav_state_check
    and gps_check
    and gps_healthy
    # and guided_mode
    and self.ekf_healthy
):
    ...
    self.drone_ready = True
```

`drone_ready` is the barrier the planner blocks on before arming and issuing moves
(`run_planner:485-513`: `while not self.drone_ready: ... self.arm() ... self.move_to_point(home)`).

**Is the gate genuinely weaker than intended?** YES. The comparison is a strict subset test
(`flags & mask == mask`), so dropping bit 16 from `mask` makes it *easier* to satisfy, not
harder. With the bug the EKF is declared healthy as soon as it has a good attitude solution
plus a **relative** horizontal position estimate. `EKF_POS_HORIZ_ABS` is the bit ArduPilot
sets when the filter has a horizontal position fix **absolutely referenced to the earth frame
(GPS-aligned)** — i.e. the estimate that GPS waypoint navigation actually depends on.
`EKF_POS_HORIZ_REL` can be set from dead-reckoning / relative sources before the absolute fix
is good. So the rover can arm and begin executing absolute lat/long `reposition` commands
(`move_to_point` → `reposition`) while its absolute position estimate is still unconverged —
exactly the failure mode "launch with weaker GPS health than intended" describes.

**Mitigations / nuance (kept for honesty):** the gate is an AND of several conditions, so
EKF is not the *only* guard. `gps_check` (`self.gps[0] != 0`, i.e. a GLOBAL_POSITION_INT has
arrived) and `gps_healthy` (`MAV_SYS_STATUS_SENSOR_GPS in sensors_health`) must also hold.
Those reduce, but do **not** eliminate, the hazard: GPS-sensor-healthy + a non-zero reported
position can coexist with an EKF whose absolute horizontal position has not yet converged
(post-arm GPS glitch, slow EKF convergence, brief HORIZ_ABS dropout). In all such windows the
intended HORIZ_ABS guard is silently absent. The duplicate-OR also strongly signals the
author *intended* a second, distinct bit here (almost certainly `EKF_POS_HORIZ_ABS`), so this
is a real typo with a real safety consequence, not a stylistic quibble.

**Trigger:** any mission start where the EKF reports attitude+rel-pos healthy but abs-pos not
yet healthy. **Reachable on a live mission:** YES — this is on the normal arm/launch path.

---

## MC-5 — Motion-state fields read in `run_planner` with no lock while the handler thread writes them

**VERDICT: CONFIRMED (no locking; reads/writes are cross-thread).  Severity: MEDIUM.
Confidence: HIGH.**

Writers (message-loop thread, `handle_HEARTBEAT`, 838-843):

```python
self.mav_states = lookup_exact(msg.system_status, mav_states_list)
self.mav_mode = custom_mode_mapping[msg.custom_mode]
self.armed = (msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED) == MAV_MODE_FLAG_SAFETY_ARMED
```

`self.gps` is written by `handle_GLOBAL_POSITION_INT` (806) on the same thread.

Readers (planner thread, `run_planner` / `move_to_point`):
- `run_planner:485` `while not self.drone_ready:` ... `:487` reads `self.gps`, `self.ekf_healthy`
- `run_planner:498` `while self.mav_mode != "ROVER_MODE_MANUAL":`
- `run_planner:503` `while self.mav_mode != "ROVER_MODE_GUIDED":`
- `run_planner:508` `if not self.armed:`
- `move_to_point:450-458` reads `self.mav_mode`, `self.armed`, `self.motor_active` together
  to make the collision-avoidance arm/disarm decision.

**No lock.** There is no `threading.Lock` anywhere in the file; `acquire`/`with self.<lock>`
matches resolve only to the two `Condition` objects, which guard the message-loop/single-op
toggling, not these fields. The planner and the message loop are confirmed distinct threads
(`set_and_start_planner:350` daemon planner thread; `process_messages` runs on the
`message_loop_thread` created at `__init__:333`). So the reads and writes are genuinely
cross-thread and unsynchronized.

**Torn multi-field decision is real.** The `move_to_point` collision logic (450-458) branches
on `mav_mode`, `armed`, and `motor_active` as if they form a consistent snapshot, but a
HEARTBEAT can land between any two of those reads and update `mav_mode`/`armed`. CPython's GIL
makes each individual attribute read/write atomic (no partial-string / partial-int tearing),
so the *severity* is bounded to logical inconsistency across fields rather than memory
corruption — e.g. acting on an old `armed` with a new `mav_mode`. The practical worst case is
a mis-timed `arm()`/`disarm()` or a missed collision-avoidance disarm for one ~0.1 s loop
iteration; the loop re-reads and self-corrects on the next pass. Hence MEDIUM, not HIGH: real
data race, real cross-thread reads, but the consequence is a transient logic glitch rather
than a hard failure, and there is no struct-tearing thanks to the GIL.

**Trigger:** routine — HEARTBEAT (writer) and the planner loop (reader) run continuously.
**Reachable on a live mission:** YES, continuously; impact is intermittent.

---

## MC-10 — `handle_HEARTBEAT` does unguarded `custom_mode_mapping[msg.custom_mode]` → KeyError kills the message thread

**VERDICT: CONFIRMED.  Severity: HIGH (safety-relevant: strands the planner / blinds all
state).  Confidence: HIGH.**

Line 838-840:

```python
self.mav_mode = custom_mode_mapping[
    msg.custom_mode
]
```

Dict access is **unguarded** — no `.get(...)`, no `in` check, no try/except. The dispatcher
`process_message` (942-948) calls the handler with **no exception handling**:

```python
if msg_type in self.message_handlers:
    self.message_handlers[msg](self, msg)   # (line 948; no try/except)
```

and the loop `process_messages` (779-791) is a bare `while True:` with no try/except either.
So a `KeyError` propagates out of the handler, out of `process_message`, out of
`process_messages`, and **terminates the daemon message-loop thread**. After that:
- no further `handle_*` runs, so `gps`, `armed`, `mav_mode`, `ekf_healthy`, `last_heartbeat`
  all freeze at their last values;
- the planner thread keeps running on stale state (e.g. blocks forever in
  `while self.mav_mode != "ROVER_MODE_GUIDED"` at 503, or keeps issuing `move_to_point`
  against a frozen `self.gps`, never reaching tolerance);
- the watchdog/`last_heartbeat` staleness is only meaningful if something reads it on another
  thread — the loop that would update it is dead.

**Which modes are unmapped.** `custom_mode_mapping` (140-153) contains keys
`{0,1,3,4,5,6,7,10,11,12,15,16}`. Any other `custom_mode` raises `KeyError`. Concretely, the
ArduPilot Rover mode numbers **absent** from the dict include:
- **2** (legacy/`LEARNING` in older Rover firmware),
- **8** (`SMART_RTL` is mapped to 12, but mode 8 itself is unmapped),
- **9**,
- **13** (`DOCK` on some builds),
- **14**,
- and anything ≥17.

So if the autopilot ever reports a mode the dict doesn't list (firmware version skew, a mode
the operator selects on the TX such as a future/unlisted mode, or a transient mode number),
the single offending HEARTBEAT kills the whole message thread.

**Trigger:** receipt of one HEARTBEAT whose `custom_mode` is not a dict key. **Reachable on a
live mission:** YES — operator-selectable modes and firmware skew make this attainable without
any code change; it requires only that the rover ever advertise an unlisted mode. Severity is
HIGH because the failure is silent (no log, no buzzer) and disables all state ingestion,
including the collision-avoidance inputs and the `last_heartbeat` liveness signal.

---

## MC-3 / MC-8 — `move_to_point` blocks with no timeout; `move_to_home.max_wait` does not bound it

**VERDICT: CONFIRMED.  Severity: HIGH (RTL/return-home can hang forever).  Confidence: HIGH.**

**MC-3 — `move_to_point` has no timeout (432-464):**

```python
def move_to_point(self, point, log_interval=5):
    self.reposition(lat=point[1], long=point[0])
    last_message = None
    while self.distance_to_target(point) > self.tolerance_in_m:
        ...
        time.sleep(0.1)
    logging.info(...)
    return True
```

The only loop exit is `distance_to_target(point) <= tolerance_in_m`. There is no time bound,
no iteration cap, no abort flag (note it does **not** check `self.planner_should_move`). If the
rover never reaches tolerance — disarmed by collision-avoidance and never re-armed (the
`elif not self.armed and not collision_soon: self.arm()` branch only fires when the distance
finder is present and clears), stuck behind an obstacle, GPS drift keeping it just outside
tolerance, motors faulted, mode switched out of GUIDED so `reposition` is ignored — this loop
spins forever. It can only return `True`; it has no failure return.

**MC-8 — `move_to_home`'s `max_wait` does not bound the inner `move_to_point` (544-570):**

```python
def move_to_home(self, max_wait=300):
    ...
    self.move_to_point(self.home)          # <-- line 558, BEFORE the timed loop

    start_time = time.time()
    while time.time() - start_time < max_wait:   # <-- timer starts AFTER move_to_point returns
        dist = self.distance_to_target(self.home)
        if dist <= self.tolerance_in_m:
            return True
        time.sleep(1)
    logging.warning("Timed out waiting for the drone to reach home.")
    return False
```

The `max_wait` clock (`start_time = time.time()`) is set on line 560, **after**
`move_to_point(self.home)` on line 558 has already returned. So `max_wait` bounds only the
post-arrival distance-polling loop — which is essentially redundant, since `move_to_point`
already guarantees `distance <= tolerance` before it returns (so the very first iteration of
the `while` typically returns `True` immediately). The `max_wait` parameter therefore provides
**no protection** for the actual blocking call. If `move_to_point(self.home)` hangs (per MC-3),
`move_to_home` hangs forever regardless of `max_wait`, the "Timed out" warning is never
reached, and RTL/return-home never completes or fails out. The doc/contract ("blocks until the
drone has reached home or the max_wait time has elapsed", 546-548) is **false** with respect to
the hang case.

**Trigger:** any return-home where the rover cannot reach `home` within tolerance.
**Reachable on a live mission:** YES — RTL/return-home is a normal end-of-mission action and a
common emergency action; combined with MC-3 this is an unbounded hang on a safety-critical
path. HIGH.

---

## Summary table

| Claim | Verdict | Final severity | Safety flag |
|---|---|---|---|
| MC-4  (EKF gate double-OR / HORIZ_ABS omitted) | CONFIRMED | HIGH | YES — weakens arm/launch GPS-health gate |
| MC-5  (unlocked cross-thread motion-state reads) | CONFIRMED | MEDIUM | YES — transient torn decisions (GIL bounds it) |
| MC-10 (unguarded `custom_mode_mapping` KeyError) | CONFIRMED | HIGH | YES — kills message thread, strands planner |
| MC-3 / MC-8 (no-timeout move; max_wait doesn't bound it) | CONFIRMED | HIGH | YES — RTL/return-home can hang forever |

All four claims are real, triggerable, and reachable on a live mission. No claim was REFUTED
or PARTIAL.
