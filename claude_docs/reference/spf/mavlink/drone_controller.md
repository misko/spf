# Reference — `spf/mavlink/mavlink_controller.py` (`Drone`) + `mavlink_send.py` + `mavparm.py`

**Scope:** the `Drone` class, `drone_get_planner`, the MAVLink message handlers, the
readiness state machine, motion commands, and the threading model. Plus the two small
sibling modules `mavlink_send.py` and `mavparm.py`.

**Pinned to commit:** `574c4548e23c18d95ff3c2b72d3b63f0b1f42251` (target file last touched
2025-04-26). All `file:line` citations are against that revision.

**Safety note:** This drives a **real rover** over a serial/MAVLink link. Several methods
issue `arm`/`disarm`/`reposition`, run `sudo shutdown`, or call `sys.exit` — and most of
them run on a **background thread**, not main. Read "Threading & safety" before trusting
any single method in isolation.

**Units / frames** (link [`00_concepts/conventions.md`](../../00_concepts/conventions.md)):
- GPS points are `[long, lat]` in **decimal degrees** (note ordering). `distance_to_target`
  calls `swap_lat_long` before `haversine` because `haversine` wants `(lat, long)`.
- MAVLink wire integers are scaled: lat/lon ×1e7, heading (`hdg`) ×100, unix time in µsec,
  boot time in msec. Handlers undo these scalings.
- `reposition` re-applies the ×1e7 scaling when sending `COMMAND_INT`.

---

## Threading & safety (read first)

There are up to **three** threads touching a `Drone`:

| Thread | Created | Started by | Body | What it does |
|---|---|---|---|---|
| **Main** | — | process | `mavlink_radio_collection.__main__` / `mavlink_controller_run` | constructs Drone, polls `drone_ready`/`is_planner_in_control`, calls `move_to_home` |
| **Message loop** | `__init__:333` (`daemon=True`) | `start()` (`392`) | `process_messages` (`779`) | blocking `recv_match`, dispatches every MAVLink msg to a `handle_*` |
| **Planner** | `set_and_start_planner:350` (`daemon=True`) | `set_and_start_planner` (`345`) | `run_planner` (`466`) | waits for ready, then issues `set_home`/`arm`/`move_to_point` |
| **Distance finder** | inside `DistanceFinderController` | `__init__:252` | (external) | ultrasonic range; read in `move_to_point` |

**Which handlers run on the message thread:** *all* of `handle_*` (dispatched by
`process_message` → `message_handlers`, `942-948`). They run on the **message-loop thread**,
never main. This is the central hazard: a handler is not just updating state, several of them
take **drastic, process-level actions**:

- `handle_RC_CHANNELS` (`897`): on `chan9_raw > 1500` runs `subprocess.run(["sudo","shutdown","0"])`
  — **powers the host off from a message handler**. On `chan7_raw > 1000` calls
  `self.reboot()` then `sys.exit(1)`. On `chan10_raw>1500` triggers a live compass calibration.
  All from the daemon message thread. See Issue **MC-1**.
- The planner thread (`run_planner`) issues motion commands while the message thread mutates the
  same `self.*` fields it reads (`drone_ready`, `mav_mode`, `armed`, `gps`, `motor_active`,
  `disable_distance_finder`). **No locks** guard these cross-thread reads/writes. See **MC-5**.

**The single-operation mechanism** (`single_operation_mode_on/off`, `process_messages`) is the
*only* synchronization primitive, and it is a cooperative pause of the message loop — not a
mutex over drone state. Used to get exclusive serial access during `set_home` (in `run_planner`)
and param load. It uses two `Condition`s; correctness depends on the message thread reaching its
check point. See **MC-6** for an analysis of its reentrancy assumptions.

**Blocking / unbounded waits that can strand a mission** (no timeout → hang forever):
- `run_planner:485` `while not self.drone_ready: sleep(10)` — never times out.
- `run_planner:498/503` `while self.mav_mode != "ROVER_MODE_MANUAL"/"GUIDED": sleep(10)` — never
  times out (waits for a human to flip the RC switch; intended, but unbounded).
- `move_to_point:436` `while distance_to_target > tolerance` — no timeout; loops until reached
  or collision-avoidance disarms. A target the rover can't reach loops forever. See **MC-3**.
- `update_all_parameters:374` blocks up to `timeout` then **`sys.exit(1)`** (kills the process).
- `single_operation_mode_on:766` waits on a Condition with no timeout.

---

## Functions

### `drone_get_planner(routine, boundary)` · `mavlink_controller.py:176` · ✅ Active

**Summary.** Factory mapping a routine name to a configured motion `Planner` over a GPS
boundary polygon. The one constructor the collection entry point calls to build the path the
rover will drive.

**Signature**
- Inputs: `routine` (str ∈ {`circle`,`center`,`bounce`,`diamond`}); `boundary` (np `Nx2`,
  `[long,lat]` degrees).
- Returns: a `CirclePlanner` / `StationaryPlanner` / `BouncePlanner` / `PointCycle`.
- Side effects: `diamond` uses **unseeded** `np.random.rand()` (`211`) and `np.flip`; unknown
  routine raises `Exception("Missing planner")` (`224`).

**Called from**
- `mavlink_radio_collection.py:256` → into `drone.set_and_start_planner(...)`.
- Re-exported & imported there (`18-22`). Fan-in: 1 prod call site (+ tests via the CLI).

**Contract**
- *Pre:* `boundary` must be a valid convex polygon (passed to `Dynamics`, which runs
  `ConvexHull` and raises `BoundingBoxIsNonConvexException` if not). `diamond` needs ≥4 points
  (`boundary_to_diamond` indexes 0-3).
- *Post:* a `Planner` whose `dynamics.bounding_box` is `boundary`; `run_planner` later reads
  `planner.dynamics.bounding_box.mean(axis=0)` as home.

**Tested by** — indirectly: `test_mavlink_radio_collect.py` (`center`), `test_in_simulator.py`
(`circle`). No direct unit test.

**Analysis** — *Correctness:* tiny magic constants (`circle_diameter=0.0003` etc.) are degrees,
tuned to the GPS scale; not validated against boundary size. *Issue:* `circle` inherits the
`CirclePlanner` deg/rad loop bug documented in the planner cluster (not this file). No issues in
this factory itself beyond the unseeded randomness (reproducibility, not safety).

---

### `Drone.__init__(connection, tolerance_in_m=5, distance_finder=None, fake=False, ignore_mode=False)` · `:236` · ✅ Active

**Summary.** Builds the controller: stashes the mavutil connection, snapshots the mode mapping,
creates (but does **not** start) the message-loop thread, initializes all the cross-thread state
fields, and starts the distance-finder thread if one was supplied.

**Signature**
- Inputs: `connection` (a `mavutil` connection, or `None` when `fake=True`); `tolerance_in_m`
  (float, meters — arrival tolerance for `move_to_point`); `distance_finder`
  (`DistanceFinderController` or None); `fake` (bool — skip all MAVLink I/O); `ignore_mode`
  (bool — bypass RC mode gating in `run_planner`).
- Returns: None.
- Side effects: **starts `distance_finder.run_in_new_thread()`** (`252`) immediately; reads
  `connection.mode_mapping()` and `connection.sysid_state[...]` (MAVLink I/O) when not fake
  (`257-260`); sets numba log level (`254`); constructs two `Condition`s and the message thread
  (unstarted, `333`); calls `reset_params`.

**Called from**
- `mavlink_radio_collection.py:229` (real) and `:234` (fake, `connection=None`).
- `mavlink_controller_run:1067/1086` (CLI). Fan-in: 4 prod call sites + tests.

**Contract**
- *Pre:* if `fake=False`, `connection` must be a live connection that has already received a
  heartbeat (so `mode_mapping()` and `sysid_state[sysid]` are populated). Callers do
  `wait_heartbeat`/`recv_match` first (`mavlink_radio_collection` relies on the connection being
  fresh; `mavlink_controller_run:1056-1078` explicitly waits).
- *Post:* message thread exists but is **not running** until `start()`. `planner`/
  `planner_in_control`/`home`/`launch_home`/`lat`/`long` attributes **do not exist yet** — they
  are created lazily by `set_and_start_planner`, `set_home`, `handle_*`. Accessing them before
  the relevant call raises `AttributeError` (see **MC-2**).

**Tested by** — both fake (`test_mavlink_radio_collect`) and real-sim (`test_in_simulator`).

**Analysis**
- *Issue **MC-4** (copy-paste, ✅ verified, P2):* `healthy_ekf_flag` (`264-268`) ORs
  `EKF_STATUS_STRING_TO_DEC["EKF_POS_HORIZ_REL"]` **twice** and omits `EKF_POS_HORIZ_ABS`. The
  flag is therefore `EKF_ATTITUDE | EKF_POS_HORIZ_REL` (value 9) — the second OR is a no-op.
  This is the threshold `handle_EKF_STATUS_REPORT` uses to declare `ekf_healthy`, which gates
  `drone_ready`. **`⚠️ no-crash` — silently wrong:** the readiness bar is *laxer* than the name
  implies (relative position only, no absolute fix required). Safety-relevant (lets the rover be
  declared "ready" with only relative EKF position), but it makes readiness *easier*, not a
  crash. Verify intent before "fixing" — downstream timing may rely on this looser gate.
- *Issue **MC-2** (lazy attributes, 🔍 analysis, P2):* `is_planner_in_control()` reads
  `self.planner_in_control`, which only exists after `set_and_start_planner`. The
  `mavlink_radio_collection` flow always calls `set_and_start_planner` before
  `is_planner_in_control`, so it is safe *there*; a different caller order would `AttributeError`.
- *Correctness:* `fake=True` path never starts the message loop (caller skips `start()`), so in
  fake mode no handler ever runs and `drone_ready` stays False. The collection main loop guards
  every `drone_ready`/`is_planner_in_control` wait with `not args.fake_drone`
  (`mavlink_radio_collection.py:241,312`), so fake mode does not hang. Confirmed.

---

### `Drone.start()` · `:392` · ✅ Active
Starts `message_loop_thread`, returns `self`. After this, handlers run on that thread. Fan-in:
`mavlink_radio_collection.py:232`, `mavlink_controller_run:1091`. Side effect: spawns the thread
(daemon). No issues.

### `Drone.set_and_start_planner(planner)` · `:345` · ✅ Active

**Summary.** Attaches a planner, initializes `planner_in_control=False`, and **starts the planner
thread** running `run_planner`. The point at which the rover may begin to move autonomously.

- Inputs: `planner` (a `Planner`). Returns: None.
- Side effects: sets `self.planner`, `self.planner_in_control`; asserts planner not None
  (`348`); **starts `planner_thread`** (`351`, daemon).
- Called from: `mavlink_radio_collection.py:255`. Fan-in: 1.
- *Analysis:* sets `planner_in_control` *before* starting the thread (`347` then `351`), so the
  `is_planner_in_control` getter is safe once this returns. No lock — but the only writer of
  `planner_in_control` after this is the planner thread itself, and the only reader is main; a
  bool read/write is atomic in CPython. ✅ no issue beyond **MC-5** general statement.

### `Drone.run_planner()` · `:466` · ✅ Active — planner thread body

**Summary.** The autonomous-mission state machine, run on the planner thread. Sets home, waits
for `drone_ready`, then (unless `ignore_mode`) waits for the human to flip the rover into MANUAL
then GUIDED via RC, arms, drives to home, and then streams planner points into `move_to_point`
forever until `planner_should_move` goes False.

- Inputs: none (reads `self`). Returns: None.
- Side effects: `single_operation_mode_on/off` around `set_home` (`475-480`); many blocking
  sleeps; `buzzer`; `arm`; repeated `move_to_point` (MAVLink reposition + arm/disarm); sets
  `self.planner_in_control` True/False; logging.
- Called from: planner thread target (`set_and_start_planner:350`). Fan-in: 1 (thread).

**Contract / Analysis**
- *Pre:* `self.planner` set (guaranteed by `set_and_start_planner`); message loop running so
  `drone_ready`/`mav_mode`/`armed` get updated — **else it blocks forever** (the message thread
  is what flips `drone_ready`).
- *Issue **MC-7** (unbounded waits, ✅ verified, P2, safety-relevant):* the three `while … sleep`
  loops (`485`, `498`, `503`) have **no timeout**. If the EKF never goes healthy, or the operator
  never switches to GUIDED, the planner thread blocks indefinitely and the mission never starts
  (main side just keeps logging "Waiting for drone to start moving"). This is by-design human
  gating, but there is no watchdog. Documented, not a crash.
- *Issue **MC-5** (state race, 🔍 analysis, P1, safety-relevant):* `run_planner` (planner thread)
  reads `self.mav_mode`, `self.armed`, `self.drone_ready`, `self.gps` while
  `handle_HEARTBEAT`/`handle_GLOBAL_POSITION_INT` (message thread) writes them, with no lock.
  Individual attribute reads are atomic in CPython, but multi-field decisions (e.g. "armed AND
  guided AND not motor_active") can observe a **torn combination** across a heartbeat boundary.
  Worst realistic effect: an extra `reposition`/`arm`/`disarm` cycle. Needs an adversarial
  confirm of exact interleavings before promotion; flagged because it is motion control.
- *Correctness:* the `ignore_mode` branch (`491-495`) sets `planner_in_control=True` and idles —
  used by `--ignore-mode` to bypass RC gating (the fake/bench path). The normal branch only
  reaches the move loop after MANUAL→GUIDED→armed. `planner_in_control` is set True *inside* the
  move loop (`533`) — i.e. only after the first `move_to_point` returns — so the main process's
  `is_planner_in_control` wait is satisfied once real motion has begun. Confirmed against
  `test_in_simulator.test_guided_mode_moving_and_recording`.

### `Drone.move_to_point(point, log_interval=5)` · `:432` · ✅ Active

**Summary.** Commands a single GUIDED reposition to a `[long,lat]` point and **blocks** until the
rover is within `tolerance_in_m`, running ultrasonic collision-avoidance (disarm/arm) in the loop.

- Inputs: `point` (`[long,lat]` deg); `log_interval` (s). Returns: `True` (only on arrival).
- Side effects: `reposition`; in-loop `disarm`/`arm` on imminent collision (`<30` from distance
  finder, `448`); sleeps; logging. Reads `self.distance_finder.distance`, `self.armed`,
  `self.mav_mode`, `self.motor_active`, `self.disable_distance_finder` (all cross-thread).
- Called from: `run_planner:513,532` and `move_to_home:558`. Fan-in: 2.

**Contract / Analysis**
- *Pre:* drone in GUIDED and armed for motion to actually occur; otherwise it still loops (it
  only acts on collision avoidance when `mav_mode == "ROVER_MODE_GUIDED"`). `distance_finder`
  may be None (guarded).
- *Issue **MC-3** (no timeout, ✅ verified, P2, safety-relevant):* the `while distance_to_target >
  tolerance` loop (`436`) has **no max-iteration/timeout**. If the rover physically cannot reach
  the point (stuck, GPS drift, persistent obstacle keeping it disarmed), this never returns and
  the planner thread is stuck here. `move_to_home` wraps a *second* timeout loop around its own
  `distance_to_target` check (`561`) but still calls the unbounded `move_to_point` first
  (`558`) — so `move_to_home`'s `max_wait` does **not** bound the inner block. See **MC-8**.
- *Correctness:* collision logic reads `self.distance_finder.distance` (updated on the distance
  thread) — three-way race but effects are conservative (disarm-on-doubt). Disarm→sleep(2)→re-arm
  hysteresis present. The `>30`/`<30` threshold is hardcoded (units = whatever the distance
  finder reports, presumably cm — ⚠️ unverified unit).

### `Drone.move_to_home(max_wait=300)` · `:544` · ✅ Active

**Summary.** Drives back to the stored `self.home` and waits (bounded by `max_wait`) for arrival.
Returns True on arrival, False on timeout or if `home` was never set.

- Inputs: `max_wait` (s). Returns: bool. Side effects: `move_to_point(self.home)` (motion);
  logging.
- Called from: `mavlink_radio_collection.py:356` (end of run). Fan-in: 1.
- *Issue **MC-8** (timeout not honored, ✅ verified, P2):* guards `hasattr(self,"home")` (good,
  prevents `AttributeError`), but then calls `self.move_to_point(self.home)` (`558`) **before**
  the timed loop — and `move_to_point` is itself unbounded. So if the rover can't reach home,
  `move_to_home` hangs inside `move_to_point` and never reaches its own `max_wait` check. The
  `max_wait` loop only ever runs *after* arrival, where it returns True almost immediately.
  Effect: `max_wait` is largely dead. Safety-relevant (end-of-mission RTL can hang). Note this is
  GUIDED-reposition "home", **not** the autopilot RTL mode (`set_rtl_mode` exists but is commented
  out at `556`).

### `Drone.is_planner_in_control()` · `:401` · ✅ Active
Returns `self.planner_in_control`. Fan-in: `mavlink_radio_collection.py:312,352`. Safe after
`set_and_start_planner`. See **MC-2** for the lazy-attribute caveat.

### `Drone.get_position_bearing_and_time()` · `:404` · ✅ Active
Snapshot dict `{gps, heading, gps_time}`. **No lock** — reads three fields written by the message
thread; `gps` is a 2-vector replaced wholesale in `handle_GLOBAL_POSITION_INT`, so a reader gets
a consistent pair (reference swap is atomic), but `heading`/`gps_time` can be from a different
message. Low impact. Fan-in: external (data collector via `position_controller`). 🔍.

### `Drone.distance_to_target(target_point)` · `:409` · ✅ Active
Haversine meters between `self.gps` and `target_point` (both `[long,lat]`; `swap_lat_long` first).
**Dead line** `415` `return np.linalg.norm(...)` after the active `return` — unreachable. Pure
otherwise. P3 (cosmetic).

### Message handlers (all run on the **message-loop thread**)

| Handler | Line | Writes | Notes |
|---|---|---|---|
| `handle_HOME_POSITION` | 793 | `self.launch_home` (÷1e7) | `[long,lat]`. ✅ |
| `handle_GLOBAL_POSITION_INT` | 803 | `lat,long,gps,heading` | `gps=[long,lat]`; `heading=hdg/100` (deg). ✅ |
| `handle_GPS_RAW_INT` | 809 | `gps_satellites,gps_fix_type` | indexes `gps_fix_type_str_to_num[int]` — **misnamed map** (int→str), works. See **MC-9**. |
| `handle_EKF_STATUS_REPORT` | 813 | `ekf_healthy` | uses the buggy `healthy_ekf_flag` (**MC-4**). |
| `handle_PARAM_VALUE` | 825 | `params,param_count` | feeds `update_all_parameters` poll. ✅ |
| `handle_HEARTBEAT` | 835 | `last_heartbeat,mav_states,mav_mode,armed,drone_ready` | the readiness state machine; see below. ✅ |
| `handle_SYSTEM_TIME` | 873 | `gps_time,time_since_boot` | usec/msec → s. ✅ |
| `handle_SYS_STATUS` | 877 | `sensors_present/enabled/health` | bitmask decode. ✅ |
| `handle_STATUSTEXT` | 888 | — | logs. ✅ |
| `handle_SERVO_OUTPUT_RAW` | 919 | `motor_active` | 1500/1500 ⇒ idle. ✅ |
| `handle_RC_CHANNELS` | 897 | `disable_distance_finder` **+ process-level side effects** | **MC-1** below. |
| `handle_NAV_CONTROLLER_OUTPUT` / `handle_COMMAND_ACK` / `handle_RC_CHANNELS_SCALED` | 798/819/893 | — | no-op stubs 🧪. |

#### `Drone.handle_HEARTBEAT(msg, log_interval=5)` · `:835` — the readiness state machine

Sets `mav_mode = custom_mode_mapping[msg.custom_mode]` (**MC-10**: an unmapped custom mode (e.g.
2, 8, 9, 13, 14) raises `KeyError` on the message thread — only modes in the dict at `140-153`
are handled; ArduPilot Rover modes 2/8/9/13/14 are absent). Computes `drone_ready` once
(latched, never cleared) when: a STANDBY/ACTIVE mav_state, a non-zero GPS, GPS sensor health,
and `ekf_healthy` (`863-871`). Note `guided_mode` is computed but the GUIDED requirement is
**commented out** (`867`) — readiness does not require GUIDED. ✅ verified.

#### `Drone.handle_RC_CHANNELS(msg)` · `:897` — ⚠️ dangerous side effects

- `chan9_raw > 1500` → `subprocess.run(["sudo","shutdown","0"])` (`899`) — **host power-off**.
- `chan10_raw > 1500` → `run_compass_calibration()`.
- `chan7_raw > 1500` → `reboot(force=True)`; `elif > 1000` → `reboot()` then `sys.exit(1)`
  (`909`) — **terminates the process** from the message thread (the daemon thread calls
  `sys.exit`, which raises `SystemExit` only in that thread; whether the process exits depends on
  it being a daemon — see **MC-1** note).
- `chan12_raw > 1000` toggles `disable_distance_finder` (disables ultrasonic collision avoidance).

**This is KNOWN_ISSUES #18 — CONFIRMED** (`899` sudo shutdown, `909` sys.exit, both reachable
from a live RC link on the message thread).

### `Drone.update_all_parameters(timeout=50)` · `:370` · ⚠️ `sys.exit` on timeout
Blocks polling `param_count` vs `len(self.params)`; on timeout logs and **`sys.exit(1)`** (`379`).
Called from `mavlink_controller_run:1143` (CLI param save/load/diff). Reachable. P2 — a transient
param-fetch stall kills the CLI rather than returning a failure the caller could handle.

### Other commands (one-liners)
`buzzer` (`353`, 5× retry, swallows `AttributeError`), `send_status` (`396`), `erase_logs`
(`416`), `set_rtl_mode` (`572`), `run_compass_calibration` (`592`), `request_home` (`608`),
`set_home` (`630`, also sets `self.home=[long,lat]`), `turn_off_hardware_safety` (`648`,
⚠️unverified intent — sends `set_mode` with `MAV_MODE_FLAG_DECODE_POSITION_SAFETY` as base mode,
likely incorrect usage), `reboot` (`655`), `disarm` (`678`), `arm` (`694`), `reposition` (`710`,
COMMAND_INT, lat/lon ×1e7), `do_mission` (`741`, 🧪 no callers), `ack` (`757`, 🧪 commented at all
call sites), `get_cmd` (`538`, caches name↔num), `set_mode` (`950`). All issue MAVLink writes on
**whatever thread calls them** (planner thread for the motion ones; main for CLI ones).

### `single_operation_mode_on/off` + `process_messages` · `:762/770/779`
Cooperative pause of the message loop for exclusive serial access. `process_messages` holds
`message_condition` and, when `single_operation` is requested, sets `message_loop=False`, notifies
`single_condition`, and waits. `single_operation_mode_on` asserts `not self.single_operation`
(`763`) — a **second** concurrent single-op request would `AssertionError`. Only `run_planner`
(around `set_home`) and the CLI param-load path use it, never concurrently in the current callers.
*Issue **MC-6** (🔍 analysis, P2):* the asserts make these non-reentrant; nested/overlapping use
would crash the calling thread. Not reached by current callers.

### Module helpers
`lookup_bits`/`lookup_exact` (`227/231`, decode mavlink bitmask/exact), `get_ardupilot_serial`
(`954`, globs `/dev/serial/by-id/usb-ArduPilot*`, returns None unless exactly one),
`get_mavlink_controller_parser` (`962`), `mavlink_controller_run` (`1038`, CLI main — many
`sys.exit`, blocking waits), `set_drone_mode` (`1172`, 3× retry GUIDED/MANUAL, `sys.exit(1)` on
bad mode).

---

## `spf/mavlink/mavlink_send.py` · 🧪 demo script

Top-level script (no functions/classes): connects to a **hardcoded** `/dev/ttyACM1` @ 57600,
waits for heartbeat, then streams a random `NAMED_VALUE_FLOAT "TEMP"` once/sec in an infinite
loop. Not imported anywhere (verified: no importers). Example/throwaway. Side effects: opens
serial, infinite loop, prints, `KeyboardInterrupt` exit. No callers, no tests.

---

## `spf/mavlink/mavparm.py` · ✅ vendored (ArduPilot pymavlink)

`MAVParmDict(dict)` — param load/save/diff/set over MAVLink. Used by `Drone.reset_params`
(`self.params = MAVParmDict()`) and the CLI param paths.

| Method | Line | Purpose | Side effects / notes |
|---|---|---|---|
| `__init__` | 16 | dict + `exclude_load` list + `mindelta=1e-6` | ✅ |
| `mavset` | 34 | set one param, wait ≤1s/retry for `PARAM_VALUE` ack | **blocking** poll; `struct.pack`; prints on timeout; mutates `self[name]`. ✅ |
| `save` | 80 | write params to file (wildcard filter) | opens/writes file. ✅ |
| `load` | 98 | read file; if `mav` given, push changed params via `mavset` | **reads file**; returns `False` on open-fail else `(count,changed)`; **stray debug** `print("WRITTING",…)` at `126` (**MP-1**, P3). |
| `show_param_value`/`show` | 153/156 | print params | ✅ |
| `diff` | 163 | count differences vs another file | returns `None` (implicit) if `other.load` is falsy (`174-175`) — **MP-2**. |

**Issues**
- **MP-2** (✅ verified, P2): `diff` returns `None` when the other file fails to load
  (`174-175`); caller `mavlink_controller_run:1151` does `sys.exit(diffs)` → `sys.exit(None)` →
  exit code 0 (success) on a *failed* diff. Reachable only if the diff file is unreadable, but the
  caller pre-checks `os.path.isfile` (`1146`), so the readable-but-empty/garbage case is the live
  trigger. Also note `test_load_and_diff_params` relies on `diff` returning a non-zero count to
  raise `CalledProcessError` — a None return would silently pass.
- **MP-3** (✅ verified, P3): `load` returns bare `False` (not a tuple) on open failure, but
  `mavlink_controller_run:1161` unpacks `count, changed = drone.params.load(...)` → `TypeError`
  if the file is missing. Caller guards with `os.path.isfile` (`1155`) first, so dead in practice.
- **MP-1** (✅ verified, P3): unconditional debug `print("WRITTING", …)` at `126`.

---

## Cluster summary

| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `drone_get_planner` | ✅ | 1 (+CLI) | ❌ indirect | unseeded `diamond` random; inherits CirclePlanner deg/rad |
| `Drone.__init__` | ✅ | 4 | ❌ indirect | **MC-4** `healthy_ekf_flag` double-OR (laxer readiness) |
| `set_and_start_planner` | ✅ | 1 | ❌ indirect | starts autonomous motion thread |
| `run_planner` | ✅ | 1 (thread) | ✅ sim | **MC-7** unbounded waits; **MC-5** state race |
| `move_to_point` | ✅ | 2 | ✅ sim | **MC-3** no timeout → can hang mission |
| `move_to_home` | ✅ | 1 | ❌ | **MC-8** `max_wait` not honored (inner block unbounded) |
| `handle_RC_CHANNELS` | ⚠️ | msg thread | ❌ | **MC-1 / KI #18** sudo shutdown + sys.exit on msg thread |
| `handle_HEARTBEAT` | ✅ | msg thread | ✅ sim | **MC-10** KeyError on unmapped custom_mode |
| `handle_EKF_STATUS_REPORT` | ✅ | msg thread | ✅ sim | depends on buggy MC-4 flag |
| `update_all_parameters` | ⚠️ | 1 (CLI) | ✅ sim | `sys.exit(1)` on timeout |
| `MAVParmDict.diff` | ✅ | 1 (CLI) | ✅ sim | **MP-2** None→exit 0 on failed diff |

**Cross-cutting themes**
1. **Handlers do far more than update state.** The message-loop thread can power off the host
   (`sudo shutdown`), `sys.exit`, reboot the autopilot, and start a compass calibration — all
   from `handle_RC_CHANNELS`. Any audit of "what can move/kill the rover" must include the
   handler table, not just the public methods. (KI #18.)
2. **No locking over cross-thread `Drone` state.** Three threads (main, message, planner) share
   `mav_mode`/`armed`/`gps`/`drone_ready`/`motor_active`/`disable_distance_finder` with only the
   cooperative single-operation pause for serial access. Atomic single-field reads keep this from
   being catastrophic, but multi-field motion decisions can observe torn state (**MC-5**).
3. **Unbounded waits everywhere.** `run_planner` (3 loops), `move_to_point` and `single_operation`
   can block forever, and a never-healthy EKF still strands the mission with no watchdog
   (**MC-7**). Two of these were narrowed on 2026-08-03: a rover that stops covering ground in
   GUIDED is now detected and handed to the operator in MANUAL (ROVER_RUNBOOK §17), and
   `move_to_home`'s `max_wait` is a shared monotonic deadline that actually bounds the drive
   instead of starting after it. `move_to_point` still has no overall deadline for a rover that
   moves without arriving (**MC-3/8**).
4. **`sys.exit` as error handling.** `update_all_parameters`, `handle_RC_CHANNELS`,
   `mavlink_controller_run`, `set_drone_mode` all exit the process instead of returning errors.
5. **Misnamed maps / dead code:** `gps_fix_type_str_to_num` is int→str (works, lies);
   `distance_to_target` dead `np.linalg.norm` tail; `MAVParmDict` debug print.

---

## NEW-ISSUES (this pass)

- **MC-1** `mavlink_controller.py:899,909` — `handle_RC_CHANNELS` runs `sudo shutdown` / `sys.exit`
  / autopilot reboot from the **message thread**. P1, ✅ verified, **safety-relevant**.
  (= KNOWN_ISSUES #18, confirmed.)
- **MC-3** `mavlink_controller.py:436` — `move_to_point` blocks with **no timeout**; an
  unreachable target hangs the planner thread. P2, ✅ verified, safety-relevant.
- **MC-4** `mavlink_controller.py:264-268` — `healthy_ekf_flag` ORs `EKF_POS_HORIZ_REL` twice,
  omits `EKF_POS_HORIZ_ABS`; readiness bar is laxer than intended. P2, ✅ verified,
  `⚠️ no-crash / verify downstream`, safety-relevant.
- **MC-5** `mavlink_controller.py:466,803,835` — unlocked cross-thread reads of motion state in
  `run_planner`; torn multi-field decisions possible. P1, 🔍 analysis, safety-relevant.
- **MC-7** `mavlink_controller.py:485,498,503` — unbounded ready/mode waits in `run_planner`; no
  watchdog. P2, ✅ verified, safety-relevant.
- **MC-8** `mavlink_controller.py:558` — `move_to_home`'s `max_wait` does not bound the inner
  unbounded `move_to_point`; RTL can hang. P2, ✅ verified, safety-relevant.
- **MC-10** `mavlink_controller.py:838` — `handle_HEARTBEAT` does `custom_mode_mapping[msg.custom_mode]`;
  an unmapped Rover custom mode (2/8/9/13/14) raises `KeyError` on the message thread, killing the
  message loop. P2, 🔍 analysis (depends on autopilot ever reporting those modes), safety-relevant
  (loss of heartbeat/state updates strands the planner waits).
- **MC-2** `mavlink_controller.py:402` — `is_planner_in_control` reads lazy attr
  `planner_in_control`; `AttributeError` if called before `set_and_start_planner`. P2, 🔍, current
  caller order is safe.
- **MC-9** `mavlink_controller.py:51,811` — `gps_fix_type_str_to_num` is an int→str map (name
  inverted); works but misleading. P3, ✅ verified.
- **MP-2** `mavparm.py:174-175` — `diff` returns `None` on other-file load failure → caller
  `sys.exit(None)` (exit 0) on a failed diff. P2, ✅ verified.
- **MP-3** `mavparm.py:104` — `load` returns `False` (not tuple) on open-fail; caller unpacks →
  `TypeError`. P3, ✅ (dead: caller pre-checks `isfile`).
- **MP-1** `mavparm.py:126` — stray `print("WRITTING", …)` debug. P3, ✅.
- **MC-dead** `mavlink_controller.py:414` — dead `np.linalg.norm` after `return`. P3, ✅.

**KNOWN_ISSUES cross-check**
- **#17 (`yaml_defaults` global `args` NameError):** CONFIRMED in
  `mavlink_radio_collection.py:55-70` — `yaml_defaults` reads `args.records_per_receiver`,
  `.routine`, `.tx_gain`, `.drone_uri`, `.inference` but `args` is not a parameter; it works only
  because `__main__` sets module-global `args` (`179`) *before* calling it (`183`). Importing and
  calling `yaml_defaults` as a library function → `NameError`. P2, ✅ verified.
- **#18 (sudo shutdown / sys.exit in `handle_RC_CHANNELS`):** CONFIRMED (= MC-1),
  `mavlink_controller.py:899` (`sudo shutdown 0`), `:909` (`sys.exit(1)`), on the message thread.
  Safety-relevant. P1, ✅ verified.
