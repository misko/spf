# PASS-1 Surface Inventory — Collection / Hardware Cluster

Read-only surface inventory (Pass 1 per `claude_docs/DOCUMENTATION_GUIDE.md`). Signature-level
truth only: inputs, outputs, side effects. Line numbers verified against source as read.

Status legend: ✅ active · ⚠️ legacy/concern · 🔴 broken (verified) · 🧪 experimental/unused ·
`⚠️unverified` = claim not confirmed in Pass 1.

Angles/units note: GPS points throughout are `[long, lat]` in **decimal degrees** (note the
ordering — many helpers `swap_lat_long` before calling `haversine`). GRBL positions are in
**motor steps / mm** in a 2D cable-robot plane. See `00_concepts/conventions.md`.

---

## `spf/mavlink/mavlink_controller.py`

Module-level constants/maps (no behavior): `EKF_STATUS_DEC_TO_STRING`/`..._STRING_TO_DEC`
(36-49), `gps_fix_type_str_to_num`/`..._num_to_str` (51-62, **misnamed** — see Notable),
`mav_states_list` (64), `mission_states` (76), `mav_mission_states` (85), `sensors_list` (104),
`custom_mode_mapping` (140), `ROVER_MODE_RTL=11` (155), `switchable_modes` (157), `tones` (162,
encoded bytes at 171), `LOG_ERASE=121` (173). Logging is configured at import (29-34) → writes
`logs.log` + stream handler (import side effect).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `drone_get_planner` | 176 | func | Build a motion Planner for a routine over a GPS boundary | `routine` (str: circle/center/bounce/diamond), `boundary` (np.ndarray Nx2 [long,lat]) | Planner subclass instance | `np.random.rand`/`np.flip` for diamond (randomness); raises `Exception("Missing planner")` on unknown | ✅ |
| `lookup_bits` | 227 | func | Decode a bitmask int into list of set flag names | `x` (int mask), `table` (list[str] mavlink attr names) | list[str] | reads `mavutil.mavlink.<name>` | ✅ |
| `lookup_exact` | 231 | func | Names in table whose mavlink value equals x | `x` (int), `table` (list[str]) | list[str] | reads `mavutil.mavlink.<name>` | ✅ |
| `Drone.__init__` | 236 | method | Construct drone controller, set up threads/state | `connection` (mavutil conn or None), `tolerance_in_m`=5, `distance_finder`=None, `fake`=False, `ignore_mode`=False | None | starts distance_finder thread (251-252); reads `connection.mode_mapping()`/sysid (MAVLink I/O) unless fake; sets numba log level; creates Condition objs + threads (not started); `reset_params` | ✅ |
| `Drone.set_and_start_planner` | 345 | method | Attach planner and start planner thread | `planner` (Planner) | None | mutates self; **starts** `planner_thread` (thread) | ✅ |
| `Drone.buzzer` | 353 | method | Play a tune on the autopilot buzzer | `tone_bytes` (bytes) | bool (success) | MAVLink `play_tune_send` x5 retries; `time.sleep(0.1)`; swallows AttributeError | ✅ |
| `Drone.reset_params` | 367 | method | Reset param dict to empty MAVParmDict | none | None | mutates `self.params` | ✅ |
| `Drone.update_all_parameters` | 370 | method | Fetch all params, block until count matches | `timeout`=50 (s) | bool (complete?) | MAVLink `param_fetch_all`; **blocking** poll loop; `sys.exit(1)` on timeout; logging | ⚠️ (sys.exit on timeout) |
| `Drone.start` | 392 | method | Start the message-processing loop thread | none | self | **starts** `message_loop_thread` | ✅ |
| `Drone.send_status` | 396 | method | Send STATUSTEXT to autopilot | `text` (str) | None | MAVLink `statustext_send` | ✅ |
| `Drone.is_planner_in_control` | 401 | method | Getter for planner_in_control flag | none | bool | none | ✅ |
| `Drone.get_position_bearing_and_time` | 404 | method | Snapshot gps/heading/gps_time | none | dict | reads self (no lock) | ✅ |
| `Drone.distance_to_target` | 409 | method | Haversine distance to a target point | `target_point` ([long,lat]) | float meters | none (dead `np.linalg.norm` line 414 after return) | ⚠️ dead code |
| `Drone.erase_logs` | 416 | method | Send LOG_ERASE command_long | none | None | MAVLink `command_long_send` | ✅ |
| `Drone.move_to_point` | 432 | method | Reposition then block until within tolerance; collision-avoid via distance finder | `point` ([long,lat]), `log_interval`=5 | True | MAVLink reposition/arm/disarm; **blocking** loop with sleeps; reads distance_finder; logging | ✅ |
| `Drone.run_planner` | 466 | method | Planner thread body: wait ready, drive to planned points | none | None | many MAVLink calls (set_home, arm, move); **blocking** sleeps/waits; buzzer; logging; sets planner_in_control | ✅ |
| `Drone.get_cmd` | 538 | method | Resolve mavlink cmd name→num, cache both ways | `cmd` (str) | int | mutates `self.mav_cmd_name2num`/`num2name`; reads mavutil | ✅ |
| `Drone.move_to_home` | 544 | method | Drive to stored home, block until reached or timeout | `max_wait`=300 (s) | bool | calls move_to_point; **blocking**; logging; returns False if no `home` attr | ✅ |
| `Drone.set_rtl_mode` | 572 | method | Set autopilot to RTL mode | none | None | MAVLink `command_long_send` (DO_SET_MODE) | ✅ |
| `Drone.run_compass_calibration` | 592 | method | Start mag calibration | none | None | MAVLink encode+send (DO_START_MAG_CAL) | ✅ |
| `Drone.request_home` | 608 | method | Request HOME_POSITION from autopilot | none | None | MAVLink encode+send (GET_HOME_POSITION) | ✅ |
| `Drone.set_home` | 630 | method | Set autopilot home + store self.home | `lat`, `long` (deg) | None | MAVLink command_long_send; mutates `self.home`=[long,lat] | ✅ |
| `Drone.turn_off_hardware_safety` | 648 | method | Decode-position safety mode set | none | None | MAVLink `set_mode_send` | ⚠️unverified intent |
| `Drone.reboot` | 655 | method | Reboot autopilot (optionally forced/bootloader) | `force`=False, `hold_in_bootloader`=False | None | MAVLink reboot/command_long_send | ✅ |
| `Drone.disarm` | 678 | method | Disarm motors | `force`=False | None | MAVLink ARM_DISARM(0) | ✅ |
| `Drone.arm` | 694 | method | Arm motors | `force`=False | None | MAVLink ARM_DISARM(1) | ✅ |
| `Drone.reposition` | 710 | method | Send DO_REPOSITION (command_int) to lat/long | `lat`, `long` (deg) | None | MAVLink `command_int_send`; lat/long scaled ×1e7 (big block commented out 711-723) | ✅ |
| `Drone.do_mission` | 741 | method | Set/restart current mission | `restart_mission`=True | None | MAVLink DO_SET_MISSION_CURRENT | 🧪 (no callers seen ⚠️unverified) |
| `Drone.ack` | 757 | method | Wait for a message type, return True if None | `keyword` (str) | bool | MAVLink `recv_match` **blocking** timeout=5 | 🧪 (commented at call sites) |
| `Drone.single_operation_mode_on` | 762 | method | Pause message loop for exclusive access | none | True | acquires `single_condition`; **blocking wait**; asserts not already single | ✅ |
| `Drone.single_operation_mode_off` | 770 | method | Resume message loop | `turn_on_messages`=True | None | notifies conditions; asserts in single mode | ✅ |
| `Drone.process_messages` | 779 | method | Message loop thread body | none | None (infinite) | holds `message_condition`; **blocking** recv_match; honors single_operation pause | ✅ |
| `Drone.handle_HOME_POSITION` | 793 | method | Store launch_home from HOME_POSITION | `msg` | None | mutates `self.launch_home` (÷1e7); logging | ✅ |
| `Drone.handle_NAV_CONTROLLER_OUTPUT` | 798 | method | No-op handler | `msg` | None | none (pass) | 🧪 stub |
| `Drone.handle_GLOBAL_POSITION_INT` | 803 | method | Update lat/long/gps/heading | `msg` | None | mutates self.lat/long/gps/heading (hdg÷100) | ✅ |
| `Drone.handle_GPS_RAW_INT` | 809 | method | Update sat count + fix type | `msg` | None | mutates self; **note** indexes `gps_fix_type_str_to_num` (int→str map) | ⚠️ naming (see Notable) |
| `Drone.handle_EKF_STATUS_REPORT` | 813 | method | Set ekf_healthy from flags | `msg` | None | mutates `self.ekf_healthy` | ✅ |
| `Drone.handle_COMMAND_ACK` | 819 | method | No-op handler | `msg` | None | none (pass) | 🧪 stub |
| `Drone.handle_PARAM_VALUE` | 825 | method | Store param value + count | `msg` | None | mutates `self.params`, `self.param_count` | ✅ |
| `Drone.handle_HEARTBEAT` | 835 | method | Update mode/armed/states; compute drone_ready | `msg`, `log_interval`=5 | None | mutates many self fields; logging; sets `drone_ready` | ✅ |
| `Drone.handle_SYSTEM_TIME` | 873 | method | Update gps_time + time_since_boot | `msg` | None | mutates self (unix usec→s, boot ms→s) | ✅ |
| `Drone.handle_SYS_STATUS` | 877 | method | Decode sensors present/enabled/health | `msg` | None | mutates self lists | ✅ |
| `Drone.handle_STATUSTEXT` | 888 | method | Log autopilot status text | `msg` | None | logging | ✅ |
| `Drone.handle_RC_CHANNELS_SCALED` | 893 | method | No-op handler | `msg` | None | none (pass) | 🧪 stub |
| `Drone.handle_RC_CHANNELS` | 897 | method | RC-switch actions (shutdown/cal/reboot/ultrasonic toggle) | `msg` | None | **`subprocess.run(["sudo","shutdown","0"])`**; reboot; `sys.exit(1)`; mutates disable_distance_finder; logging | ⚠️ shutdown/exit side effects |
| `Drone.handle_SERVO_OUTPUT_RAW` | 919 | method | Infer motor_active from servo PWM | `msg` | None | mutates `self.motor_active` | ✅ |
| `Drone.message_handlers` | 925 | class attr | Dict msg-type→handler | — | dict | none | ✅ |
| `Drone.process_message` | 942 | method | Dispatch one message to handler | `msg` (or None) | None | sleeps 0.01 if None; calls handler | ✅ |
| `Drone.set_mode` | 950 | method | Delegate set_mode to connection | `mode` | None | MAVLink `connection.set_mode` | ✅ |
| `get_ardupilot_serial` | 954 | func | Find single ArduPilot serial device | none | str path or None | globs `/dev/serial/by-id/`; logging | ✅ |
| `get_mavlink_controller_parser` | 962 | func | Build argparse parser for CLI | none | ArgumentParser | none | ✅ |
| `mavlink_controller_run` | 1038 | func | CLI main: connect, dispatch to buzzer/params/mode/etc | `args` (Namespace) | None | MAVLink connection; file reads/writes; many `sys.exit`; **blocking** waits; logging | ✅ |
| `set_drone_mode` | 1172 | func | Switch to GUIDED/MANUAL with retries | `drone`, `mode` (str) | return_code int (0 ok) | `drone.set_mode`; sleeps; `sys.exit(1)` on invalid mode | ✅ |

**Notable**
- 🔴/⚠️ **Naming bug:** `gps_fix_type_str_to_num` (51) and `gps_fix_type_num_to_str` (62) are
  **inverted** — the "str_to_num" dict maps int→str, and `handle_GPS_RAW_INT` (811) indexes it by
  the integer `msg.fix_type` to get a *string*. Works, but names lie. Flag for Pass 3.
- `Drone.__init__:266-268` `healthy_ekf_flag` ORs `EKF_POS_HORIZ_REL` **twice** and never includes
  `EKF_ATTITUDE`-vs-`EKF_POS_HORIZ_ABS` etc — likely a copy-paste bug. Flag for Pass 3.
- `distance_to_target:414` has dead `np.linalg.norm` after the `return`.
- `handle_RC_CHANNELS` triggers `sudo shutdown`, `sudo halt`-style behavior and `sys.exit` from a
  message-handler thread — surprising for a handler. Safety-critical, flag for Pass 3.
- Device-mapping parsing is **duplicated** here vs the two radio-collection files (see cluster note).
- `do_mission`, `ack`, and the `handle_*` stubs (NAV_CONTROLLER_OUTPUT, COMMAND_ACK,
  RC_CHANNELS_SCALED) appear effectively unused/no-op — 🧪 pending Pass 2 call-graph.

---

## `spf/mavlink/mavlink_send.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| (module script body) | 1-57 | script | Connect to Pixhawk on `/dev/ttyACM1`@57600 and stream random NAMED_VALUE_FLOAT "TEMP" 1/s | none (hardcoded) | — | opens serial; MAVLink send; `random.uniform`; infinite loop with `time.sleep(1)`; prints; KeyboardInterrupt exit | 🧪 demo/example |

**Notable**
- No functions/classes — pure top-level demo script. Hardcoded serial port and baud. Not imported
  anywhere (⚠️unverified — confirm in Pass 2). Treat as example code.

---

## `spf/mavlink/mavparm.py`

Vendored from ArduPilot pymavlink (header line 1). `MAVParmDict(dict)`.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `MAVParmDict.__init__` | 16 | method | Init dict + exclude_load list + mindelta | `*args` | None | sets `self.exclude_load`, `self.mindelta=1e-6` | ✅ |
| `MAVParmDict.mavset` | 34 | method | Set one param over MAVLink, wait for ACK | `mav`, `name`, `value`, `retries`=3, `parm_type`=None | bool | MAVLink `param_set_send`/`recv_match`; **blocking** poll (≤1s/retry); struct pack; mutates self[name]; prints on timeout | ✅ |
| `MAVParmDict.save` | 80 | method | Save params to file (wildcard filter) | `filename`, `wildcard`="*", `verbose`=False | None | **opens/writes file**; prints if verbose | ✅ |
| `MAVParmDict.load` | 98 | method | Load params from file; optionally push via mav | `filename`, `wildcard`="*", `mav`=None, `check`=True, `use_excludes`=True | (count,changed) tuple, or False on open-fail | **reads file**; if mav: calls mavset (MAVLink I/O); many prints incl debug "WRITTING" | ⚠️ debug print left in (126) |
| `MAVParmDict.show_param_value` | 153 | method | Print one name/value | `name`, `value` | None | print | ✅ |
| `MAVParmDict.show` | 156 | method | Print all params matching wildcard | `wildcard`="*" | None | prints | ✅ |
| `MAVParmDict.diff` | 163 | method | Count/print diffs vs another param file | `filename`, `wildcard`="*", `use_excludes`=True, `use_tabs`=False, `show_only1/2`=True | int diffs (or None if load fails) | loads other file; prints each diff | ✅ |

**Notable**
- `load:126` prints `"WRITTING", line, numeric_value` unconditionally — stray debug, flag.
- `load` returns `False` (not a tuple) on open failure while caller in mavlink_controller unpacks
  `count, changed = drone.params.load(...)` (1161) — would crash if file missing, but caller
  guards with `os.path.isfile` first (1155). Note for Pass 3.
- `diff` returns `None` (no return) if `other.load` is falsy; caller `sys.exit(diffs)` (1151) would
  `sys.exit(None)` → exit 0. Edge case, flag.

---

## `spf/grbl/grbl_interactive.py`

Module constants: `home_pA/home_pB` (25-26), `home_bounding_box` (27), calibration points (37-39,
78), `circle_center` (40), circle diameter min/max (41-42), `run_grbl=True` global (44),
`GRBL_STEP_SIZE=8` (53). Coordinates in **motor steps/mm** in the cable-robot plane.

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `stop_grbl` | 47 | func | Set global `run_grbl=False` | none | None | **mutates global** `run_grbl`; logging | ✅ |
| `CalibrationV1Planner.__init__` | 82 | method | Calibration planner (raster sweep) | `dynamics`, `start_point`, `step_size`=8, `y_bump`=150 | None | super().__init__ | 🧪 (v1 calib) |
| `CalibrationV1Planner.get_planner_start_position` | 88 | method | Return tx_calibration_point | none | np.ndarray | reads module const | ✅ |
| `CalibrationV1Planner.yield_points` | 91 | generator | Raster sweep between bounce walls | none | yields points (np 2-vec) | reads module consts; bounce via dynamics; **infinite** | 🧪 |
| `GRBLDynamics.__init__` | 132 | method | Cable-robot kinematics with calibration | `calibration_point`, `pA`, `pB`, `bounding_box`, `unsafe`=False | None | super().__init__ (ConvexHull check) | ✅ |
| `GRBLDynamics.from_steps` | 144 | method | Motor steps → xy position (forward kinematics) | `state` = (a_steps,b_steps) | np 2-vec xy (rounded 1dp) | none (pure); `np.sqrt` (could NaN if r1²<x²) | ✅ |
| `GRBLDynamics.to_steps` | 173 | method | xy → motor steps (inverse); bounds-check both ways | `p` (xy) | (a_steps,b_steps) | raises `PointOutOfBoundsException` unless unsafe; calls from_steps to re-validate | ✅ |
| `FakeStream.__init__` | 202 | method | Queue-backed fake serial stream | `cq`, `rq` (queues) | None | none | 🧪 test/fake |
| `FakeStream.write` | 206 | method | Put command on queue | `x` | None | mutates queue | 🧪 |
| `FakeStream.readline` | 209 | method | Block-get response from queue | none | bytes | **blocking** queue.get | 🧪 |
| `FakeStream.close` | 212 | method | No-op | none | None | none | 🧪 |
| `GRBLController.__init__` | 217 | method | Open real or fake GRBL serial; init position | `serial_fn` (str/None/"none"), `dynamics`, `channel_to_motor_map` (dict) | None | **opens pyserial @115200** OR spawns 2 fake-grbl threads; writes "?"; `update_status`; sleeps | ✅ |
| `GRBLController._should_move` | 257 | method | Any axis target≠current? | none | bool | reads self | 🧪 fake-only |
| `GRBLController._move_fake_grbl` | 263 | method | Fake-grbl motion thread body | none | None (infinite) | **blocking** queue.get; mutates fake position | 🧪 fake-only |
| `GRBLController._run_fake_grbl` | 296 | method | Fake-grbl command responder thread | none | None (infinite) | **blocking** queue.get; puts responses; print | 🧪 fake-only |
| `GRBLController.push_reset` | 312 | method | Send Ctrl-X reset to GRBL | none | str response | **serial write/read** | ✅ |
| `GRBLController.update_status` | 316 | method | Query "?" and parse MPos into xy | `skip_write`=False | position dict | **serial write/read**; sleeps; **recursive** on parse fail/"ok"; logging; mutates self.position | ⚠️ (recursion, see Notable) |
| `GRBLController.wait_while_moving` | 356 | method | Block until two reads show no motion | none | None | calls update_status (serial); sleeps; reads global run_grbl | 🔴 (uses wrong keys — see Notable) |
| `GRBLController.move_to` | 369 | method | Emit G0 move for each channel's point | `points` (dict channel→xy) | None | `to_steps` (may raise OOB); **serial write/read**; reads global run_grbl; sleeps; logging | ✅ |
| `GRBLController.set_current_position` | 393 | method | G92 set-origin for a channel | `motor_channel`, `steps`=(a,b) | None | **serial write**; update_status | ✅ |
| `GRBLController.distance_to_targets` | 407 | method | Per-channel distance current→target | `target_points` (dict) | dict channel→float | update_status (serial) | ✅ |
| `GRBLController.targets_far_out` | 414 | method | Any channel beyond tolerance? | `target_points`, `tolerance`=50 | bool | update_status (serial) | ✅ |
| `GRBLController.move_to_iter` | 421 | method | Drive iterators of points until done/stopped | `points_by_channel` (dict of generators), `return_once_stopped`=False | None | move_to (serial); **blocking** poll; reads global run_grbl; sleeps | ✅ |
| `GRBLController.close` | 436 | method | Close serial | none | None | **closes serial** | ✅ |
| `get_next_points` | 440 | func | Advance each channel generator one step | `channel_iterators` (dict) | dict (drops StopIteration channels) | iterator consumption | ✅ |
| `GRBLManager.__init__` | 451 | method | Bind controller+routine; build routine table; lock | `controller`, `routine` (str/None) | None | acquires `self.ready` lock; creates (unstarted) planner_thread | ✅ |
| `GRBLManager.has_planner_started_moving` | 470 | method | Getter | none | bool | none | ✅ |
| `GRBLManager.run_planner` | 473 | method | Planner thread: setup routine, get_ready, run | none | None | calls routine fn; serial via controller; logging; sleeps; swallows Exception | ✅ |
| `GRBLManager.start` | 498 | method | Start planner thread | none | None | **starts thread** | ✅ |
| `GRBLManager.get_ready` | 501 | method | Move all channels into routine start positions | none | None | StationaryPlanner; move_to_iter (serial); **blocking**; reads global run_grbl; logging | ✅ |
| `GRBLManager.run` | 526 | method | Run the routine planners' point streams | none | None | move_to_iter (serial); mutates planner start_points; logging | ✅ |
| `GRBLManager.bounce` | 534 | method | Set both channels to BouncePlanner | none | None | mutates self.planners | ✅ |
| `GRBLManager.tx_circle` | 544 | method | ch0 stationary center, ch1 circle | none | None | mutates self.planners | ✅ |
| `GRBLManager.rx_circle` | 561 | method | ch0 circle, ch1 stationary center | none | None | mutates self.planners | ✅ |
| `GRBLManager.v4_calibrate` | 578 | method | ch0 bounce, ch1 stationary rx_cal | none | None | mutates self.planners | ✅ |
| `GRBLManager.rx_random_circle` | 593 | method | ch0 random-diameter circle, ch1 stationary | none | None | `random.randint` (randomness); mutates planners | ✅ |
| `GRBLManager.v1_calibrate` | 612 | method | ch0 stationary rx_cal, ch1 CalibrationV1 | none | None | mutates self.planners | 🧪 v1 |
| `GRBLManager.close` | 627 | method | Close controller | none | None | closes serial | ✅ |
| `get_default_dynamics` | 631 | func | Build GRBLDynamics with home consts | `unsafe`=False | GRBLDynamics | none | ✅ |
| `get_default_gm` | 641 | func | Build full GRBLManager (dynamics+controller) | `serial_fn`, `routine`, `unsafe`=False | GRBLManager | opens serial via controller; channel map `{0:"XY",1:"ZA"}` | ✅ |
| `exit_fine` | 653 | func | Stop planners + global run_grbl, sys.exit(0) | `gm` | None (exits) | mutates global; `sys.exit(0)`; print | ✅ |
| (`__main__` REPL) | 662-742 | script | Interactive GRBL shell (status/move/routines/timer) | stdin | — | serial I/O; `select` on stdin; threads; `sys.exit` | ✅ |

**Notable**
- 🔴 `wait_while_moving` (356-367) indexes `old_pos["a_motor_steps"]`/`["b_motor_steps"]` but
  `update_status` returns keys `a0_motor_steps/b0_motor_steps/a1_motor_steps/b1_motor_steps` — those
  keys **do not exist** → `KeyError` if ever called. Looks dead (no callers in this file). Flag 🧪/🔴.
- `update_status` (316) recurses on parse failure / "ok" / empty response, toggling `skip_write`
  each time — could recurse deeply / unboundedly on a chatty or stuck device. Pass 3 hazard.
- `from_steps` (165) `np.sqrt(r1²−x²)` can produce NaN if the geometry is inconsistent (point
  outside reachable workspace). No guard. Pass 3.
- Motor-step parse order in `update_status` (335): `b0,a0,b1,a1` — non-obvious axis ordering tied
  to `channel_to_motor_map={0:"XY",1:"ZA"}`. Units/frame note for conventions.md.
- `move_to`/`set_current_position` write `b_motor_steps` to motors[0] and `a_motor_steps` to
  motors[1] — A/B swap relative to `from_steps`. Verify consistency in Pass 3.

---

## `spf/grbl/run_grbl.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| (module script body) | 1-51 | script | Stream a g-code file to GRBL over serial | `argv[1]`=device, `argv[2]`=gcode file | — | opens serial@115200 + file; serial write/read per line; prints; `sys.exit(1)` on bad argv | 🧪 standalone util |

**Notable**
- No functions. `_line[0] == ";"` (39) will IndexError on a blank line. Minor, flag.

---

## `spf/grbl/mm_per_step.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| (module script body) | 1-29 | script | Compute steps/mm from gearing and print GRBL `$` config | none (hardcoded constants) | — | **print** of `$100/$101/$110/...` config | 🧪 one-shot calc |

**Notable**
- Pure computation + print; gear_ratio `26+103/121`, microstepping 16. No callable API.

---

## `spf/gps/gps_utils.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `swap_lat_long` | 6 | func | Swap (long,lat)↔(lat,long); handles 2D arrays | `x` (tuple or np Nx2) | swapped tuple/array | none | ⚠️ (array branch suspect — see Notable) |
| `degnorm` | 13 | func | Normalize degrees to [-180,180) | `x` (deg) | deg | none (pure) | ✅ |
| `pinorm` | 17 | func | Normalize radians to [-π,π) | `x` (rad) | rad | none (pure) | ✅ |
| `calc_bearing` | 26 | func | Bearing (deg from north) p0→p1 | `p0`,`p1` (np Nx2 [long,lat] deg) | np array deg | asserts shapes; pure | ✅ |
| `calc_relative_bearing` | 48 | func | Relative bearing facing/current/desired | `p_facing`,`p_current`,`p_desired` (Nx2) | (rel_deg, facing_bearing, desired_bearing) | pure | ✅ |

**Notable**
- 🔴 **Likely bug:** `swap_lat_long` array branch (8-9) does `np.hstack([x[:,[0]], x[:,[1]]])`
  which **does not swap** — it returns columns in the same order. The scalar/tuple branch (10)
  *does* swap `(x[1], x[0])`. Inconsistent. High-priority Pass 3 verification (could silently
  corrupt boundary/haversine inputs).

---

## `spf/gps/boundaries.py`

GPS boundary polygons (np Nx2 `[long, lat]` deg): `crissy_boundary_convex` (9), `crissy_boundary`
(18), `franklin_safe` (35), `franklin_boundary` (44), `fort_baker_boundary` (65),
`fort_baker_right_boundary` (74), `fort_baker_left_boundary` (99). `boundaries` dict (121).

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `boundary_to_diamond` | 54 | func | Midpoints of boundary edges → diamond (4 pts) | `boundary` (np Nx2, uses idx 0-3) | np 4x2 | none (pure) | ✅ |
| `find_closest_boundary_with_distance` | 130 | func | Nearest named boundary (by centroid haversine) | `gps` ([long,lat]) | (distance_m, name) or (inf,None) | swap_lat_long+haversine; catches ValueError | ✅ |
| `find_closest_boundary` | 150 | func | Nearest boundary name within cutoff | `gps`, `cutoff`=inf | name str or None | calls above | ✅ |

**Notable**
- `boundary_to_diamond` hardcodes indices 0-3; only valid for ≥4-point boundaries.
- `find_closest_boundary` returns the **name** (str); callers in radio collection index
  `boundaries[name]` — confirm the str-vs-array contract in Pass 2 (mavlink_radio_collection uses
  both `find_closest_boundary` and `boundaries[...]`).

---

## `spf/motion_planners/dynamics.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `BoundingBoxIsNonConvexException` | 7 | class | Exception type | — | — | none | ✅ |
| `PointOutOfBoundsException` | 11 | class | Exception type | — | — | none | ✅ |
| `chord_length_to_angle` | 14 | func | Chord length → central angle | `chord_length`, `radius` | rad | pure; `arcsin` (NaN if chord/2r>1) | ✅ |
| `a_to_b_in_stepsize_np` | 18 | func | Points from a→b spaced by step_size (vectorized) | `a`,`b` (np 2-vec), `step_size` | np Mx2 (or [b]) | pure; last point forced to b | ✅ |
| `a_to_b_in_stepsize` | 31 | func | Wrapper → calls `_np`; loop body is dead | `a`,`b`,`step_size` | list/np | pure; **dead code after return (33-44)** | ⚠️ dead code |
| `Dynamics.__init__` | 47 | method | Bounding box → convex-hull validated Path | `bounding_box` (Nx2), `unsafe`=False, `bounds_radius`=1e-5 | None | ConvexHull (raises NonConvex); logging; sets self.polygon (None if <3 pts) | ✅ |
| `Dynamics.to_steps` | 68 | method | Bounds-check point (base: no transform) | `p` | None | raises PointOutOfBounds unless unsafe | ✅ |
| `Dynamics.from_steps` | 78 | method | Identity (base) | `state` | state | none | ✅ |
| `Dynamics.binary_search_edge` | 81 | method | Bisect along direction to find boundary distance | `left`,`right`,`xy`,`direction`,`epsilon` | float distance | **recursive**; calls to_steps (may raise, caught) | ✅ |
| `Dynamics.distance_from_point_to_segment` | 93 | method | Point→segment distance via rotation | `p`,`v0`,`v1` (2-vec) | float | pure; `assert np.isclose(_v1[1],0)` | ✅ |
| `Dynamics.get_boundary_vector_near_point` | 114 | method | Unit vector of nearest boundary edge | `p` | np 2-vec | raises ValueError if no polygon; pure | ⚠️ dead `score` lines (see Notable) |

**Notable**
- `a_to_b_in_stepsize` (31): immediately `return a_to_b_in_stepsize_np(...)`; lines 33-44 are
  unreachable dead code (old loop impl).
- `get_boundary_vector_near_point` (125-131): first `score` (cosine) is computed then **overwritten**
  by `distance_from_point_to_segment` (131) — the dot-product block is dead. Also note `p - v1, v1 - v1`
  (129) computes dot with zero vector (always 0); harmless since overwritten but confusing.
- `from_steps`/`chord_length_to_angle` arcsin/sqrt domains unguarded (NaN risk). Pass 3.

---

## `spf/motion_planners/planner.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `Planner.__init__` | 10 | method | Base planner state + RNG | `dynamics`,`start_point`,`step_size`,`epsilon`=1,`seed`=None | None | `np.random.default_rng(seed)`; sets self.running | ✅ |
| `Planner.get_bounce_pos_and_new_direction` | 19 | method | Reflect direction off boundary edge | `p`,`direction` (2-vec) | (bounce_point, new_direction) | binary_search_edge (recursive); pure | ✅ |
| `Planner.random_direction` | 35 | method | Random unit direction | none | np 2-vec | **RNG** (self.rng) | ✅ |
| `Planner.yield_points` | 39 | abstractmethod | Contract: yield path points | none | generator | abstract | ✅ |
| `Planner.get_planner_start_position` | 43 | method | Default start pos (None) | none | None | none | ✅ |
| `Planner.stop` | 46 | method | Set running=False | none | None | mutates self | ✅ |
| `BouncePlanner.single_bounce` | 51 | method | One bounce segment + noisy new direction | `direction`,`p` | (to_points, new_direction) | RNG (uniform); pure otherwise | ✅ |
| `BouncePlanner.yield_points` | 66 | generator | Yield infinite bounce path | none | yields points | delegates to bounce | ✅ |
| `BouncePlanner.bounce` | 69 | generator | Bounce loop with stall recovery | `current_p`, `total_bounces`=-1 | yields points | **RNG**; logging; **`print("BOUNCING")` every iter**; asserts; honors self.running | ⚠️ noisy print |
| `StationaryPlanner.__init__` | 119 | method | Planner that parks at a point | `dynamics`,`start_point`,`stationary_point`,`step_size` | None | super | ✅ |
| `StationaryPlanner.yield_points` | 125 | generator | Move to point then yield it forever | none | yields points | **infinite** while running | ✅ |
| `StationaryPlanner.get_planner_start_position` | 136 | method | Return stationary_point | none | 2-vec | none | ✅ |
| `PointCycle.__init__` | 141 | method | Cycle through a fixed point list | `dynamics`,`start_point`,`points`,`step_size` | None | super | ✅ |
| `PointCycle.get_planner_start_position` | 153 | method | First point | none | 2-vec | none | ✅ |
| `PointCycle.yield_points` | 156 | generator | Loop forever over points | none | yields points | **infinite**; honors running | ✅ |
| `CirclePlanner.__init__` | 175 | method | Circle traversal planner | `dynamics`,`start_point`,`step_size`,`circle_diameter`,`circle_center`=[0,0] | None | **`np.random.rand`** for direction (randomness, not seeded rng) | ⚠️ unseeded random |
| `CirclePlanner.get_planner_start_position` | 191 | method | Position at angle 0 (top) | none | 2-vec | none | ✅ |
| `CirclePlanner.angle_to_pos` | 194 | method | Angle → xy on circle | `angle` (rad) | 2-vec | pure | ✅ |
| `CirclePlanner.yield_points` | 199 | generator | Traverse circle 0→360 | none | yields points | honors running; `while current_angle<360` (deg vs rad — see Notable) | ⚠️ unit mix |

**Notable**
- `CirclePlanner.yield_points` (205): loop condition `while current_angle < 360` but
  `angle_increment` is **radians** (from `chord_length_to_angle`) and `angle_to_pos` treats angle
  as radians. Comparing a radian accumulator to `360` means it loops ~57× past a full circle. Looks
  like a deg/rad bug — high-priority Pass 3.
- `CirclePlanner.__init__` uses unseeded `np.random.rand()` for direction (186) — not reproducible
  via the `seed` plumbing in `Planner`.
- `BouncePlanner.bounce` prints `"BOUNCING"` every iteration (84) — log spam.

---

## `spf/grbl_radio_collection.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `grbl_radio_parser` | 19 | func | Build argparse for GRBL radio collection CLI | none | ArgumentParser | none | ✅ |
| `grbl_radio_main` | 78 | func | Orchestrate GRBL+SDR data collection run | `args` (Namespace) | None | reads YAML + device-mapping file; configures logging (file handler); writes yaml copy; starts GRBLManager thread; **blocking** polls; `os.rename` files; sleeps; many asserts | ✅ |
| (`__main__`) | 204 | script | Parse args → grbl_radio_main | — | — | as above | ✅ |

**Notable**
- **Device-mapping parsing** (90-105) is duplicated almost verbatim with
  `mavlink_radio_collection.yaml_defaults` (32-53) and conceptually with the one in
  `mavlink_controller`. Same `pluto://usb:...` URI construction, 2-or-3-field line format. Prime
  dedup candidate.
- Hard asserts: `routine is not None` (87), `emitter type=="sdr"` for tx-gain (108), every receiver
  has `motor_channel` (126-127), `data-version` ∈ {2,5} else `DataVersionNotImplemented`.

---

## `spf/mavlink_radio_collection.py`

| Symbol | Line | Kind | Purpose | Inputs | Outputs | Side effects | Status |
|---|---|---|---|---|---|---|---|
| `yaml_defaults` | 32 | func | Fill yaml config from device-mapping + CLI args | `yaml_config` (dict), `device_mapping_fn` (path) | dict (mutated config) | **reads device-mapping file**; mutates yaml_config; **references global `args`** (not a param!) | ⚠️ global capture (see Notable) |
| `parse_args` | 74 | func | Build+parse argparse for mav radio collection | none | Namespace | parses sys.argv | ✅ |
| (`__main__`) | 177 | script | Full drone radio-collection run (connect, plan, collect, RTL) | — | — | MAVLink connection; serial; reads/writes files; spawns Drone threads; optional NN inference dataset; `os.rename`; `sudo sync` on pi; many `sys.exit`; **blocking** loops | ✅ |

**Notable**
- 🔴 **Latent bug:** `yaml_defaults` reads `args.records_per_receiver`, `args.routine`, `args.tx_gain`,
  `args.drone_uri`, `args.inference` (55-70) but `args` is **not a parameter** — it relies on the
  module-global `args` set in `__main__` (179). Importing/calling `yaml_defaults` as a library
  function would `NameError`. Flag for Pass 3.
- **Device-mapping parsing** (34-53) duplicates `grbl_radio_collection` (90-105). Dedup candidate.
- Only `data-version == 4` supported (294) else `DataVersionNotImplemented`. `--realtime` wires an
  NN inference dataset (`v5inferencedataset` + `v5spfdataset_nn_wrapper`).
- `subprocess.getoutput("sudo sync")` (361) at end on pi; the `sudo halt` is commented out (360).

---

## Cluster summary

**Symbol counts by status** (approx; module-level script bodies counted once each):

| Status | Count | Notes |
|---|---|---|
| ✅ Active | ~95 | bulk of Drone/GRBL/planner/gps API |
| ⚠️ Concern/legacy/dead-code | ~14 | dead code, debug prints, unit mixes, global capture |
| 🔴 Broken (verified) | 4 | `swap_lat_long` array branch, `wait_while_moving` keys, `yaml_defaults` global `args`, `CirclePlanner` deg/rad |
| 🧪 Experimental/unused | ~18 | mavlink_send demo, fake-grbl, v1 calibration, handler stubs, mm_per_step, run_grbl util |

(Counts are Pass-1 estimates; exact tallies to be reconciled in Pass 2 with the call graph.)

**Top 5 deep-dive candidates (Pass 2/3 priority):**
1. **`swap_lat_long` array branch** (`gps_utils.py:8-9`) — appears to NOT swap columns while the
   tuple branch does; feeds haversine/boundary math repo-wide. Highest blast radius.
2. **`CirclePlanner.yield_points` deg/rad** (`planner.py:205`) — radian accumulator compared to
   `360`; governs all circle routines (rx_circle/tx_circle/rx_random_circle).
3. **`yaml_defaults` global `args`** (`mavlink_radio_collection.py:32-71`) — uses undefined-as-param
   `args`; only works as `__main__` script, not importable. Plus duplicated device-mapping parse.
4. **`Drone.handle_RC_CHANNELS` + `update_all_parameters`** (`mavlink_controller.py:897, 370`) —
   safety-critical side effects (`sudo shutdown`, `sys.exit` from handler thread; `sys.exit(1)` on
   param timeout). Verify thread-safety and intent.
5. **`GRBLController.update_status` recursion + `wait_while_moving` dead keys**
   (`grbl_interactive.py:316, 356`) — unbounded recursion risk on chatty serial; `wait_while_moving`
   would KeyError. Plus A/B motor-step ordering consistency vs `from_steps`/`move_to`.

**Cross-cutting themes:**
- *Duplicated device-mapping parsing* across `grbl_radio_collection.py`, `mavlink_radio_collection.py`,
  and the URI logic — single dedup opportunity.
- *Heavy hidden side effects in hardware controllers*: many `sys.exit`/`subprocess`/`sudo` calls and
  blocking waits live inside methods/handlers, not just `__main__`.
- *Unit/frame fragility*: GPS `[long,lat]` ordering (constant `swap_lat_long` dance) and GRBL motor
  A/B-step ordering are the recurring correctness risk; deg-vs-rad surfaces in the circle planner.
- *Dead/legacy code*: `a_to_b_in_stepsize` loop body, `get_boundary_vector_near_point` cosine block,
  `distance_to_target` tail, v1 calibration path, `mavlink_send`/`run_grbl`/`mm_per_step` scripts.
