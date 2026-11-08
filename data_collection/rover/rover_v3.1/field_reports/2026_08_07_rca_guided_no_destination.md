# Root Cause Analysis — 2026-08-07 rover mission: rovers recording in GUIDED without making progress

**Status:** mechanism established with high confidence for *why the rovers stayed stopped*; one residual field-unverifiable alternative remains open. Scope of the incident is materially narrower than the field report implies.

Every statement below is tagged **[M] measured** (I or a named investigator computed it from the artifacts / from source / from a SITL run) or **[I] inferred**. Numbers are quoted with the artifact they came from. All mission artifacts were opened read-only; nothing was written to `/mnt/qnap01` or `/mnt/md*`.

Clock convention: collector logs are Pi-local BST (UTC+1); `receivers/r0/gps_timestamp` in the zarr is absolute UTC from `SYSTEM_TIME.time_unix_usec`, so log↔zarr alignment is exact by construction, not fitted. `system_timestamp` is seconds-since-collector-start and is **not** a wall clock — three investigators independently confirmed that using it misplaces events by 50–55 s.

---

## 1. What actually happened

### 1.1 Ground truth: the rovers really were parked, and it was not a telemetry artifact

Full-capture GPS path speed (path length ÷ span), computed directly from `receivers/r0/{gps_lat,gps_long}` in all 17 zarr stores that carry usable GPS **[M]**:

| capture | rover | m/s | verdict |
|---|---|---|---|
| 00_08_04 (.tmp) | RO1 | 1.29 | healthy |
| 00_08_06 | RO3 | 1.33 | healthy |
| 00_08_33 | RO2 | 1.40 | healthy |
| 00_16_43 | RO4 | 1.36 | healthy |
| 00_40_09 | RO1 | 1.36 | healthy |
| 00_41_05 | RO2 | 1.43 | healthy |
| 00_44_09 (.tmp) | RO3 | 1.30 | healthy |
| 00_45_44 | RO4 | 1.33 | healthy |
| 00_55_11 | RO3 | 1.36 | healthy |
| 01_12_26 | RO2 | 1.44 | healthy |
| **01_15_15** | **RO1** | **0.78** | **degraded** |
| **01_24_04 (.tmp)** | **RO4** | **0.65** | **degraded** |
| **01_27_43 (.tmp)** | **RO3** | **0.55** | **degraded** |
| 01_38_45 (.tmp) | RO3 | 1.28 | healthy |
| 01_44_09 (.tmp) | RO2 | 1.38 | healthy |
| 01_48_38 (.tmp) | RO1 | 0.76 | degraded, **but by a different mechanism** (§4.3) |

A purely geometric "parked" detector (maximal windows whose entire point cloud fits inside a 6 m circle, ≥30 s long, no internal sampling gap >5 s) **[M]**:

- **RO1 01:15:15** — 817 s parked out of 1789 s (46%), in 8 windows, longest **250.5 s** starting 01:41:11.
- **RO4 01:24:04** — 559 s of 1214 s (46%), longest 147 s.
- **RO3 01:27:43** — 235 s of 446 s (53%).
- **RO1 01:48:38** — 121 s of 333 s.
- Every healthy capture: 0–46 s total, and those windows are stall hand-over waits or startup.

The feed was live and the vehicle was inert in *both* position and attitude **[M]**:
- Longest run of bit-identical `(gps_lat, gps_long)` anywhere in any mission capture: **3 records (1.6 s)**. Median inter-sample step inside the freezes: 2.2 cm (RO1), 3.4 cm (RO3) — 1–3 LSB of the 1.113 cm GPS quantum.
- `GLOBAL_POSITION_INT.hdg` (written per record) inside the long freezes: median |dheading/dt| **0.00–0.11 °/s**, max deviation from mid-window 0.3–4.7°. RO1's 250 s freeze holds 255.8° essentially constant. This kills "pivoting/spinning in place" and "high-centred with wheels turning" — a vehicle doing either would show attitude change.

### 1.2 The incident, capture by capture

**RO1 `rover_2026_08_07_01_15_15_nRX2_bounce_spacing0p035_tag_RO1` — the one *listed* mission capture that shows the symptom** **[M]**

- Planner starts issuing moves 01:15:30.244; recording starts 01:16:30.723; first zarr record 01:16:32.267.
- **Phase A, 01:16:30.7 → 01:31:48.9 (918.2 s):** 1252.7 m of GPS path at 1.364 m/s, 31 waypoints reached. One genuine physical stall at 01:21:16.952 (motor commanded, no motion), cleared by operator hand-back at 01:21:58.451 (41.5 s); it then reached 12 more waypoints normally.
- Last leg issued **01:31:18.358** to `[-122.47856146, 37.83506867]`; last "Reached target" 01:31:18.341.
- The rover drove that leg: **41.73 m of GPS path in the first 30 s at 1.39 m/s**, distance 48.99 → 18.715 m by 01:31:43.710 with `motor_active=True`, GUIDED.
- **First MANUAL sample 01:31:48.882.** After that it never reached another waypoint and never moved under GUIDED again.
- **Phase B, 816.9 s:** 147.1 m of path (0.180 m/s). Dwell budget: GUIDED/motor-off/still **627.6 s**, MANUAL/motor-off/still 103.5 s, MANUAL/motor-on/moving 50.3 s, GUIDED/motor-off/moving 30.5 s, GUIDED/motor-on/still 5.0 s.
- The 8 GUIDED/motor-off spans total **627.5 s (10 min 27 s)**, longest 204.8 s from 01:41:09. Independent zarr-side four-state budget: **653 s covering 53 m** (median 0.05 m/s). Both agree within ~4%.
- **939 s on a single waypoint** (01:31:18 leg issue → 01:46:57 last log line). Zero STALL lines in that entire window.
- **1101 of 3000 records (36.7%)** were written inside the 8 dead spans; 1429 (47.6%) after the takeover.
- Whole capture: 1401 m against 2220–2628 m for the healthy captures.

**RO4 `01_24_04` (`.log.tmp` / aborted — NOT one of the nine listed mission captures)** **[M]**

- Watchdog fires a genuine stall at 01:28:13 (distance decaying 57.96 → 53.88 m with `motor_active=True`); operator returns control 01:29:04.362; planner issues a fresh leg 01:29:04.367 (4 ms later).
- That fresh leg drove: **22.68 m of GPS path in 20 s at 1.14 m/s**, 54.24 → 33.40 m.
- **First MANUAL sample 01:29:24.669.** Never reached another waypoint. **1030 s on one leg** (01:29:04 → log end 01:46:14).
- GUIDED/motor-off 598–605 s. Parked 559 s of 1214 s.
- **Caveat, important:** RO4's parked GPS is genuinely ~7× noisier than RO1/RO3 (quietest-window median step 15.3 cm/sample ≈ 0.29 m/s pseudo-speed vs 2.2 cm for RO1). Its "183 m of path in GUIDED/motor-off" is largely noise. Net displacement 01:41:01–01:44:19 is **4.8 m in 198 s**, max radius from centroid 3.7 m. RO4 path-length statistics are not directly comparable to RO1/RO3 without this correction.

**RO3 `01_27_43` (`.log.tmp`)** **[M]** — leg issued 01:30:47.445, closed 49.13 → 29.29 m at 1.42 m/s, MANUAL at 01:31:07.676, then 91 s of GUIDED/motor-off and 235 s parked out of a 446 s capture. The operator then drove it ~35 m *away* from its target and parked it again.

### 1.3 The rovers were not slow — they were stopped

This is the single most decision-relevant measurement, because it eliminates the whole battery/traction/drivetrain family **[M]**:

- Median 5 s GPS net displacement while `mav_mode=GUIDED` and `motor_active=True`, per capture, all 17 captures including the degraded ones: **5.94, 5.95, 5.98, 6.15, 6.21, 6.23, 6.24, 6.28, 6.31, 6.33, 6.33, 6.34, 6.35, 6.36, 6.38, 6.40, 6.44, 6.46 m** — flat to within 6% across 100 minutes and four rovers.
- Log-only closing rate `-d(dist)/dt` within a leg (needs no clock alignment at all), p50 by 10-minute mission bin: 1.46, 1.45, 1.44, 1.43, 1.43, 1.42, 1.44, 1.45, 1.40, 1.43, 1.43 m/s. Per rover first/mid/last third: RO1 1.48/1.48/1.46, RO2 1.40/1.39/1.39, RO3 1.48/1.47/1.48, RO4 1.40/1.38/1.36. Worst decline **−2.9%** over 91 minutes.
- GUIDED + `motor_active=False`, 251 clean intervals / 1284 s: path speed p50 **0.10 m/s**, net speed p50 0.07 m/s, closing rate p50 **−0.00 m/s**, heading rate p50 **0.0 °/s**. 51% of intervals move distance-to-target by <0.25 m; net displacement p50 **0.37 m** against **6.16 m** for motor-on intervals.
- Time budget across the fleet, before vs after mission minute 65 (≈01:15 local): before — GUIDED/motor-on 12897 s (99.3%), GUIDED/motor-off **31 s (0.2%)**, MANUAL 56 s (0.5%). After — GUIDED/motor-on 4667 s (70.7%), GUIDED/motor-off **1253 s (19.0%)**, MANUAL/motor-off 552 s (8.4%).

**The "fade" is entirely composition.** Drive effort when commanded never degraded; the fraction of time at commanded-neutral throttle went from 0.2% to 19%.

Corroborating: in the same degraded captures, MANUAL + `motor_active=True` median 5 s displacement is 7.78 m (RO1 01_15_15), 6.48 m (RO4 01_24_04), 8.25 m (RO3 01_27_43). **The operator could drive the vehicle at full speed in the same minute the autopilot could not.** That is direct evidence against flat battery, bogging, obstruction, or drivetrain failure, and for a command-side failure. **[M]**

---

## 2. Root cause

### 2.1 The mechanism (verified at source on both sides, reproduced three times independently in SITL)

**ArduPilot side** — read from the ArduRover source inside `csmisko/ardupilotspf:latest` (ArduRover V4.5.7, HEAD 52bed8d542) **[M]**:

- `Rover/mode_guided.cpp:3-20` — `ModeGuided::_enter()`: `if (rover.is_boat()) { if (!start_loiter()) start_stop(); } else { start_stop(); }`, then `g2.wp_nav.init()`.
- `Rover/mode_guided.cpp:392-395` — `start_stop()` sets `_guided_mode = SubMode::Stop`.
- `grep -rn "start_stop\|_guided_mode = " Rover/` shows `ModeGuided::start_stop()` has **exactly two call sites, both inside `_enter()`**. Every other `_guided_mode =` assignment installs a *live* target. So `SubMode::Stop` inside GUIDED is reachable **only** by re-entering GUIDED — no other route exists.
- `g2.wp_nav.init()` ends in `set_origin_and_destination_to_stopping_point()` (`libraries/AR_WPNav/AR_WPNav.cpp`), so the destination is discarded twice over. The boat branch (loiter) does not pursue the old destination either — the discard is frame-class-independent.
- `Rover/mode.cpp:336-363` — `stop_vehicle()` takes throttle from `get_throttle_out_stop(...)` and sets `steering_out = 0` once stopped. **This is the source of `servo1_raw == servo3_raw == 1500`, and also of the single transient `motor_active=True` sample observed at GUIDED re-entry while the vehicle is still braking.** Both halves of the field signature come from the same six lines.
- No staleness timeout can rescue it: the 3 s "target not received" timeouts in `ModeGuided::update()` apply only to HeadingAndSpeed / TurnRateAndSpeed / SteeringAndThrottle. `SubMode::WP` has none.
- `git tag --contains dd6ce1fa73` in the image → present in **Rover-4.2.0 onward**, absent from every Rover-4.1.x tag. This is not a 4.5.7 quirk.

**spf side** — verified at the *deployed* commits, not just HEAD **[M]**. The nine incident logs name two commits, `ccc1c0283d` (RO2/RO3) and `1488ce5531` (RO1/RO4); `git show <c>:spf/mavlink/mavlink_controller.py | md5sum` is **identical** for both (`b35faab989cba50d9a3a799390a00a80`), and `move_to_point`/`_stall_verdict` diff clean against HEAD. HEAD line numbers below; field-commit equivalents in parentheses.

- `spf/mavlink/mavlink_controller.py:1516` (field :1437) — `self.reposition(lat=point[1], long=point[0])` is issued **exactly once per waypoint**, before the loop.
- `:1522` (field :1443) — `while self.distance_to_target(point) > self.tolerance_in_m:` — **no re-issue, no deadline, no mode check inside the loop.** (Contrast `move_to_home` at `:1704`/`:1718`, which *does* have a deadline and *does* re-issue on `MOVE_SKIPPED`.)
- `:1543` — the only in-loop reposition re-issue (`:1558-1561`, "Are we sleeping somwehere?") and the only in-loop `arm()` (`:1557`) are both nested inside `if self.distance_finder is not None:`.
- `spf/mavlink_radio_collection.py:405` — `if is_pi() and args.ultrasonic and not args.fake_drone:` — `distance_finder` stays `None` under `--no-ultrasonic`.
- **All 23 mission logs' `# argv:` provenance headers carry `--no-ultrasonic` and `--no-crash-recovery`** (binary-safe grep; four `.tmp` files contain NULs and return nothing to plain grep). Independent confirmation the block never ran: **zero occurrences of "Are we sleeping somwehere?" in any of the 23 logs.**
- `:1888-1918` — `reposition()` is a bare `command_int_send` with no return value; `:2029-2033` — `handle_COMMAND_ACK` is a bare `pass`. No command in this controller ever checks its result.
- `--no-crash-recovery` means `_stall_verdict` can never return `STALL_ESCAPE` (`if not crash_recovery: return STALL_MANUAL`), so the only recovery path is `_hand_over_to_manual` → `MOVE_SKIPPED` → `run_planner:1659-1669` drops a waypoint and issues a **fresh** reposition. **That is the code-level reason a fresh reposition is the only thing that ever unfroze a rover — it is the only reposition the code can still emit.**

**SITL causal test, run independently three times by three investigators, on separate containers (all removed; the developer's container on 14590-14595 untouched)** **[M]**:

| phase | servo1/servo3 | armed | mode | distance |
|---|---|---|---|---|
| P1 reposition issued, driving | 1872–1875 | True | GUIDED | 1350.8 → 1200.3 m |
| P2 GUIDED→MANUAL(2 s)→GUIDED, **nothing else sent** | **1500 / 1500 exactly** | **True** | **GUIDED** | **frozen at 1197.6 m for 40 s (<0.1 m change)** |
| P3 identical `DO_REPOSITION` re-issued | 1874/1875 within ~2 s | True | GUIDED | 1197.6 → 902.2 m |

A HOLD→GUIDED round trip behaves identically (frozen at 155.6 m for 20 s, resumed on re-issue). `set_mode("GUIDED")` while *already* in GUIDED does **not** clear the destination.

### 2.2 The field discriminator

The strongest field evidence is a clean 2×2 over all 23 logs, classifying every GUIDED re-entry by whether a fresh `reposition()` was **code-guaranteed** to follow (a planner leg line, or a stall resume, since `MOVE_SKIPPED` always re-enters `move_to_point`) **[M]**:

- **With a fresh reposition: 10/10 events closed 13.9–35.2 m in the next 30 s** (median 31.1 m).
- **Operator CH8 hand-back with no fresh reposition: 26 events, median 0.5 m closed**, only 2 above 10 m — and both resolve (one is the GPS-anomalous truncated capture, the other has 1 of 6 samples motor-active, i.e. one braking transient).
- Dose-response: handbacks vs GUIDED/motor-off sample count across 19 captures — **0 handbacks → 0–2 samples (10 captures); 2 → 10, 11; 3 → 7, 19; 8 → 121, 138.**
- **Negative control:** three captures (`01_38_45` RO3, `01_44_09` RO2, `01_48_38` RO1) have 15–28 MANUAL samples but **zero MANUAL→GUIDED handbacks** — the operator took the rover and kept it. Their GUIDED/motor-off counts are **1, 0, 1**, with no spans at all. **MANUAL time alone does not produce the freeze; only hand-back into GUIDED does.**
- **Within-capture control on the incident rover itself**, same night, minutes apart: RO1's 01:21:16 stall → collector hand-over → operator returns 01:21:58 → planner issues a **fresh leg** → resumes immediately, 15 more waypoints in 10 minutes at ~1.5 m/s. Ten minutes later, an operator takeover at 01:31:48 with **no** fresh leg → 15 minutes of nothing, while the same operator still drove it 9.1 m in 10 s in MANUAL at 01:34:26.

**Statement of root cause.** Every ArduRover ≥ 4.2.0 discards the GUIDED destination and commands neutral throttle on re-entry into GUIDED (`Rover/mode_guided.cpp:3-20,392-395` → `Rover/mode.cpp:336-363`). spf's `move_to_point` issues `reposition()` once per waypoint (`mavlink_controller.py:1516`), never re-issues inside its loop, never checks a COMMAND_ACK (`:1888-1918`, `:2029-2033`), never notices a mode change, and its only self-healing branch (`:1543-1561`) is dead code under the fleet-standard `--no-ultrasonic`. An operator mode excursion mid-leg therefore strands the vehicle in GUIDED with no destination, indefinitely, while the collector keeps writing records. **This explains why the rovers stayed stopped.** [mechanism: **M**; that it is what happened in the field: **M** for the correlation and the code, **I** for the last link, because no autopilot state was recorded — see §2.4.]

### 2.3 Two distinct triggers, and one of them is not the operator's fault

This is a real correction to the framing carried through the investigation **[M]**:

- **RO1 01:15:15** — the trigger was an **unprompted operator takeover** at 01:31:48, with the rover 18.7 m from target and closing at 1.5 m/s. Nothing in the log precedes it: no STALL, no warning, no error.
- **RO4 01:24:04** — the trigger was a **genuine physical stall** the watchdog correctly detected at 01:28:13. The collector's own hand-over is **self-healing by construction** (`MOVE_SKIPPED` → fresh reposition; 11/11 stall resumes fleet-wide produced immediate motion), and it did heal: the fresh leg at 01:29:04 closed 20.8 m. What stranded RO4 was a **second, unprompted** CH8 takeover 20 s later at 01:29:24, with no fresh leg after it.

So the collector's stall hand-over should **not** be listed as a stranding trigger. Only an operator excursion that is not followed by a planner leg strands the vehicle.

### 2.4 The one candidate that is not fully closed

The field artifacts record no autopilot state. `handle_HEARTBEAT`'s status line is inside `if not self.drone_ready:` (`:2057-2075`), so `armed` is logged **exactly once per capture, at startup** — RO1 01:15:15 contains exactly one HEARTBEAT line. The zarr has no arm/mode/servo/planner array or attribute. Therefore:

- **C1 (dominant): GUIDED destination discarded on re-entry, never re-issued.** Everything above.
- **C2 (not formally excludable from the artifacts): the vehicle was disarmed in GUIDED.** A disarmed rover also outputs neutral servos in GUIDED and also defeats the watchdog's `armed` gate, and the only mid-leg `arm()` is in the same dead `--no-ultrasonic` block — so it would be equally unrecoverable and observationally identical. **Two independent lines close it short of proof:** (i) Rover emits "Throttle armed"/"Throttle disarmed" on every arm/disarm (`Rover/AP_Arming.cpp:135,155`) and spf forwards STATUSTEXT verbatim (`:2111-2123`); across **all 23 logs there are 7 "Throttle armed" and exactly ONE "Throttle disarmed"** — RO4 01:24:04 at 01:36:17.430, followed by "Throttle armed" at 01:36:17.907 (+0.48 s). RO1 01:15:15 has neither, so it entered already armed and no disarm was ever announced during its 15-minute dead tail. (ii) SITL control: forced disarm in GUIDED then re-arm with **no mode change** resumed driving to the **original** destination (383 → 0.7 m), matching `AR_WPNav::update`'s disarmed early-return which zeroes outputs but preserves `_orig_and_dest_valid`. **A disarm cannot produce a persistent freeze; only mode re-entry can.**
- **C3 (refuted for this mission, live latent hazard): the reposition was denied at issue.** `Rover/GCS_Mavlink.cpp:743-747` rejects `DO_REPOSITION` with `MAV_RESULT_DENIED` when `!in_guided_mode() && !change_modes`; spf sends `param2=0` so `change_modes` is false, and the ack is discarded. **Refuted by measurement for 2026-08-07:** across all 383 legs in the mission, 369 with ≥5 GPS fixes in the first 20 s after issue, net displacement min **2.76 m**, p1 4.46 m, p5 10.09 m, median 20.11 m — **zero legs with <2.0 m of motion.** All three incident freeze onsets drove 20–40 m at 1.14–1.42 m/s on the leg they later froze on. There is no instance in the whole mission of "DO_REPOSITION issued, vehicle does not move."

**What would settle C1 vs C2 definitively:** the autopilot's own DataFlash `.bin` logs from RO1 and RO4 for 2026-08-07 01:15–01:47. `MODE` (with `ModeReason`: RC vs GCS_COMMAND) at 10 Hz, `ARM`/`DISARM` events, `RCIN` ch8, `RCOU` servo1/servo3, and `GUID`/`NTUN`/`PSC` target records would prove all of it on the real vehicle rather than in SITL, in one pass. If those SD cards have been recycled, this is unrecoverable for 2026-08-07 and only a controlled bench reproduction (§5) can close it.

---

## 3. Why it was not caught

### 3.1 The stall watchdog's `driving` gate is structurally blind to exactly this state

`spf/mavlink/mavlink_controller.py:1262-1264` (field `:1183-1187`) **[M]**:

```python
driving = self.armed and self.motor_active and self.mav_mode == "ROVER_MODE_GUIDED"
if not driving:
    self._reset_stall_anchor()
    return STALL_OK
```

and `motor_active` comes from `handle_SERVO_OUTPUT_RAW` (`:2201-2205`): `False` **iff** `servo1_raw == 1500 and servo3_raw == 1500` — i.e. iff the autopilot is commanding neutral throttle on both skid channels.

The failure state *is* commanded-neutral throttle. So the watchdog **resets its own clock, once per pass, on precisely the signature it exists to catch.** Field confirmation: **zero STALL lines occur inside any of the long frozen spans**, including RO1's 15.6-minute and RO4's 17.2-minute dead tails, while the same watchdog fired **11 times** that night on the `motor_active=True` physical stalls and cleared them in 10–152 s. `STALL_DETECT_SECONDS = 10.0`. **[M]**

Note what this means for evidence handling: **a stall can only ever be reported with `motor_active=True`, whether or not any given hypothesis is true.** "All 8 mission stalls show motor_active True" therefore carries zero discriminating power and must not be cited as evidence for anything.

### 3.2 Three further blind spots on the same path

- **No leg-level deadline.** `move_to_point`'s loop (`:1522`) has no timeout. `move_to_home` (`:1704`, `:1718`) has one. RO1 held a single waypoint for **939 s** and RO4 for **1030 s** against healthy bounce leg durations of 34–56 s; nothing in the code notices. **[M]**
- **No command acknowledgement anywhere.** `handle_COMMAND_ACK` is `pass` (`:2029-2033`); `arm()`'s ack check is commented out at `:1886`; `reposition`, `disarm`, `reverse`, `set_rtl_mode`, `do_mission`, `set_mode` are all fire-and-forget. There is no path by which a rejected command reaches the planner. **[M]**
- **The health accounting does not exist in these captures.** None of the 22 zarr stores carries a `planner_control_lost_seconds` attribute or any planner/mode/arm key — I checked the full key list. So even the aggregate that was supposed to flag this is absent. **[M]** (This is consistent with already-fixed defect (b).)
- **`planner_control_loss_reason` (`:1041-1050`) does not test `armed`** — a vehicle in GUIDED but never armed records indefinitely and is reported healthy. Latent, not observed to have fired here. **[M]**

---

## 4. What is contradicted

### 4.1 "Rovers 1 and 4 were sitting in GUIDED not moving" — materially overstated in scope

Of the **nine listed mission captures**, exactly **one** shows the signature: RO1 01:15:15. **[M]**

- **RO4's two complete mission captures are clean.** `00:16:43` — 2220 m, 1.36 m/s, 0.8% stationary, longest stationary span 6.3 s, **zero MANUAL samples**, 47 waypoints. `00:45:44` — 2242 m, 1.33 m/s, 0.0% stationary, longest span 4.1 s, zero MANUAL samples, 46 waypoints. The "rover 4 was not moving" observation **cannot refer to either of them.**
- RO1's other listed capture, `00:40:09`, is also clean: 2494 m, 1.36 m/s, 57 waypoints during recording, 384/386 Dist samples `motor_active=True`, zero MANUAL samples, one 10 s physical stall cleared in 10.1 s.
- The RO4 instance of the incident is in the **aborted** `01:24:04.log.tmp`, which is not part of the nine listed artifacts. The RO3 instance (`01:27:43.tmp`) is likewise unlisted — and RO3 was not named in the report at all.

**Consequence for the record: the incident is 3 captures on 3 rovers, of which 2 captures are aborted `.tmp` runs, and the affected set includes RO3 but not most of RO4's mission.**

### 4.2 The already-fixed reporting defects account for a real, substantial share of the symptom

This is the honest half of the "it's an artifact" reading, and it should not be buried **[M]**:

- **MANUAL + `motor_active=False` stationary time is ~590 s fleet-wide** (RO4 01_24_04 183 s, RO3 01_27_43 137 s, RO1 01_48_38 107 s, RO1 01_15_15 96 s, RO3 01_38_45 67 s) against ~1350 s of GUIDED + motor-off. In **RO3 01_27_43 (137 s vs 91 s)** and **RO1 01_48_38 (107 s vs 0 s)** the MANUAL-stationary bucket is the **larger** one.
- 429 of RO4's 6000 clean-capture records (7.15%) were written during operator MANUAL takeovers; 1429 of RO1 01:15:15's 3000 records (47.6%) came after its takeover.
- That recording is exactly defect (b) — the collector never re-checked planner control — and the mislabelling as planner-controlled is exactly defect (a).
- A competing and equally-supported reading of the late captures: **from ~01:31 the operators may have been retrieving the rovers, not rescuing them.** RO3 01_27_43 was *healthy* (34.5 m from target, closing at 1.5 m/s) when it was taken at 01:31:07, parked 86 s, driven 35 m *away* from its target, parked again. RO1 01_48_38 and RO2 01_44_09 were taken and kept, with zero handbacks and zero freezes.

**So: "rovers collecting radio data but not moving in guided mode" is partly the real GUIDED-with-no-destination bug and partly the recording-through-operator-possession defect, and the split cannot be determined from the artifacts.** RO1 01:15:15's pattern (a 3–7 m nudge every 60–90 s, then back to GUIDED) does read as repeated rescue attempts, which favours the bug for that capture. The operator's account of the 01:15–01:47 window is the only thing that settles the proportion.

### 4.3 Specific claims made during this investigation that the evidence contradicts

- ❌ **"Every GPS-bearing capture from 01:15:15 onward is degraded."** False. `01_38_45` RO3 = 1.28 m/s and `01_44_09` RO2 = 1.38 m/s, both squarely inside the healthy band, both with zero GUIDED/motor-off time. **[M]**
- ❌ **`01_48_38` RO1 and `01_38_45` RO3 counted as instances of this failure.** False. Both have **zero** GUIDED + `motor_active=False` intervals. `01_48_38`'s entire 121 s stationary window is MANUAL + motor-off in all 21–23 overlapping log samples; `01_38_45`'s 75 s window is 100% MANUAL. Their non-progress is operator-held stops. **[M]**
- ❌ **"19 spans in 6 captures, ~1352 s."** Overstated as a scope figure. ~96% of the GUIDED/motor-off mass is three captures: RO1 01_15_15 (653 s), RO4 01_24_04 (598 s, noise-inflated), RO3 01_27_43 (91 s), plus 51 s in the truncated RO4 00_08_09. Every other capture has ≤5 s. **[M]**
- ❌ **"Every long non-progress span is `motor_active=False`, never True."** False. Three spans ≥15 s have `motor_active=True` with <3 m of distance change: RO4 `00_45_44` @01:14:12 (55.0 s), RO3 `00_55_11` @01:26:08 (31.8 s), RO3 `00_44_09` @00:54:05 (28.0 s). Those are genuine physical stalls — exactly what the watchdog is for — and all recovered. **[M]**
- ❌ **"Each freeze is preceded within one 5 s log interval by a MANUAL sample."** 17/19 spans have one within ≤10.2 s but only 13/19 within one interval. The extra interval is explained: `stop_vehicle()` produces one transient `motor_active=True` sample while braking, and GPS shows a monotonic coast-down (RO4 01_24_04: 12.2 m over 25 s decelerating from 2.5 m in the first 2 s to 0.3 m in the last 1.5 s). The prediction needs relaxing to two intervals. **[M]**
- ❌ **"Servos were pinned at neutral for the whole span."** Overstates 5 s point sampling. 14 of 276 GUIDED/motor-off intervals show >3 m of real GPS motion (RO4 01_24_04 closes 27.3 → 22.4 → 17.6 m across two consecutive False-labelled intervals). 13 of the 14 lie within ±25 s of a MANUAL sample, so the coast-down/operator-burst account survives — but `motor_active` read at the print instant does not certify neutral throttle for the intervening 5 s. **[M]**
- ❌ **Battery / traction fade (H4).** Refuted: speed while commanded is flat to −2.9% across the mission (§1.3), and the non-progress happens at commanded-zero throttle with the vehicle inert in position *and* heading. **Not** excluded as a background factor — there is no battery/voltage/current channel anywhere in the logs or zarr, and `handle_SYS_STATUS` reads only the three sensor bitmasks and discards `msg.voltage_battery`/`msg.current_battery`. What is excluded is battery sag as the explanation for the non-progress. **[M]**
- ❌ **Rejected-reposition-at-issue (H2).** Refuted for this mission (§2.4, C3). Still a real latent defect.
- ⚠️ **`docs/learnings.md:639-655` (WP_RADIUS no-op) did not reproduce on ArduRover 4.5.7.** With `WP_RADIUS` readback 5.0, a 3 m forward target drove net 2.94 m; 4 m → 3.80; 6 m → 5.63; 8 m → 7.51. Reverse escape leg: 3 m → net 2.71 m with real reverse servo output (1354/1351). This matters because 741 of 19999 bounce legs (3.7%) are shorter than `tolerance_in_m=5`. **This learning needs re-testing on field firmware before it is relied on again.** **[M]**

---

## 5. Fixes

### 5.1 Fixes the cause

**F1 — Re-issue `reposition()` on any re-entry into GUIDED. (highest value, smallest change)**
Latch `self.mav_mode` at the top of `move_to_point`'s loop (`mavlink_controller.py:1522`); on any observed transition *into* `ROVER_MODE_GUIDED`, re-send `self.reposition(lat=point[1], long=point[0])`. `move_to_home` (`:1705-1715`) already implements exactly this pattern on `MOVE_SKIPPED` — the repo contains the correct idiom in one place and not the other.
**Test that proves it:** SITL A/B with the *only* difference being the patch. Drive a leg, flip GUIDED→MANUAL→GUIDED mid-leg with nothing else sent. The unpatched control must stay frozen at neutral for ≥40 s (already measured three times); the patched build must resume within one re-issue interval and reach the *original* target. Then a bench capture with a deliberate CH8 flip mid-leg, checked for zero GUIDED/motor-off spans >10 s.

**F2 — Unconditional periodic re-issue while in the waypoint loop.**
Re-send the destination every ~5 s regardless of `distance_finder`, and delete the `--no-ultrasonic`-gated dead branch at `:1543-1561` (or move the reposition out of it). This is defence-in-depth over F1 and also covers packet loss on the original command.
**Test:** SITL soak — flip modes at random 20 times over 10 minutes and require every leg to complete. **Watch for:** repeated `DO_REPOSITION` re-arming ArduPilot's input-shaping / speed profile. Measure achieved ground speed with and without periodic re-issue over 10 legs; if the re-issue costs speed, back off to F1's edge-triggered form only.

**F3 — Set `MAV_DO_REPOSITION_FLAGS_CHANGE_MODE` (param2 bit 0) in the `command_int` at `:1888-1918`.**
spf currently sends `param2=0`, so `Rover/GCS_Mavlink.cpp:743-747` denies the command whenever the vehicle is not already in GUIDED. Setting the bit makes the reposition self-healing for free and closes the C3 hazard.
**Test:** SITL — send `reposition()` while in HOLD, MANUAL and disarmed; require `MAV_RESULT_ACCEPTED` and a mode change to GUIDED, versus the currently-measured `MAV_RESULT_DENIED`.

**F4 — Actually read `COMMAND_ACK`.**
Replace the bare `pass` at `:2029-2033` with a per-command-id record of the last result; have `reposition()` and `arm()` wait briefly for their ack and log an ERROR on anything other than `MAV_RESULT_ACCEPTED`; un-comment/restore the arm ack at `:1886`.
**Test:** SITL — force a denied reposition (send it in HOLD without F3) and require an ERROR line naming `MAV_CMD_DO_REPOSITION` and `DENIED`. Regression: a normal mission emits zero such lines.

### 5.2 Would have caught it sooner

**G1 — Fix the stall watchdog's `driving` gate (`:1262-1264`).**
Drop `motor_active` from the gate. Gate on `mav_mode == GUIDED` **and** `planner_in_control` **and** "a destination has been issued and not yet reached", and derive no-progress from GPS regardless of servo state. Keep the existing 10 s threshold for the motor-on case; use a longer one (20–30 s) for the motor-off case so waypoint pivots and the measured `stop_vehicle()` coast-down (up to ~12 m over 25 s) do not trip it.
**Test:** replay the recorded `(mav_mode, motor_active, GPS)` traces from all 23 aug6 captures through `_stall_verdict` offline. Requirement: it fires within 30 s of RO1's 01:32:29 onset and RO4's 01:29:44 onset, **and produces zero new firings on the nine clean captures** (whose worst non-tripping GUIDED pivots already run 6.00–9.66 s against a 10.0 s threshold — margin is thin, so this regression check is mandatory).

**G2 — Record autopilot state per record in the zarr.**
Add `mav_mode`, `armed`, `servo1_raw`, `servo3_raw`, `planner_in_control`, and the last-issued destination alongside `gps_lat/gps_long/heading`. Today the zarr carries only RF + `gps_lat/gps_long/heading/gps_timestamp/system_timestamp` — every state conclusion in this RCA had to be joined from 5 s log prints, and the armed state could not be recovered at all.
**Test:** re-derive the four-state dwell budget for a bench capture **from the zarr alone**, with no `.log` — and require it to match the log-joined budget within 5%.

**G3 — Leg-level watchdog: alarm and abort when one waypoint is held longer than N × the routine's median leg duration.**
RO1 held one waypoint 939 s and RO4 1030 s, against 34–56 s (bounce) and ~5 s (circle) medians. A 5× threshold would have fired within ~4 minutes.
**Test:** same offline replay; require firing on the two incident legs and zero firings across the nine clean captures (whose longest legs are 76 s).

**G4 — Log `armed` in the per-waypoint Dist line (`:1538-1540`) and emit a periodic HEARTBEAT status line.**
Currently the status line lives inside `if not self.drone_ready:` (`:2057-2075`), so `armed` appears exactly once per capture. This is the single reason C2 in §2.4 could not be closed by measurement.
**Test:** grep a bench capture and require ≥1 armed sample per minute.

**G5 — Make the FC DataFlash `.bin` a routine post-mission artifact.** Copy it off the SD card into the capture directory. Everything left unknown in §6 becomes measurable.

**G6 — Explicitly gate the fleet flags.** `--no-ultrasonic` silently disables the only self-healing reposition *and* the only mid-leg re-arm. Either make those paths independent of `distance_finder` (F2) or fail loudly at startup when a recovery path is disabled.

---

## 6. What remains unknown

Stated as gaps, not filled in.

1. **The field ArduPilot firmware version and FC parameters.** No `AUTOPILOT_VERSION`, boot banner, or param dump exists in any capture. The destination-discard behaviour is verified present in **every Rover release from 4.2.0 (Dec 2021) onward**, and every code path was checked against plausible settings of `FRAME_CLASS`/`GUID_OPTIONS`/`WP_RADIUS`/`SERVOn_FUNCTION`/`FS_CRASH_CHECK`/`OA_TYPE` — but that is analysis, not measurement on the field vehicles. **[I]**
2. **Whether the vehicles were armed during the field freezes.** Not recorded after startup (§2.4). Closed indirectly two ways, not measured. Only the DataFlash `ARM`/`MODE` records can settle it.
3. **Whether the mode excursions were operator CH8 input or something else.** The mode change is visible only as a `mav_mode` field flip at 5 s sampling; no RC_CHANNELS or CH8 line is logged. That a *human* was on the sticks is supported by 3–9 m of real motion during MANUAL windows, but nothing records intent. `FS_THR_ENABLE` is forced to 0 by `rover3_base_parameters.params:8`, which argues against an RC failsafe, but is not proof.
4. **Operator intent in the 01:15–01:47 window — rescue or retrieval?** This decides how much of the reported symptom belongs to the bug versus already-fixed defect (b) (§4.2). The artifacts cannot answer it. **Ask the operators.**
5. **One measured span fits nothing.** RO4 `00_08_09.log.tmp`, onset 00:13:23, 15.2 s, on a leg issued 6 s earlier while already in GUIDED, with **no MANUAL sample anywhere earlier in the capture** (the first is 21 s later). Worse: its distance series swings ±10 m per 5 s interval **with `motor_active=False`** — 126.45, 124.75, 133.66, 122.98 m. A coast-down cannot move 10 m up and then 10 m back down. Either real ~2 m/s motion at commanded-neutral throttle (which contradicts the mechanism) or a position/EKF glitch (which would make that capture's distance arithmetic untrustworthy). Its zarr is a truncated `.tmp` with only 457 valid fixes over 397 s, so it could not be tested from GPS. **This is the author's own falsification criterion and it is unresolved.** Every span ≥20 s does have a documented excursion.
6. **~590 s of MANUAL + motor-off stationary time is unexplained.** Nothing in the artifacts says why the operator parked rovers for 60–180 s at a time.
7. **A third, wholly unexplained failure mode at 02:19–02:20.** RO4 `02:19:55`, RO1 `02:19:58` and RO2 `02:20:04` emitted 372 / 213 / 130 "waiting for rover to move into guided mode" lines with **zero** Dist samples and zero waypoints (RO4 waited until 02:32:28, ~12 min). Their zarrs are zero-fill preallocation — 3000/3500 records with `system_timestamp`, `gps_timestamp` and `gps_lat` all exactly 0. Five stores in total are zero-fill (RO1 01:12:14, RO3 01:36:17, RO4 02:19:55, RO1 02:19:58, RO2 02:20:04) and RO3 02:19:54 has no zarr at all. These rovers never entered GUIDED; nothing here explains why.
8. **Deployed-tree drift.** 8 of the 9 incident captures ran a `-dirty` tree (`ccc1c02`/`1488ce55`). `mavlink_controller.py` is anchored — identical md5 across both named commits, and `move_to_point`/`_stall_verdict` diff clean against HEAD — but other files could differ from their committed form. RO4's four captures ran a **clean** `1488ce55` and are the only fully anchored ones.
9. **`docs/learnings.md:639-655` (WP_RADIUS no-op) is now in doubt** (§4.3). Until re-tested on field firmware, whether the 741/19999 sub-5 m bounce legs are harmless creeps or hard freezes is open.
10. **Two concurrent planner loops.** RO1 01:15:15 shows two interleaved Dist streams ~0.9 s apart with different targets (46.2 m and 60.6 m) from 01:46:31, after "Moving files to final location". The stall-resume path also appears to issue the planner leg twice (RO4 01:24:04 at 01:29:04.362/.367, targets 3.9 m apart; RO1 at 01:21:58.455/.460). This is outside the recorded data window and cannot explain 01:31–01:45, but **two concurrent loops repositioning the same flight controller with contradictory destinations is an independent freeze mechanism that was not tested.** It deserves its own investigation.
11. **Battery is not positively excluded as a background factor** — only as the explanation for the non-progress. No battery telemetry exists anywhere; `handle_SYS_STATUS` discards it. A tripped low-battery failsafe would have appeared as a STATUSTEXT (the logs do carry FC text verbatim) and none did — but that only rules out a *tripped* failsafe, and only if `BATT_MONITOR` was configured, which is unverified.
12. **Whether the RF payload recorded during the stationary spans is usable.** Not examined. 1101 of RO1 01:15:15's 3000 records (36.7%) were captured with the vehicle parked; the data-curation implications were not assessed here.