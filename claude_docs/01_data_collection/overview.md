# 01 · Data Collection — overview

How SPF physically gathers data: rigs that move radios through space while recording raw IQ
and ground-truth positions, producing the `.zarr` datasets the rest of the pipeline consumes.

## The two platform families

| Family | Motion control | Positioning | Data format | Orchestrator | Status |
|---|---|---|---|---|---|
| **Wall array** (CNC gantry) | GRBL stepper controller | XY in mm (gantry geometry) | v5 | `spf/grbl_radio_collection.py` | ✅ v2 active |
| **Rover / drone** (mobile) | MAVLink → ArduPilot | GPS lat/long → XY | v4 | `spf/mavlink_radio_collection.py` | ✅ v3.1 active |

Both drive radios in a **routine** (bounce / circle / center / diamond) and record from **two
2-element receivers** + an emitter.

## Hardware generations

| Platform | What it is | Status |
|---|---|---|
| Wall array **v1** | Fixed RX wall; emitter on suspended belt rig | ⚠️ archived (superseded by v2) |
| Wall array **v2** | Two independently moving arrays, TMC2208 drivers, PoE, laser calibration | ✅ active |
| Rover **v1** | Early custom chassis | ⚠️ abandoned (motor/gear failures) |
| Rover **v2** | GoBilda Recon chassis | ⚠️ deprecated |
| Rover **v3.1** | RPi + ArduPilot FMUv3 over MAVLink, 2× PlutoPlus, GPS boundary enforcement | ✅ active |

Details + bill-of-materials live in the per-platform READMEs under `data_collection/`.

## Motion-control stacks

- **GRBL** (`spf/grbl/`) — drives stepper-motor XY gantries. Key classes: `GRBLDynamics`
  (Cartesian ↔ motor-step inverse kinematics), `GRBLController` (serial interface to the GRBL
  board), `GRBLManager` (runs a planner in a thread). Entry: `get_default_gm(serial, routine)`.
- **MAVLink** (`spf/mavlink/`) — drives ArduPilot rovers/drones. Key class: `Drone`
  (`spf/mavlink/mavlink_controller.py`) — waits for GPS 3D fix + EKF health + armed, then runs a
  planner. Supports `fake=True` for hardware-free testing. `drone_get_planner(routine, boundary)`
  builds the motion plan; GPS boundaries come from `spf/gps/boundaries.py`.

## Motion planners (shared)

`spf/motion_planners/planner.py` provides the trajectory generators used by **both** stacks:
`BouncePlanner` (random billiard with boundary reflection), `CirclePlanner`, `StationaryPlanner`
(go to a point and hold — the "center" routine), `PointCycle` (waypoint loop — "diamond").
`spf/motion_planners/dynamics.py` handles boundary geometry / collision checks.

## Radio targets (emitters)

`spf/radio_targets/` — the signal sources: **WiFi** (ESP32, FCC test mode), **LoRa** (Heltec,
915 MHz), and **5.8 GHz FPV VTX**. The emitter can also be an SDR in TX mode. Selected in the
YAML `emitter:` block (`type: sdr | esp32 | o4`).

## How a collection run works (end to end)

```
load YAML config + device-mapping (USB port → SDR URI)
        │
        ▼
build position controller (GRBLManager | Drone)  ──▶ set_and_start_planner(routine)
        │                                                     │
        ▼                                                     ▼
DataCollector.radios_to_online()   (blocking: init + verify all SDRs)
        │
        ▼
wait until the platform is actually moving (poll planner status)
        │
        ▼
DataCollector.start()  ──▶ per-receiver threads: capture IQ → beamform → tag position+time → write
        │
        ▼
poll is_collecting() until n-records-per-receiver reached (or --run-for-seconds)
        │
        ▼
DataCollector.done()  ──▶ finalize; move temp/ files to output dir; (rover) return-to-home
```

The collector classes live in `spf/data_collector.py`: `DroneDataCollectorRaw` (rover/v4),
`GrblDataCollectorRaw` (wall/v5), and `GrblDataCollector` (legacy v2). Output is a `.zarr` +
`.yaml` (config copy) + `.log`, handed off to [`03_datasets`](../03_datasets/overview.md).

Optional **real-time inference** (`--realtime --checkpoint`) runs a trained model during
collection on the rover/RPi5 — see `data_collection/rpi5_inference/setup.md`.

## Known issues / tech debt (spot-verified)

- **Duplicated device-mapping parser** — near-identical logic in `grbl_radio_collection.py` and
  `mavlink_radio_collection.py`; candidate for a shared util.
- **Asymmetric controller APIs** — `GRBLManager.has_planner_started_moving()` vs
  `Drone.is_planner_in_control()`; no common base class despite parallel roles.
- **Routine naming inconsistency** — wall array uses `v1_calibrate`/`v4_calibrate`/`rx_circle`/
  `tx_circle`/`bounce`; rover uses `bounce`/`circle`/`center`/`diamond`.
- ⚠️ **`GrblDataCollector`** (`spf/data_collector.py:624`) is the legacy v2 collector; current
  configs use `GrblDataCollectorRaw` (v5). *(The exact failure mode for v2 was not verified — do
  not assume it raises; confirm before relying on it.)*

## Reference inventory (current source of truth)

No per-module deep dive exists yet for collection. Until the leaf pages below are written, the
verified surface inventory is the source of truth for this section's symbols — signature-level
truth (inputs, outputs, side effects) with line numbers for every symbol across `spf/mavlink/`,
`spf/grbl/`, `spf/gps/`, `spf/motion_planners/`, and the two `*_radio_collection.py` orchestrators:

- [`../reference/_inventory/collection_hw.md`](../reference/_inventory/collection_hw.md)

## Planned leaf pages

*(All to be written — see the reference inventory above for verified symbol details in the meantime.)*

- `hardware_platforms.md` — per-generation BOM, wiring, status.
- `motion_control.md` — GRBL vs MAVLink controller APIs in depth.
- `motion_planners.md` — planner classes + GPS boundaries.
- `radio_targets.md` — emitter setup and config.
- `orchestrators.md` — the two `*_radio_collection.py` entry points, full CLI, + RPi5 edge inference.

## Known issues

Collection-relevant entries from [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) (severities as
re-triaged there):

- **#17** 🟠 P1 — `yaml_defaults` reads a module-global `args` (not a parameter) → `NameError` if
  imported as a library (`spf/mavlink_radio_collection.py:32-71`).
- **#18** 🟠 P1 — `Drone.handle_RC_CHANNELS` runs `sudo shutdown` / `sys.exit` from the MAVLink
  message thread (`spf/mavlink/mavlink_controller.py:897-909`).
- **#4** 🟡 P2 (dead branch) — `swap_lat_long` 2-D-array branch is a no-op, but no live caller passes
  a 2-D `(N,2)` array, so the buggy branch is never reached; live rover GPS math is correct today
  (`spf/gps/gps_utils.py:8-10`).
- **#5** 🟡 P2 — `CirclePlanner` radian-increment vs `< 360`. Overshoot (~57×) is real **but does not
  manifest**: `angle_to_pos` uses periodic `sin/cos` so every lap is the correct circle, and
  collection ends by `n-records-per-receiver`, not generator exhaustion
  (`spf/motion_planners/planner.py:205`).
- **#29** 🟡 P2 (dead) — `wait_while_moving` indexes `a/b_motor_steps` keys that `update_status`
  never sets → `KeyError`; also dead (no callers) (`spf/grbl/grbl_interactive.py:363`).
- **#30** 🟡 P2 — device-mapping parsing duplicated verbatim across the two collection orchestrators
  (`spf/grbl_radio_collection.py:94-102`, `spf/mavlink_radio_collection.py:37-45`).
