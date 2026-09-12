# ArduPilot rover CLI

`ardu_cli.py` is the guarded command-line front end for ArduPilot inspection
and sensor calibration on an SPF rover.

Run it without arguments for the authoritative on-rover command cheatsheet:

```bash
python -m spf.ardupilot.ardu_cli
```

The flight-controller USB serial link has one owner. Stop production before a
direct-serial command and restore it afterward:

```bash
sudo systemctl stop mavlink_controller.service
source ~/spf-virtualenv/bin/activate

python -m spf.ardupilot.ardu_cli status
python -m spf.ardupilot.ardu_cli compass
python -m spf.ardupilot.ardu_cli prearm

sudo systemctl start mavlink_controller.service
```

All inspection commands support `--json` and `--json-output FILE`. `compass`
can also inspect an export without opening the vehicle:

```bash
python -m spf.ardupilot.ardu_cli compass --params rover.params --json
```

The compass report lists the priority IDs and every configured/detected slot,
including its full bus-aware device ID, external classification, and yaw-use
flag. Exact duplicate nonzero device IDs fail policy because ArduPilot cannot
use an ID-keyed priority list to distinguish them.

If exactly one detected compass is marked external and it has the fleet's
expected device ID, the CLI can repair only the priority order and
`COMPASS_USE*` selection:

```bash
python -m spf.ardupilot.ardu_cli compass --repair --yes \
  --parameter-timeout 60
```

Repair refuses an armed rover, incomplete parameter downloads, duplicate IDs,
extra unconfigured compasses, zero/multiple external compasses, and an unknown
external device ID. It never guesses detection, external classification,
orientation, or calibration. Priority writes require an ArduPilot reboot;
reboot and run the read-only `compass` and `prearm` checks again before use.

Production boot evaluates the same policy while verifying managed vehicle
parameters. It prints all compass slots and priorities to the systemd journal,
writes `/home/pi/compass_ready.json`, and refuses collection/motion on failure.
Inspect recent boot evidence with:

```bash
journalctl -u mavlink_controller.service -b --no-pager | grep -E 'Compass|compass'
```

## Compass calibration

Calibration changes persistent flight-controller state. Put the assembled
rover on a safe support, verify it is disarmed, move away from steel, magnets,
high-current wiring, SDRs, and the Pi, and then run:

```bash
python -m spf.ardupilot.ardu_cli magcal start \
  --yes --mask 1 --retry --monitor-seconds 300
# Successful calibration is autosaved by default. To abort an active run:
python -m spf.ardupilot.ardu_cli magcal cancel --yes --mask 1
```

Progress is printed as it arrives. `--monitor-seconds` is a maximum; the CLI
returns early when ArduPilot sends the terminal calibration report.

Use `--no-autosave` on `magcal start` only for a deliberate review-before-save
workflow; after reviewing the reports, save it with `magcal accept --yes`.

## Accelerometer calibration

The CLI supports ArduPilot's full interactive six-position calibration. Stop
the production service first because it normally owns the flight-controller
USB serial link, make the assembled rover safe and disarmed, then run:

```bash
sudo systemctl stop mavlink_controller.service
python -m spf.ardupilot.ardu_cli accelcal start --yes
```

The command prints the complete pose plan, then waits for ArduPilot to request
each pose: level, left side, right side, nose down, nose up, and upside down.
It waits for Enter before acknowledging each stable pose and fails closed on an
unexpected pose, rejected acknowledgement, timeout, or missing terminal result.
After a successful save, reboot ArduPilot and run `ardu_cli prearm` before
restoring production.

### Investigating an unconfirmed calibration result

On Rover 4 (2026-08-04), the CLI reported no terminal result after the six-pose
sequence, and nondefault offsets/scales were observed afterward. Those values
alone do not establish whether that attempt succeeded: they can be from an
earlier calibration. The cause of that field incident has not been confirmed.

The CLI now tolerates repeated pose requests while waiting for the final result,
using one fixed timeout. Previously, a buffered request for pose 6 could abort
the wait immediately, leaving a following SUCCESS message unread. This sequence
is reproduced in a regression test; it is not assumed to explain the field run.
The accepted success/failure messages are unchanged, and missing confirmation
still stops the combined calibration procedure before compass calibration.

```bash
# Read stored parameter values without writing calibration state.
python -m spf.ardupilot.ardu_cli accelcal verify

# Capture messages during a physically supervised calibration.
python -m spf.ardupilot.ardu_cli accelcal start --yes \
  --trace --trace-output accelcal_trace.txt
```

`accelcal verify` inspects the offset/scale parameters reported for IMU slots
1–3. Each slot is labeled `default_values`, `nondefault_values`, `unknown`
(missing axes), or `invalid_values` (nonfinite values or scales outside
0.8–1.2). It does not infer which sensors are active, validate their identity,
or prove that any values were written by the latest run. Unity scales with
nonzero offsets are simply nondefault values, not a calibration failure.

Its exit codes describe **parameter inspection**, not calibration readiness:

- **0:** download complete, all reported slots have six finite values, and scales
  are in range. This includes factory-default slots.
- **1:** a complete inspection found invalid numeric values.
- **2:** the download or a reported slot is incomplete, or no offset/scale
  parameters were received. The overall status is `unknown` even if one complete
  slot was received. Invalid values already observed are still listed.

Human-readable and JSON output both include this limitation. Require a confirmed
calibration result and ArduPilot pre-arm checks before restoring production.

`--trace` records calibration messages to stderr, with command IDs, parameters,
status text, and relative timestamps. Routine telemetry is counted by type.
During pose and terminal waits, tracing receives messages without a type filter
so unexpected message types can be inspected. It does not expand the success
matcher. `--trace-output` implies tracing and saves the transcript when the
calibration returns or raises an error. Both options also apply to the
accelerometer stage of `calibrate`.

The CLI refuses direct-serial access while `mavlink_controller.service` is
active. Calibration actions additionally refuse an armed vehicle and require
the explicit `--yes` acknowledgement. `--allow-active-service` exists only for
an expert using a deliberately shared endpoint; it should not be used against
the flight-controller serial device. Once connected, the CLI also claims the
TTY exclusively at the kernel level so a delayed production-service start
cannot become a second reader and steal calibration messages.
