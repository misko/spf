# ArduPilot rover CLI

`ardu_cli.py` is the guarded command-line front end for ArduPilot inspection
and compass calibration on an SPF rover.

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

## Compass calibration

Calibration changes persistent flight-controller state. Put the assembled
rover on a safe support, verify it is disarmed, move away from steel, magnets,
high-current wiring, SDRs, and the Pi, and then run:

```bash
python -m spf.ardupilot.ardu_cli magcal start --yes --monitor-seconds 120
# Successful calibration is autosaved by default. To abort an active run:
python -m spf.ardupilot.ardu_cli magcal cancel --yes
```

Use `--no-autosave` on `magcal start` only for a deliberate review-before-save
workflow; after reviewing the reports, save it with `magcal accept --yes`.

The CLI refuses direct-serial access while `mavlink_controller.service` is
active. Calibration actions additionally refuse an armed vehicle and require
the explicit `--yes` acknowledgement. `--allow-active-service` exists only for
an expert using a deliberately shared endpoint; it should not be used against
the flight-controller serial device.
