# Rover 3.1 Boot Profiling

This document defines the boot endpoint, shows the measured Rover 2 and Rover 3
critical paths, and records the safe optimizations applied in PR #13.

## Endpoint

For indoor, repeatable testing, boot is considered ready when
`mavlink_radio_collection.py` first logs:

```text
Drone startup wait for drone ready: ...
```

This proves that firmware preparation, radio attestation, repository checking,
vehicle-parameter verification, and MAVLink startup completed. GPS fix, EKF
health, and GUIDED readiness are not suitable indoor timing endpoints because
they depend on sky view and vehicle state.

## Measured baseline

All times below are monotonic seconds since kernel boot.

| Milestone | Rover 2, one Pluto | Rover 3, two Plutos |
|---|---:|---:|
| Pluto preparation starts | 10.1 | 10.0 |
| Pluto preparation completes | 21.2 | 31.4 |
| Parameter verification completes | 103.6 | 59.5 |
| GPS-time helper completes | Did not; manually stopped at 211.2 | 62.4 |
| First indoor GPS/EKF wait | About 231.6 after manual stop | 85.4 |

Rover 2 spent about 53 seconds updating packages and reinstalling an unchanged
SPF checkout, then entered a 180-second GPS-time wait. Rover 3 already had GPS
UTC and therefore did not pay the timeout, but heavy imports still preceded the
first readiness message.

## Optimized behavior

The normal boot path now:

1. Performs the repository update check.
2. Skips `apt` and `pip install -e` when `HEAD` is unchanged.
3. Performs the full dependency refresh, editable install, unit reconciliation,
   and reboot when an update actually changes `HEAD`.
4. Verifies all committed vehicle parameters exactly as before.
5. Uses a plausible RTC/NTP system time for boot filenames and defers GPS UTC
   refresh until after a successful capture.
6. Starts MAVLink readiness reporting before importing the heavy collector,
   Torch, Zarr, and SDR modules.
7. Emits the GPS-time tune once when that helper is explicitly needed, rather
   than once per polling iteration.

No firmware, radio, parameter, update, GPS-time, or collection function is
removed.

## Rover 2 result

On commit `920bc6380a03ebf88b70166866aa415d676b0206`, a complete Rover 2
reboot reached the indoor GPS/EKF wait at 56.3 seconds:

| Milestone | Seconds since kernel boot |
|---|---:|
| Pluto preparation completes | 20.0 |
| Unchanged repository check completes | 22.0 |
| Parameter verification reports zero differences | 46.3 |
| GPS UTC is deferred due to plausible clock | 46.8 |
| First indoor GPS/EKF wait | 56.3 |

Compared with the approximately 231-second pre-change path, this is about 175
seconds faster (76%).

## Audible-tone verification

A RØDE VideoMic GO II (`19f7:001c`) was used as an independent check.
Recordings were made at 48 kHz, mono, signed 16-bit:

```bash
arecord -q -D hw:3,0 -f S16_LE -r 48000 -c 1 \
  -d 150 /tmp/rover2_boot.wav
```

The test captured:

- a 15-second quiet baseline;
- one deliberately requested GPS tune;
- the old `--get-time` loop, which produced repeating matching tone clusters
  until the process was terminated;
- a patched 150-second full reboot.

The patched full reboot had only bounded startup/parameter notification groups,
all ending by 45.3 seconds, and no matching GPS-tune clusters after 60 seconds.

## Profiling commands

Show the current boot milestones:

```bash
sudo journalctl -b \
  -u spf-pluto-direct-usb.service \
  -u mavlink_controller.service \
  --no-pager -o short-monotonic
```

Show the current state:

```bash
systemctl is-active \
  spf-pluto-direct-usb.service \
  mavlink_controller.service
```

Measure the systemd portion:

```bash
systemd-analyze
systemd-analyze critical-chain mavlink_controller.service
```

## Remaining bottlenecks

The largest remaining measured costs are:

- Pluto preparation: about 11 seconds for one radio and 21 seconds for two;
- full MAVLink parameter download/verification: about 18–28 seconds;
- cold lightweight collection startup: about 10 seconds on Rover 2.

Future work should preserve exact verification. The safest next architectural
optimization is to run independent Pluto preparation and vehicle-parameter
verification in parallel, then gate collection on both successful results.

## Rollback

PR #13 is isolated on `agent/boot-critical-path`. To return a test rover to
production after validation:

```bash
sudo systemctl stop mavlink_controller.service
git -C /home/pi/spf checkout main
git -C /home/pi/spf pull --ff-only
sudo systemctl restart spf-pluto-direct-usb.service
sudo systemctl start mavlink_controller.service
```

Restarting the Pluto service is required after changing SPF commits because the
ready manifest intentionally records and verifies `spf_git_sha`.
