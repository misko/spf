# Rover 3.1 pre-field acceptance checklist

Use this checklist before taking Rover 1, 2, or 3 into the field. It is the
short, pass/fail companion to [ROVER_RUNBOOK.md](./ROVER_RUNBOOK.md), which
remains the detailed operational and recovery reference.

Do not substitute one gate for another:

- the fake-radio pytest checks the Python collection pipeline;
- SITL checks ArduPilot/MAVLink motion and file completion;
- a real-radio `--fake-drone` capture checks each physical Rover and Pluto
  without moving the vehicle;
- a restrained physical dry-run checks the assembled vehicle.

Every box in the release-level section must pass for the exact SPF commit being
deployed. Every box in the per-Rover section must pass independently on each
physical Rover.

## Acceptance record

Create one copy of this table in the field log for each test campaign.

| Item | Value |
|---|---|
| Test date / operator | |
| SPF commit (`git rev-parse HEAD`) | |
| Rover | `1` / `2` / `3` |
| Pi model and serial | |
| ArduPilot version | |
| Pluto transport | `iio` / `direct_usb_v2` |
| Pluto serial(s) and USB path(s) | |
| Pluto installed QSPI version(s) | |
| RAM image SHA-256, if used | |
| Capture config | |
| Zarr path | |
| Frames stored per receiver | |
| Median frame rate per receiver | |
| SITL report for this SPF commit | |
| Overall result | `PASS` / `FAIL` |

A failed or unrecorded required item means the Rover is not field-ready.

## 1. Choose and record one Pluto firmware path

Do not mix the two paths within a capture.

| | Legacy production path | Gain/RSSI direct-USB path |
|---|---|---|
| Pluto boot image | Installed QSPI `v0.37-dirty` | Experimental v0.38-based image, RAM boot only |
| IQ transport | Standard USB-IIO | Vendor interface 6, direct USB bulk |
| Radio configuration | Standard USB-IIO / pyadi | Standard USB-IIO / pyadi before direct streaming |
| Gain and RSSI | Host reads after IQ | Pluto-local observations in the same transfer as IQ |
| YAML | Recovery/manual only | Canonical `rover<id>_production_v7.yaml`; V7 implies direct USB/V2 |
| Dataset | v4 | v7: v4 fields plus complete protocol-v2 metadata |
| Current qualification | Production path | Rover 1: simultaneous two-Pluto bench/boot qualification |
| Rollback | Already running QSPI | Reset/power-cycle restores unchanged QSPI |

### Canonical production path

Use the committed production configuration selected by `drone_run.sh`:

| Rover | Required real-radio configuration | Receivers |
|---|---|---:|
| 1 | `capture_configs/rover1_production_v7.yaml` | 2 |
| 2 | `capture_configs/rover2_production_v7.yaml` | 1 |
| 3 | `capture_configs/rover3_production_v7.yaml` | 2 |

The production RF/frame contract is 5.766 GHz LO, 30 MS/s, 3 MHz RF
bandwidth, 524,288 complex samples per channel per frame, two RX channels per
Pluto, four kernel RX buffers, slow-attack gain control, and a 0.5 second
snapshot interval.

Pass:

- every installed Pluto reports the expected installed firmware;
- standard USB-IIO enumerates and captures both RX channels;
- the production config has the correct physical antenna spacing;
- the custom direct-USB interface is not required.

Fail:

- a Rover's physical spacing and YAML disagree;
- a radio is missing, maps to the wrong receiver port, or has only one RX
  channel;
- the intended legacy capture accidentally selects `direct_usb`.

### Gain/RSSI direct-USB path

Follow [the direct-USB firmware and capture runbook](../../../docs/direct_usb_gain_benchmark.md).
Use the committed checksum-pinned, serial/path-aware loader. For Rover 1 or 3,
which each require two Plutos:

```bash
cd /home/pi/spf
data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh download
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh \
  check-config-all 2
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh load-all 2
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh verify-all 2
```

It obtains the exact hardware-tested image from:

```text
https://github.com/misko/plutosdr-fw/releases/tag/v0.38-plutoplus-spf-gain-rssi-v2

f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4
```

The multi-radio loader keeps both Plutos attached. It identifies each by USB
serial and physical path, moves each duplicate `192.168.2.1` USB-network
interface into a temporary network namespace, and targets DFU with the exact
physical USB path. It writes only to RAM. Run `verify-all` again immediately
before collection; loading is required after every Pluto reset or full power
cycle, not before every frame or capture.

Rover 1 uses
`capture_configs/rover1_receiver_config_pi_3mhz_35mm_direct_usb_v2.yaml`.
It runs both radios simultaneously, negotiates protocol v2, writes data
version 7, preserves the legacy `signal_matrix`, `gains`, and `rssis` fields,
and also stores complete start/end gain/RSSI and stream metadata.

Pass:

- the image was RAM-booted, not written to QSPI;
- standard USB-IIO and vendor interface 6 both enumerate;
- `iiod` and `sdr_usb_gadget` both run on the Pluto;
- the normal simultaneous 100-frame-per-radio Zarr capture and v7 validator
  pass;
- gain and RSSI are finite for both channels;
- every receiver stores a unique Pluto serial and physical USB path;
- a reset restores the original QSPI firmware and removes interface 6.

Fail:

- magic, version, header size, CRC, transfer length, sequence, gain, or RSSI
  validation fails;
- either IQ channel is zero or non-finite;
- the host performs per-frame IIO gain/RSSI reads;
- QSPI was flashed before RAM-boot and rollback acceptance;
- equal gain endpoints are described as proof of no in-frame transition.

Rover 1 passed the simultaneous two-Pluto, 100-frame-per-receiver bench test,
automatic reboot preflight, stock-QSPI rollback, and RAM-image restoration on
2026-07-26. This is hardware evidence for Rover 1 only. Rover 2 and Rover 3
still require their own committed physical configurations and per-Rover
qualification before using direct USB in the field.

### Rover 1 boot qualification and production modes

The normal production unit now always requires the RAM-firmware preparation
unit first. Migrate an existing installation without changing its capture
profile with:

```bash
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh \
  production-default
```

The boot qualification workflow is deliberately separate from the
motion-capable `mavlink_controller.service`. Enabling it installs two units:

1. `spf-pluto-direct-usb.service` verifies the persistent AD9361/2r2t
   configuration and four dual-RX DMA scan elements, checksum-verifies and
   RAM-loads every attached/configured Pluto,
   regenerates `~/device_mapping` after final USB enumeration, and writes
   `/run/spf/direct_usb_ready.json`.
2. `spf-direct-usb-preflight.service` requires that ready stamp, runs exactly
   100 motion-free fake-drone frames per receiver, reopens the final v7 Zarr,
   validates it, and writes `PASS` plus `validation.json`.

Enable the mode without starting either capture immediately:

```bash
cd /home/pi/spf
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh qualify
sudoedit /etc/spf/direct_usb_boot.env
sudo reboot
```

The qualify command stops and disables `mavlink_controller.service`; this is a
safety requirement, because that legacy unit starts the production mission
loop. After reboot:

```bash
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh status
systemctl status spf-pluto-direct-usb.service \
  spf-direct-usb-preflight.service --no-pager
python3 -m json.tool /run/spf/direct_usb_ready.json
find /home/pi/preflight/boot_direct_usb -maxdepth 2 \
  \( -name PASS -o -name validation.json \) -print
```

A Pi-only reboot may leave USB power applied to the Plutos. The loader still
reloads the exact checksum-pinned image because a vendor interface alone does
not prove its build. The pre-load gate checks persistent `ad9361`/`2r2t`
U-Boot values but deliberately does not interpret the active RAM
`/opt/VERSIONS` value as QSPI identity. The stock-QSPI version allowlist remains
mandatory for the separate, explicit operation that writes persistent U-Boot
configuration. A full Rover power cycle or a tested `rollback-all` returns the
Plutos to QSPI.

To prove rollback without enabling motion:

```bash
sudo systemctl stop spf-direct-usb-preflight.service \
  spf-pluto-direct-usb.service
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh \
  rollback-all 2
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh status-all 2
iio_info -s
```

Pass only when both radios report `direct_usb=false`, standard USB-IIO is
present, the vendor interface/gadget process is absent, and `/opt/VERSIONS`
matches the recorded QSPI version. Restore qualification mode with
`systemctl start spf-pluto-direct-usb.service` followed by
`systemctl start spf-direct-usb-preflight.service`.

To opt out of the default RAM image and return to the stock-QSPI legacy IIO
workflow, first run `rollback-all`, then:

```bash
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh \
  restore-legacy
sudo reboot
```

`restore-legacy` sets `SPF_DIRECT_USB_DISABLE=1`, enables but deliberately does
not immediately start the motion-capable service, and retains the base
firmware-before-MAVLink ordering. The prerequisite service exits without a RAM
load in this explicit rollback mode.

After qualification, restore production through direct USB without starting
it immediately:

```bash
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh \
  production-v7
sudoedit /etc/spf/rover_collection.env
```

For the first reboot set `SPF_SKIP_SELF_UPDATE=1` and
`SPF_BOOT_VALIDATE_ONLY=1`. Pass only when:

- the loader completes before `mavlink_controller.service`;
- every serial/path and verified firmware identity appears in
  `/run/spf/direct_usb_ready.json`;
- the old `spf-direct-usb-preflight.service` is disabled and creates no Zarr;
- a real heartbeat reports MANUAL and `armed=false`; and
- the launcher exits before parameter writes, collection, planner, arm, or
  motion.

When the Rover is physically safe to move and the normal MANUAL→GUIDED operator
procedure is ready, set `SPF_BOOT_VALIDATE_ONLY=0`. The canonical V7 config
writes full endpoint metadata and retains each Rover's established routine and
record count.

## 2. Release-level software gates

Run these once for the exact SPF commit that will be installed on all Rovers.
Record the commit and test output. Re-run after any code, configuration, Docker
image, or ArduPilot parameter change.

### 2.1 Fake radios plus fake drone

This is fast and needs neither Docker nor hardware:

```bash
cd /home/pi/spf
source /home/pi/spf-virtualenv/bin/activate
export PYTHONBREAKPOINT=0
python3 -m pytest tests/test_mavlink_radio_collect.py -v
```

Pass:

- all tests pass;
- the fake collector produces a final `.zarr`, `.yaml`, and `.log`;
- v4 and v6 compatibility/schema guards pass.

Fail:

- any test fails, times out, or leaves only a partial file unexpectedly.

### 2.2 ArduPilot SITL

SITL validates MAVLink, parameters, mode transitions, simulated motion, and
final-file promotion. It uses fake radios and does not validate any physical
Rover or Pluto.

```bash
docker pull csmisko/ardupilotspf:latest
python3 -m pytest tests/test_in_simulator.py -v -s
```

Pass:

- all seven SITL tests pass;
- MANUAL remains stationary and leaves temporary files;
- GUIDED produces simulated movement and final `.zarr/.yaml/.log` files;
- the final receiver data checked by the test is finite.

Fail:

- Docker/image setup fails;
- MANUAL moves;
- GUIDED does not move or does not finish a recording;
- parameter load/diff, reboot, clock, or buzzer tests fail.

Because SITL contains no physical Rover, one passing report may cover all
Rovers only when they deploy the exact recorded SPF commit. If SITL is also run
on each Pi to test its local Docker/Python installation, record those results
separately; they still do not replace the real-radio gate.

## 3. Per-Rover static bench inspection

Put the Rover on blocks so the wheels cannot drive it off the bench. Stop the
automatic mission service before testing:

```bash
sudo systemctl stop mavlink_controller.service
cd /home/pi/spf
source /home/pi/spf-virtualenv/bin/activate
export PYTHONBREAKPOINT=0

git rev-parse HEAD
cat ~/rover_id
cat /proc/device-tree/model
lsusb
lsusb -t
cat ~/device_mapping
iio_info -s
df -h /
vcgencmd measure_temp
```

Also verify:

- [ ] Rover ID, static IP, SiK NetID, and ArduPilot `SYSID_THISMAV` are unique.
- [ ] Rover 1/3 show two Plutos; Rover 2 shows one.
- [ ] USB physical ports agree with `device_mapping` and the receiver-port
      assignments.
- [ ] Both channels and antenna cables are connected to every receiver.
- [ ] YAML antenna spacing equals the measured physical spacing.
- [ ] Pi Wi-Fi is disabled.
- [ ] Battery is charged and the low-voltage disconnect is set correctly.
- [ ] GPS cable is routed away from the Pi and SDRs.
- [ ] There is enough free storage for the test and field mission.
- [ ] System time is sane and the CPU governor can enter `performance`.
- [ ] No unexpected reboot, USB error, undervoltage, or thermal warning is
      present in `dmesg`.

Run the normal read-only Pluto environment check and regenerate the mapping:

```bash
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh \
  check-config-all 2
bash data_collection/rover/rover_v3.1/device_mapping.sh > ~/device_mapping
cat ~/device_mapping
```

If this fails, do not repair a radio during collection boot. In a controlled
provisioning session, inspect and then explicitly apply the serial-aware plan:

```bash
sudo data_collection/rover/rover_v3.1/check_and_set_pluto.sh --dry-run
sudo data_collection/rover/rover_v3.1/check_and_set_pluto.sh --apply
```

Pass only when the expected number of radios is present after the check and
the receiver-port association regenerates correctly after one unplug/replug or
full Rover power cycle. USB bus/device numbers may change; serial number and
physical port assignment must not.

## 4. Per-Rover real-radio capture with a fake drone

This is the required no-motion hardware acceptance. It uses each Rover's real
Pluto(s), normal collector, real YAML, and LMDB-backed Zarr writer while
replacing only the vehicle/GPS controller with `Drone(fake=True)`.

Select the production configuration:

```bash
case "$(cat ~/rover_id)" in
  1) CONFIG=data_collection/rover/rover_v3.1/capture_configs/rover_receiver_config_pi_3mhz_35mm.yaml; EXPECTED_RX=2 ;;
  2) CONFIG=data_collection/rover/rover_v3.1/capture_configs/rover_single_receiver_config_pi_3mhz.yaml; EXPECTED_RX=1 ;;
  3) CONFIG=data_collection/rover/rover_v3.1/capture_configs/rover_receiver_config_pi_3mhz_43mm.yaml; EXPECTED_RX=2 ;;
  *) echo "Invalid rover_id"; exit 1 ;;
esac
```

Capture 100 stored frames per receiver:

```bash
OUT=/home/pi/preflight/$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

python3 spf/mavlink_radio_collection.py \
  --fake-drone \
  --no-ultrasonic \
  --yaml-config "$CONFIG" \
  --device-mapping ~/device_mapping \
  --routine center \
  --records-per-receiver 100 \
  --temp "$OUT" \
  --tag "PREFLIGHT_RO$(cat ~/rover_id)"
```

The production configs exercise all installed radios simultaneously. Do not
replace them with `tests/test_config.yaml`; that file uses fake radios and a
small 4,096-sample frame.

Validate every receiver and print its measured cadence:

```bash
ZARR=$(find "$OUT" -maxdepth 1 -name '*.zarr' -print -quit)
python3 - "$ZARR" 100 "$EXPECTED_RX" <<'PY'
import sys

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

path, expected_frames, expected_receivers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
z = zarr_open_from_lmdb_store(path)
try:
    assert z.attrs["sdr_identity_version"] == 1
    receiver_names = sorted(z.receivers.keys())
    assert len(receiver_names) == expected_receivers, receiver_names
    receiver_serials = []
    receiver_paths = []
    for name in receiver_names:
        rx = z.receivers[name]
        assert rx.attrs["sdr_family"] == "pluto", dict(rx.attrs)
        serial = rx.attrs["sdr_serial"]
        usb_path = tuple(rx.attrs["usb_port_path"])
        assert serial, f"{name}: missing Pluto serial"
        assert usb_path, f"{name}: missing Pluto USB path"
        receiver_serials.append(serial)
        receiver_paths.append(usb_path)
        signal = rx.signal_matrix
        assert signal.shape == (expected_frames, 2, 524288), signal.shape
        assert signal.dtype == np.dtype("complex64"), signal.dtype
        assert rx.gains.shape == (expected_frames, 2), rx.gains.shape
        assert rx.rssis.shape == (expected_frames, 2), rx.rssis.shape
        assert np.isfinite(rx.gains[:]).all(), f"{name}: invalid gain"
        assert np.isfinite(rx.rssis[:]).all(), f"{name}: invalid RSSI"
        for frame_index in range(expected_frames):
            frame = signal[frame_index]
            assert np.isfinite(frame).all(), f"{name}: frame {frame_index} non-finite"
            assert np.any(frame[0]), f"{name}: frame {frame_index} RX1 all-zero"
            assert np.any(frame[1]), f"{name}: frame {frame_index} RX2 all-zero"
        intervals = np.diff(rx.system_timestamp[:])
        assert np.all(intervals > 0), f"{name}: timestamps not increasing"
        median_hz = float(1.0 / np.median(intervals))
        p99_seconds = float(np.percentile(intervals, 99))
        print(f"{name}: frames={expected_frames} median_hz={median_hz:.3f} "
              f"interval_p99_s={p99_seconds:.3f} "
              f"serial={serial} usb_path={usb_path}")
        assert median_hz >= 1.8, f"{name}: capture too slow"
        assert p99_seconds <= 1.0, f"{name}: excessive frame stalls"
    assert len(receiver_serials) == len(set(receiver_serials)), receiver_serials
    assert len(receiver_paths) == len(set(receiver_paths)), receiver_paths
finally:
    z.store.close()
print("PASS")
PY
```

Pass:

- the process exits zero;
- the output is final `.zarr/.yaml/.log`, not `.tmp`;
- every configured receiver stores exactly 100 frames;
- every receiver has `complex64[100,2,524288]` IQ;
- every frame has finite, nonzero RX1 and RX2 samples;
- gain and RSSI arrays are finite and shaped `[100,2]`;
- every receiver records a unique Pluto serial and physical USB path;
- receiver timestamps are strictly increasing;
- setup completes without a radio-configuration assertion, and the resolved
  YAML records the intended LO, sample rate, bandwidth, FIR, gain mode, and
  frame size;
- median cadence is at least 1.8 frames/s and interval p99 is at most 1.0 s;
- logs contain no radio retry exhaustion, overflow, USB error, or writer
  failure.

The measured one-radio reference on this codebase is approximately 2.06
frames/s for USB-IIO and 2.02 frames/s for direct USB v2. The capture is paced
at roughly 2 frames/s, so a much larger reported number is not expected.
Investigate a result below the pass threshold rather than silently lowering
the threshold.

For Rover 1 direct-USB v2/data-v7 qualification, validate with:

```bash
python3 -m spf.scripts.validate_direct_usb_v7_zarr \
  /path/to/final_capture.zarr \
  --expected-frames 100 \
  --expected-receivers 2
```

This validator checks both receiver groups, IQ shape/content, metadata
validity and unsafe flags, legacy gain/RSSI compatibility, endpoint flags,
stream sequences, protocol version, and unique serial/physical-path identity.
It does not enforce cadence; apply the median and p99 gates above separately.

The Rover 1 direct captures on 2026-07-26 sustained median rates of
1.93–1.99 frames/s. Each 100-frame run also contained one post-start,
synchronized 1.2–2.0 second host-side stall, plausibly when the 2 GB Pi began
background writeback of the 294–360 MB LMDB file. The strict p99 ≤1.0 second
field gate therefore remains open even though the transport/data validator
passes. Do not hide that distinction or lower the field threshold without a
separate performance decision.

## 5. Per-Rover controls and restrained physical dry-run

Keep the vehicle on blocks for the initial control checks.

- [ ] MANUAL / RTL / GUIDED switch positions are confirmed as
      **Manual / RTL / Guided**, in that order.
- [ ] Arm/disarm operates on CH5.
- [ ] CH7 reboot behavior is understood and cannot be triggered accidentally.
- [ ] CH9 cleanly shuts down the Pi.
- [ ] CH10 starts compass calibration only when intended.
- [ ] CH12 enables/disables the ultrasonic stop as intended.
- [ ] Left/right motor direction and neutral PWM are correct.
- [ ] GPS has a real 3D fix and the EKF has absolute-position convergence.
- [ ] Compass calibration is current and heading agrees with the vehicle.
- [ ] Ultrasonic sensing stops GUIDED motion at the configured distance.
- [ ] Ground-station telemetry works on the Rover's assigned SiK NetID.

Then perform a short, supervised, low-speed outdoor run inside the intended
convex boundary. Use a small record count and keep an operator ready to
disarm. Pass only if GUIDED motion, recording, return/home behavior, and final
file promotion all complete without manual file repair.

Do not use `debug_drone_run.sh` as proof of production readiness: it currently
contains stale config/path selections and uses a fake drone. Invoke
`mavlink_radio_collection.py` with the intended production config explicitly.

## 6. Inspect, archive, and sign off

For every Rover, archive:

- the resolved YAML sidecar generated by the collector;
- the complete log;
- the final Zarr;
- the acceptance table from the top of this document;
- `git rev-parse HEAD`;
- `~/device_mapping`;
- Pluto serial, USB path, firmware version, and RAM-image hash if applicable;
- frame count, median frame rate, p99 interval, and any USB/overflow count;
- SITL output for the exact deployed SPF commit.

Before leaving:

- [ ] All required release-level gates are green.
- [ ] Every physical Rover has its own passing 100-frame real-radio Zarr.
- [ ] Every physical radio and both RX channels were exercised.
- [ ] Multi-radio Rovers passed with their radios running simultaneously.
- [ ] The intended firmware path is written in the field log.
- [ ] New-firmware Rovers passed RAM boot, metadata validation, and rollback.
- [ ] Batteries, spare storage, cables, stock firmware, DFU recovery image,
      tools, and the rollback instructions are packed.
- [ ] Exactly one boot workflow is enabled after bench work. Select canonical
      V7 production or the stock-QSPI recovery state without starting a mission
      immediately:

```bash
# New firmware, full V7 endpoint metadata:
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh \
  production-v7
# Stock firmware recovery; automatic production remains disabled:
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh \
  restore-legacy
# Start now only when the rover is safe to move; otherwise reboot in the field.
sudo systemctl start mavlink_controller.service
```

Final pass means the exact software, configuration, radios, storage path,
MAVLink behavior, and assembled Rover intended for the field were tested.
