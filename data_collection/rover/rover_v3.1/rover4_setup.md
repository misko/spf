# Rover 4 setup

Provision ROVER04 as `rover_id 4` on static `192.168.1.44`, functionally
identical to Rover 1, using Git `main` as the transport for every script.

Prepared 2026-08-03 from a read-only audit of Rover 1 (`192.168.1.41`),
Rover 4 (`192.168.1.183`), and this repository.

---

## 1. Transport model: Git `main` is the only channel

**No script is ever copied to a rover by `scp`, pasted into a shell, or edited
in place.** Every change follows one path:

```
edit on workstation  ->  commit  ->  push origin main  ->  git pull on rover4  ->  run from /home/pi/spf
```

This is already the fleet's mechanism: `update_spf_before_boot.sh` runs at boot,
does a bounded `git fetch origin main` and a `git merge --ff-only`, and refuses
to proceed if the checkout does not fast-forward. Rover 1 tracks `main` this way.

Two consequences that constrain everything below.

**1.1 Pushing to `main` deploys to the whole fleet.** Rovers 1–3 fast-forward to
`origin/main` on their next boot. Every change made for Rover 4 must therefore be
backward-safe for the existing rovers. In practice:

- widening a `rover_id` guard from `^[1-3]$` to `^[1-4]$` is a superset — safe
- adding `rover4_production_v7.yaml` is additive — safe
- adding new scripts is additive — safe
- **changing an existing rover's behaviour is not**, and must not be bundled
  into this work

**1.2 The Rover 4 checkout must stay clean.** `update_spf_before_boot.sh` aborts
on a dirty working tree or a non-fast-forward HEAD. Never `vi` a script on the
rover to "just try something" — the rover will silently stop self-updating and
drift from the fleet. Fix it on the workstation, push, pull.

**Bootstrap.** Rover 4 already has working network (eth0 DHCP `192.168.1.184`
and wifi `192.168.1.183`), so the initial `git clone` happens over that before
any network reconfiguration. The chicken-and-egg only bites if networking is
changed first — which §5 explicitly avoids.

---

## 2. Rover 4 identity

"Like Rover 1" means functionally identical with Rover-4 identity substituted.
Most of it derives from `~/rover_id`; only four items are set by hand.

| Item | Rover 1 | **Rover 4** | Set by |
|---|---|---|---|
| `~/rover_id` | 1 | **4** | hand (§6.2) |
| hostname | `roverpi1` | **`roverpi4`** | hand (§6.1) — currently `ROVER04` |
| eth0 static | 192.168.1.41 | **192.168.1.44** | hand (§5) — `setup.sh` formula `40+id` agrees |
| capture config | `rover1_production_v7.yaml` | **`rover4_production_v7.yaml`** | hand (§4.3) |
| dataset tag | `RO1` | **`RO4`** | auto — `drone_run.sh:309` `--tag "RO${rover_id}"` |
| `SYSID_THISMAV` | 1 | **4** | auto — `rover3_base_parameters.params:26`, substituted at `drone_run.sh:228` |
| routine / n / radios | `bounce` / 3000 / 2 | same | from the capture config |
| **`rest-offset-m`** | `[1.0, 1.0]` | **`[-1.0, -1.0]`** | hand (§4.3) — **see below** |
| telemetry ports | 14571 / 14581 | **14574 / 14584** | base-station Mac (`telem.sh`), not the rover |

SiK NetID is out of scope for this work.

### `rest-offset-m` is safety-relevant, not cosmetic

Each rover rests in its own quadrant so two vehicles never park on the same
point, and `tests/test_rest_offset.py::test_production_rovers_rest_apart`
enforces ≥1.99 m separation:

| Rover | offset |
|---|---|
| 1 | `[ 1.0,  1.0]` |
| 2 | `[ 1.0, -1.0]` |
| 3 | `[-1.0,  1.0]` |
| **4** | **`[-1.0, -1.0]`** ← the one free quadrant |

**A verbatim copy of `rover1_production_v7.yaml` would give Rover 4 Rover 1's
offset and park both vehicles at the same home point.** The separation test
would not catch it, because it iterates `EXPECTED_OFFSETS` rather than globbing
the config directory — so Rover 4 is invisible to it until explicitly added.
Adding `4: [-1.0, -1.0]` to `EXPECTED_OFFSETS` is what makes the offset
load-bearing.

---

## 3. Platform baseline (post-reflash)

Rover 4 was reflashed on 2026-08-03 and now matches Rover 1 on everything that
previously blocked provisioning.

| | Rover 1 | Rover 4 | |
|---|---|---|---|
| Model | Pi 4B Rev 1.5 | Pi 4B Rev 1.5 | ✅ |
| OS | Debian 12 bookworm | Debian 12 bookworm | ✅ |
| Image variant | `stage2` (Lite) | `stage2` (Lite) | ✅ |
| Python | 3.11.2 | 3.11.2 | ✅ |
| Arch | aarch64 / arm64 | aarch64 / arm64 | ✅ |
| Boot target | multi-user.target | multi-user.target | ✅ |
| Hardware | 2× Pluto + PX4 FMU | 2× Pluto + PX4 FMU | ✅ |
| Kernel | 6.1.0-rpi7 | 6.12.93+rpt | image dates 2023-12-11 vs 2026-06-18 |
| `ifupdown` | installed (`ii`) | **not installed** (`un`) | §5 |
| `/boot/config.txt` | symlink → `firmware/` | **stub, "DO NOT EDIT"** | §5 |

`NetworkManager.conf` is already byte-identical on both
(`plugins=ifupdown,keyfile`, `[ifupdown] managed=false`), so installing
`ifupdown` is sufficient for NetworkManager to release `eth0` — no NM edits.

### Two traps in `setup.sh` on this image

1. Its Pi-4 network block writes `/etc/network/interfaces`, which is inert
   without `ifupdown`.
2. Its Pi-4 wifi block writes `/boot/config.txt`, which on the 2026 image is a
   stub reading *"DO NOT EDIT THIS FILE — moved to /boot/firmware/config.txt"*.
   Wifi would remain enabled and the stub would be corrupted.

`setup.sh` also runs both board blocks ungated, disables wifi and reboots before
any static address is proven, and is not re-runnable. **It is not invoked as a
whole.** Its steps are executed individually, in the safe order below.

---

## 4. Phase A — repository work (merge before touching Rover 4)

### 4.1 Four guards reject `rover_id 4`

All must be widened or the rover cannot boot a mission.

| Location | Current | Effect if missed |
|---|---|---|
| `spf/scripts/rover_capture_config.py` `CANONICAL_CONFIGS` | `{1,2,3}` | `resolve_capture_plan(4)` raises |
| `configure_direct_usb_boot.sh:39` | `^[1-3]$` | §7.3 dies |
| **`drone_run.sh:60`** | `^[1-3]$` | **every mission launch dies** |
| **`prepare_direct_usb_boot.sh:59`** | `^[1-3]$` | boot preparation dies |

```bash
cd data_collection/rover/rover_v3.1
sed -i 's/\^\[1-3\]\$/^[1-4]$/' \
    configure_direct_usb_boot.sh drone_run.sh prepare_direct_usb_boot.sh
grep -c '\^\[1-4\]\$' configure_direct_usb_boot.sh drone_run.sh prepare_direct_usb_boot.sh
```
Expect `1` from each file.

### 4.2 Canonical map
```python
CANONICAL_CONFIGS = {1: "rover1_production_v7.yaml", ..., 4: "rover4_production_v7.yaml"}
```

### 4.3 Capture config
```bash
cd capture_configs
sed 's/^rest-offset-m: \[1\.0, 1\.0\]$/rest-offset-m: [-1.0, -1.0]/' \
    rover1_production_v7.yaml > rover4_production_v7.yaml
diff rover1_production_v7.yaml rover4_production_v7.yaml   # expect ONLY line 6
```
Rover 4 shares Rover 1's geometry (2× Pluto, 0.035 m, ports 1/2, 5.766 GHz), so
everything except the rest offset is copied. **Confirm the intended role before
merging** — if Rover 4 is an emitter, this config is wrong.

### 4.4 Extend the rover enumerations

Four places enumerate rovers. Extending them is what makes Rover 4 *covered*
rather than silently skipped:

| File | Change |
|---|---|
| `tests/test_rest_offset.py` | `EXPECTED_OFFSETS` += `4: [-1.0, -1.0]`; rename `test_the_three_production_rovers_rest_apart` → `test_production_rovers_rest_apart` |
| `tests/test_pluto_multi_firmware.py:600` | `(1, 2, 3)` → `(1, 2, 3, 4)` |
| `tests/test_rover_capture_profile.py:201` | `[1, 2, 3]` → `[1, 2, 3, 4]` |
| `tests/test_calibration_firmware_pins.py` | `(1, 2, 3)` → `(1, 2, 3, 4)` — **only on the `fix/calibration-firmware-pin-drift` branch; apply when that merges** |

### 4.5 Fleet compatibility — rovers 1–3 boot `main`

Every change above is additive or a superset. Verified against what Rovers 1–3
execute on boot:

| Change | Effect on Rovers 1–3 | |
|---|---|---|
| `CANONICAL_CONFIGS` += `4:` | each looks up its own id; the dict merely grows | ✅ additive |
| `^[1-3]$` → `^[1-4]$` (×3 scripts) | `1`/`2`/`3` still match | ✅ superset |
| new `rover4_production_v7.yaml` | never read by 1–3 | ✅ additive |
| new `configure_rover_network.sh`, `audit_rover.sh`, `compare_rovers.sh` | not in any boot path | ✅ additive |
| test enumerations | test-only, never executed on a rover | ✅ |

**Nothing modifies behaviour for Rovers 1–3.** That property is what makes this
safe to push to `main` given they fast-forward on boot, and it is the standing
constraint for adding Rover 5.

### 4.6 New automation (§9)

### 4.7 Gate
```bash
python -m pytest -q tests/test_calibration_firmware_pins.py \
  tests/test_rover_capture_profile.py tests/test_radio_missing_shutdown.py \
  tests/test_pluto_ready_manifest.py
python -c "from spf.scripts.rover_capture_config import resolve_capture_plan as r; \
  p=r(4); print(p.rover_id, p.expected_radios, p.firmware_release_tag)"
```
**PASS:** tests green; resolver prints `4 2 v0.38-...-v3`.
**FAIL → STOP.** Do not touch Rover 4.

### 4.8 Push, and reserve the address
```bash
git push origin main
```
`192.168.1.44` sits inside the LAN DHCP pool. §2 of `ROVER_RUNBOOK.md` records
observed IP squatting on `.41/.42/.43`. **Reserve `.44` on the router now.**

---

## 5. Phase B — network cutover ⚠️ highest risk

Rover 4's only irreversible step is disabling wifi. The ordering exists so that
`.44` is proven while two other paths are still up.

### 5.1 Confirm three access paths
```bash
ping -c1 192.168.1.183   # wifi
ping -c1 192.168.1.184   # eth0 DHCP
```
Plus physical console as last resort. **Fewer than two remote paths → STOP.**

### 5.2 Clone the repo (bootstrap, over existing network)
```bash
ssh pi@192.168.1.183 'cd /home/pi && git clone https://github.com/misko/spf.git && \
  git -C /home/pi/spf rev-parse --short HEAD'
```

### 5.3 Static address, wifi untouched
```bash
ssh pi@192.168.1.183 'sudo /home/pi/spf/data_collection/rover/rover_v3.1/configure_rover_network.sh 4 --stage static-only'
```

### 5.4 GATE — prove `.44` from a new session
```bash
ssh -o BatchMode=yes pi@192.168.1.44 'hostname; ip -brief addr show eth0; ip route | head -2'
```
**PASS:** answers, shows `192.168.1.44/24`, default via `192.168.1.1`.
**FAIL → `--stage rollback`. Wifi is still up. Do not continue.**

### 5.5 Disable wifi (only now)
```bash
ssh pi@192.168.1.44 'sudo /home/pi/spf/data_collection/rover/rover_v3.1/configure_rover_network.sh 4 --stage disable-wifi'
ssh pi@192.168.1.44 'sudo reboot'
sleep 45 && ssh pi@192.168.1.44 'ip -brief addr; ip link show wlan0 2>/dev/null || echo "wlan0 GONE (expected)"'
```
**PASS:** `.44` returns, `wlan0` absent.
**FAIL → physical console required.** This is the only step with no remote
fallback, which is why §5.4 precedes it.

---

## 6. Phase C — base provisioning

All over `.44`.

### 6.1 Hostname
```bash
ssh pi@192.168.1.44 'sudo hostnamectl set-hostname roverpi4 && \
  sudo sed -i "s/\bROVER04\b/roverpi4/g" /etc/hosts && hostname'
```
Fleet convention is `roverpiN` (verified: `roverpi1`, `roverpi2`, `roverpi3`).
A stale `/etc/hosts` entry makes `sudo` emit DNS warnings.

### 6.2 Identity, deps, venv
```bash
ssh pi@192.168.1.44 'echo 4 > ~/rover_id
bash /home/pi/spf/data_collection/rover/rover_v3.1/install_deps.sh
python3 -m venv ~/spf-virtualenv
~/spf-virtualenv/bin/pip install -e /home/pi/spf
~/spf-virtualenv/bin/pip install RPi.GPIO'
```

### 6.3 Shell environment (Rover 1's exact lines)
```bash
ssh pi@192.168.1.44 'grep -v spf ~/.bashrc | grep -v lsusb > /tmp/bashrc && mv /tmp/bashrc ~/.bashrc
echo export PYTHONPATH=/home/pi/spf >> ~/.bashrc
echo "test -z \"\$VIRTUAL_ENV\" && source ~/spf-virtualenv/bin/activate" >> ~/.bashrc
echo "lsusb -t | grep usb-storage | sed '"'"'s/.*Port \([0-9]*\): Dev \([0-9]*\),.*/\1 \2/g'"'"' > ~/device_mapping" >> ~/.bashrc'
```
Only the **Pi 4** `device_mapping` line, matching Rover 1. The Pi 5
`lsusb | grep PLUTO` variant is deliberately skipped.

### 6.4 GATE
```bash
ssh pi@192.168.1.44 'bash -lc "python3 -c \"import spf; print(spf.__file__)\"; cat ~/device_mapping"'
```
**PASS:** imports from `/home/pi/spf`; `device_mapping` has 2 lines
(Rover 1 reference: `1 3` / `2 4`).

---

## 7. Phase D — firmware and services

### 7.1 ArduPilot (Rover 4.5.0 fmuv3)
```bash
ssh pi@192.168.1.44 'bash /home/pi/spf/data_collection/rover/rover_v3.1/flash_ardupilot.sh'
```

### 7.2–7.3 Pluto provisioning and production boot
```bash
ssh pi@192.168.1.44 'sudo /home/pi/spf/data_collection/rover/rover_v3.1/check_and_set_pluto.sh --apply
sudo /home/pi/spf/data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh production-default'
```
**`setup.sh`'s `plutosdr-fw-v0.37-dirty` Dropbox blob is not used.** It predates
the direct-USB campaign; the pinned v3 release in the capture config is
installed by the two commands above.

### 7.4 Stock service trim
```bash
ssh pi@192.168.1.44 'sudo systemctl disable --now ModemManager.service bluetooth.service \
  rpi-eeprom-update.service e2scrub_reap.service 2>/dev/null
sudo systemctl mask NetworkManager-wait-online.service'
```

### 7.5 GATE
```bash
ssh pi@192.168.1.44 'lsusb | grep -c PLUTO; sudo head -5 /run/spf/direct_usb_ready.json'
```
**PASS:** 2 Plutos; `ready_manifest_version: 2`, 2 radios (Rover 1 reference).

---

## 8. Phase E — audit against Rover 1

```bash
./data_collection/rover/rover_v3.1/compare_rovers.sh 192.168.1.41 192.168.1.44
```
**PASS:** identical on OS, image stage, Python, git branch, Pluto count,
ready-manifest version and all four unit states. Differs only on `rover_id`,
hostname, eth0 address, kernel, and the documented deltas in §10.

Then run the existing gates before any field use:
1. `PRE_FIELD_CHECKLIST.md` — per-rover real-radio Zarr and fake-drone gates
2. `run_direct_usb_boot_preflight.sh` — 100-frame receive check
3. Compare against Rover 1's 100-frame result in
   `field_reports/2026_08_02.md` §"Rover 1 100-frame receive check"

---

## 9. Automation added for this work

Most of the deep work was already scripted and idempotent — `check_and_set_pluto.sh`,
`configure_direct_usb_boot.sh`, `reconcile_rover_boot_units.sh` (exit codes
0/10/75), `run_direct_usb_boot_preflight.sh`, `flash_ardupilot.sh`,
`install_deps.sh`, and the `check_*` audits. The gap was the thin provisioning
shell around them, which is exactly the OS-version-sensitive part.

### `configure_rover_network.sh <rover_id> --stage <stage>`
Replaces the inline network/wifi logic in `setup.sh`.

- **detects image layout** — `/boot/config.txt` symlink vs stub, writes the real
  file
- **detects `ifupdown`** — installs and uses it when matching Rover 1; falls
  back to NetworkManager otherwise
- stages: `static-only`, `disable-wifi`, `rollback`, `--verify-only`
- **refuses `disable-wifi` unless the static address already answers**
- idempotent

### `audit_rover.sh` / `compare_rovers.sh`
Emits a machine-readable rover fingerprint — OS, image stage, Python, `rover_id`,
hostname, eth0, wifi state, Pluto count, ready-manifest version, git HEAD and
branch, the four SPF unit states, the five stock services — and diffs two rovers.

This would have surfaced today's divergences (Rover 1's still-enabled
ModemManager, the `/boot/config.txt` stub) automatically.

### Deferred: `setup.sh` hardening
Board detection instead of two ungated blocks, idempotent/resumable
`--only`/`--skip`, dropping the v0.37 blob, and delegating to the two scripts
above. **Left until after Rover 4 is commissioned** — refactoring the provisioner
while commissioning hardware changes two variables at once, and `setup.sh` is not
in the path when its steps are called individually.

---

## 10. Known deltas vs Rover 1

Rover 1 was provisioned by an older `setup.sh` and does **not** match what the
script does today. Where they disagree, Rover 4 follows the current script.

| Item | Rover 1 | Rover 4 | Why |
|---|---|---|---|
| ModemManager, bluetooth, rpi-eeprom-update, e2scrub_reap | enabled | **disabled** | `setup.sh:192-196`. ModemManager AT-probes `/dev/ttyACM*` and the Pluto CDC gadget during re-enumeration — a documented suspect in "expected N Plutos, found N−1". |
| NetworkManager-wait-online | enabled | **masked** | keeps `network-online.target` off the boot critical path |
| arduino PATH in `~/.bashrc` | absent | **present** | `setup.sh:177` |
| udev `usb_device` rule | absent | **present** | `setup.sh:91-92` |
| `/boot/config.txt` | symlink | stub | shipped by the image; not fixable |
| Kernel | 6.1.0-rpi7 | 6.12.93+rpt | different image dates |

**These deltas are deliberate.** The alternative — replicating Rover 1 exactly —
would carry forward a configuration the maintained script has already moved away
from. Backporting the service trim to Rovers 1–3 is the better reconciliation and
is tracked separately.

---

## 11. Abort conditions

- §4.6 tests fail → repo change wrong, do not touch the rover
- §5.4 `.44` unreachable → `--stage rollback`, wifi still up
- §6.4 `import spf` fails → dependency problem, rover still fully recoverable
- §5.5 no return after reboot → physical console, revert
  `/boot/firmware/config.txt`
- §7.5 fewer than 2 Plutos, or manifest ≠ v2 → enumeration/hardware issue; do
  not proceed to field use

## 12. Estimated time

Phase A ~20 min plus review · B ~15 min · C ~20 min (pip is slow on a Pi) ·
D ~15 min · E ~10 min. **≈1.5 h**, excluding PR review and the DHCP reservation.
