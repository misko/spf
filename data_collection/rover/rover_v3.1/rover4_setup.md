# Rover 4 setup

**STATUS: COMPLETE (2026-08-03).** Provision ROVER04 as `rover_id 4` on static `192.168.1.44`, functionally
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
| Taranis RxNum | `01` | **`04`** | hand (§14.2) — bench, **outstanding** |
| R9 SX pin 6 | `SBUS` | **`SBUS`** | hand (§14.5) — **not set by binding**; ✅ confirmed 2026-08-04 |
| SiK NetID | `25` | **`46`** | hand (§14.9) — bench, **outstanding** |

The last two rows were out of scope for Phases A–E and are still unset: Rover 4
has no RC link and no telemetry pairing, so **it cannot yet be driven**. See
[§14](#14-phase-f--rc-link-and-sik-telemetry-outstanding-as-of-2026-08-04).

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
any static address is proven, and is not re-runnable.

**`setup.sh` is DEPRECATED as of 2026-08-03 and refuses to run** (echo + `exit 1`
at the top). It is kept only as a historical reference for its firmware and
parameter URLs. Use `provision_rover.sh`.

### Quick start — `provision_rover.sh`

Bootstrap (two commands that cannot be scripted from inside the repo):

```bash
sudo apt-get update && sudo apt-get install -y git
cd /home/pi && git clone https://github.com/misko/spf.git
```

Then, on the rover:

```bash
cd /home/pi/spf/data_collection/rover/rover_v3.1
sudo ./provision_rover.sh 4 --stage all        # identity + network + base
# --- from ANOTHER machine, prove the address (wifi is still up) ---
ssh pi@192.168.1.44 'hostname; ip -brief addr show eth0'
sudo ./provision_rover.sh 4 --stage wifi-off && sudo reboot
sudo ./provision_rover.sh 4 --stage firmware  && sudo reboot
./provision_rover.sh 4 --stage audit
./compare_rovers.sh 192.168.1.41 192.168.1.44
```

Every stage is idempotent, so a re-run after a failure or reboot is safe.
`--stage all` deliberately stops before `wifi-off` (needs the human gate above)
and `firmware` (needs a reboot first). The detailed per-step rationale for each
stage is in §5–§8 below.

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

**`git` is not on the 2026 Lite image and must be installed first.** `setup.sh`
has the same gap in reverse order — line 15 clones the repo, line 17 runs
`install_deps.sh` which is what installs `git`. On an image that shipped `git`
this never surfaced.

```bash
ssh pi@192.168.1.183 'sudo apt-get update -qq && sudo apt-get install -y -qq git'
ssh pi@192.168.1.183 'cd /home/pi && git clone https://github.com/misko/spf.git && \
  git -C /home/pi/spf rev-parse --short HEAD'
```

### 5.3 Static address, wifi untouched
```bash
ssh pi@192.168.1.183 'sudo /home/pi/spf/data_collection/rover/rover_v3.1/configure_rover_network.sh 4 --stage static-only'
```

**DNS.** `static-only` also writes a plain `/etc/resolv.conf`
(`8.8.8.8` + `1.1.1.1`), matching Rover 1. This is required, not cosmetic: once
`eth0` moves to ifupdown, NetworkManager has no DNS to publish and leaves an
empty `# Generated by NetworkManager` stub, after which `apt` and `git` fail
with *"Temporary failure resolving deb.debian.org"*. The `dns-nameservers` line
in `/etc/network/interfaces` does **not** cover this — it only takes effect via
the `resolvconf` package, which is not installed on Rover 1 either.

If a rover is ever left in that state, DNS must be restored by hand before it
can `git pull` the fix — a genuine bootstrap deadlock, and the one sanctioned
exception to §1's "never edit on the rover" rule:

```bash
sudo rm -f /etc/resolv.conf
printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf
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

### `provision_rover.sh <rover_id> --stage <stage>`
The end-to-end provisioner, written after Rover 4 was commissioned by hand so
that it encodes a sequence that is known to work rather than one that is merely
plausible.

Stages: `identity`, `network`, `wifi-off`, `base`, `firmware`, `audit`, and
`all`. The split is not cosmetic — `wifi-off` and `firmware` each end in a
reboot, and `wifi-off` is gated on a human proving the static address first.
It **delegates** to the existing, separately-tested scripts
(`configure_rover_network.sh`, `install_deps.sh`, `flash_ardupilot.sh`,
`check_and_set_pluto.sh`, `configure_direct_usb_boot.sh`, `device_mapping.sh`,
`audit_rover.sh`) rather than reimplementing them, and covers the eight steps
that were ad-hoc during Rover 4: git bootstrap check, `~/rover_id`, venv,
editable install, `RPi.GPIO`, `~/.bashrc`, the udev rule, and `device_mapping`.

`device_mapping` matters more than it looks: `~/.bashrc` only regenerates it on
an **interactive** login, so a provisioning run over ssh would otherwise leave
it missing, and `prepare_direct_usb_boot.sh` fails without it.

`setup.sh` is deprecated and refuses to run. `tests/test_provision_rover.py`
asserts that its `exit 1` precedes every executable side effect, that every
documented stage is dispatched, that `network` never disables wifi, and that
`--stage all` stops before `wifi-off` and `firmware`.

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

## 12. Execution log — 2026-08-03 (COMPLETE)

Rover 4 was provisioned end to end on 2026-08-03. Rover 1 was read but never
modified. Final audit: **31 fleet fields compared, 2 differ**, both benign.

| | Rover 1 | Rover 4 |
|---|---|---|
| `rover_id` / hostname | 1 / `roverpi1` | **4 / `roverpi4`** |
| eth0 | 192.168.1.41/24 | **192.168.1.44/24** |
| capture config | `rover1_production_v7.yaml` | **`rover4_production_v7.yaml`** |
| rest offset / tag | `[1.0, 1.0]` / RO1 | **`[-1.0, -1.0]` / RO4** |
| OS / image / Python | bookworm / stage2 / 3.11.2 | identical |
| Plutos / FMU | 2 / `ardupilot-fmuv3` | identical |
| ready manifest | v2, 2 radios | identical |
| units (4) | enabled ×3, preflight disabled | identical |
| stock services | all enabled | identical |

Remaining differences: `git_head` (Rover 1 is simply behind until its next
boot — `spf-rover-update` self-heals it) and `boot_config_layout`
(symlink vs split, a 2023-vs-2026 image artifact, not correctable without
reflashing Rover 1).

**The stock-service trim in §7.4 was deliberately NOT applied.** Rover 1 has
those services enabled, and the goal was parity with Rover 1. Trimming them is
a fleet-wide decision to make once, for all four rovers, not a divergence to
introduce on the newest one. §10 stands as the rationale for doing it later.

### Defects found and fixed during execution

Provisioning Rover 4 surfaced four real bugs, three of which affect the whole
fleet. All are fixed on `main`.

| Defect | Scope | Fix |
|---|---|---|
| **Three stale resolver field-count checks.** `d79fa576` added a 16th field (`device-fw`) and updated only one of four call sites. `configure_direct_usb_boot.sh`, `run_direct_usb_boot_preflight.sh` (the 100-frame preflight in `PRE_FIELD_CHECKLIST`) and `run_direct_usb_restart_soak.sh` all still asserted 15 and were **failing on every rover** | **Rovers 1–4** | `29cf15e` — `-ge 16`, so a future field cannot break them again |
| `flash_ardupilot.sh` ran plain `python`, which has no pyserial on a Lite image → `ModuleNotFoundError: No module named 'serial'` | Rovers 1–4 | `1fedadb` — prefer the venv interpreter |
| Moving eth0 to ifupdown leaves NetworkManager publishing no DNS → empty `resolv.conf`, apt and git stop resolving | new provisioning | `4b8c753` — write a static resolv.conf, matching Rover 1 |
| `git` absent from the 2026 Lite image, and `setup.sh` clones before `install_deps.sh` installs it | new provisioning | documented in §5.2 |

Two bugs were also caught in the new scripts *before* they could do harm, by
validating them read-only against Rover 1: `command -v ifup` reports absent
because `/sbin` is not in the `pi` user's ssh PATH, and a `q()`-wrapped
conditional always returned 0 (`q()` ends in `|| true`), which misreported an
unflashed FMU as flashed.

### Deviations from the plan as written

- **Pluto provisioning (§7.2) partially refused, correctly.** One radio was
  already provisioned; the other runs the direct-USB firmware persistently, and
  `check_and_set_pluto.sh` refuses to write U-Boot while that image is active.
  Provisioning is meant to happen on stock firmware *before* direct-USB is
  installed. `configure_direct_usb_boot.sh production-default` then succeeded
  and the boot chain came up clean, so no further action was taken. **If that
  radio ever needs re-provisioning it must first be reverted to stock.**
- Wifi was disabled before base provisioning rather than after, so that the
  static address was proven to survive a reboot before ~20 minutes of `pip`
  work was invested.

## 13. Estimated time

Phase A ~20 min plus review · B ~15 min · C ~20 min (pip is slow on a Pi) ·
D ~15 min · E ~10 min. **≈1.5 h**, excluding PR review and the DHCP reservation.

---

## 14. Phase F — RC link and SiK telemetry (outstanding as of 2026-08-04)

Phases A–E left Rover 4 with a provisioned Pi, flashed FC, working Plutos and a
capture config, but **no RC link and no SiK NetID**. It cannot be driven. This
phase closes both. It is a bench operation — every step needs the rover powered
and in front of you.

Fleet context and the meaning of each transmitter setting are in
[`ROVER_RUNBOOK.md`](./ROVER_RUNBOOK.md) §3.5.1. Rover 4's assignments:

| Item | Rover 4 | Rationale |
|---|---|---|
| RxNum | **`04`** | free; `00`/`01`/`05` are taken by rovers 3/1/2 |
| Receiver slot | **1** | standardize new binds on slot 1 |
| SiK NetID | **`46`** | continues the 25/32/39 step-of-7 pattern |

### 14.0 Prerequisites

- An **R9 SX** receiver physically installed, its `CH6/SBUS OUT` port wired to
  the FC's RCIN, and **pin 6 switched to SBUS** (§14.5 — binding does not do
  this).
- A **flight battery connected**, not just USB. The receiver is powered from the
  servo/RCIN rail; a USB-only rover has a dead receiver and an empty RC stream.
- The **R9M ACCESS** module in the transmitter's external bay.
- Every other rover **powered off** for §14.1–14.6. This is not optional: it is
  what makes a mis-set RxNum fail loudly (nothing responds) instead of silently
  driving the wrong vehicle.

### 14.1 Create the model — copy, do not build from blank

Model Select → copy an existing rover model → rename to `Rover 4`.

Copying carries the CH1–12 map, mixes and switch assignments across intact. The
map is safety-critical (arm on CH5, mode on CH8, shutdown on CH9) and retyping it
by hand is the likeliest way to produce a rover that arms on the wrong switch.

### 14.2 Set RxNum before binding anything

In the new model: `External RF: R9M ACCESS`, `Ch Range CH1-16`, **`RxNum 04`**.

**Clear the receiver slots the copy inherited.** A copied model carries the
source model's bind, so until they are cleared Rover 4's model is still addressing
*Rover 1's* receiver.

RxNum reaches a receiver **only at bind time** — it is written into the receiver
and stored there. Editing it later does not push over the air; it just breaks the
link until you re-bind. Hence: set it now, bind after.

### 14.3 Register the receiver (skip if already registered)

ACCESS splits pairing into registration and bind. Registration is one-time per
receiver and transmitter-wide; it persists across later re-binds.

Power the R9 SX with its F/S button held to enter registration mode, then
External RF → **Register** on the transmitter. Note that the button is for
*registration only* — the bind in §14.4 needs the receiver powered normally. The
button-hold procedure in `README.md` is the older ACCST/X8R one and does not
apply here.

### 14.4 Bind into slot 1

Power the receiver normally. Receiver 1 → **Bind**, select the R9 SX. The slot
should then show the receiver name instead of `[Bnd]`.

### 14.5 Switch the receiver's pin 6 to SBUS — **binding does not do this**

**This is the step that cost a bench session on 2026-08-04**, and switching it
is what finally made Rover 4 respond to the radio. A registered, bound
R9 SX with its SBUS lead correctly in the flight controller's RCIN emits
*nothing the FC can decode*, because the pin is still a PWM channel.

The R9 SX ships with all six pins as **PWM channels CH1–CH6**. The port
silkscreened `CH6/SBUS OUT` outputs **PWM channel 6** until you change it. Per
the [R9 SX manual](https://www.frsky-rc.com/wp-content/uploads/Downloads/Manual/R9%20SX/R9%20SX-Manual.pdf):
"6 standard servo connectors (default PWM channel)", "Switchable CH5 / CH6 into
S.Port / SBUS Output channels".

Symptom when this is wrong — the bind looks perfect and the vehicle is deaf:

```
rover ardupilot rc
  the flight controller reports 0 channels (rssi 255) - no receiver input
```

**Where the setting lives.** Not the module options — those are the R9M's own
(RF power, telemetry) and show only a power setting. Cursor onto the **receiver
line itself**, the row showing `R9SX1` under `Receiver <n>`, press ENTER there,
and choose **Options** from that popup. The screen is `REC OPTIONS R9SX`:

| Pin | Default | Wanted |
|---|---|---|
| Pin1–Pin4 | `CH1`–`CH4` | unchanged |
| Pin5 | `CH5` | `S.PORT` (telemetry) |
| **Pin6** | `CH6` | **`SBUS`** |

The transmitter reads these **over the air from the receiver**, so the receiver
must be powered and linked before the screen will populate. On a bench that means
a real flight battery — the servo/RCIN rail is not powered by the Pi's USB, so a
USB-only rover has an unpowered receiver and this menu will not load.

Rover 2 is the known-good reference; its pin screen is what Rover 4's should
match.

### 14.6 GATE — model match, both directions

1. With only Rover 4 powered, move the sticks. **PASS:** its servos respond.
2. Select **Rover 1's** model. **PASS:** Rover 4 does *not* respond.

Step 2 is the one that catches a duplicate or unchanged RxNum, and it is the one
that gets skipped. Do not proceed past a failure here.

Confirm what the flight controller actually receives, rather than inferring it
from servo movement:

```bash
sudo systemctl stop mavlink_controller.service && rover ardupilot rc
```

Populated channels prove receiver → FC. `0 channels` means §14.5 is not done, or
the receiver has no power. **CH16 carries RSSI** (the R9 SX spec is "6 PWM / 16
SBUS (CH16 outputs RSSI)"), so a CH16 that moves on its own is signal strength,
not a stray control.

**LED reference** for telling the three failure states apart at the bench, from
the manual: solid green = bind mode; **green flashing, red off = bound and
working normally**; nothing lit = no power.

### 14.7 Verify the channel map against §3.5.2

Run ArduPilot RC calibration, then confirm each control does what
[`ROVER_RUNBOOK.md`](./ROVER_RUNBOOK.md) §3.5.2 says — with the rover on stands,
wheels clear of the ground:

- **CH5 (SF)** arms and disarms.
- **CH8 (SA)** mid position reads **RTL**, not Guided. The boot-enforced
  `rover3_base_parameters.params` sets `MODE4=11`/`MODE6=15`; the README's older
  ordering is wrong and has already caused confusion once (see `docs/learnings.md`,
  2026-07-23).
- **CH9 (SH)** shuts the Pi down only after a release, then a >2 s hold while
  disarmed.
- **CH7 (SD)** reboots the FC — confirm before trusting it, it also kills the
  collector in the lower band.

### 14.8 Receiver failsafe — decide, do not inherit

The R9 SX's own failsafe (No Pulses / Hold / Custom) and the FC's
`FS_THR_ENABLE` are one decision, not two. `rover3_base_parameters.params` ships
`FS_THR_ENABLE 0`, so ArduPilot takes **no** action on RC loss regardless of the
receiver setting — a receiver left on "Hold" means a rover that loses its link
keeps executing its last command. On a 900 MHz link the rover can be well past
visual range when that happens.

Protection requires **both** the receiver on No Pulses **and** `FS_THR_ENABLE`
enabled. Set this deliberately for Rover 4 and record what you chose; it is
tracked as an open item in §3.5.1.

### 14.9 SiK NetID 46

Set both ends of Rover 4's SiK pair to **NetID 46** via Mission Planner (see
`README.md` "SikRadio"; copy the settings across to the second radio). Non-unique
NetIDs cross-talk.

> **Unresolved:** the SiK band is not recorded anywhere in-tree. If these radios
> are 915 MHz they share a band with the R9M on FCC firmware — two transmitters
> per vehicle, four vehicles per field. That degrades range and link quality in a
> way that reads as a hardware fault. Confirm the band and record it here.

### 14.10 Exit criteria

- §14.6 passes in both directions, confirmed with `rover ardupilot rc`.
- Every control in §14.7 verified on stands.
- Receiver failsafe chosen and written down.
- NetID 46 on both radios, telemetry link up.
- §3.5.1's fleet table updated with Rover 4's row.
