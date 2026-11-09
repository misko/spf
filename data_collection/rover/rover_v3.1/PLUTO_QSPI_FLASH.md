# Persistent QSPI flashing of the gain/RSSI Pluto firmware

Goal: stop RAM-loading the direct-USB gain/RSSI firmware on every boot (~88–104 s
and the dominant boot-stability risk). Instead flash it **once** to the Pluto's
QSPI so each radio cold-boots its own firmware, and on every boot only **verify
the version and re-flash if it differs**.

Everything below is traced to the plutosdr-fw Makefile and the on-device
`/sbin/update.sh` flasher read off a live rover Pluto (`192.168.2.1`).

---

## 1. The QSPI partition map (read from a live PlutoPlus)

| MTD | Name | Contents | Flashed by |
|---|---|---|---|
| `mtd0` | `qspi-fsbl-uboot` | **FSBL + U-Boot (bootloader)** | `boot.frm` → `handle_boot_frm()` → `dd of=/dev/mtdblock0` |
| `mtd1` | `qspi-uboot-env` | U-Boot environment | `boot.frm` → `dd of=/dev/mtdblock1` |
| `mtd2` | `qspi-nvmfs` | non-volatile FS | — |
| `mtd3` | `qspi-linux` | **FIT: kernel + FPGA bitstream + rootfs** | `pluto.frm` → `handle_frimware_frm()` → `dd of=/dev/mtdblock3` |

**The gain/RSSI features live entirely in the FIT (`mtd3`).** The bootloader
(`mtd0`) is what bricks PlutoPlus units when a bad v0.38 build is written to it.
The hardware-qualified RC16 image reports
**`device-fw v0.38-plutoplus-spf-gain-series-v4-rc12-9-g867e1`** from `mtd3`
while retaining the v0.37 QSPI bootloader in `mtd0`. This exact FIT passed the
complete two-radio RAM campaign and a serial-scoped persistent-QSPI canary,
including a second reboot and USB/IP/TX/V7 gates.

Canonical production V7 configs declare `pluto-firmware.boot-mode: qspi`.
Boot preparation follows that setting, verifies the active version over
USB-IIO, and writes only on mismatch. `SPF_PLUTO_RAM_LOAD=1` is the explicit
volatile recovery/qualification override; it must not be set in the normal
Rover environment.

## 2. Why the historical flash bricked, and the safe path

`/sbin/update.sh` main loop, in order:
1. If a `pluto-fw-*.zip` is on the mass storage → it **unzips all `*.frm`** from it.
2. If `pluto.frm` present → `handle_frimware_frm` → **`dd → /dev/mtdblock3`** (firmware only). ✅ safe
3. If `boot.frm` present → `handle_boot_frm` → **`dd → /dev/mtdblock0` + `mtdblock1`** (bootloader). ⚠️ brick risk

The old `setup.sh` copied the **`.zip`**, which contains `boot.frm` → it rewrote
the v0.37 bootloader with the build's bootloader. **The safe flash copies ONLY
`pluto.frm`** — never the zip, never `boot.frm` — so `mtd0` is never touched and
the known-good v0.37 bootloader stays put.

Both `handle_frimware_frm` and `handle_boot_frm` verify a trailing md5 and (for
firmware) require the magic `ITB PlutoSDR (ADALM-PLUTO)` before writing.

## 3. Creating the boot image (`pluto.frm`) — no rebuild needed

plutosdr-fw builds `pluto.frm` and `pluto.dfu` from the **same** FIT (`pluto.itb`):

    pluto.dfu = pluto.itb + 16-byte DFU suffix        (dfu-suffix -a)
    pluto.frm = pluto.itb + md5(pluto.itb) trailer     (cat itb md5 > frm)

The release only publishes `pluto.dfu`, so recover the `.itb` (strip the DFU
suffix) and re-append the md5 trailer. [`make_pluto_frm.sh`](./make_pluto_frm.sh)
does this and validates the result the same way the on-device flasher does:

```bash
# download the exact image the boot currently RAM-loads, then convert it
gh release download v0.38-plutoplus-spf-gain-series-v4-rc16 \
  --repo misko/plutosdr-fw \
  --pattern "plutoplus-spf-main-867e18542311-pluto.dfu" \
  -D /tmp/fw
bash data_collection/rover/rover_v3.1/make_pluto_frm.sh \
  /tmp/fw/plutoplus-spf-main-867e18542311-pluto.dfu \
  /tmp/fw/pluto.frm
```

Verified: the generated RC16 `pluto.frm` is 12,725,820 bytes, has SHA-256
`8d2623b6f8b5e5fd69d214afed20fe48dce4cd4aa0fe4714fc9825f1dccad415`,
carries the FRM_MAGIC, and has a self-consistent MD5 trailer. The on-device
`handle_frimware_frm` therefore accepts it and writes only `mtd3`.

## 4. Flashing via the mass-storage ("mount the drives") path

The Pluto exposes its updater volume as a USB mass-storage disk (`/dev/sda`,
`/dev/sdb` on the Pi). Same mechanism as the v0.37 provisioning flash, but with
`pluto.frm` (firmware only) instead of the `.zip`:

```bash
mount_point=/media/pluto ; sudo mkdir -p "$mount_point"
for dev in /dev/sda /dev/sdb; do          # one per attached Pluto
  [ -b "$dev" ] || continue
  sudo mount "${dev}1" "$mount_point"
  sudo cp /tmp/fw/pluto.frm "$mount_point"/pluto.frm   # ONLY pluto.frm — never boot.frm / the zip
  sudo eject "$dev"                        # eject triggers update.sh → dd to mtd3 → reset
  while [ ! -b "${dev}1" ]; do sleep 2; done   # wait for the Pluto to flash + re-enumerate
done
```

Alternative (equivalent, firmware-partition only): DFU — `device_reboot sf`
then `dfu-util -a firmware.dfu -D pluto.dfu`. Both ultimately go through U-Boot's
`dfu_sf` targeting the firmware partition.

After flashing, the Pluto cold-boots the gain-series FIT from QSPI and
`grep device-fw /opt/VERSIONS` reads
`v0.38-plutoplus-spf-gain-series-v4-rc12-9-g867e1`.

## 5. Boot flow: check version over USB-IIO, flash only on mismatch

Replace the unconditional per-boot RAM-load with a verify-first check. Expected
version comes from the canonical config manifest (the release's `device-fw`
string). The active version is a standard libiio context attribute, so the
normal path does not mount the updater volume and does not use the duplicate
`192.168.2.1` USB-network address. Per attached Pluto:

```
uri="usb:${bus}.${device_address}.5"   # bus/address bound to this USB serial
running=$(iio_attr -T 2000 -u "$uri" -C fw_version |
          sed -n 's/^fw_version:[[:space:]]*//p')
if [ "$running" = "$EXPECTED_DEVICE_FW" ]; then
    : # correct: do not open the updater volume
else
    mount only this serial's updater volume and flash pluto.frm (section 4)
    reset + wait for re-enumerate
    re-verify over USB-IIO that running == EXPECTED_DEVICE_FW (else fail closed)
fi
```

Result: the **first** boot after provisioning flashes QSPI; **every subsequent
boot** finds the version already correct and does nothing — no RAM-load, no DFU
re-enumeration flap, no updater-volume mount, and no ~104 s. The ready manifest
then independently checks the vendor direct-USB gadget SHA, protocol, and
metadata capabilities before collection is authorized.

### A RAM-booted radio defeats this check — reboot to QSPI before flashing

The comparison above reads the **active** firmware, which equals the QSPI
contents only when nothing has been RAM-loaded. That assumption breaks in
exactly the situation where you most want to flash: straight after a RAM-boot
acceptance campaign, the radios are already running the candidate you are about
to install. `fw_version` matches, the gadget SHA matches, every radio is
skipped, the script reports success — and QSPI still holds the old firmware.
The next power cycle silently reverts the fleet.

Nothing errors, so this is invisible unless you look for it. It bit during the
gain-series-v4 promotion: v4 ships gadget
`2e8e40ade5dcf3c7880a5ebb58419ad7c37ed552`, the *same* SHA RC17 records, so even
the gadget check could not separate the RAM image from the stale QSPI one.

- **Reboot every radio to QSPI first** and confirm it comes back on the
  *installed* version before running the flasher.
- **Do not accept `/opt/VERSIONS` as proof of a successful flash.** A RAM-booted
  radio reports the new string no matter what is in `mtd3`. Proof requires a
  full power cycle followed by a re-read.

### `DEFAULT_APPROVED_QSPI_DEVICE_FW` does not tell you what is installed

`spf/scripts/pluto_multi_firmware.py` defaults to
`DEFAULT_APPROVED_QSPI_DEVICE_FW = ("v0.37-dirty",)`. That constant is a policy
knob, **not** a description of the fleet. Measured on both bench radios
(2026-08-11), QSPI holds `v0.38-plutoplus-spf-gain-series-v4-rc12-9-g867e1` —
the build recorded in §1 above.

The gate `_require_approved_qspi_version()` runs from `provision_config_all()`
only, **not** from `check_config_all()`. Since
`tests/radio_hardware/run_gain_series_v3_candidate.sh` uses `check-config-all`,
its gates pass regardless of the installed QSPI version, and a persistent flash
does not break the bench campaign. What it does affect is
`provision-config-all`, which the rover path already parameterises via
`load_direct_usb_firmware.sh` `--approved-qspi-version`.

To learn what is actually installed: reboot the radio so no RAM image is
active, then read `fw_version` over USB-IIO.

## 6. Safety / recovery

- Only `mtd3` is written (verified: `handle_frimware_frm` → `dd of=/dev/mtdblock3`).
  The v0.37 FSBL/U-Boot in `mtd0` — the brick source — is never touched.
- The FIT being flashed is byte-identical to the one already RAM-booting on the
  fleet, so it is known to run under the installed bootloader.
- If a flash is ever interrupted or a bad FIT is written, recovery is a re-flash
  of a good `pluto.frm` (mtd3) via mass storage or DFU. Only a `boot.frm`/`mtd0`
  mistake needs the URST→MIO52 DFU jumper — which this procedure never writes.
- **Qualify on one radio first** with someone at the bench before flashing the
  fleet: flash `pluto.frm`, full power-cycle, confirm `device-fw` +
  dual-RX + gain/RSSI + vendor interface-6, then roll out.
