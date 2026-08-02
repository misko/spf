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
The production devices run
**`device-fw v0.38_plutoplus_with_timestamping-9-g7b02`** from `mtd3` while
retaining the v0.37 QSPI bootloader in `mtd0`. This exact v0.38 FIT has passed
normal-reset/direct-USB qualification on the two-radio development bench.

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
gh release download v0.38-plutoplus-spf-gain-rssi-fingerprint-v2 \
  --repo misko/plutosdr-fw --pattern "pluto.dfu" -D /tmp/fw
bash data_collection/rover/rover_v3.1/make_pluto_frm.sh /tmp/fw/pluto.dfu /tmp/fw/pluto.frm
```

Verified: the generated `pluto.frm` (13,733,956 B) carries the FRM_MAGIC and a
self-consistent md5 trailer, so `handle_frimware_frm` will accept it and write
`mtd3`.

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

After flashing, the Pluto cold-boots the gain/RSSI FIT from QSPI and
`grep device-fw /opt/VERSIONS` reads `v0.38_plutoplus_with_timestamping-9-g7b02`.

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
