# Rover 1 two-Pluto direct-USB boot acceptance

Date: 2026-07-26  
Host: `roverpi1`, `192.168.1.41`  
Tested SPF boot commit: `cd006ea675ad93e4e441dae16fae117bce45f3fe`

## Hardware and immutable inputs

```text
physical USB 1-1.1  104000d02597000b16003400eb98846432
physical USB 1-1.2  10400090fd950014020005008faf192e5a
installed QSPI       device-fw v0.37-dirty on both radios
release              v0.38-plutoplus-spf-gain-rssi-v2
DFU asset            plutoplus-spf-direct-usb-gain-rssi-v2-pluto.dfu
DFU SHA-256          f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4
```

The direct image was loaded into RAM only. No QSPI write command was used.

## Capture contract

The committed profile was
`data_collection/rover/rover_v3.1/capture_configs/rover1_receiver_config_pi_3mhz_35mm_direct_usb_v2.yaml`.

```text
receivers                 2 Plutos, simultaneously
RX channels per Pluto     2
LO                        5.766 GHz
sample rate               30 MS/s
RF bandwidth              3 MHz
samples/channel/frame     524,288
kernel RX buffers         4
gain mode                 slow_attack
snapshot interval         0.5 s
frames/receiver           100
transport/protocol        direct USB / v2
dataset                   v7
```

## Gates and evidence

| Gate | Result | Evidence |
|---|---|---|
| Legacy baseline | Pass | Stock-QSPI IIO captured 100 frames on both radios; median 2.071/2.026 Hz |
| Multi-radio identity | Pass | Two unique serials and physical paths; duplicate-address networks isolated |
| Persistent radio configuration | Pass | Both U-Boot environments reported `ad9361` and `2r2t` |
| RAM load | Pass | Both physical DFU paths loaded the checksum-pinned image; interface 6, IIO, `iiod`, and `sdr_usb_gadget` verified |
| Explicit direct profile | Pass | Both radios used protocol v2 and data version 7 |
| Direct capture | Pass | 100 finite, nonzero `complex64[2,524288]` frames per receiver |
| Gain/RSSI metadata | Pass | Start/end arrays finite and valid on every frame; unsafe flags absent; legacy fields equal end values |
| Stream metadata | Pass | Stream/buffer/sample rules accepted by the strict validator |
| Reboot | Pass | Linux boot ID changed from `58442329-2d8d-49b7-b41d-9977fc367b59` to `b559577b-611c-4e73-847b-7fd53a41db51`; boot services passed |
| Zarr reopen | Pass | `validation_boot_reboot.json` has `status: pass` |
| QSPI rollback | Pass | Both radios returned to `v0.37-dirty`; direct interfaces/processes absent and standard IIO present |
| RAM restore | Pass | Both stock radios were backed up, path-targeted through DFU, and returned to direct USB |
| Post-rollback capture | Pass | `validation_after_rollback_restore.json` has `status: pass` |
| Boot safety | Pass | `mavlink_controller.service` disabled/inactive; qualification units enabled |

Reboot capture:

```text
/home/pi/preflight/boot_direct_usb/20260726_060246_rover1/
```

Post-rollback/restoration capture:

```text
/home/pi/preflight/boot_direct_usb/20260726_061050_rover1/
```

The committed service-mode helper was then exercised in both directions on
Rover 1. `restore-legacy` stopped/disabled both qualification units and
enabled—but did not start—the legacy unit. `enable` reversed those states,
preserved the existing environment, and reinstalled the committed units. A
final capture from SPF commit `11c025aa6ed45d27c54c0605813093b51dc19877`
passed:

```text
/home/pi/preflight/boot_direct_usb/20260726_062540_rover1/
validation_final_deployed_commit.json
```

The large LMDB/Zarr files remain on Rover 1 and are not committed.

## Stock-to-RAM boot proof

The initial reboot proved service ordering and automatic capture, but retained
Pluto RAM because a Pi-only reboot did not remove USB power. A stronger final
test therefore reset both Plutos to stock QSPI immediately before rebooting
the Pi:

```text
boot ID before  b559577b-611c-4e73-847b-7fd53a41db51
boot ID after   fbbe101d-8448-429e-a958-cfb7556f338d
```

On the new boot, the journal records both checksum-pinned, physical-path DFU
loads (`dfu-util -p 1-1.1` and `-p 1-1.2`), a separate pre-load stock backup
for each serial, both post-load direct-USB verification passes, post-
enumeration mapping regeneration, and only then the dependent capture.

The resulting exact-commit capture passed:

```text
/home/pi/preflight/boot_direct_usb/20260726_063517_rover1/
validation_stock_to_ram_boot.json
```

This closes both idempotent cases: retained correct RAM is verified/skipped,
while stock QSPI is automatically RAM-loaded at boot.

## Rollback observations

After `rollback-all 2`, runtime discovery reported:

```text
serial=104000d02597000b16003400eb98846432 usb=1-1.1 direct_usb=false
serial=10400090fd950014020005008faf192e5a usb=1-1.2 direct_usb=false
runtime_count=2 expected_count=2
```

Both `/opt/VERSIONS` files began with `device-fw v0.37-dirty`; `iiod` was
running, `sdr_usb_gadget` was absent, and USB-IIO contexts were available at
`usb:1.6.5` and `usb:1.7.5`.

After the subsequent loader service, both radios reported
`direct_usb=true`, mapping regenerated to USB addresses 9 and 11, and the
ready stamp recorded the tested SPF commit and firmware SHA-256.

## Performance qualification

The direct runs sustained roughly the requested two stored frames per second:

```text
reboot capture median         1.990 / 1.977 Hz
post-rollback median          1.936 / 1.932 Hz
```

The data/protocol validator passes. The separate field-performance gate does
not: each direct run had one post-start synchronized interval above one
second, yielding p99 values of 1.2–2.0 seconds. File sizes were 294–360 MB on
a 2 GB Pi with default dirty-page thresholds; host storage/writeback pressure
is a plausible cause but was not instrumented sufficiently to prove it.
Retain the existing p99 ≤1.0 second field criterion until that is investigated
or explicitly re-decided.
