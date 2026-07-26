# Rover direct-USB gain/RSSI capture

This runbook records a Rover v7 Zarr while each Pluto+ sends device-local gain
and RSSI observations in the same direct-USB transfer as its IQ frame. The
experimental firmware is RAM-booted; these instructions never flash QSPI.

For deployment decisions and the old-versus-new firmware acceptance matrix,
use the Rover 3.1
[pre-field checklist](../data_collection/rover/rover_v3.1/PRE_FIELD_CHECKLIST.md).
The one-Pluto development tests remain useful, but the primary hardware
qualification is now Rover 1 with two simultaneously attached Plutos.

## Fixed capture contract

```text
RX channels                 2
samples per channel/frame   524288
sample rate                 30000000 samples/s
RF bandwidth                3000000 Hz
LO                          5766000000 Hz
gain mode                   slow_attack
kernel RX buffers           4
snapshot interval           0.5 s
frames                      100
wire protocol               v2, 96-byte header
IQ payload                  4,194,304 bytes
complete USB transfer       4,194,400 bytes
dataset                     v7 (v4 fields plus protocol-v2 metadata)
```

The two-radio Rover 1 configuration is:

```text
data_collection/rover/rover_v3.1/capture_configs/
    rover1_receiver_config_pi_3mhz_35mm_direct_usb_v2.yaml
```

Its RF, frame, pacing, gain mode, and 35 mm geometry match the legacy Rover 1
IIO configuration. The transport changes to `direct_usb`, protocol v2, and
data version 7. Existing callers may still use:

```python
signal_matrix = pplus.rx()
rssis = pplus.rssis()
gains = pplus.gains()
```

In direct mode, `rx()` receives one header-plus-IQ transfer and caches its end
metadata. `rssis()` and `gains()` return cached `float64[2]` values and perform
no IIO transaction. The v7 collector calls `rx_with_metadata()` and also stores
the start/end values, validity flags, read durations, stream/buffer/sample
sequences, endpoint-change flags, and IQ power. Legacy `gains` and `rssis`
remain the frame-end values.

## Published source

Repository:

```text
https://github.com/misko/plutosdr-fw
```

Hardware-tested release:

```text
https://github.com/misko/plutosdr-fw/releases/tag/v0.38-plutoplus-spf-gain-rssi-v2
```

Pins:

```text
firmware main / integration
    dd6b1f4db710abc20693888db08e8da2427e0dc3
Buildroot
    6d5b0298364dc03ae9fb1c0754b83355960b4d63
USB gadget
    54610e01c6fd6a69df77f148ea0dc88f9cb18063
```

The exact hardware-tested RAM image is published as:

```text
plutoplus-spf-direct-usb-gain-rssi-v2-pluto.dfu
size: 13733347 bytes
SHA-256:
f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4
```

The source commits were created after that image was tested and capture the
same source content. A clean rebuild has different embedded Git version text,
so its binary hash is expected to differ.

## Build

Clone the published integration branch recursively:

```sh
git clone \
  --branch v0.38_plutoplus_timestamp_gain_metadata \
  --recurse-submodules \
  https://github.com/misko/plutosdr-fw.git
cd plutosdr-fw
```

Use the Vivado version required by the Quantulum base branch. The original
firmware PDF documents a stale XSA checksum; follow its explicit checksum
exception for that XSA only. Do not disable checksum validation globally.

Buildroot obtains the gadget from the pinned `usb-gadget-gain-metadata` commit.
The pin can be verified independently:

```sh
make -C buildroot sdr_usb_gadget-dirclean
make -C buildroot sdr_usb_gadget
file buildroot/output/target/usr/sbin/sdr_usb_gadget
```

Then build the normal firmware target:

```sh
make
sha256sum build/pluto.dfu
```

## Download, identify, and RAM boot

Before changing boot state, record the installed firmware, serial, USB path,
and U-Boot environment:

```sh
iio_info -s
lsusb -t
cat ~/device_mapping
ssh root@192.168.2.1 cat /opt/VERSIONS
ssh root@192.168.2.1 fw_printenv > plutoplus_uboot_env_backup.txt
```

The preferred Rover loader downloads the release asset, verifies its SHA-256,
backs up `/opt/VERSIONS` and `fw_printenv`, enters DFU, loads the image into
RAM, and verifies the resulting composite USB device. Its multi-radio mode
selects each Pluto by serial and physical USB path while isolating the
duplicate `192.168.2.1` USB-network interfaces:

`setup.sh` installs its `curl`, `dfu-util`, `sshpass`, `lsusb`, and
`iio_info` prerequisites through `install_deps.sh`.

```sh
cd /home/pi/spf
data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh download
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh \
  check-config-all 2
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh load-all 2
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh verify-all 2
```

Keep both Plutos attached. `load-all 2` refuses to continue unless exactly two
unique runtime serials exist, hashes the cached image, targets DFU with each
physical sysfs path, and verifies standard USB-IIO, vendor interface 6,
`iiod`, and `sdr_usb_gadget` on both radios. The single-radio `load`, `verify`,
and `rollback` commands remain available for isolated development rigs.

The verified image is cached at:

```text
~/.cache/spf/firmware/plutoplus-spf-direct-usb-gain-rssi-v2-pluto.dfu
```

Pre-load device records are written under:

```text
/var/lib/spf/pluto-firmware/
```

For diagnosis on a rig with exactly one isolated Pluto, the equivalent manual
RAM-load sequence is:

```sh
sha256sum \
  ~/.cache/spf/firmware/plutoplus-spf-direct-usb-gain-rssi-v2-pluto.dfu

ssh root@192.168.2.1 /usr/sbin/device_reboot ram

dfu-util -d 0456:b673,0456:b674 \
  -a firmware.dfu \
  -D ~/.cache/spf/firmware/plutoplus-spf-direct-usb-gain-rssi-v2-pluto.dfu

dfu-util -d 0456:b673,0456:b674 \
  -a firmware.dfu \
  -e
```

Pass only if standard USB-IIO and vendor interface 6 both enumerate and both
`iiod` and `sdr_usb_gadget` run. Do not flash an image that has not passed RAM
boot and rollback.

This is a boot-time prerequisite, not a per-capture operation. Load once after
each Pluto power cycle or reset, then run any number of direct-USB captures
until the next Pluto reset. A Pi-only reboot may leave USB power applied and
the RAM image resident; the loader detects, verifies, and skips a redundant
load. The collector itself never silently loads firmware. Run:

```sh
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh verify-all 2
```

before starting a direct-USB capture; it fails closed if the expected
interface or on-device processes are absent.

## Smoke test

Configure the radio through pyadi/USB-IIO, then request a small frame:

```sh
source /home/pi/spf-virtualenv/bin/activate
python -m spf.sdrpluto.direct_usb_smoke \
  --serial 104000f6ad020002fdff3a00bba2f096a1 \
  --protocol-version 2 \
  --samples 4096 \
  --frames 1
```

Repeat at the Rover frame size:

```sh
python -m spf.sdrpluto.direct_usb_smoke \
  --serial 104000f6ad020002fdff3a00bba2f096a1 \
  --protocol-version 2 \
  --samples 524288 \
  --frames 1
```

Pass conditions:

- exact transfer length and CRC;
- `(2, 524288)` `complex64` IQ with two finite, nonzero channels;
- valid gain-dB and RSSI endpoint pairs;
- sequence zero for a new one-frame stream;
- no dummy, overflow, or metadata-read-failure flags.

## Normal 100-frame capture

```sh
python spf/mavlink_radio_collection.py \
  --fake-drone \
  --no-ultrasonic \
  --yaml-config data_collection/rover/rover_v3.1/capture_configs/rover1_receiver_config_pi_3mhz_35mm_direct_usb_v2.yaml \
  --device-mapping ~/device_mapping \
  --routine center \
  --records-per-receiver 100 \
  --temp /home/pi/preflight/direct_usb_v7 \
  --tag DIRECT_USB_V7_RO1
```

Validate the final Zarr:

```sh
python -m spf.scripts.validate_direct_usb_v7_zarr \
  path/to/final_capture.zarr \
  --expected-frames 100 \
  --expected-receivers 2
```

Each v7 receiver group preserves the legacy arrays:

```text
signal_matrix   complex64[100, 2, 524288]
gains           float64[100, 2], frame-end dB
rssis           float64[100, 2], frame-end positive dB magnitude
```

It adds:

```text
gain_db_start/end             float32[100, 2]
rssi_db_start/end             float32[100, 2]
gain_metadata_valid           bool[100]
rssi_metadata_valid           bool[100]
gain_endpoints_equal          bool[100, 2]
gain_metadata_flags           uint32[100]
stream_id                     uint64[100]
buffer_sequence               uint64[100]
sample_sequence               uint64[100]
gain/rssi read durations      uint32[100]
first_gain_change_sample      int32[100, 2]
iq_power_dbfs                 float32[100, 2]
```

Every Pluto receiver group also records capture-time identity attributes:

```text
sdr_identity_version
sdr_family
sdr_serial
usb_vendor_id
usb_product_id
usb_bus_at_capture
usb_address_at_capture
usb_port_path
iio_uri_at_capture
rx_transport
```

Direct-USB captures additionally retain the established `direct_usb_*` and
gain-metadata capability attributes. `sdr_serial` is the durable device
identity; `usb_port_path` records its physical Rover connection. Bus, address,
and IIO URI are diagnostics that may change after re-enumeration. Collection
fails before frame zero if two receiver entries resolve to the same Pluto
serial or physical USB path.

The validator rejects bad IQ, missing/invalid metadata, unsafe firmware flags,
legacy/end-value disagreement, sequence gaps within a stream, wrong transport
or protocol, and duplicate identities. A new stream may start at sequence zero
because the current compatibility client makes one finite START request per
stored Rover frame.

On Rover 1 the two-radio runs on 2026-07-26 sustained median rates of
1.93–1.99 frames/s. They passed the data validator, but each run had one
post-start synchronized host-side interval above one second. Keep the
pre-field checklist's cadence gate separate from the protocol/data pass.

## Default firmware prerequisite and opt-in qualification

Normal production now requires `spf-pluto-direct-usb.service` before
`mavlink_controller.service`. With no `/etc/spf/direct_usb_boot.env`, the
loader defaults to enabled and derives the expected radio count from
`/home/pi/rover_id`. Capture transport and Zarr schema remain selected
separately by `SPF_CAPTURE_PROFILE`.

For an existing installation, apply that ordering without changing the capture
profile:

```sh
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh \
  production-default
```

Install the boot units and environment while disabling the legacy,
motion-capable mission service:

```sh
cd /home/pi/spf
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh enable
sudoedit /etc/spf/direct_usb_boot.env
sudo reboot
```

At boot, `spf-pluto-direct-usb.service` verifies configuration, conditionally
loads both RAM images, regenerates `~/device_mapping`, verifies both radios,
and writes `/run/spf/direct_usb_ready`.
`spf-direct-usb-preflight.service` then runs the exact two-radio 100-frame
capture above, invokes the v7 validator, and writes a `PASS` file under
`/home/pi/preflight/boot_direct_usb/<run>/`.

Inspect:

```sh
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh status
systemctl status spf-pluto-direct-usb.service \
  spf-direct-usb-preflight.service --no-pager
cat /run/spf/direct_usb_ready
```

Never enable this boot qualification alongside
`mavlink_controller.service`; that legacy service starts the production
mission workflow.

## Semantics

The Pluto reads raw RX1/RX2 gain indices locally, converts them through the
active full gain table, and sends whole-dB values. Raw indices do not cross the
v2 interface; they are retained locally for endpoint-change flags. RSSI is the
local Linux-driver `rssi` attribute encoded in quarter dB and retains SPF's
historical positive-magnitude convention.

The observations are buffer-associated ARM reads. Equal endpoints mean only
that no endpoint difference was observed; a transition and return inside the
frame remains possible. RSSI is not dBm, calibrated antenna-input power, or a
whole-buffer statistic.

## Soak

The low-storage soak exercises the public compatibility API without writing
IQ to disk:

```sh
python -m spf.sdrpluto.direct_usb_soak \
  --uri usb:1.1.5 \
  --serial 104000f6ad020002fdff3a00bba2f096a1 \
  --duration-seconds 3600 \
  --interval-seconds 0.5
```

Fail on any invalid metadata, malformed IQ, sequence/USB exception, all-zero
channel, or continued RSS growth.

## Rollback

A Pluto reset or full power cycle discards the RAM image and restores stock
QSPI. With both Rover 1 radios attached:

```sh
sudo systemctl stop spf-direct-usb-preflight.service \
  spf-pluto-direct-usb.service
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh \
  rollback-all 2
sudo data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh status-all 2
```

Pass rollback only when both serials report their original QSPI version,
standard USB-IIO remains present, and interface 6 plus `sdr_usb_gadget` are
absent. To leave the Rover on legacy IIO at its next boot:

```sh
sudo data_collection/rover/rover_v3.1/configure_direct_usb_boot.sh \
  restore-legacy
```

That command sets `SPF_DIRECT_USB_DISABLE=1` and enables but does not
immediately start the motion-capable mission service. The prerequisite service
still runs first but deliberately skips RAM loading. To restore the direct
qualification state instead, select `qualify` (which resets the disable value)
or start
`spf-pluto-direct-usb.service` and then
`spf-direct-usb-preflight.service`.

See `docs/direct_usb_gain_completion_audit.md` for the final evidence matrix
and the two physical RF characterizations that require additional bench
equipment.
