# Rover 3 direct-USB gain/RSSI capture

This runbook records the normal Rover 3 v4 Zarr dataset while the Pluto+ sends
device-local gain and RSSI observations in each direct-USB IQ transfer. The
experimental firmware is RAM-booted; these instructions never flash QSPI.

For deployment decisions and the old-versus-new firmware acceptance matrix,
use the Rover 3.1
[pre-field checklist](../data_collection/rover/rover_v3.1/PRE_FIELD_CHECKLIST.md).
The direct-USB work documented here is qualified on one Pluto. It does not by
itself qualify a simultaneous two-Pluto Rover 1/3 field configuration.

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
dataset                     existing v4 schema
```

The configuration is:

```text
data_collection/rover/rover_v3.1/capture_configs/
    rover3_one_radio_benchmark_direct_usb.yaml
```

It is a near-copy of the IIO benchmark config. The only behavioral change is
`rx-transport: direct_usb` with protocol v2. The normal collector still calls:

```python
signal_matrix = pplus.rx()
rssis = pplus.rssis()
gains = pplus.gains()
```

In direct mode, `rx()` receives one header-plus-IQ transfer and caches its end
metadata. `rssis()` and `gains()` return cached `float64[2]` values and perform
no IIO transaction. Existing `DataSnapshotV4`, Zarr fields, readers, and models
remain transport-unaware.

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
RAM, and verifies the resulting composite USB device:

`setup.sh` installs its `curl`, `dfu-util`, `sshpass`, `lsusb`, and
`iio_info` prerequisites through `install_deps.sh`.

```sh
cd /home/pi/spf
data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh download
data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh status
data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh load
```

For safety, `load` requires exactly one attached Pluto. On a Rover with two
Plutos, disconnect or power down one, load and verify the remaining unit, then
repeat for the other unit. The first loader version deliberately refuses to
select a radio using a transient bus/device number.

The verified image is cached at:

```text
~/.cache/spf/firmware/plutoplus-spf-direct-usb-gain-rssi-v2-pluto.dfu
```

Pre-load device records are written under:

```text
~/.local/state/spf/pluto-firmware/
```

For diagnosis, the equivalent manual RAM-load sequence is:

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
until the next reboot. The collector does not silently load firmware. Run:

```sh
data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh verify
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
  --exit \
  --yaml-config data_collection/rover/rover_v3.1/capture_configs/rover3_one_radio_benchmark_direct_usb.yaml \
  --device-mapping ~/device_mapping \
  --routine center \
  --records-per-receiver 100 \
  --temp artifacts/direct_usb_gain_metadata/rover3_one_radio/direct_v2_100 \
  --tag DIRECT_USB_V2_100
```

Validate the final Zarr:

```sh
python -m spf.scripts.validate_direct_usb_compat_zarr \
  path/to/final_capture.zarr \
  --expected-frames 100
```

The v4 receiver group must have exactly the same keys, shapes, and dtypes as an
IIO v4 capture:

```text
signal_matrix   complex64[100, 2, 524288]
gains           float64[100, 2], dB
rssis           float64[100, 2], positive dB magnitude
```

Array compatibility does not discard hardware provenance. Every Pluto
receiver group records capture-time identity attributes:

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

A normal reboot or power cycle leaves RAM firmware and restores stock QSPI:

```sh
data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh rollback
```

Pass rollback only when stock firmware reports its original version, standard
USB-IIO captures two channels, and the custom protocol is absent.

See `docs/direct_usb_gain_completion_audit.md` for the final evidence matrix
and the two physical RF characterizations that require additional bench
equipment.
