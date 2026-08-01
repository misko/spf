# Direct-USB stability bring-up - 2026-08-01

This report records the first execution of the attached-radio pytest gates and
the finite-buffer firmware candidate described in
[`DIRECT_USB_STABILITY_PLAN.md`](../../../data_collection/rover/rover_v3.1/DIRECT_USB_STABILITY_PLAN.md).

## Radios

| Serial | Physical USB path | Historical note |
|---|---:|---|
| `104000bac4950008230026001b440a003a` | `1-1.1` | `.17`; candidate qualification radio |
| `1040007c4a94000211000b009186843ef2` | `1-1.2` | `.18`; control radio |

USB device addresses changed during reboots and were not used as identity.

Both radios were persistently updated, one at a time, with the published
fingerprint-v1 image using `pluto.frm` only. The updater wrote the Linux FIT to
`mtd3`; it did not write `boot.frm`, the firmware ZIP, `mtd0` or the U-Boot
environment. Both radios then passed a normal `device_reboot reset` and returned
with direct USB present, proving that the published image boots from QSPI.

Published persistent image:

- Release: `v0.38-plutoplus-spf-gain-rssi-fingerprint-v1`
- DFU SHA-256: `0a6a8939b31babed2ad7093d83941ebc809323d69804adcd8da5bcae0e48d3e9`
- Device firmware: `v0.38_plutoplus_with_timestamping-9-g7b02`
- Gadget build: `a1e6417d07188bd72be70692e28c5d6ae9a5ec62`

## Baseline and host limit

The first production-size run used 524,288 samples per channel, CS16 dual RX,
and protocol v2 gain/RSSI metadata.

Passed:

- identity and protocol-v2 capability query on both radios;
- ten repeated production-size one-frame START/STOP cycles per radio;
- exact IQ shape/type and non-zero channels;
- gain/RSSI validity and header CRC validation;
- protocol-v2 to V7 LMDB-Zarr write, close, reopen and field validation;
- simultaneous one-frame capture from both radios.

The host has `usbcore.usbfs_memory_mb=16`. An intentionally deeper test that
queued three 4,194,400-byte framed transfers on both radios failed during host
submission with `LIBUSB_ERROR_NO_MEM`: about 25 MiB was requested. No gadget
START or IQ transfer had occurred. Production queues one frame per radio (about
8 MiB total), which passed. Multi-frame ordering is therefore tested one radio
at a time, while simultaneous testing uses the production queue depth.

Ten-cycle two-radio host RSS grew by 20,512 KiB and remained below the declared
64 MiB gate.

After both radios returned to the published persistent image, the final opt-in
hardware suite passed all five tests: identity/capabilities, contiguous
multi-frame sequence, five production-sized lifecycle cycles per radio,
simultaneous two-radio capture, and a three-record-per-radio V7 close/reopen
round trip. The command selected both radios by serial and required an exact
count of two.

## Stability candidate

The candidate changes only two resource policies:

1. A finite request allocates `min(frame_count, 16)` gadget USB buffers. A
   production one-frame request now allocates one ~4 MiB buffer instead of
   sixteen (~64 MiB).
2. Gadget debug logging defaults off when `sdr_usb_gadget_debug` is absent.

It does not change the USB descriptors, protocol, metadata layout, capability
bits, IQ layout or standard USB-IIO function.

Candidate provenance:

- Gadget: `4872a1c3d67011858ba37a5db21a50591250d428`
- Buildroot: `680b1ad3761d2f00ceb81a40c88b1b23ce6e4956`
- Firmware: `cbf9fb8c90be74e7452ddea5f228731b7dc4f8fb`
- Candidate DFU SHA-256: `c867bbe107139eb6065b3a3282b76c38ad8bd0a5315505738c1e743c773e49bc`
- Candidate device firmware: `v0.38_plutoplus_with_timestamping-10-gcbf9`

Build/test results:

- Six gadget host-native tests passed, including the new buffer-policy test.
- The actual ARM daemon cross-compiled against Buildroot libaio/libiio.
- The complete `pluto.dfu` image built with one job.
- The candidate was RAM-loaded on `.17`; `.18` stayed on the persistent
  published image as a control.
- Candidate `.17` passed 50 production-sized one-frame cycles.
- Pluto used memory was 11,848 KiB before and 10,484 KiB after those cycles.
- No default `/var/log/sdr_usb_gadget.log` was created.
- A mixed candidate/control run passed 20 production-size cycles per radio,
  simultaneous dual-radio capture and five V7 records per radio.

The first candidate RAM-load attempt encountered the known USB-network
namespace discovery race before DFU download (`expected one USB-network
interface ... found []`). The loader's bounded second attempt succeeded. This
is recorded as a boot-loader robustness issue, not a capture or candidate
firmware failure.

## Production collector and provenance

The canonical Rover 3 V7 configuration was exercised through the real
`mavlink_radio_collection.py --fake-drone` path with both persistent radios:

| Run | Result |
|---|---|
| 10 records per receiver | Clean finalization and fresh-process validation |
| 100 records per receiver | Clean finalization; 200 total IQ frames validated |
| QSPI provenance smoke, 3 records per receiver | Clean finalization; manifest and both receiver groups record `qspi` and `firmware_verified=true` |

The 100-record artifact was
`/tmp/spf-stability-production100/rover_2026_08_01_05_28_10_nRX2_center_spacing0p043_tag_STABILITY100.zarr`
and occupied 356,593,728 bytes. Every signal frame had shape `(2, 524288)`,
finite, non-zero complex64 IQ, valid gain/RSSI metadata and the expected radio
identity. Median frame periods were 0.555 seconds (`.18`) and 0.551 seconds
(`.17`). The receiver spans were 59.319 and 59.125 seconds respectively.

During review, the production YAMLs were found to declare `boot-mode: ram`
while boot preparation actually defaulted to persistent QSPI. This did not
change the received samples but made the readiness/Zarr provenance false. The
three production V7 configs now declare `qspi`; the resolver accepts the
explicit `ram` and `qspi` modes; boot preparation follows the config unless
`SPF_PLUTO_RAM_LOAD` is explicitly set. A no-write hardware run confirmed both
active images matched, generated a `qspi` readiness manifest, and the QSPI
provenance smoke capture round-tripped that value into both receiver groups.

## Current disposition

- Both radios' persistent QSPI contains the published fingerprint-v1 image.
- Both radios currently run that published persistent image after normal reset.
- The candidate is not yet a release and must not be made persistent until its
  commits/image are published and the Rover config manifest is updated.

Remaining gates:

1. Exercise an intentionally interrupted capture and verify its incomplete
   state and primary-error provenance.
2. Run a longer lifecycle/dual-radio soak and inspect device/host memory and
   kernel log deltas.
3. Publish and pin the candidate, then repeat a cold-boot two-radio acceptance
   before considering persistent rollout.
