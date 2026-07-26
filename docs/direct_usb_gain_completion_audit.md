# Direct-USB gain/RSSI completion audit

Date: 2026-07-25, extended for dual-radio Rover 1 on 2026-07-26

This audit distinguishes the completed implementation from physical RF bench
characterization that cannot be inferred from transport tests.

## Completed and directly evidenced

| Requirement | Result | Authoritative evidence |
|---|---|---|
| Preserve standard USB-IIO for setup | Pass | Hardware configuration and subsequent direct capture passed; stock and RAM images both enumerated IIO |
| Use a separate vendor bulk interface for RX | Pass | Interface 6, bulk IN `0x89`, bulk OUT `0x07` exercised on hardware |
| Two RX channels, 524,288 samples/channel | Pass | Saved v4 Zarr is `complex64[100,2,524288]` |
| Version and negotiate metadata | Pass | Capability range 1..2 and explicit v2 START exercised |
| Keep protocol v1 available | Pass | Hardware two-frame v1 rollback smoke, sequences 0 and 1 |
| Fixed header and payload alignment | Pass | 96-byte header plus 4,194,304-byte IQ payload equals exact 4,194,400-byte transfer |
| Validate magic/version/size/CRC/layout/sequence | Pass | C/Python golden vectors and malformed-frame tests |
| Read RX1/RX2 raw gain state locally | Pass | Manual channel-specific register tests and zero on-device read failures |
| Convert raw state through active gain table | Pass | Full-table/range/mode validation plus exact manual 20/20 and 20/40 dB results |
| Preserve raw-state endpoint-change semantics | Pass | Flags are computed on Pluto before dB conversion; equal rounded dB is not used for comparison |
| Read RX1/RX2 RSSI locally | Pass | Local attributes matched stable host diagnostics at 0.25 dB resolution |
| Attach gain, RSSI, and IQ to one transfer | Pass | Device log, exact hardware transfer completion, CRC, and IQ-offset checks |
| Avoid per-frame host IIO metadata reads | Pass | Direct `PPlus` test makes IIO accessors raise while capture succeeds |
| Preserve Python `rx/rssis/gains` API | Pass | Real hardware facade returned `(2,N)`, `float64[2]`, `float64[2]` |
| Preserve existing v4 Zarr schema | Pass | IIO and direct v2 groups have the same 13 arrays and matching dtypes/shapes |
| Record 100 normal Rover frames | Pass | Normal YAML → collector → LMDB Zarr capture reopened and validated 100/100 |
| Manual equal and unequal gains | Pass | Five-frame v2 captures returned exactly 20/20 and 20/40 dB |
| Slow-attack AGC | Pass | 100-frame capture and 7,200-frame soak observed plausible independent motion |
| Throughput budget | Pass | Direct 2.024 Hz versus IIO 2.058 Hz, about 1.7% lower |
| Bounded memory and explicit cleanup | Pass | Ten-minute and one-hour runs reached the same stable RSS plateau |
| One-hour soak | Pass | 7,200 frames, zero invalid headers/IQ/read failures |
| RAM boot and stock rollback | Pass | QSPI remained unchanged; reset restored stock v0.37 and removed interface 6 |
| Publish source and tested image | Pass | Firmware, Buildroot, gadget, and SPF commits are on GitHub; the exact hardware-tested DFU is a checksum-pinned release asset |
| Fetch/build from published pin | Pass | Buildroot fetched gadget commit from GitHub, cross-built it, and installed ARM ELF |

Test evidence:

```text
SPF scoped acceptance suite   163 passed
transport-only rerun          154 passed
gadget C suite                5 passed
ARM cross-build               passed
published-pin Buildroot       passed
```

Earlier bring-up and correction records are retained in:

```text
artifacts/direct_usb_gain_metadata/2026-07-24/
artifacts/direct_usb_gain_metadata/2026-07-25/timestamp_validation.md
artifacts/direct_usb_gain_metadata/rover3_one_radio/
    2026-07-25_iio_baseline_repeat/report.md
```

Published commits:

```text
SPF                            5461f247240d46b5b874191a3eea628338abc0a1
firmware main/master           dd6b1f4db710abc20693888db08e8da2427e0dc3
Buildroot                      6d5b0298364dc03ae9fb1c0754b83355960b4d63
USB gadget                     54610e01c6fd6a69df77f148ea0dc88f9cb18063
```

Published hardware-tested image:

```text
release
    https://github.com/misko/plutosdr-fw/releases/tag/v0.38-plutoplus-spf-gain-rssi-v2
asset
    plutoplus-spf-direct-usb-gain-rssi-v2-pluto.dfu
size
    13733347 bytes
SHA-256
    f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4
```

## Ambient 2.4 GHz functional exercise

On 2026-07-25/26, a passive spectrum sweep found strong bursty
20 MHz-class activity centered near 2457 MHz (Wi-Fi channel 10). The
hardware-tested v2 image was RAM-booted and exercised through the normal Rover
3.1 collector and v4 Zarr writer using:

```text
LO                              2457 MHz
sample rate                     30 MS/s
RF bandwidth                    20 MHz
samples per RX per frame        524,288
stored frames per run           100
```

The fixed-gain run used 10/10 dB. It passed 100/100 frames with exact
10.0/10.0 dB metadata, a median 2.018 frame/s, RSSI magnitudes from 38.5 to
66.75 dB, no clipped frames, and peak I/Q components of 579 and 638 counts.
Frame-average IQ power ranged from -63.4 to -22.0 dBFS on RX1 and -62.5 to
-20.7 dBFS on RX2. This provides a clean real-signal transport and
frame-association check.

The matched slow-attack run also passed 100/100 frames at a median 2.006
frame/s. Stored gain ranged from 18 to 71 dB on RX1 and 17 to 58 dB on RX2,
with changes at 70 and 84 stored-frame boundaries respectively. A bounded
two-buffer protocol trace returned continuous buffer/sample sequences and
valid start/end gain and RSSI pairs; both channel endpoint-change flags were
set with no dummy, overflow, or metadata-read-failure flags.

Slow-attack AGC is not suitable for clean capture of this particular bursty
ambient signal at these settings: quiet periods drive gain high, and a
subsequent burst clipped in 71/100 RX1 frames and 95/100 RX2 frames. The
fixed 10 dB capture did not clip.

The captures and their resolved YAML files are beneath:

```text
artifacts/direct_usb_gain_metadata/rover3_one_radio/
    2026-07-25_wifi_ch10/manual_100/
    2026-07-25_wifi_ch10/slow_attack_100/
```

This is useful cross-band and live-AGC evidence, but it does not close either
calibrated RF gate below: ambient transmitter power, packet duty cycle, and
multipath are uncontrolled.

## Physical characterization still missing

### 1. Calibrated stepped-input RSSI test

Required evidence:

- hold frequency and receiver configuration fixed;
- step RF input power by known amounts using a calibrated attenuator or signal
  generator;
- demonstrate the expected positive-magnitude RSSI direction and report error
  versus the commanded steps.

Why current evidence is insufficient:

- exact agreement with the Linux `rssi` attribute proves compatibility, but it
  does not independently characterize RF-input response;
- the configured `o4` source is external and bursty;
- no programmable attenuator or controllable RF source is attached.

Connected-device inspection found only the Pluto composite USB device. The
Pluto's `/dev/ttyACM0` is its own ACM function, not an attenuator controller.

### 2. Coherent phase-versus-gain-change test

Required evidence:

- split one coherent CW source into RX1/RX2;
- induce controlled gain transitions;
- compare relative phase and IQ power for endpoint-changed and unchanged
  frames;
- preserve the rule that equal endpoints do not prove in-frame stability.

Why current evidence is insufficient:

- ambient/o4 phase statistics are not a controlled coherent phase reference;
- the transport proves frame association but cannot substitute for a common-CW
  RF setup.

## Exact next bench sequence

1. Attach a common CW source through a two-way splitter to RX1/RX2.
2. Insert a calibrated programmable attenuator before the splitter.
3. RAM boot the tested v2 firmware; do not flash it.
4. Capture fixed input levels, for example 0, 5, 10, 15, and 20 dB additional
   attenuation, with at least 20 frames per level.
5. Verify RSSI monotonicity and quarter-dB agreement with occasional
   out-of-stream IIO diagnostics.
6. Run manual RX1-only, RX2-only, and AGC transitions while recording the
   coherent relative phase.
7. Archive commands, attenuator identity/calibration, raw datasets, and
   analysis beneath the existing artifact root.
8. Reset to stock QSPI and re-run the standard IIO smoke.

Until those two RF stimuli are available, the firmware and software delivery
is operationally complete, but the full physical characterization gates are
not proven.

## Rover 1 dual-radio boot and v7 extension

The later Rover 1 qualification extends the original one-radio/v4
compatibility result without invalidating it.

Hardware:

```text
Rover                         1, roverpi1, 192.168.1.41
Pluto physical path 1-1.1    104000d02597000b16003400eb98846432
Pluto physical path 1-1.2    10400090fd950014020005008faf192e5a
installed QSPI, both          device-fw v0.37-dirty
tested SPF boot commit        cd006ea675ad93e4e441dae16fae117bce45f3fe
RAM image SHA-256             f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4
```

The serial/path-aware multi-loader kept both duplicate-address USB-network
devices attached, verified `ad9361`/`2r2t`, loaded each image through its exact
DFU physical path, regenerated the IIO mapping after final enumeration, and
verified IIO plus direct USB on both devices.

The explicit Rover 1 profile uses protocol v2 and dataset v7. In addition to
the v4 compatibility fields, v7 stores both gain and RSSI endpoint pairs,
validity and raw endpoint-change flags, local read durations, stream/buffer/
sample sequences, first-change placeholders, IQ power, and radio identity.

Three service-shaped two-radio captures and one earlier direct capture passed
the strict data validator. The actual reboot run produced:

```text
capture
  /home/pi/preflight/boot_direct_usb/20260726_060246_rover1/
frames
  100 per receiver
IQ
  complex64[100,2,524288] per receiver; finite and nonzero
gain/RSSI metadata
  valid on every frame; legacy fields equal frame-end fields
identity
  two unique serials and physical USB paths
stream metadata
  valid; 100 finite one-frame stream generations per receiver
validator
  status=pass
median cadence
  1.990 Hz and 1.977 Hz
```

A real Raspberry Pi reboot changed the Linux boot ID, left the legacy
motion-capable service disabled, ran the loader unit, regenerated the changed
USB-IIO mapping, and automatically ran and reopened the 100-frame Zarr.
Because a Pi-only reboot kept USB power applied, the radios retained their RAM
images and the idempotent loader verified rather than reloaded them.

Rollback was therefore tested separately and more strongly:

1. both radios were reset by serial into their installed QSPI images;
2. both reported `direct_usb=false`, standard USB-IIO, `v0.37-dirty`, and no
   `sdr_usb_gadget`;
3. the loader then backed up each stock state, entered each physical DFU path,
   reloaded the checksum-pinned RAM image, and restored both direct interfaces;
4. another two-radio 100-frame v7 capture passed.

The transport/data objective is complete. The stricter field cadence gate is
still open: median throughput is near the intended 2 Hz, but each 100-frame
direct run had one post-start synchronized interval of 1.2–2.0 seconds, so
the existing p99 ≤1.0 second criterion was not met. The timing pattern is
consistent with host-side storage/writeback pressure on the 2 GB Pi and is
not evidence of malformed USB metadata, but that inference still requires a
dedicated writer/USB trace before field sign-off.
