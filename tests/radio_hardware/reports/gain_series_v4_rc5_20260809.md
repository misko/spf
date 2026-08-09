# Gain-series v4 RC5 hardware gate — 2026-08-09

## Disposition

**FAIL — do not promote, release, or write this image to QSPI.**

RC5 passed its complete offline build, routed timing, bus-skew, CDC, packaging,
checksum, and provenance-attestation gates. It passed the two-radio TX2
loopback gate on the first volatile boot, but reproduced boot-dependent TX
starvation on the second volatile boot. The campaign stopped at that mandatory
gate; downstream candidate USB/IP/Zarr tests were not run.

Both radios were subsequently returned to their preserved production firmware,
both TX channels were verified muted, and the production direct-USB baseline
passed.

## Candidate identity

- Firmware repository merge commit: `a24728c7ccbcf5aa541d02a233ed5ed3d57d01a7`
- HDL Quantulum commit: `12b0af1c232f9afe0b6b0883fdb970f338b0798a`
- GitHub Actions run: `31319112802`
- Artifact: `plutoplus-main-a24728c7ccbcf5aa541d02a233ed5ed3d57d01a7-31319112802-1`
- DFU: `plutoplus-spf-main-a24728c7ccbc-pluto.dfu`
- DFU SHA-256: `9326ccf38ad4a37f6cc537ce6dae8a8345a052c4269e99a52c8b2ceb82ee6118`
- Active candidate device-fw: `v0.38-plutoplus-spf-gain-series-v4-rc2-10-ga2472`
- Boot mode used: volatile DFU/RAM only

The downloaded bundle sidecar and every file named by its internal
`SHA256SUMS` passed local verification.

## Bench setup

Two Pluto+ radios were attached over USB. On each radio, TX2 was routed through
the declared 30 dB attenuated splitter path to RX1 and RX2.

| Radio | USB physical path |
|---|---|
| `104000bac4950008230026001b440a003a` | `1-1.1` |
| `1040007c4a94000211000b009186843ef2` | `1-1.2` |

Before loading the candidate:

- both persistent AD9361/2R2T configuration checks passed;
- TX1 and TX2 read back as `-80 dB` on both radios;
- the production direct-USB baseline passed 6/6 targeted tests.

## Offline gates

The candidate bundle reported and local inspection confirmed:

- routed setup WNS: `0.010 ns`;
- routed hold WHS: `0.010 ns`;
- all user timing constraints met;
- timestamp FIFO bus-skew constraints passed;
- no routed `CDC-10` combinational-before-synchronizer paths;
- coherent-counter and deterministic FIFO-reset RTL simulations passed;
- DFU suffix, FIT layout, XSA layout, packaged rootfs, and ARM gadget checks
  passed;
- GitHub bundle verification and provenance attestation passed.

## Hardware results

### Volatile boot epoch 1 — pass

Both radios produced a coherent TX2 loopback tone on RX1/RX2.

| Radio | RX1 tone | RX2 tone | coherence | phase std | AGC gain reduction RX1/RX2 |
|---|---:|---:|---:|---:|---:|
| `…a003a` | `-41.81 dBFS` | `-27.73 dBFS` | `0.9999997` | `0.028 deg` | `29/28 dB` |
| `…43ef2` | `-41.62 dBFS` | `-28.19 dBFS` | `0.9999995` | `0.032 deg` | `28/28 dB` |

The test completed its manual-gain and slow-attack AGC checks, and explicit
post-test readback confirmed TX1/TX2 were muted on both radios.

### Volatile boot epoch 2 — fail

Radio `104000bac4950008230026001b440a003a` produced no usable TX2 tone:

- RX1/RX2 tone: `-103.97/-110.80 dBFS`;
- RX1/RX2 tone SNR: `-40.43/-48.94 dB`;
- cross-channel coherence: `0.1826`;
- within-capture phase standard deviation: `84.65 deg`;
- measured peak frequency error: `+2200.83 Hz`.

These values are noise-floor behavior, not a weak but valid cabled tone. The
test failed on the first radio and therefore did not claim an epoch-2 result
for the second radio. Both TX paths were explicitly muted immediately after
the failure.

## Interpretation

RC5 proves that the routed CDC-10 issue in RC4 can be fixed cleanly, but the
new source-registered FIFO-reset crossing is not sufficient to eliminate the
original boot-dependent TX starvation. One successful boot followed by a
noise-floor boot is the same state-dependent signature isolated during the
RC2 FPGA/userspace bisection. The result rejects the hypothesis that this FIFO
reset change alone fixes the fault.

The next FPGA investigation should observe the TX stream at successive
boundaries after a failing boot: DMA valid/ready, async-FIFO write/read reset
busy, FIFO empty/full, timestamp/upack valid/ready, and DAC valid/data. That
will locate the first boundary where progress stops instead of adding another
reset based only on inference.

## Recovery evidence

Both radios returned to the preserved firmware:

`v0.38-plutoplus-spf-gain-rssi-fingerprint-v2-8-gf53d`

Final checks:

- both radios enumerated at their original physical paths;
- TX1 and TX2 read back as `-80 dB` on both radios;
- production direct-USB baseline passed 6/6 targeted tests;
- QSPI was never modified.

Local campaign evidence was written under:

`/tmp/spf-gain-series-v4-rc5-20260809T151534Z`
