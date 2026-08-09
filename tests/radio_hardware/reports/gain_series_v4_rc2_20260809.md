# Gain-series v4 RC2 hardware promotion result — 2026-08-09

## Verdict

**FAIL — do not promote RC2 or write it to QSPI.**

The receive and metadata paths passed, including protocol-v2 compatibility,
simultaneous protocol-v3 USB capture, a production-sized V7 Zarr round trip,
and direct-IP parity. The explicit TX2 loopback gate failed on both radios:
the FPGA DDS tone is present through the production firmware's direct-RX path
but disappears into the noise floor after RC2 direct RX starts.

## Candidate identity

| Field | Value |
|---|---|
| Release | `v0.38-plutoplus-spf-gain-series-v4-rc2` |
| Source commit | `aed638ffb7953601b642c438288addb859ab1e8d` |
| DFU asset | `plutoplus-spf-main-aed638ffb795-pluto.dfu` |
| DFU SHA-256 | `8aa50f092e5465b2ab78c865b851153cfd4745765702ed6eed03419fb83399c0` |
| Boot mode | Volatile RAM only |

The tested DFU was downloaded from GitHub Actions run `31288652045` and was
confirmed byte-identical to the RC2 release asset. RC1 (`d0e29715...`) was not
used in this campaign.

## Bench

Two PlutoPlus radios were attached over USB. On each radio, TX2 was connected
through 30 dB of physical attenuation and a splitter to that radio's RX1 and
RX2. TX1 remained at -80 dB. The radios were identified by immutable serial:

- `104000bac4950008230026001b440a003a`
- `1040007c4a94000211000b009186843ef2`

## Gate results

| Gate | Result | Evidence |
|---|---:|---|
| Production protocol-v2 baseline | PASS | 6 tests |
| Persistent AD9361/2R2T configuration | PASS | Both radios |
| Exact RC2 RAM boot | PASS | Both serials and physical USB paths |
| RC2 protocol-v2 compatibility | PASS | 6 tests |
| RC2 protocol-v3 USB + simultaneous streams | PASS | 2 tests |
| RC2 V7 production-size round trip | PASS | 100 frames/radio, 2^19 samples/channel/frame, 120.52 s |
| RC2 direct-IP parity | PASS | Resolved LAN address by serial, 1 test |
| RC2 TX2 loopback | **FAIL** | Tone absent on both radios |
| Exit TX mute | PASS | TX1/TX2 read back -80 dB on both radios |
| QSPI untouched | PASS | Candidate was RAM booted only |

At 2412 MHz, 3 MS/s, 3 MHz bandwidth, a +100 kHz DDS tone, and the required
one-time pyadi RX-DMA prime before direct RX:

| Firmware | Radio | RX1 tone dBFS | RX2 tone dBFS |
|---|---|---:|---:|
| Persistent production | `...a003a` | -20.89 | -21.06 |
| RC2 | `...a003a` | -105.15 | -107.89 |
| RC2 | `...43ef2` | -103.88 | -106.12 |

The RC2 observations had low SNR, low cross-channel coherence, and unstable
phase, as expected for noise rather than the commanded tone. The production
control used the same radio, cable harness, configuration, and primed handoff.

## Reproduction

With both radios wired through at least 30 dB attenuation:

```bash
SPF_V3_IMAGE_SHA256=8aa50f092e5465b2ab78c865b851153cfd4745765702ed6eed03419fb83399c0 \
tests/radio_hardware/run_gain_series_v3_candidate.sh \
  --with-tx-loopback \
  --loopback-attenuation-db=30 \
  /path/to/plutoplus-spf-main-aed638ffb795-pluto.dfu \
  DIRECT_IP_HOST
```

The test is fail-closed: TX cannot start without both explicit TX opt-in and a
declared attenuation of at least 30 dB. It independently mutes all selected
radios before testing, after testing, and from the outer campaign exit trap.

## Firmware follow-up

The failure is below the Python metadata parser. On RC2, both protocol-v2 and
protocol-v3 direct RX frames can lose the DDS tone after the direct streaming
path starts, while standard pyadi RX can observe the transmitter before that
handoff.

### Hybrid firmware bisection

Two RAM-only FIT images were assembled from already-built production and RC2
components; no source was rebuilt for this bisection:

| Rootfs/software | FPGA | Result |
|---|---|---|
| RC2 | Production | TX passes on both radios |
| Production | RC2 | One radio passes and one radio fails |
| RC2 | RC2 | The failed radio can change between boots |

Repeated protocol-v2 captures on a single full-RC2 boot were stable within
that boot: one radio failed all three epochs while the other passed all three.
The failure therefore follows the RC2 FPGA image and its boot-time state, not
the USB/IP gadget userspace or the metadata parser. The routed RC2 design meets
all declared timing constraints, but its CDC report identifies the new 64-bit
TX timestamp crossing.

### Hypothesis disposition

| Hypothesis | Status | Evidence |
|---|---|---|
| Python parser shifted or corrupted IQ | Rejected | The same loss occurs through the protocol-v2 host path; RX metadata and Zarr gates pass. |
| USB/IP gadget userspace disables TX | Rejected | RC2 userspace with the production FPGA passes on both radios. |
| Bench cable, attenuator, or DDS configuration is wrong | Rejected | Production controls on the same radios and harness show a roughly -21 dBFS tone. |
| One physical radio is defective | Rejected | The failing serial changes between RC2-FPGA boots. |
| Ordinary routed timing failure | Rejected | RC2 has positive setup/hold/pulse-width slack and meets all declared constraints. |
| TX FPGA power-up state | Confirmed fault domain | Any tested image containing the RC2 FPGA can select a silent radio at boot; behavior is stable within a boot. |
| Unreset TX asynchronous FIFO | Leading causal mechanism | The TX XPM FIFO reset was tied permanently inactive; the symptom is boot-selected starvation and changes with FPGA implementation. RC4 is the controlled fix test. |

### RC3 result

RC3 retained the new RX sample-counter HDL and reverted only the TX timestamp
Gray-code source register. The firmware image itself built, but the post-route
CDC gate correctly rejected the routed design for `CDC-10`
combinational-before-synchronizer paths. The register is therefore required for
a valid Gray-code CDC implementation. RC3 produced no accepted artifact and
was never loaded on hardware.

### RC4 controlled fix

RC4 restores the registered Gray-code crossing and changes only the TX
timestamp FIFO reset behavior. The previously inactive XPM FIFO reset now:

1. synchronizes the DAC reset level into the FIFO write-clock domain;
2. starts asserted deterministically after FPGA configuration;
3. remains asserted for at least four complete write-clock cycles; and
4. reasserts after a runtime DAC reset.

The focused startup/runtime reset simulation and the existing coherent RX
counter simulation pass. RC4 remains RAM-boot-only and unpromoted until it
passes three independent TX boot epochs on both radios, followed by the full
protocol-v2, protocol-v3 USB/IP, simultaneous-stream, and V7 Zarr gates.

RC4 passing every boot would convert the unreset-FIFO explanation from a strong
mechanism supported by bisection into the demonstrated root cause. Any RC4 TX
failure would falsify that explanation and require instrumentation of FIFO busy,
empty, reset, and transfer-start state in the next FPGA candidate.

### Test-infrastructure issue found during cleanup

The original multi-radio rollback path had a separate race: it waited for USB
product `b673`, which was already present before reboot, without first requiring
the old device to disappear. It also treated presence of direct-USB interface 6
as proof that a RAM candidate was active, although the current production QSPI
firmware also exposes that interface.

Rollback now preserves the first serial-specific pre-load `/opt/VERSIONS`
record, requires USB disappearance and re-enumeration on the same physical
path, and verifies that the returned firmware version matches that preserved
record. Focused tests cover repeated RAM boots, already-restored radios, and a
real reboot. This issue affected cleanup diagnostics; it did not cause the RC2
TX failure.

A replacement candidate must pass this same two-radio test before promotion.

After bisection, both radios were verified on unchanged persistent production
firmware `v0.38-plutoplus-spf-gain-rssi-fingerprint-v2-8-gf53d`, and TX1/TX2
were read back at -80 dB on both radios.
