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
protocol-v3 direct RX frames lose the DDS tone after the direct streaming path
starts, while standard pyadi RX can observe the transmitter before that
handoff. Investigate the RC2 HDL/direct-gadget interaction with the TX DDS and
DMA ownership. A replacement candidate must pass this same two-radio test
before promotion.

After the campaign, both radios were reset to their unchanged persistent
production firmware and both TX channels were verified muted at -80 dB.
