# Gain-series v4 RC16 hardware gate — 2026-08-10

## Disposition

**PASS for the complete two-radio RAM promotion gate, including direct USB,
direct IP, TX2 loopback, protocol-v2 compatibility, and V7 Zarr. Proceed to a
controlled one-radio QSPI canary; do not perform a fleet rollout yet.**

RC16 restores maximum finite-burst direct-IP throughput to the same practical
20–22 MiB/s range measured by direct USB on this host. The exact public-build
DFU transferred 320 maximum-size frames (1.25 GiB) at 21.33 MiB/s with no
missing frames, sequence gaps, partial reassemblies, rejected fragments,
duplicate fragments, or receive-queue overflows.

The candidate was tested only through volatile DFU/RAM boot. QSPI was not
modified. After testing, both radios were explicitly reset into the preserved
production QSPI image, TX1/TX2 were verified at `-80 dB`, and the production
protocol-v2 baseline passed 6/6.

## Candidate identity

- Firmware main commit: `867e18542311c25f5c1980bbdcfffc823e56f0a2`
- USB gadget commit: `2e8e40ade5dcf3c7880a5ebb58419ad7c37ed552`
- Direct-IP gadget commit: `7cae12eb62cfb2fb656169bd1cfe7da2a0aff583`
- Buildroot commit: `ac893d30559baf00df34c38e0b9e5c9e62500f46`
- HDL commit: `5360aeac83013e8e3aa0b5d4e0a9b32280fe1677`
- HDL Quantulum commit: `55803cd3d7f74dccd7bf43dc713f832d1eeaf2e6`
- Linux commit: `d798b0d821b85ebd51ecffbfa68d8e4d69b77132`
- U-Boot commit: `1ff0468e9bea29b0a768a7bf52db8d025c521b9a`
- GitHub Actions run: [31360123948](https://github.com/misko/plutosdr-fw/actions/runs/31360123948)
- Release: [v0.38-plutoplus-spf-gain-series-v4-rc16](https://github.com/misko/plutosdr-fw/releases/tag/v0.38-plutoplus-spf-gain-series-v4-rc16)
- Artifact: `plutoplus-main-867e18542311c25f5c1980bbdcfffc823e56f0a2-31360123948-1`
- Bundle: `plutoplus-spf-main-867e18542311.tar.gz`
- Bundle SHA-256: `b026afe9bed713dea58d51789749dcebfc5ebbcf580883188937fd4db07b40d9`
- DFU: `plutoplus-spf-main-867e18542311-pluto.dfu`
- DFU SHA-256: `27aca40915fd75fbcabfadef88fee96ff422c6058f83fab8a57a09b8d1eae911`
- Packaged device-fw: `v0.38-plutoplus-spf-gain-series-v4-rc12-9-g867e1`
- Tested boot mode: volatile DFU/RAM only

The GitHub build, independent verification job, adjacent bundle checksum,
`SHA256SUMS`, and `PAYLOAD_SHA256SUMS` all passed. The packaged gadget binaries
were verified as ARM32 executables from the pinned source graph.

After publication, every asset was downloaded again into a fresh directory.
The release bundle matched its adjacent SHA-256 sidecar, all 30 entries in the
nested `SHA256SUMS` passed, and all four deployable payloads matched
`PAYLOAD_SHA256SUMS`, including the tested DFU digest above.

Routed timing passed with setup WNS `+0.504 ns`, hold WHS `+0.014 ns`, and
zero failing setup, hold, or pulse-width endpoints.

## Bench fixture

Each radio had TX2 routed through the declared 30 dB attenuated splitter path
to that radio's RX1 and RX2. Tests activate one TX2 at a time, hold TX1 at
`-80 dB`, and verify both transmitters at `-80 dB` after every stage.

| Radio serial | USB physical path | Additional transport |
|---|---|---|
| `104000bac4950008230026001b440a003a` | `1-1.1` | USB IIO and direct USB |
| `1040007c4a94000211000b009186843ef2` | `1-1.2` | USB IIO, direct USB, and LAN direct IP |

## Promotion gates

Campaign root:

`/tmp/spf-gain-series-v4-rc16-rmem128-full-20260810T073500Z-hardware`

| Gate | Result |
|---|---|
| Persistent AD9361/2R2T configuration | Both serials passed |
| Shared-hub RC16 RAM loads | 3/3 epochs passed |
| Immediate post-boot TX mute | Both radios, all epochs passed |
| Internal cyclic TX through timestamp FIFO | Both radios, all epochs passed |
| External TX2 loopback and live gain series | Both radios, all epochs passed |
| Explicit post-TX mute | Both radios, all epochs passed |
| Protocol-v2 compatibility at 524,288 samples | 6 passed |
| Protocol-v3 USB | Passed on both radios |
| Fresh protocol-v3 START stress | 100 streams/radio, 600 frames total passed |
| Simultaneous protocol-v3 USB | Passed |
| Production-sized V7 Zarr | 100 records/radio written and reopened in 3:28 |
| Malformed direct-IP datagrams | Survived; subsequent valid request passed |
| Protocol-v3 direct-IP frame | Passed with 162 gain observations |
| Maximum direct-IP finite burst | 20 cycles, 320 frames, 1.25 GiB passed |
| Post-rollback production baseline | 6 passed |
| Final TX state | TX1/TX2 `-80 dB` on both radios |

Key evidence SHA-256 values:

- repeated USB starts: `1898609cc5a3cac13fdbec84605504a1b35a5c2b77209fd8f8fbeb4209a031a0`;
- malformed direct IP: `0bae317f667521cd7d8aeb1f4f5a8450e837802452ec30948a8ff1dceeb011a2`;
- direct-IP frame: `3a2ef73877e656b21406a04b64bbc5590455a8d78fd2ea7f5bfe168517c1ea35`;
- direct-IP maximum burst: `521c21ec99f1cd82773757b3b3c1226fbe9a094226527327a0a67a76c208a02f`.

## Direct-IP result and root cause

RC12 coupled DMA capture to paced UDP transmission. At about 11.36 MiB/s,
later DMA blocks could become too old to associate with retained gain
observations, so multi-frame requests correctly failed closed with a sample
gap. RC16 separates the operations:

1. capture the complete finite request into eight kernel IIO buffers and a
   bounded on-radio queue;
2. snapshot gain metadata while capture is active;
3. drain the queue over MTU-safe UDP using GSO and absolute pacing; and
4. validate the common inner V3 frame, sequence, CRC, and metadata on the host.

The first exact-RC16 campaign then exposed a host-only queue-sizing issue. A
64 MiB IQ request produced a nominal 128 MiB effective `SO_RCVBUF`, but Linux
charges UDP socket memory by skb allocation, not just payload bytes. Tens of
thousands of 1,472-byte datagrams exhausted that queue and left 12/16 complete
frames plus three partial frames. This was not a radio DMA, gain-series, or
wire-rate regression.

The qualified host setting requests 128 MiB with `net.core.rmem_max` at least
128 MiB. Linux reports a 256 MiB effective socket queue, leaving room for the
64 MiB IQ payload and skb overhead. The campaign restores the original sysctl
on every exit.

| Measurement | Result |
|---|---:|
| Radio burst sample rate | 20 MS/s |
| Frames per finite request | 16 |
| Samples per channel per frame | 524,288 |
| IQ payload per request | 64 MiB |
| Burn-in requests | 20 |
| Total frames / payload | 320 / 1.25 GiB |
| Aggregate payload throughput | 21.33 MiB/s |
| Minimum / maximum request throughput | 18.68 / 22.07 MiB/s |
| Effective host receive queue | 256 MiB |
| Duplicate / expired / rejected fragments | 0 / 0 / 0 |
| Socket receive-queue overflows | 0 |

Before the final gate, three independent exact-binary burn-ins with the same
256 MiB effective queue transferred another 960 frames / 3.75 GiB without a
sequence, reassembly, or kernel UDP error. Their aggregate rates were 21.89,
21.95, and 21.58 MiB/s.

This qualifies the maximum finite burst. It does not claim that 20 MS/s of
dual-CS16 can stream indefinitely over this transport: that source rate is
160 MB/s, while the finite request is captured first and drained afterward.

## TX, timing, and gain-series evidence

All six radio/epoch TX measurements passed tone frequency, SNR, clipping,
coherence, phase stability, manual-gain metadata, slow-attack response, and
mute checks. Coherence was at least `0.9999965`, within-capture phase standard
deviation was at most `0.080 degrees`, and AGC reduced both channels by 28 or
29 dB. The smallest measured active-to-muted tone reduction was 67.4 dB.

The requested gain-observation interval was 2,048 samples. ARM/SPI snapshots
remain best-effort; their FPGA sample-counter brackets are authoritative. This
does not claim sample-exact CTRL_OUT event capture.

- Sequential USB sample-time uncertainty: at most `0.482 ms`.
- Simultaneous USB sample-time uncertainty: at most `0.547 ms`.
- Direct-IP sample-time uncertainty: `0.379 ms`.
- Direct-IP 524,288-sample frame: 162 valid gain observations.

All timing results remain below the 5 ms promotion threshold. Host realtime is
not asserted to be GNSS-synchronized UTC.

## Candidate history

| Candidate | Finding | Resolution |
|---|---|---|
| RC14 | USB path passed; IP retained the old coupled capture/send failure | Added bounded capture-first queue and GSO sender |
| RC15 source build | 320 frames / 1.25 GiB passed at 21.94 MiB/s | Sent exact source graph to the public builder |
| RC15 public build | Buildroot headers lacked the `UDP_SEGMENT` definition | Added the Linux ABI-compatible fallback value and rebuilt |
| RC16 first exact campaign | 64 MiB host request was marginal after skb accounting | Raised request to 128 MiB / 256 MiB effective and added overflow diagnostics |
| RC16 final campaign | Every USB, IP, TX, V7, compatibility, and recovery gate passed | Ready for controlled QSPI canary |

## Recovery automation correction

The campaign also exposed an ambiguity in the rollback helper. If a candidate
was already RAM-booted before the campaign, the saved pre-load version string
could equal the active version and `rollback-all` could skip the reset even
though the boot mode remained volatile. The helper now treats an explicit
rollback literally: it always resets each serial into QSPI and then verifies
the preserved version and physical USB identity. A focused regression test
covers the equal-version/unknown-boot-mode case.

Both hardware radios were finally restored to:

`v0.38-plutoplus-spf-gain-rssi-fingerprint-v2-8-gf53d`

## Promotion recommendation

1. Publish a release containing only the exact DFU identified above.
2. Update the production manifest/config pins only after the release asset and
   its checksums are independently re-downloaded and verified.
3. Persist the image to one radio, reboot from QSPI, and rerun USB, V7, TX,
   malformed-IP, and maximum-burst gates.
4. Keep the prior production DFU and QSPI image as rollback.
5. Promote to the remaining radios only after the persistent canary passes.
