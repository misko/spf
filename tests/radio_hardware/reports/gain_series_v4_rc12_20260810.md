# Gain-series v4 RC12 hardware gate — 2026-08-10

## Disposition

**PASS for direct USB and bounded one-frame direct IP — RC12 completed the
two-radio RAM-boot promotion campaign. Continuous multi-frame direct IP is
rate-limited as documented below and is not qualified at the rover's 3 MS/s
rate.**

RC12 retains RC11's gain-series, timestamp, USB-startup, and TX fixes and
hardens the direct-IP control socket against malformed UDP datagrams. The
exact candidate passed three independent two-radio RAM boots, physical TX2
loopback on both radios after every boot, protocol-v2 compatibility,
protocol-v3 USB stress, simultaneous USB, a 100-record-per-radio V7 round
trip, malformed direct-IP traffic, and a real direct-IP frame.

The candidate was loaded only into volatile RAM. Both radios were restored to
the preserved QSPI production firmware after testing and passed the production
direct-USB baseline again.

## Candidate identity

- Firmware main commit: `fa5f95f0af2a0586c80b54eff7ae04512cb96f7f`
- USB gadget commit: `e14eae63ac6b7fe51828e85de34a8d4e1c50d49e`
- Direct-IP gadget commit: `e44821f6d2737e3cdf452a017787f11c451fe04e`
- Buildroot commit: `950b336d3a933c3f56ecd4ec20266502c775cf54`
- HDL commit: `5360aeac83013e8e3aa0b5d4e0a9b32280fe1677`
- HDL Quantulum commit: `55803cd3d7f74dccd7bf43dc713f832d1eeaf2e6`
- Linux commit: `d798b0d821b85ebd51ecffbfa68d8e4d69b77132`
- U-Boot commit: `1ff0468e9bea29b0a768a7bf52db8d025c521b9a`
- GitHub Actions run: [31342525077](https://github.com/misko/plutosdr-fw/actions/runs/31342525077)
- Artifact: `plutoplus-main-fa5f95f0af2a0586c80b54eff7ae04512cb96f7f-31342525077-1`
- Bundle: `plutoplus-spf-main-fa5f95f0af2a.tar.gz`
- Bundle SHA-256: `34264a491a387cd82aba178428069bf66b0b1ec67c28ba886f2627d77c67a457`
- DFU: `plutoplus-spf-main-fa5f95f0af2a-pluto.dfu`
- DFU SHA-256: `2209e23ccc76b9748b0b4435ff706f8edbd7ad0ce1b950ff4065a399d7de52d4`
- Packaged device-fw: `v0.38-plutoplus-spf-gain-series-v4-rc11-2-gfa5f9`
- Tested boot mode: volatile DFU/RAM only

The public Kalman build and GitHub's independent verification/attestation job
both passed. A fresh download passed its adjacent bundle checksum, every entry
in `SHA256SUMS`, and every entry in `PAYLOAD_SHA256SUMS`.

Routed timing passed with setup WNS `+0.504 ns`, hold WHS `+0.014 ns`, and
zero failing setup, hold, or pulse-width endpoints.

## RC11 persistent-canary findings

RC11 had passed the original RAM-only release campaign. A subsequent
persistent-QSPI canary exposed two additional operational issues before fleet
promotion:

1. A Pluto reboot restored both TX hardware-gain readbacks to `-10 dB`. No DDS
   or DMA waveform was active, so this was not observed RF emission, but it was
   not a fail-closed boot state. SPF now removes stale mute evidence and runs
   `spf.scripts.mute_pluto_tx` after every RAM/QSPI transition and before
   publishing rover readiness. A mute failure leaves readiness absent and
   blocks collection.
2. A four-byte unknown UDP control datagram caused RC11's `sdr_ip_gadget` to
   return a fatal epoll-handler result and terminate. RC12 classifies control
   envelopes before legacy parsing and silently drops undersized or unknown
   datagrams. Native tests cover null, short, legacy, protocol-v3, time-anchor,
   and unknown envelopes.

An earlier manual direct-IP timeout was separately traced to a stale DHCP
address: the LAN radio moved from `192.168.1.179` to `192.168.1.181` after a
reboot. The automated campaign resolves the candidate address from the radio's
USB serial and successfully followed it to `192.168.1.163`. IP address is never
treated as radio identity.

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

`/tmp/spf-gain-series-v4-rc12-20260810T000826Z-hardware`

| Gate | Result |
|---|---|
| Production protocol-v2 baseline | 6 passed |
| Persistent AD9361/2R2T configuration | Both serials passed |
| Shared-hub RC12 RAM loads | 3/3 epochs passed |
| Immediate post-boot TX mute | Both radios, all epochs passed |
| Internal cyclic TX through timestamp FIFO | Both radios, all epochs passed |
| External TX2 loopback and live gain series | Both radios, all epochs passed |
| Explicit post-TX mute | Both radios, all epochs passed |
| Protocol-v2 compatibility at 524,288 samples | 6 passed |
| Protocol-v3 USB | Passed on both radios |
| Fresh protocol-v3 START stress | 100 streams/radio, 600 frames total passed |
| Simultaneous protocol-v3 USB | Passed |
| Production-sized V7 Zarr | 100 records/radio written and reopened |
| Malformed direct-IP datagrams | Lengths 0, 1, 3, 4, 4, and 80 survived |
| Protocol-v3 direct-IP frame | Passed; 162 gain observations |
| Post-rollback production baseline | 6 passed |
| Final TX state | TX1/TX2 `-80 dB` on both radios |

The repeated-start evidence SHA-256 is
`bfdcdbab3419fcd39ac94a5c46edf27156777378f8c7a8e4683ad5ef45d25db0`.
The malformed-IP evidence SHA-256 is
`9059c9e60ee49d102abde228b3b29d2c6ded2686fde53422fe06a52839b318f8`.
The real direct-IP frame evidence SHA-256 is
`b65fbda3d0a6d181627829e8fa2bc3858ec15a6c4def60e9714d6658dcacaa1c`.

## TX measurements

All six radio/epoch measurements passed frequency, SNR, clipping, coherence,
phase stability, manual-gain metadata, slow-attack response, and mute checks.

| Epoch | Radio | RX1/RX2 tone (dBFS) | Coherence | Phase std | AGC reduction RX1/RX2 | Mute delta RX1/RX2 |
|---:|---|---|---:|---:|---|---|
| 1 | `…0a003a` | `-41.54/-27.33` | `0.9999903` | `0.111°` | `29/29 dB` | `76.1/76.0 dB` |
| 1 | `…843ef2` | `-41.48/-27.88` | `0.9999992` | `0.030°` | `28/28 dB` | `83.9/96.8 dB` |
| 2 | `…0a003a` | `-41.52/-27.29` | `0.9999992` | `0.023°` | `29/29 dB` | `84.4/68.6 dB` |
| 2 | `…843ef2` | `-41.41/-27.84` | `0.9999971` | `0.076°` | `28/28 dB` | `77.6/98.0 dB` |
| 3 | `…0a003a` | `-41.51/-27.30` | `0.9999877` | `0.149°` | `29/29 dB` | `77.3/79.9 dB` |
| 3 | `…843ef2` | `-41.48/-27.84` | `0.9999992` | `0.028°` | `28/28 dB` | `81.6/89.5 dB` |

## Timing and gain-series evidence

The requested gain observation interval was 2,048 samples. The ARM/SPI reads
remain best-effort and their actual FPGA sample-counter bounds are
authoritative; this does not claim sample-exact CTRL_OUT event capture.

- Sequential USB sample-time uncertainty: at most `0.456 ms` in the smoke run.
- Simultaneous USB sample-time uncertainty: at most `0.509 ms`.
- Direct-IP sample-time uncertainty: `0.374 ms`.
- Direct-IP 524,288-sample frame: 162 valid gain observations.

All timing results remain below the 5 ms promotion threshold. Host realtime is
not asserted to be GNSS-synchronized UTC.

## Recovery and automation corrections

The candidate runner now:

- mutes all attached radios before testing, immediately after every RAM boot,
  after each TX stage, and on every exit—including receive-only failures;
- executes the malformed-IP survival test before normal direct-IP capture; and
- prints a rollback command containing the exact candidate image, SHA, serial
  count, and campaign-specific state root.

The original generic rollback wrapper correctly refused to guess when its
default directory contained multiple historical state records. Using the
campaign-specific state root restored both radios to:

`v0.38-plutoplus-spf-gain-rssi-fingerprint-v2-8-gf53d`

Both then reported gadget SHA
`2072e1d0823ef6db3bc141dd733a90d76e23fc33`, TX1/TX2 at `-80 dB`, and passed
the production baseline 6/6.

## Extended direct-IP burn-in

An additional RAM-only campaign exercised the uniquely routed LAN radio after
the initial promotion gate. It injected malformed control datagrams throughout,
validated every CRC-backed protocol-v3 frame and gain-observation series, and
restored both radios to their unchanged QSPI image afterward.

The bounded one-frame path was stable:

- 1,000 fresh START/STOP cycles passed with 1,000 unique stream IDs;
- mixed frame sizes were 16,384, 32,768, 131,072, and 524,288 samples/channel;
- 1,409,024,000 IQ bytes (1.312 GiB) were validated in 340.0 seconds;
- all 600 injected malformed control datagrams were survived;
- duplicate fragments, expired frames, rejected frames, sequence regressions,
  metadata errors, and daemon deaths were all zero;
- median/max time-anchor round trip was 0.527/3.603 ms; and
- median/max local gain-read duration was 0.384/12.564 ms.

Continuous finite streaming exposed a transport-rate boundary hidden by the
original one-frame IP gate. The default 1,472-byte UDP mode deliberately sends
eight datagrams and then waits 1 ms, or roughly 11.8 MB/s before protocol
overhead. Dual-channel CS16 requires eight bytes per time sample:

| Radio sample rate | IQ payload rate | 16 contiguous frames |
|---:|---:|---|
| 1.0 MS/s | 8.0 MB/s | Pass |
| 1.2 MS/s | 9.6 MB/s | Fail closed: sample-sequence gap |
| 1.4 MS/s | 11.2 MB/s | Fail closed: sample-sequence gap |
| 1.5 MS/s | 12.0 MB/s | Fail closed: sample-sequence gap |
| 1.75–3.0 MS/s | 14.0–24.0 MB/s | Fail closed: sample-sequence gap |

At 1.0 MS/s, a longer 200-START burn-in returned 3,200/3,200 contiguous
frames, 400 MiB of IQ, with zero failures in 92.9 seconds. At the rover's
3 MS/s rate, increasing the requested UDP datagram size from 1,472 through
8,192 bytes did not prevent sequence gaps. Sizes from 16,384 through 65,507
bytes instead produced whole-frame timeouts, so relying on IP fragmentation is
not a safe workaround.

Debug output establishes the failure mechanism. While an IP frame is paced to
the host, DMA and the FPGA sample counter continue advancing. The next queued
DMA block can then predate all retained gain observations. The v3 worker
correctly refuses that IQ because it cannot associate the required gain series;
the host independently rejects any resulting sample discontinuity. The gadget
currently advertises up to 16 finite frames without advertising a rate limit,
so host-side capability discovery alone cannot predict this boundary.

Evidence remains in `/tmp/spf-rc12-ip-burnin-20260810` on the hardware host:

- one-frame burn-in SHA-256:
  `fe229da26e4f224961c123591501edc7b37cc842cbb605fb7b52fb0c3b3cbce0`;
- 1 MS/s continuous burn-in SHA-256:
  `7b43324bb70b004cf42ba04b99449a6d52f6a000dd5603643f327b1451df44e0`;
- sample-rate ladder SHA-256:
  `ae426ed962931b033c2afd99092b0572404713c52944cf701f09959ce6a59c50`;
- datagram-size ladder SHA-256:
  `3c0e41948ba49b2e2ef63a94aad100fe7534fc78338f8b5ea18a60746096b50a`;
  and
- on-radio debug trace SHA-256:
  `583e3b804094d91aefb37b28c4ea8fae1f4be3fd948c9fc93f176506719c7fae`.

## Promotion recommendation

Publish and pin only the exact DFU identified above for the qualified direct-USB
path. Retain the previous production release and hash as rollback. Do not claim
continuous direct-IP parity at the rover's 3 MS/s rate; use one-frame bounded
IP requests or at most the demonstrated 1 MS/s continuous mode until the IP
transport advertises and enforces a safe rate or is redesigned to avoid stale
DMA blocks. After SPF CI passes, perform a controlled one-radio QSPI canary,
reboot it from QSPI, run the V7/USB/TX gates plus the bounded IP gate, and only
then roll the same bytes to the remaining radios.
