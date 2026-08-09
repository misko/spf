# Gain-series v4 RC11 hardware gate — 2026-08-09

## Disposition

**PASS — RC11 completed the two-radio RAM-boot promotion campaign.**

RC11 fixes the intermittent protocol-v3 startup sample loss found in RC10.
The exact RC10 failure was first converted into a repeatable red hardware
test. The same test then passed 100 fresh streams per radio on RC11. RC11 also
passed the production protocol-v2 baseline, repeated volatile boots, internal
and external TX tests, V7 Zarr round-trip, direct-IP parity, runtime-counter
audit, and post-rollback production checks.

The candidate was loaded only into volatile RAM. QSPI was not modified. Both
radios were returned to the preserved production firmware after testing.

## Candidate identity

- Firmware main commit: `8fd497c350b46b6129ab6a80c1f89e1b5f3eb9d3`
- Gadget commit: `e14eae63ac6b7fe51828e85de34a8d4e1c50d49e`
- Buildroot commit: `2e6a9f392c71e5a45d813a88f0ca2a30dab47ba1`
- HDL commit: `5360aeac83013e8e3aa0b5d4e0a9b32280fe1677`
- HDL Quantulum commit: `55803cd3d7f74dccd7bf43dc713f832d1eeaf2e6`
- Direct-IP gadget commit: `11d0cbb4c5d39e572f1b6f01e949840d1120d97f`
- GitHub Actions run: `31333752921`
- Artifact: `plutoplus-main-8fd497c350b46b6129ab6a80c1f89e1b5f3eb9d3-31333752921-1`
- Bundle: `plutoplus-spf-main-8fd497c350b4.tar.gz`
- Bundle SHA-256: `fd792e93b7c1c50dd3a5d704b6f5f679cfb8f37de39a4567337997778cc50fa7`
- DFU: `plutoplus-spf-main-8fd497c350b4-pluto.dfu`
- DFU SHA-256: `4caca323f35f4e7852add5d5b35637c84c10a74f7cbb49ecb82e49328f2e872a`
- Candidate device-fw: `v0.38-plutoplus-spf-gain-series-v4-rc2-27-g8fd49`
- Boot mode tested: volatile DFU/RAM only
- Hardware-qualified prerelease tag: `v0.38-plutoplus-spf-gain-series-v4-rc11`
- Prerelease: <https://github.com/misko/plutosdr-fw/releases/tag/v0.38-plutoplus-spf-gain-series-v4-rc11>

The freshly downloaded outer bundle sidecar, every entry in the internal
`SHA256SUMS`, and every entry in `PAYLOAD_SHA256SUMS` passed. GitHub's separate
verification job downloaded the exact artifact, verified it, and published a
build-provenance attestation. After hardware qualification, the exact tested
bundle and DFU were published as the RC11 prerelease and downloaded again into
a fresh directory. The bundle sidecar passed and the downloaded DFU retained
the expected SHA-256 above.

## Bench fixture

Each radio had TX2 routed through the declared 30 dB attenuated splitter path
to that radio's RX1 and RX2. Tests activate only one TX2 at a time and hold TX1
at `-80 dB`.

| Radio | USB physical path | Additional transport |
|---|---|---|
| `104000bac4950008230026001b440a003a` | `1-1.1` | USB IIO and direct USB |
| `1040007c4a94000211000b009186843ef2` | `1-1.2` | USB IIO, direct USB, and LAN direct IP |

The LAN address is DHCP-provided and changed between production and candidate
boots. It was resolved by serial rather than treated as radio identity.

## Offline build gates

The Kalman x86-64/Vivado 2022.2 build reported:

- source graph and host preflight passed;
- coherent-counter HDL simulation passed;
- clean FPGA rebuild and XSA export passed;
- routed setup WNS `+0.504 ns` and hold WHS `+0.014 ns`;
- all user timing constraints met;
- timestamp-FIFO bus-skew constraints passed;
- no routed CDC-10 combinational-before-synchronizer paths;
- DFU suffix, FIT layout, XSA layout, rootfs identity, and ARM gadget checks
  passed; and
- final package hashes passed.

## RC10 root-cause analyses

### 1. Shared-hub DFU failure

The original campaign transitioned both radios into DFU concurrently below
the same VIA USB hub. During the second epoch the hub emitted error `-71` and
both radios disappeared from USB, although both remained alive over LAN. A
hub-port power cycle recovered them; there was no evidence of radio damage.

Root cause: simultaneous runtime-to-DFU re-enumeration on one physical parent
hub.

Fix in SPF commit `3c7a429`:

- group radios by immediate USB parent;
- serialize RAM loads within a shared parent; and
- retain concurrency only across independent parents.

Eight RC11 two-radio RAM-load epochs subsequently completed without `-71`,
hub enable, descriptor-read, or enumeration errors.

### 2. Intermittent 32,768-sample loss

RC10 intermittently produced:

```text
sample sequence discontinuity:
expected 10505641763, got 10505674531
```

The difference is exactly `32,768` samples. Relaxed-parser diagnostics showed:

- `buffer_sequence` remained `0, 1, 2`;
- the gap occurred only between frame 0 and frame 1 of a fresh stream;
- later boundaries were contiguous;
- runtime counters reported no USB error, IIO refill error, dropped frame,
  starvation, gain failure, or RSSI failure; and
- changing the requested gain-observation interval did not remove the gap.

Root cause: the gadget created the live IIO buffer before allocating and
registering all USB/AIO resources. The pinned local libiio backend supplied
only four kernel blocks by default. An occasional startup scheduling delay
could exhaust that backlog before the steady event loop began, losing one DMA
block while the gadget's own USB buffer sequence remained continuous.

RC11 gadget changes:

- derive scan size before enabling DMA;
- allocate and register USB/AIO resources first;
- configure eight bounded IIO kernel blocks;
- create the live IIO buffer only immediately before the event loop; and
- discard exactly one timestamp-aligned startup block after the gain sampler
  is ready and immediately before streaming.

### 3. Red/green startup regression

SPF commits `2e6f8c9` and `6d640af` add a release gate that creates a new
protocol-v3 receiver and START for every cycle, records the exact failing
serial/cycle, and persists partial evidence.

On RC10 it failed on radio `…0a003a`, cycle 31, with the exact 32,768-sample
gap above. On RC11 it passed:

- 100 fresh streams per radio;
- three frames per stream;
- 200 total streams and 600 total frames; and
- zero sequence discontinuities or metadata errors.

The RC11 repeated-start JSON SHA-256 is
`0842ee0bfb130af98ae099f1c56214675b6544cd8ed4bbb9aa9fb054a789c4f0`.

## TX-gate test RCA

The first RC11 campaign stopped during boot epoch 3 because one RX channel's
muted tone-bin delta was `12.6 dB`, below the `15 dB` threshold. This was a
test-design error, not a TX failure:

- the active reference used fixed manual RX gains `(20, 35) dB`;
- the test then changed both channels to slow-attack AGC;
- after TX was stopped, AGC raised the receiver gain toward maximum; and
- the test compared that amplified muted frame against the earlier manual-gain
  frame.

Across three boots the confounded muted bin rose from roughly `-93 dBFS` to
`-68 dBFS` and then `-57 dBFS`, matching variable AGC timing.

SPF commit `6ab76a9` restores and verifies the same manual RX gains, captures a
fresh active reference, then mutes TX and compares the muted frame at identical
receiver gain. The corrected red/green result was:

| Radio | RX1 mute delta | RX2 mute delta |
|---|---:|---:|
| `…0a003a` | `78.4 dB` | `90.9 dB` |
| `…843ef2` | `76.3 dB` | `93.5 dB` |

The complete campaign was then restarted from the production baseline rather
than treating partial stages as a pass.

## Clean RC11 hardware campaign

Campaign root:

`/tmp/spf-gain-series-v4-rc11-20260809T2154Z-hardware/full-campaign`

Campaign-console SHA-256:

`1a2bb338e4220e09ee0a4ace7f09936f47516f285af5b4c5496200af828a3e99`

| Gate | Result |
|---|---|
| Production protocol-v2 baseline | 6 passed |
| Persistent AD9361/2R2T configuration | Both serials passed |
| Shared-hub candidate RAM load | 3/3 campaign epochs passed |
| Internal cyclic TX through timestamp FIFO | Both radios, all epochs passed |
| External TX2 loopback and live gain series | Both radios, all epochs passed |
| Explicit post-TX mute | Both radios, all epochs passed |
| Candidate protocol-v2 compatibility at 524,288 samples | 6 passed |
| Protocol-v3 USB and simultaneous USB | 3 passed |
| Fresh START stress | 100 streams per radio passed |
| V7 production-sized Zarr | 100 records per radio passed and reopened |
| Protocol-v3 direct IP | Passed by serial-resolved LAN radio |
| Final candidate status | Both serials present, correct physical paths |

Two additional RAM-load/TX/mute epochs passed after the campaign. Across the
two RC11 campaigns and extended checks, eight independent RC11 RAM loads were
exercised. The final clean promotion evidence contains five consecutive
all-green boot/TX epochs: three in the restarted campaign and two extended
epochs.

### TX measurements from the clean three-epoch campaign

All twelve radio/epoch channel tones passed frequency, level, SNR, clipping,
coherence, phase stability, manual-gain metadata, and slow-attack gain response
checks.

| Epoch | Radio | RX1/RX2 tone | Coherence | Phase std | AGC reduction RX1/RX2 | Mute delta RX1/RX2 |
|---:|---|---|---:|---:|---|---|
| 1 | `…0a003a` | `-41.53/-27.36 dBFS` | `0.9999991` | `0.036 deg` | `29/29 dB` | `90.8/74.9 dB` |
| 1 | `…843ef2` | `-41.57/-28.06 dBFS` | `0.9999984` | `0.068 deg` | `28/28 dB` | `82.3/88.5 dB` |
| 2 | `…0a003a` | `-41.58/-27.38 dBFS` | `0.9999988` | `0.043 deg` | `29/29 dB` | `83.5/80.3 dB` |
| 2 | `…843ef2` | `-41.57/-28.02 dBFS` | `0.9999980` | `0.068 deg` | `28/28 dB` | `77.4/99.6 dB` |
| 3 | `…0a003a` | `-41.56/-27.39 dBFS` | `0.9999885` | `0.112 deg` | `28.5/25 dB` | `80.8/93.8 dB` |
| 3 | `…843ef2` | `-41.52/-28.02 dBFS` | `0.9999992` | `0.023 deg` | `28/24 dB` | `85.7/95.0 dB` |

## V7 and timing evidence

The reopened hardware-backed store contained, for each radio:

- `signal_matrix` shape `(100, 2, 524288)`, dtype `complex64`;
- `gain_observation_index` shape `(100, 256, 2)`;
- 100 unique stream IDs and buffer sequence zero for every one-frame stream;
- strictly increasing FPGA sample sequences;
- valid gain and RSSI metadata for every record;
- valid sample-time metadata for every record; and
- the expected serial, physical USB path, and gadget build ID.

The requested observation interval was `2048` samples. At the tested 3 MS/s
configuration, each frame contained 161 to 163 valid ARM observations. The
wire/Zarr capacity is 256, but `2048` is a requested cadence, not a claim that
the ARM/SPI path can always complete 256 reads. Each actual read retains its
FPGA sample-counter interval and is authoritative. Reliable every-2048-sample
event detection at higher rates still requires the planned FPGA CTRL_OUT path.

Maximum measured host-time uncertainty was:

- sequential USB: `0.452 ms`;
- simultaneous USB: `0.928 ms`;
- V7 records: `0.454 ms`; and
- direct IP: `0.369 ms`.

All are below the 5 ms promotion threshold. Realtime values describe the
host's clock and do not assert UTC synchronization.

The compact V7 audit SHA-256 is
`6fc9941bb56a0c006054868831d6f9feb97d10d8731abdba005a35e2756acd35`.

## Runtime and USB health

On the final candidate boot, both gadgets reported:

- lifecycle `IDLE` and last-error subsystem `NONE`;
- zero dropped frames;
- zero IIO refill errors;
- zero USB submit errors and short writes;
- zero buffer starvation;
- zero gain and RSSI read failures;
- zero control errors and stop timeouts; and
- gadget build ID `e14eae63ac6b7fe51828e85de34a8d4e1c50d49e`.

The runtime-status JSON SHA-256 is
`c9eea92846160bf7d6dbb29308f8b62713f7d273c098b18bea25899bce242380`.

The host kernel log contained zero `-71`, hub-enable, descriptor-read, or
enumeration failures during RC11. It recorded 18 `Synchronize Cache(10)`
warnings from the Pluto mass-storage LUN disappearing during deliberate DFU
re-enumeration; these were expected detach messages and did not affect either
radio interface.

## SPF test changes and CI

Relevant SPF commits:

- `3c7a429` — serialize shared-hub RAM loads;
- `a80ec83` — model the required TX calibration attribute in the test fake;
- `2e6f8c9` — add the repeated fresh-START hardware gate;
- `6d640af` — persist serial/cycle details on startup-stress failure; and
- `6ab76a9` — compare TX active/muted levels at fixed RX gain.

Full SPF CI at commit `2e6f8c9` passed `1459` tests, skipped `30`, and retained
the two expected xfails in run `31334348151`. The two later commits affect
opt-in attached-radio test evidence and were also exercised directly against
both radios. Their latest full CI run must still be green before publishing a
production release.

## Recovery evidence

After the eighth candidate boot, both radios were restored to:

`v0.38-plutoplus-spf-gain-rssi-fingerprint-v2-8-gf53d`

Final post-rollback checks:

- both serials returned at physical paths `1-1.1` and `1-1.2`;
- TX1 and TX2 read back at `-80 dB` on both radios;
- the production direct-USB baseline passed 6/6 tests; and
- QSPI was never modified.

## Promotion recommendation

RC11 has passed the hardware candidate gates and the exact tested bytes are
published in the hardware-qualified prerelease above. Do not rebuild or
substitute a different binary during promotion. Pin the tested release tag and
DFU hash, retain the current production image as rollback, then perform the
existing controlled persistent-flash rollout. Protocol-v3 consumers must
continue to interpret ARM observations as sample-bracketed, best-effort
measurements—not sample-exact proof that AGC was stable between observations.
