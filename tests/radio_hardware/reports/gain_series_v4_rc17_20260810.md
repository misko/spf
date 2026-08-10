# Gain-series v4 RC17 hardware gate — 2026-08-10

## Disposition

**PASS for volatile two-radio qualification. RC17 fixes the direct-IP control
lifecycle failure found after RC16 and passes USB, IP, TX2 loopback, repeated
restart, protocol-v2 compatibility, and production V7 Zarr gates. Publish it
as a prerelease; do not promote it to rover QSPI from this report alone.**

No QSPI partition was modified during this campaign. Both radios were left
running the candidate from RAM, with TX1 and TX2 explicitly muted at `-80 dB`.

## Candidate identity

- Firmware commit: `1f3fe0cbe865df0a8793e0fd0096368d02d28a14`
- USB gadget commit: `2e8e40ade5dcf3c7880a5ebb58419ad7c37ed552`
- Direct-IP gadget commit: `b066059e54817ad9a140c3549fcee0bf39dadc81`
- Buildroot commit: `56b7bc54d47a16c55e9cb71ed519544892ac63db`
- GitHub Actions run: [31419302756](https://github.com/misko/plutosdr-fw/actions/runs/31419302756)
- Artifact: `plutoplus-main-1f3fe0cbe865df0a8793e0fd0096368d02d28a14-31419302756-1`
- Bundle: `plutoplus-spf-main-1f3fe0cbe865.tar.gz`
- DFU: `plutoplus-spf-main-1f3fe0cbe865-pluto.dfu`
- DFU SHA-256: `88a606f1a19f493e031989b8fc76cc77644ae5473e5d627b850252c9a615c54e`
- Rootfs SHA-256: `be9ac420f628551066ac9c4f437bf60a70a532c4722e3be5d71f7cb7aca3d6df`
- XSA SHA-256: `a8b0dc57bec281b9652c55a3e58750527322a9c4ec828e04a60aa1c4706f3758`
- Packaged device-fw: `v0.38-plutoplus-spf-gain-series-v4-rc16-7-g1f3fe`

The adjacent bundle checksum and every enclosed `SHA256SUMS` entry passed.
GitHub's independent build-verification and artifact-attestation jobs passed.
Routed timing passed with setup WNS `+0.504 ns`, hold WHS `+0.014 ns`, and no
failing setup, hold, or pulse-width endpoints.

## Defect fixed

RC16's direct-IP control thread performed slow IIO setup and teardown while
servicing UDP control requests. At low sample rates, cleanup could remain in
progress long enough for the next `START_RX` to time out. Retry semantics also
did not distinguish a duplicate request from a new request safely.

RC17 introduces an explicit nonblocking lifecycle:

`IDLE -> STARTING -> ARMED -> RUNNING -> STOPPING -> REAPABLE -> IDLE`

Slow IIO work runs outside the control loop. START is acknowledged only after
the worker is ready, STOP only after ownership has been released. A bounded
peer-scoped replay window coalesces exact duplicates, rejects request-ID
collisions and stale requests, and retains a completed-stream tombstone.
Legacy RX START is refused while protocol v3 owns or releases the DMA path.

The full design and red/green invariants are documented in
[`../../../docs/direct_ip_firmware_state_machine.md`](../../../docs/direct_ip_firmware_state_machine.md)
and
[`../../../docs/direct_ip_firmware_state_machine_test_plan.md`](../../../docs/direct_ip_firmware_state_machine_test_plan.md).

## Bench fixture

TX2 on each radio was connected through the declared 30 dB attenuated splitter
to that same radio's RX1 and RX2. The test activates one TX2 at a time and
verifies both transmitters are muted after every stage.

| Radio serial | USB physical path | Candidate transports |
|---|---|---|
| `104000bac4950008230026001b440a003a` | `1-1.1` | USB IIO, direct USB, direct IP |
| `1040007c4a94000211000b009186843ef2` | `1-1.2` | USB IIO, direct USB, direct IP |

The LAN DHCP addresses changed across volatile boots. Every test resolved the
current address from the radio serial rather than trusting a stale IP address.

## Promotion results

Campaign root: `/tmp/spf-rc17-final.keUTR6/promotion`

| Gate | Result |
|---|---|
| Source graph and native state-machine tests | Passed; 15 USB and 4 IP tests |
| Persistent AD9361/2R2T configuration | Both serials passed |
| Independent RAM loads | 3/3 epochs on both radios |
| Immediate and post-test TX mute | Both radios, every epoch |
| Internal cyclic TX and external TX2 loopback | 2 radios x 3 epochs passed |
| Protocol-v2 compatibility at 524,288 samples | 6 passed |
| Protocol-v3 USB and simultaneous USB | Passed on both radios |
| Fresh USB START/STOP stress | 100 streams per radio passed |
| Production V7 Zarr | 100 records/radio written and reopened |
| Malformed direct-IP datagrams | Survived; next valid request passed |
| Protocol-v3 direct-IP metadata frame | Passed |
| Buffered direct-IP burst | 20 cycles x 16 frames passed |
| Final device count and direct-USB presence | 2/2 passed |
| QSPI writes | None |

All six radio/epoch TX captures passed tone, SNR, clipping, coherence, phase
stability, manual-gain metadata, slow-attack response, and mute checks.

Key evidence SHA-256 values:

- repeated USB starts: `823c7ca3b764c3569635c0d56207af739573e27ab1b392eeb9c0a1b9792a55c0`;
- simultaneous USB: `157e1d4085c69de4cd5ccef6a9acfb8c849a3c1a4e7b0212cf74b159b935ad33`;
- malformed direct IP: `0bae317f667521cd7d8aeb1f4f5a8450e837802452ec30948a8ff1dceeb011a2`;
- direct-IP frame: `c0344549574d7dfc5964e387702373d96b235cb1a1a677f5f938b355f1f47b80`;
- buffered direct-IP burst: `dfda25925961e7892f491ef16bc3b30b33916a3c61bb2742269153cfa3b78406`.

## Parallel direct-IP lifecycle and rate ladder

The low-rate regression gate ran ten fresh, simultaneous requests per rate on
both radios. Every one of the 120 radio sessions passed at 1, 1.25, 1.5, 2,
2.5, and 3 MS/s. This directly closes RC16's low-rate START/cleanup failure.
The report SHA-256 is
`2abc8205dac884499decac349afd5a340e6b1a6853c468a89860eb67dc652c56`.

The wider ladder ran three simultaneous requests per radio at each rate from
1 through 30 MS/s. All 66 radio sessions passed frame integrity and lifecycle
checks with no application-level duplicate fragments, rejected fragments,
expired frames, sequence faults, or receive-queue overflows. Its report
SHA-256 is
`71ff01f5746c11671a65ab0666e423fbd14d05e6f0f60a80cea8834919ff1a93`.

| Sample rate | Minimum estimated drain, radio A/B | Minimum realtime headroom, radio A/B |
|---:|---:|---:|
| 1.00 MS/s | 10.04 / 11.09 MiB/s | 1.316 / 1.454 |
| 1.25 MS/s | 10.38 / 10.33 MiB/s | 1.089 / 1.083 |
| 1.50 MS/s | 10.72 / 10.26 MiB/s | 0.937 / 0.897 |
| 3.00 MS/s | 10.42 / 10.87 MiB/s | 0.455 / 0.475 |
| 10.0 MS/s | 12.21 / 10.72 MiB/s | 0.160 / 0.140 |
| 20.0 MS/s | 11.69 / 11.19 MiB/s | 0.077 / 0.073 |
| 30.0 MS/s | 11.36 / 10.21 MiB/s | 0.050 / 0.045 |

The distinction is important: finite buffered requests remain correct through
30 MS/s, but two radios on this host can drain continuously in real time only
through about 1.25 MS/s each. At 1.5 MS/s and above, acquisition is faster than
network drain, so the bounded capture-first design adds idle time between
finite requests rather than dropping or silently corrupting frames.

The first ladder attempt incorrectly programmed raw IIO sample rate directly,
which the AD9361 rejects below 2.083333 MS/s without FIR. The production pyadi
setter enables the FIR path for low rates. The ladder now uses that same pyadi
configuration path; focused tests pass and the corrected live gate above
starts at 1 MS/s.

## Residual risks and next gates

An additional same-boot campaign subsequently passed 6,848 principal frames
and 21.69 GiB across long USB lifecycle churn, 12-frequency USB/IP switching,
manual/slow-attack gain changes, a longer parallel-IP ladder, and a
500-record-per-radio V7 reopen test. It also characterized shared-hub EP0
time-anchor contention and the expected 30 MS/s USB throughput boundary. See
[`gain_series_v4_rc17_extended_burn_20260810.md`](gain_series_v4_rc17_extended_burn_20260810.md).

- Add a bounded cleanup watchdog and supervisor recovery for a kernel IIO
  teardown that never returns.
- Add a wire-level lifecycle/diagnostic query rather than relying only on host
  inference and device logs.
- Consider a client-session nonce in a future protocol revision to make replay
  identity survive arbitrary client request-ID resets.
- Before production deployment, perform a controlled persistent-QSPI canary,
  power-cycle it, repeat the selected USB/IP/Zarr gates, then update rover
  firmware pins in a separate reviewed change.
