# Gain-series v4 RC17 extended burn-in — 2026-08-10

## Disposition

**PASS for the extended single-boot lifecycle, frequency, transport, RF-state,
and V7 storage campaign.** The principal passing gates validated 6,848 IQ
frames carrying 21.69 GiB of IQ payload without an RC17 ownership, lifecycle,
metadata, sequence, reassembly, or storage failure.

This extends, but does not replace, the RC17 promotion report. It does not
claim completion of the one-hour and injected-process-failure requirements in
Gate 11 of the state-machine test plan. QSPI was not modified. Both radios were
left online on the volatile RC17 image with TX1/TX2 verified at `-80 dB`.

## Candidate and fixture

- Release: [v0.38-plutoplus-spf-gain-series-v4-rc17](https://github.com/misko/plutosdr-fw/releases/tag/v0.38-plutoplus-spf-gain-series-v4-rc17)
- Firmware commit: `1f3fe0cbe865df0a8793e0fd0096368d02d28a14`
- DFU SHA-256: `88a606f1a19f493e031989b8fc76cc77644ae5473e5d627b850252c9a615c54e`
- USB gadget: `2e8e40ade5dcf3c7880a5ebb58419ad7c37ed552`
- Direct-IP gadget: `b066059e54817ad9a140c3549fcee0bf39dadc81`
- Radio A: `104000bac4950008230026001b440a003a`, USB `1-1.1`, LAN `192.168.1.165`
- Radio B: `1040007c4a94000211000b009186843ef2`, USB `1-1.2`, LAN `192.168.1.175`

TX2 on each radio was connected through the declared 30 dB attenuated splitter
to that radio's RX1 and RX2. Tests enabled one cabled tone per radio and the
safety wrappers muted both transmitters on every exit.

## Passing gates

| Gate | Scope | Result |
|---|---:|---|
| Fresh direct-USB lifecycle | 500 starts/radio, 3 maximum frames/start | 1,000/1,000 streams and 3,000/3,000 frames passed |
| USB payload | 524,288 samples/channel, dual CS16 | 11.72 GiB passed in 24m36s |
| Mixed LO/transport/gain burn | 12 LOs x 3 shuffled epochs | 36/36 cells passed in 11m00s |
| Mixed transport sessions | Three sessions/cell on both radios | 216/216 radio sessions passed |
| Mixed frame payload | Eight 131,072-sample frames/session | 1,728/1,728 frames, 1.69 GiB passed |
| LO changes | Both radios at every cell | 72/72 retunes passed |
| Gain-mode changes | manual 26 -> slow attack -> manual 41 | 216/216 radio-state assignments passed |
| Parallel direct-IP ladder | 1, 1.25, 1.5, 3, 10, 20, 30 MS/s | 70/70 paired cycles passed in 6m32s |
| Parallel IP frames | 10 cycles/rate, 8 maximum frames/radio | 1,120/1,120 frames, 4.38 GiB passed |
| Post-USB parallel IP return | 3 MS/s, 8 frames/radio | Passed |
| Production V7 write/reopen | 500 maximum frames/radio | 1,000/1,000 records passed in 17m29s |
| V7 IQ payload | 524,288 samples/channel, dual CS16 wire payload | 3.91 GiB wire / 7.81 GiB complex64 array passed |

The principal gates above moved 21.69 GiB over 6,848 fully validated frames.
Additional diagnostic USB and final handoff frames are excluded from that
total.

## Frequency and RF-state coverage

The mixed runner used a deterministic shuffle per epoch, so frequencies were
not repeated immediately and every revisit followed a different transition
history:

`868, 915, 1280, 1300, 1301, 2412, 2467.1, 4000, 4001, 5766, 5804, 5866 MHz`

The 1300/1301 and 4000/4001 MHz pairs deliberately cross the AD936x gain-table
band boundaries. Every frequency cell alternated either:

`USB -> IP -> USB`

or:

`IP -> USB -> IP`

while changing gain control through:

`manual 26 dB -> slow_attack -> manual 41 dB`.

Across all 1,728 mixed frames:

- minimum two-channel tone SNR: `12.966 dB`;
- minimum coherence: `0.997843`;
- maximum within-frame phase standard deviation: `0.3248 degrees`;
- duplicate, expired, rejected, and receive-queue-overflow counts: all zero;
- gain/RSSI metadata, observation-series validity, CRC, frame sequence, and
  sample sequence: all passed.

## Direct-IP result

The extended ladder performed ten fresh simultaneous requests at each of seven
rates. All 140 radio sessions passed. Kernel UDP `InErrors`, `RcvbufErrors`,
`SndbufErrors`, `InCsumErrors`, and `MemErrors` were zero across the campaign.
Application duplicate, expired, rejected, and queue-overflow counts were also
zero. There was no integrity failure and no control-rearm failure.

This remains a bounded finite-capture result. The earlier measured continuous
two-radio drain limit of approximately 1.25 MS/s per radio is unchanged.

## V7 result

The V7 test created one fresh protocol-v3 stream per record, wrote every IQ
frame and gain observation, closed the LMDB store, reopened it, and compared
the stored arrays against the captured values.

| Receiver | Records | Gain observations/frame | Maximum time uncertainty | Unique stream IDs |
|---|---:|---:|---:|---:|
| Radio A | 500 | 160-163 | 0.686 ms | 500 |
| Radio B | 500 | 161-164 | 0.527 ms | 500 |

The final store reported `capture_status=complete`, progress `[500, 500]`,
protocol v3, correct serials and USB paths, monotonically increasing sample
counters, valid sample times, and nonzero IQ for every record. The complex64
signal arrays have 7.81 GiB of logical data; the sparse LMDB store occupied
797 MiB of allocated disk blocks on this fixture.

## Limits deliberately exposed

### Shared usbfs transfer memory

The first mixed attempt queued eight 1 MiB transfers on each radio at once.
Transfer overhead pushed the pair beyond Linux's common 16 MiB usbfs allocation
and libusb returned `LIBUSB_ERROR_NO_MEM`. This occurred before firmware
lifecycle churn and both radios remained present. The committed runner now
uses one queued USB transfer per radio while retaining one contiguous eight-
frame firmware stream. The full rerun passed.

### Simultaneous USB host-time anchors

Sequential post-burn timing remained precise:

- Radio A: `0.481 ms` uncertainty;
- Radio B: `0.546 ms` uncertainty.

When both radios streamed bulk USB concurrently through the shared parent,
EP0 anchor round trips were nondeterministic. Three-frame runs observed maxima
of `5.137 ms` and `13.836 ms`; 16-frame diagnostic runs reached `22.821 ms`
and, in a separate run, `39.895 ms`. IQ frames and FPGA sample counters stayed
valid. The limitation is host-realtime correlation under shared-bus bulk
contention, not relative sample sequencing.

The 5 ms production gate was not relaxed. A diagnostic rerun used a 100 ms
ceiling only to preserve the measurements above. Future work should schedule
anchor requests outside competing bulk intervals or add a better on-device
time-transfer mechanism.

### Unsustainable 30 MS/s direct USB stream

At 30 MS/s, one dual-CS16 radio produces 240 MB/s; two produce 480 MB/s, well
beyond USB 2.0. A simultaneous 16-frame diagnostic detected an exact
524,288-sample gap and failed closed. This is the expected physical throughput
boundary and confirms the parser does not silently accept loss. Finite direct-
IP capture remains correct at 30 MS/s because capture and network drain are
decoupled.

### Protocol finite-frame bound

A deliberately oversized 100-frame USB request was rejected before START
because the negotiated protocol maximum is 16. Longer tests must use repeated
bounded requests, as the passing 500-start campaign does.

## Evidence

- USB 500x3 report SHA-256: `7264478b2c68a250a6cd925ed60f29016248caa18f8049a536e7dd3b5be3621e`
- Mixed frequency report SHA-256: `3dc6b8400162fee6b52c42fcc9deb409bcba36585bbf085e2f53c1e146de90ce`
- Extended IP ladder SHA-256: `f82f3bf3575ae8fa1cd75c8ce78fb8cb69cc92e56ff01200fa2c1acfbbe60272`
- Sequential timing report SHA-256: `c1a5e587b670eb22bb6dbeda505ba2676aea4d96772d310c1b89f4e4d46630f1`
- Final USB-to-IP report SHA-256: `6a5f296b28676a5332d13197d801fc147048ed9454773d55d32ec51e102e8905`
- V7 pytest log SHA-256: `b427f042bdf0c19970d7903828d9f9e4c42156ed5d25f2802aa2c962c213c785`

Campaign root: `/tmp/spf-rc17-extended-burn`

## Remaining promotion work

RC17's state-machine test plan still calls for:

1. a one-hour two-radio direct-IP soak;
2. deliberate host interruption during START, capture, drain, and STOP;
3. deliberate gadget-process restart and supervisor recovery;
4. a bounded on-device cleanup watchdog test; and
5. a persistent-QSPI canary followed by power-cycle and repeat gates before
   changing rover firmware pins.
