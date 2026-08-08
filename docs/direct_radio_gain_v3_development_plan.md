# Direct-radio gain series v3: development and test plan

Status: active development plan; protocol v2 remains the production default.

## Outcome

Deliver the same versioned RX frame through direct USB and direct IP. Each
frame contains dual-channel IQ plus a bounded series of sample-associated
RX1/RX2 gain observations. SPF must decode the same inner bytes and write the
same V7 fields regardless of transport.

The first implementation uses local ARM register reads. It improves temporal
resolution but remains interval-associated rather than sample-exact. FPGA
CTRL_OUT event capture is a later, independent promotion step.

## Fixture

Both local radios use the same isolated loopback:

```text
TX2 -> approximately 30 dB attenuation -> two-way splitter -> RX1 and RX2
```

The radios are identified only by serial number:

| Fixture | Serial | USB port path | Additional IP path |
| --- | --- | --- | --- |
| A (`.17` historical label) | `104000bac4950008230026001b440a003a` | `1.1` | USB network; duplicate `192.168.2.1` address |
| B (`.18` historical label) | `1040007c4a94000211000b009186843ef2` | `1.2` | `192.168.1.174` |

Only one TX2 may be active at a time. Never connect a transmitter directly to
an RX input. The two USB-network functions both use `192.168.2.1`, so they are
not a valid two-radio IP test without separate network namespaces. Use the
unique `192.168.1.174` path for the first direct-IP hardware gate.

## Architecture decisions

1. Keep pyadi/libiio as the radio configuration and diagnostics plane.
2. Do not extend the standard IIO sample ABI for the first delivery. Doing so
   would require kernel scan elements, DMA/HDL changes, and pyadi compatibility
   work while still not solving UDP framing.
3. Produce one transport-neutral v3 frame on the Pluto ARM:

   ```text
   v3 metadata prefix + gain-observation records + reserved event records + IQ
   ```

4. Send that frame unchanged:
   - direct USB: one complete frame per bulk transfer;
   - direct IP: versioned UDP fragments whose 32-bit offsets and counts can
     represent the largest supported frame.
5. Protect the complete metadata-plus-IQ frame with an outer direct-IP CRC.
   Never expose partial, corrupt, or expired frames.
6. Use one capture transport at a time per radio. IIO configuration finishes
   before direct streaming starts.
7. Preserve protocol v2 decoding and production configs until every v3 gate
   below passes.

## Initial observation policy

- Default requested interval: `32768` samples.
- At 30 MS/s this requests one observation every 1.092 ms and nominally gives
  16 observations in a `2**19`-sample frame.
- The observed local paired register-read duration is roughly 0.49 ms median
  and 0.54 ms p99. Every record therefore carries the FPGA sample counter from
  immediately before and after the read.
- A request for `2048` samples is legal but does not make the ARM/SPI reads
  complete every 68.3 microseconds. Measured counter intervals remain the
  truth.
- Endpoint equality or equal adjacent observations means only that no
  difference was observed. It is not proof that AGC did not change and return.

## Stepwise delivery gates

### Gate 0 - preserve the v2 baseline

Run receive-only capture on both radios with small and `2**19`-sample frames,
then reopen a short V7 store.

Pass:

- both serials enumerate over IIO and direct USB;
- concurrent sequences are continuous;
- two complex channels have the expected shape and order;
- the V7 store reopens with valid v2 gain/RSSI metadata; and
- no queue, protocol, or USB error occurs.

Fail action: stop. Fix or roll back the existing setup before changing firmware.

### Gate 1 - qualify each RF fixture

For each serial, run the committed TX2-off/TX2-on probe sequentially at an
established LO. Verify the peak at the configured `+100 kHz` offset.

Pass:

- both RX channels exceed the declared TX-on/off threshold;
- tone frequency, SNR, clipping, coherence, and phase-stability checks pass;
- the other radio remains muted.

Fail action: do not use the fixture for phase or gain conclusions. Power-cycle
and re-probe because these radios have a documented intermittent FPGA-DDS arm
state. Receive-only transport work may continue.

### Gate 2 - freeze the common v3 inner frame

Complete the Python and C golden-vector definitions for the fixed prefix,
bounded observation records, sentinels, sizes, flags, CRC, and sequence rules.

Pass:

- Python and C emit byte-identical golden frames;
- v1/v2 parsing remains unchanged;
- unsupported versions and malformed sizes fail closed;
- zero, one, full-capacity, and overflow observation cases are covered.

Fail action: do not implement a transport-specific workaround or second schema.

### Gate 3 - complete the direct-IP envelope and control negotiation

Use the `SIP1` outer fragment envelope around the exact inner frame. Add a
versioned capability query plus acknowledged START/STOP request containing the
inner protocol version, requested features, observation interval/capacity,
samples per channel, scan mask, host data port, and maximum datagram size.

Pass:

- a production-sized frame reassembles after packet reordering and duplicates;
- missing, overlapping, conflicting, corrupt, stale, or oversized fragments
  discard the whole frame and increment explicit counters;
- retrying a control request is idempotent;
- the old 8-bit packet-count protocol cannot accidentally parse as `SIP1`.

Fail action: no IP hardware deployment.

### Gate 4 - create one native frame producer

Refactor the gadget so IIO refill, sequence assignment, endpoint snapshots,
observation attachment, and inner-frame serialization are shared. USB and IP
own only their output adapters. Allocate all steady-state buffers before the
capture loop.

Pass:

- focused native tests pass under normal and sanitizer builds;
- USB and IP output identical inner-frame fixture hashes;
- one failed gain read marks only the record invalid and does not shift IQ;
- overflow and dropped-frame counters are explicit.

Fail action: keep the candidate off the radios.

### Gate 5 - attach CPU-side observations

Run one bounded observation worker per active RX stream. Associate reads with a
frame only when their before/after sample-counter interval overlaps that
frame's half-open IQ interval.

Pass:

- counters are monotonic and records are ordered;
- every accepted record overlaps its owning frame;
- manual equal gains report correct RX1/RX2 indices and dB values;
- requested cadence, achieved cadence, read duration, invalid reads, and
  capacity overflows are measured and stored;
- IQ streaming continues when an individual metadata read fails.

Fail action: mark v3 unavailable and continue using v2.

### Gate 6 - integrate the common Python result and V7 writer

Both receivers return the same `PlutoRxBuffer` fields. The collector must not
care whether the bytes arrived over USB or IP.

Pass:

- synthetic USB and IP frames yield equal NumPy IQ and metadata;
- V7 writes/reopens counts, padded arrays, flags, intervals, serial, firmware,
  transport, and protocol provenance;
- legacy V7/v2 files remain readable;
- malformed metadata cannot be silently replaced with endpoint values.

Fail action: do not select v3 in a capture config.

### Gate 7 - RAM-boot hardware acceptance

Test the candidate from RAM first. For each radio run USB, then stop it and run
IP. Finally run both radios concurrently over USB. Never stream the same radio
through both transports simultaneously.

Pass:

- standard IIO configuration still works before and after each direct stream;
- USB and IP IQ/tone/phase statistics agree within measured repeatability;
- 100 production frames per radio have continuous sequences and no drops;
- repeated START/STOP and recovery tests leave TX muted and DMA reusable;
- short V7 captures reopen and contain the declared observation series.

Fail action: reboot to the known-good persistent image and retain v2 configs.

### Gate 8 - soak and promotion

Run one-hour single-radio USB and IP soaks, then a one-hour concurrent two-radio
USB soak. Record CPU, memory, throughput, queue depth, fragment loss, register
read duration, observation density, and radio temperature.

Pass:

- no unexplained loss or unbounded memory growth;
- all induced loss is detected and counted;
- RF loopback phase remains consistent with the v2 baseline;
- rollback has been demonstrated.

Fail action: keep v3 experimental. Persistent flashing and rover config changes
are prohibited until this gate passes.

## Focused test matrix

| Layer | Tests run locally | Hardware required |
| --- | --- | --- |
| Inner protocol | golden bytes, sizes, CRC, sentinels, observation ordering/overflow, v1/v2 compatibility | No |
| IP envelope | reorder, duplicate, loss, timeout, overlap, conflict, full-frame CRC, memory bounds, ~4 MiB frame | No |
| Native gadget | serializer parity, allocation/error paths, fake IIO/gain source, sanitizer build | No |
| Python integration | common USB/IP result, sequence reset rules, bounded queue, synthetic socket | No |
| V7 | write/reopen, padding/count consistency, old-file compatibility, provenance | No |
| Receive hardware | small and `2**19` frames, both serials, repeated START/STOP | Yes |
| RF hardware | TX2 off/on, expected tone, channel order, phase/coherence | Yes |
| Transport parity | sequential USB/IP capture of the same fixture and configuration | Yes |
| Endurance | 100 frames, one-hour single transport, one-hour dual USB | Yes |

Run only focused tests locally. The complete repository suite remains a CI job.

## Deferred HDL gate

The ARM implementation cannot guarantee detection of every fast-attack AGC
transition. When the x86-64 HDL build server is available, add coherent sample
counter CDC validation and CTRL_OUT index-8 event capture as a separate change.
That extension may add exact in-buffer change events to the reserved v3 event
records; it must not change the USB/IP transport contract.

## Current evidence snapshot (2026-08-08)

- Protocol-v2 concurrent direct-USB capture passed on both radios with small
  and `2**19`-sample frames.
- A short production-sized V7 store passed write/reopen validation.
- The existing direct-IP gadget returned dual-channel IQ through
  `192.168.1.174`; this proves the legacy transport only, not v3 metadata.
- Synthetic `SIP1` reassembly passes production-size, reorder, duplicate, loss,
  corruption, overlap, and resource-bound tests.
- The RF probe currently fails on both fixtures because TX2/DDS is silent. The
  same serials and fixtures previously passed at approximately 67-72 dB
  TX-on/off separation, so a clean power-cycle and re-probe is required before
  RF acceptance resumes.
