# Direct-USB capture stability plan

This plan stabilizes the complete PlutoPlus-to-Zarr path used by Rover V7:

```text
AD9361 RX DMA -> Pluto direct-USB gadget -> libusb/Python -> collector -> V7 Zarr
```

It is intentionally incremental. A candidate advances only after the pass
conditions for its current gate are met. Candidate firmware is RAM-loaded
until every hardware and Zarr gate passes; the published production image is
persistent in QSPI and remains the rollback.

Execution status on 2026-08-02:

| Gate | Status | Evidence |
|---|---|---|
| 0 baseline | Pass | Two serial-selected radios; production-size lifecycle, sequence and simultaneous tests |
| 1 Python observability | Pass | Focused host integration is on `main`; SIGINT/SIGTERM/SIGKILL coverage and exact recovery attestation pass |
| 2 allocation/logging candidate | Pass for promoted scope | Runtime status, finite bounded streaming, supervisor evidence and zero-device polling are merged; optional extended counters remain future work |
| 3 RAM candidate | Pass for quick gate | Candidate/control comparison and 50-cycle candidate test pass |
| 4 V7 integrity | Pass for clean and interrupted one-radio paths | Writer lost-update race fixed; 100-frame recovery and two exhaustive 3,500-frame V7 validations pass |
| 5 soak/fault injection | Partial | Dual-radio quick/lifecycle gate and 3/3 supervised SIGKILL recovery per radio pass; real-radio `SIGSTOP`, 20,000-cycle and eight-hour gates remain |
| 6 Rover rollout | In progress | v3 release is published and production manifests are pinned; persistent one-rover and mission acceptance remain |

## Capture-incident hardening candidate (2026-08-02)

The Rover 3 incident exposed failures spanning USB, MAVLink, motion and artifact
ownership. The candidate keeps one incident owner and makes every boundary
fail closed. The software tests below are deliberately focused; the real-radio
`SIGSTOP` row is an explicit hardware gate and is not part of ordinary CI.

| Failure | Red reproduction | Fix | Green / pass condition |
|---|---|---|---|
| Queued bulk transfer misses its deadline | Event context never completes one submitted transfer | Raise `DirectUsbTransferTimeoutError` with pending/requested counts, deadline, serial and physical path; treat it as transport loss | Exact typed error; no metadata/protocol misclassification |
| Recovered USB stream is appended to an existing Zarr | Re-attested replacement returns sequence zero after transport loss | Mark the returned capture as a new stream epoch; SPF rejects it before IQ use | Supervisor may create a new artifact only after complete radio re-attestation |
| Two readers and the collector print competing tracebacks | Inject one receiver failure through a full result queue | Atomically assign one incident ID/source; readers and cleanup emit subordinate one-line records | One owning traceback at process termination; Zarr/status carry incident provenance |
| Ordinary failure hangs Python behind native libusb threads | Leave a non-daemon thread alive after a normal exception | Finalize store/radios, flush the owning traceback, then use a bounded hard exit | Subprocess exits `1` within five seconds |
| Rover continues an already-issued waypoint after capture loss | Fail capture while `move_to_point` is active | Abort planner, request `HOLD`, retry HOLD after a fresh heartbeat, and bound in-process wait | Waypoint loop exits and HOLD is sent; launcher refuses retry unless HOLD succeeds |
| Flight-controller handler crashes without message context | Recreate the observed `NoneType` item-assignment failure in `PARAM_VALUE` | Preserve the original exception as the cause, attach MAVLink message type, mark controller unhealthy | Main capture sees a typed failure and owns the complete chained traceback |
| Completed Zarr is relabeled failed when return-home fails | Final artifact exists, then post-capture navigation raises | Keep artifact/status `complete`; record a separate post-capture incident/exit code | Complete data remains complete while the service still exits non-zero |
| Misleading RC12 ultrasonic chatter with `--no-ultrasonic` | Feed changing RC12 values with no distance-finder object | Ignore RC12 when absent; otherwise apply hysteresis plus three-sample debounce | No phantom logs/actions; real switch changes once after stable samples |
| Rover 3-style shared host stall lacks evidence | Suspend capture longer than both USB/MAVLink deadlines | Independent 1 Hz JSONL watchdog records process/host/USB/storage/capture domains | Hardware `sigstop` test exits within five seconds after resume and leaves a readable incomplete prefix |
| Retry could hide wrong radio, firmware or configuration | First capture fails and attachment changes | Allow at most one new-artifact retry after exact-count discovery, atomic mapping rebuild, signed manifest write/verify and fresh process configuration | Any identity/config/firmware mismatch terminates the service; no frame is retried in-place |

The software-visible `SIGSTOP` reproducer does **not** claim that CPU starvation
was the physical cause on Rover 3. It recreates the observed simultaneous USB
deadline and MAVLink heartbeat loss so cleanup and evidence collection can be
verified deterministically. Distinguishing USB power/reset, host scheduling and
firmware stalls still depends on the independent watchdog plus kernel/voltage
evidence from a real recurrence.

## Current production baseline

- Two complex RX channels, CS16 interleaved.
- 524,288 samples per channel per frame.
- 4,194,304 IQ bytes plus a 96-byte protocol-v2 metadata header per frame.
- 30 MS/s ADC sample rate, 3 MHz RF bandwidth, slow-attack AGC.
- One finite direct-USB START/STOP request per recorded frame.
- V7 records IQ, gain/RSSI endpoint metadata, stream identity, sequence and
  hardware provenance.

The most important current risk is lifecycle churn. Each Python `rx()` creates
a finite one-frame transfer and sends START followed by STOP. The gadget then
creates and destroys its receive thread, local IIO context, DMA buffer, AIO
context and USB buffers. The current gadget reserves sixteen maximum-sized USB
buffers even when the finite request asks for one frame: about 64 MiB for a
4 MiB production payload. This is a testable root-cause hypothesis, not yet a
claim that it explains every observed disconnect.

The initial hardware baseline also exposed a separate host limit: this Pi has
`usbcore.usbfs_memory_mb=16`. Queuing three 4 MiB frames on each of two radios
at once fails at `libusb_submit_transfer()` with `LIBUSB_ERROR_NO_MEM` (about
25 MiB requested). Production queues one frame per radio (about 8 MiB total)
and passes. Tests therefore keep multi-frame sequence validation per radio and
use the real one-frame depth for simultaneous-radio validation. Any future
increase in queued depth must either raise/document the usbfs limit or reduce
the framed transfer size; it must not be mistaken for a gadget failure.

## Gate 0 - Freeze identity and reproduce the baseline

Actions:

1. Record USB serial, physical port path, firmware SHA, gadget SHA and protocol
   capabilities for every attached radio.
2. Run 20 repeated production-sized one-frame requests per radio.
3. Run one multi-frame request per radio and one simultaneous dual-radio test.
4. Record elapsed time, host RSS growth, USB errors and kernel log delta.

Pass:

- The explicitly expected radio count is present and serials are unique.
- Every frame has the exact byte count, shape `(2, 524288)`, `complex64` IQ,
  valid CRC, gain metadata, RSSI metadata and non-zero channels.
- Multi-frame sequences are contiguous within one stream.
- No USB disconnect, kernel error or unbounded host-memory growth occurs.

Fail:

- Missing/extra radio, ambiguous identity, invalid metadata, short transfer,
  sequence discontinuity, all-zero channel, timeout, USB reset or process RSS
  growth beyond the declared bound.

Output: a timestamped JSON baseline report. A failure is preserved verbatim;
it is not hidden by retrying a frame.

## Gate 1 - Make failures observable and fail cleanly in Python

Actions:

1. Preserve the original libusb/protocol exception through `ThreadedRX` and
   the collector instead of returning `None` and producing a secondary
   `TypeError`.
2. Mark the Zarr `incomplete`, store the primary error type/message/errno and
   the number of completely written records per receiver.
3. Always STOP, release the claimed interface, close libusb and mute TX during
   cleanup; preserve cleanup errors separately from the primary error.
4. Do not retry a failed direct-USB frame inside the same Zarr capture.

Pass:

- Focused red/green tests inject a USB failure and assert that the original
  error reaches the collector and incomplete-store attributes.
- Cleanup runs exactly once and a subsequent process can immediately reopen
  the same radio.

Executed evidence:

- Synthetic subprocess SIGTERM produces a readable `incomplete` LMDB-Zarr
  with `CaptureInterrupted`; SIGKILL leaves `in_progress`, never a falsely
  complete store. Both retain only monotonically safe committed-record counts.
- The real production-size Rover 2 V7 collector was stopped after two complete
  records. It exited with status 143 in 21 seconds end to end, left no final
  `.zarr`, and a fresh direct-USB request immediately reclaimed the same serial.
- A two-radio attempt exposed a real host-kernel USB disconnect on physical
  path `1-1.1`. That radio did not re-enumerate after a hub-port cycle because
  it is externally powered; this is a hardware recovery result, not a passing
  dual-radio interruption gate. Repeat the exact-count-two gate after an
  external power cycle.

Fail:

- The failure becomes `None`, `TypeError`, a hang, a falsely complete Zarr or
  an interface that remains claimed.

## Gate 2 - Reduce gadget allocation and logging risk

First candidate firmware changes:

1. Allocate only the number of USB buffers needed by a finite request, bounded
   by the existing maximum. A one-frame request must not allocate sixteen
   production frames.
2. Default gadget debug logging to off. Enabling it remains an explicit
   diagnostic action, and any persistent log must be bounded.
3. Convert receive-thread exits to one cleanup path so partially initialized
   IIO, AIO, epoll and USB resources are released deterministically.
4. Add counters/status for START, STOP, completed frames, short writes, IIO
   refill errors, gain/RSSI read failures, overflow and last error.
5. Bound STOP/join behavior; a stuck worker must produce a visible failure
   rather than hanging the USB control endpoint indefinitely.

Pass:

- Native serialization/unit tests and cross-build pass.
- For a one-frame production request, measured peak gadget allocation drops
  by approximately fifteen frame buffers without throughput regression.
- Forced failures release all resources and the next START succeeds.
- Metadata bytes and IQ payload layout remain wire-compatible.

Fail:

- Protocol/layout change without negotiation, lower sustained frame rate,
  leaked resources, unbounded STOP, or loss of standard IIO configuration.

## Gate 3 - RAM-boot candidate and compare against baseline

Actions:

1. Verify DFU recovery and record the known-good image hash.
2. RAM-load the candidate on one radio only; leave the second radio as the
   control.
3. Repeat Gate 0 on candidate and control.
4. Swap roles or RAM-load both only after the first comparison passes.

Pass:

- Candidate meets every Gate 0 requirement and is no slower by more than 5%.
- Standard USB IIO can still configure LO, sample rate, bandwidth, gain mode
  and channel mask before direct streaming.
- Power cycle returns to the known-good persistent image.

Fail:

- Enumeration/layout changes, configuration loss, throughput regression,
  metadata mismatch, recovery failure or new kernel errors.

## Gate 4 - V7 Zarr integrity

Actions:

1. Capture 10 frames per receiver with the production YAML and fake drone.
2. Reopen the LMDB-backed Zarr in a fresh process.
3. Validate schema, dimensions, dtypes, serial/USB/firmware provenance,
   metadata validity, finite IQ, per-receiver record count and capture state.
4. Repeat at 100 frames per receiver.
5. Interrupt a separate capture deliberately and validate `incomplete` plus
   its error/progress attributes.
6. Run the automated interruption matrix at multiple durable-record boundaries:
   graceful `SIGINT`, graceful `SIGTERM`, and uncatchable `SIGKILL`.
7. After every interruption, claim each serial for a fresh direct-USB frame;
   after the complete matrix, write and strictly validate a clean 100-record
   production V7 Zarr.

Pass:

- A clean run is marked complete and all expected records round-trip.
- Serial and firmware identity match the radio that produced each receiver
  group.
- An interrupted/failed run is readable but never marked complete.
- Graceful signals exit with their conventional status and record
  `CaptureInterrupted`; `SIGKILL` remains `in_progress` with no invented error
  or completion state.
- Every safely committed prefix has monotonic timestamps and valid gain/RSSI
  metadata, and no interruption strands a vendor interface.

Fail:

- Missing fields, swapped serials/channels, unwritten records accepted as
  valid, invalid metadata silently accepted, or a corrupt/unopenable store.

## Gate 5 - Soak and fault injection

Run only after the quick gates pass:

- 20,000 one-frame lifecycle cycles per radio.
- Eight-hour simultaneous dual-radio V7 capture.
- Controlled cable disconnect/reconnect between captures.
- Gadget process termination, short transfer, IIO refill error and writer
  backpressure injections in a development image/test harness.
- Configuration-attestation mismatch before any attempted recovery.

Pass:

- Zero silent corruption and zero falsely complete captures.
- Every loss is counted and associated with the affected stream/receiver.
- Host and Pluto memory reach a plateau.
- A post-soak quick test passes without rebooting either endpoint.

Fail:

- Memory growth, unexplained sequence gap, stale radio configuration,
  cross-radio identity swap, recovery into the same Zarr after uncertainty,
  or a required hard power cycle.

## Gate 6 - Rover rollout

Roll out one rover at a time. Run quick hardware tests, a 100-frame fake-drone
capture and one normal mission rehearsal before advancing. Preserve hashes and
reports in the field report. Roll back immediately on any earlier gate failure.

## Hardware pytest interface

Hardware tests live under `tests/radio_hardware/` and are never enabled by an
ordinary pytest invocation.

Quick two-radio gate:

```bash
pytest tests/radio_hardware \
  --radio-hardware \
  --radio-expected-count=2 \
  --radio-samples=524288 \
  --radio-cycles=20
```

Select exact radios when needed:

```bash
pytest tests/radio_hardware \
  --radio-hardware \
  --radio-expected-count=2 \
  --radio-serial=SERIAL_A \
  --radio-serial=SERIAL_B
```

The longer soak and V7 artifact gates use separate explicit options documented
next to their tests. CI should collect these tests but skip them unless a
dedicated radio runner opts in.

Run the bounded production interruption campaign with:

```bash
data_collection/rover/rover_v3.1/run_interrupted_capture_campaign.sh
```

It performs `sigterm:2`, `sigint:10`, `sigkill:25`, and `sigterm:100` by
default. Every partial artifact, pytest result, and kernel-log delta is kept
under `/home/pi/preflight/interrupted_capture`; a final `PASS` marker is
written only after the post-interruption production capture validates. The
matrix can be overridden without editing code through `SPF_INTERRUPT_CASES`.

Run the restart-separated duration gate with:

```bash
data_collection/rover/rover_v3.1/run_direct_usb_restart_soak.sh
```

The default requires at least two independently fingerprinted sessions and at
least 3,600 seconds of aggregate recorded time. It restarts every configured
radio between sessions, re-attests identity/configuration/firmware, records
host-resource samples and kernel-log deltas, validates each Zarr, and stops
fail-closed on any mismatch.

## Decision order

1. Establish a reproducible baseline before changing firmware.
2. Fix exception provenance before fault injection.
3. Reduce one-frame gadget allocation and logging risk before redesigning the
   transport lifecycle.
4. Consider a persistent stream/pool only after the smaller change has been
   measured. It may improve stability and throughput, but it is a larger
   protocol/lifecycle change and must not be mixed into the first experiment.
