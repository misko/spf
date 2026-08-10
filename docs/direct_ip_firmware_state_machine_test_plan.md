# Direct-IP firmware state-machine test plan

Status: RC17 candidate test and promotion plan for
[`direct_ip_firmware_state_machine.md`](direct_ip_firmware_state_machine.md).

The goal is not merely to make the observed low-rate test pass. The tests must
prove lifecycle safety under delay, loss, duplication, cancellation, process
shutdown, and resource failure without weakening IQ or metadata integrity.

Run focused tests locally. Repository-wide tests remain a CI responsibility.
Hardware candidates are RAM-booted until every gate passes.

The RC17 source currently passes the four focused native gadget tests and the
SPF stale-control-response regression. Those are implementation gates, not
hardware promotion: the single- and two-radio sections below remain required.

## Current red case

Fixture:

```text
two Pluto+ radios
unique Gigabit Ethernet addresses
dual RX CS16
524,288 samples per channel per frame
four frames per finite request
protocol v3 gain series
```

Observed on RC16:

| Rate | Repeated request result | IQ/UDP integrity |
| ---: | --- | --- |
| 1.0 MS/s | Next START failed after one clean capture | Clean |
| 1.25 MS/s | Third START failed after two clean captures | Clean |
| 1.5 MS/s | Third START failed after two clean captures | Clean |
| 2.0 MS/s | Third START failed after two clean captures | Clean |
| 2.5 MS/s | Five of five requests passed | Clean |
| 3.0 MS/s | Five of five requests passed | Clean |
| 6–30 MS/s | Bounded requests passed | Clean |

Pass for the fix: ten consecutive finite START/STOP cycles on both radios at
every rate from 1 through 3 MS/s, with no control failure, duplicated side
effect, sequence gap, partial frame, or leaked RX owner.

## Test layers

| Layer | Purpose | Hardware |
| --- | --- | --- |
| Pure transition model | Prove all state/event pairs have explicit behavior | No |
| Native gadget unit tests | Replay cache, generation events, queues, deadlines | No |
| Fake worker/IIO integration | Deterministic slow startup, cleanup, and failure | No |
| Python protocol tests | Stale replies, retry horizons, fail-closed parsing | No |
| Network fault injection | UDP loss, duplicate, reorder, delay, old datagrams | Optional radio or namespace fixture |
| Single-radio hardware | Lifecycle and resource reuse across rates | Yes |
| Two-radio hardware | Parallel contention, host memory, independent identity | Yes |
| Supervisor recovery | Fatal cleanup causes bounded daemon restart | Yes |
| Soak/promotion | Long-run reliability and provenance | Yes |

## Gate 0: freeze the transition table

Represent the firmware transitions as a pure function in the native test
target:

```c
transition_result_t transition(
    process_state_t process,
    rx_state_t rx,
    resource_owner_t owner,
    request_state_t request,
    event_t event);
```

Table-drive every valid and invalid state/event pair.

Pass:

- every pair returns an explicit next state, reply action, and worker action;
- no default branch silently ignores a side-effecting request;
- invalid pairs leave ownership unchanged and return a stable error;
- shutdown has a path from every state;
- every non-idle RX state reaches either idle or fatal under a bounded event
  sequence.

Fail action: do not add threads or IIO calls until this gate is complete.

## Gate 1: request idempotency and replay

### Required native tests

1. Duplicate START while pending creates one worker and one stream ID.
2. Duplicate START after completion returns the original `STARTED` bytes.
3. Duplicate STOP while cleanup is pending emits one stop signal.
4. Duplicate STOP after completion returns the original `STOPPED` bytes.
5. Same request ID with different bytes returns `-EALREADY`.
6. Delayed old START after a newer STOP never starts another stream.
7. Old STOP never stops a newer stream.
8. Wrong stream ID returns `-ENOENT` without state change.
9. Requests older than the replay window return `-ESTALE`.
10. Cache eviction remains bounded and preserves the high-water mark.
11. A STOP response-send failure retains replay state; a START response-send
    failure remains armed and does not release capture until retry succeeds.
12. Two peers using the same request ID remain independent.
13. Peer inactivity expiry cannot resurrect an old pending operation.
14. Request-ID wrap behavior is deterministic, even though wrap is not
    expected operationally.

Pass: every side effect has an execution count of exactly zero or one as
declared by the transition table.

## Gate 2: deterministic slow startup and cleanup

Add test-only delay injection at these boundaries:

- before worker ready;
- during the discarded startup refill;
- before sender completion;
- while stopping the gain sampler;
- while destroying the IIO buffer;
- immediately before worker-done notification.

The production binary must default all injections off. Prefer a separately
compiled test backend over an unrestricted production environment variable.

Red reproduction:

1. Set cleanup delay above the host control timeout.
2. Send STOP three times with the same request ID.
3. Send the next START while cleanup continues.
4. Confirm the old implementation either blocks control or leaves duplicate
   replies ahead of the new response.

Green requirements:

- capability queries continue to receive bounded replies during cleanup;
- duplicate STOPs are consumed/coalesced without duplicate cleanup;
- the new START receives `-EBUSY` while ownership is releasing, then succeeds
  once idle;
- no stale reply is interpreted as the new request;
- control-loop service latency remains below its declared bound;
- cleanup duration may exceed a host retry interval without corrupting state.

## Gate 3: generation and event safety

Inject and verify:

- late `READY` from generation N after generation N+1 exists;
- duplicate `WORKER_DONE`;
- eventfd notification without a queued event;
- queued event without an eventfd notification, recovered by watchdog;
- event-queue full;
- stale run/quit eventfd count;
- worker exits between state check and stop request;
- shutdown and natural completion arriving together.

Pass:

- stale events are counted and ignored;
- no event changes another generation;
- each worker is joined exactly once;
- shared arguments are not overwritten before join;
- event-queue failure goes to bounded fatal cleanup, not reuse.

Run the native targets under AddressSanitizer, UndefinedBehaviorSanitizer, and
ThreadSanitizer where the cross toolchain permits it.

## Gate 4: resource failure matrix

Fail each acquisition step once:

| Injection | Required outcome |
| --- | --- |
| Frame-size arithmetic overflow | Reject before allocation |
| Frame-slot allocation failure | `-ENOMEM`, no DMA owner |
| Local IIO context failure | `-EIO`, complete cleanup |
| RX or PHY lookup failure | `-EIO`, complete cleanup |
| Channel/scan-mask failure | `-EINVAL` or `-EIO`, no capture |
| Timestamp register read failure | `-EIO`, no untracked modification |
| Timestamp register write failure | `-EIO`, restore if modified |
| Kernel-buffer configuration failure | `-EIO`, destroy context |
| IIO-buffer creation failure | `-EIO`, release ownership |
| Startup refill short/error | `-EIO`, no frame emitted |
| Gain sampler start failure | `-EIO`, no dangling thread |
| Sender start failure | `-EIO`, stop sampler and DMA |
| IIO refill short/error mid-capture | Failed session; no partial frame accepted |
| Queue full | Explicit overflow/failure; bounded cleanup |
| UDP partial/error | Failed session; incomplete outer frame rejected |
| Timestamp restore failure | Fatal ownership fault; daemon exits |
| IIO-buffer destroy stalls | Cleanup watchdog; daemon exits |

After every recoverable injection, a clean capture must start immediately. A
fatal ownership failure must be recovered only by daemon restart.

## Gate 5: control-plane fault injection

Use a synthetic UDP peer or network namespace to inject:

- loss of the first STARTED reply;
- loss of the first STOPPED reply;
- three duplicate STARTs before READY;
- three duplicate STOPs during cleanup;
- reordered old STOPPED before a new STARTED;
- reordered old STARTED after STOPPED;
- corrupted and truncated control datagrams;
- valid datagrams from a second source port;
- delayed requests beyond replay retention;
- request-ID collision with different content;
- malformed flood within a bounded rate.

Pass:

- retries recover when the operation is safe;
- no duplicate worker or stream is created;
- no old request changes current state;
- memory and replay-cache occupancy remain bounded;
- queries remain responsive;
- malformed traffic cannot starve worker completion events.

The SPF client should independently ignore responses for other request IDs
without spending a retry attempt. That is defense in depth, not a substitute
for firmware idempotency.

## Gate 6: frame-pipeline integrity

For zero, one, and maximum-capacity metadata, verify:

- exact v3 header size and CRC;
- IQ begins at exactly `header_bytes`;
- immutable queued frame slots;
- contiguous buffer and sample sequences;
- gain observations overlap their declared sample interval;
- sender never sees an uncommitted slot;
- partial UDP frames never become host frames;
- stop during capture and stop during drain have explicit canceled-frame
  counts;
- late datagrams from a completed stream do not enter the next stream;
- capture-first memory remains within the declared fixed bound.

USB and IP serialization fixtures must remain byte-identical before outer IP
fragmentation.

## Gate 7: TX and cross-plane ownership

Verify:

1. Legacy RX cannot start while v3 RX owns or releases DMA.
2. V3 RX cannot start while legacy RX owns or releases DMA.
3. Standard IIO configuration works before and after direct capture.
4. A competing standard IIO RX stream fails explicitly; it does not corrupt
   direct IQ.
5. TX start/stop follows its own bounded lifecycle.
6. RX+TX concurrency is either explicitly qualified or rejected with
   `-EBUSY`.
7. Shutdown releases both lanes.
8. TX buffer teardown is verified and host-side TX mute remains mandatory.

The attenuated TX2 loopback gate remains explicitly opt-in.

## Gate 8: single-radio hardware lifecycle

RAM-boot the candidate and run each rate independently:

```text
1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0 MS/s
```

At each rate:

- run at least ten finite requests;
- use four maximum-size frames for the lifecycle gate;
- validate every v3 header, frame sequence, sample sequence, gain series, RSSI,
  IQ shape, and CRC;
- record startup, capture, drain, cleanup, and control latency;
- query capabilities between requests;
- restore the original sample rate and TX mute on every exit.

Pass: no control, lifecycle, ownership, frame, or metadata error. Throughput
headroom is reported separately and does not turn a correct bounded capture
into a sustained-rate claim.

## Gate 9: two-radio parallel ladder

Use the committed runner:

```bash
SPF_IP_LADDER_CYCLES=10 \
tests/radio_hardware/run_direct_ip_parallel_ladder.sh \
  HOST_A HOST_B /tmp/spf-two-radio-ip-state-machine
```

Pass:

- both serials are unique and firmware provenance matches;
- every rate completes ten lifecycle cycles;
- no `control_rearm_failure` or `integrity_failure` occurs;
- all expected frames and fragments reconstruct;
- kernel UDP `InErrors`, `RcvbufErrors`, and `MemErrors` remain zero;
- application duplicate, expired, rejected, and queue-overflow counts remain
  zero;
- per-radio cleanup latency stays within its declared deadline;
- radio sample rates, host receive-buffer limit, and TX mute are restored.

The first rate with real-time headroom below one is recorded but is not an IQ
integrity failure.

## Gate 10: interruption and supervisor recovery

Exercise these points on one radio, then both:

- kill host after START send but before STARTED;
- kill host during capture;
- kill host during drain;
- kill host after last data frame but before STOP;
- terminate gadget during capture;
- trigger the cleanup watchdog using test firmware;
- stop the service during STARTING, CAPTURING, DRAINING, and CLEANING;
- restart the daemon and perform a clean standard-IIO query and direct capture.

Pass:

- finite orphaned streams self-clean;
- no daemon restart claims success before USB/IP endpoints are usable;
- unreleased ownership causes non-zero exit and supervised restart;
- a new process nonce, generation, and stream ID distinguish restarted state;
- no stale UDP request restarts an old stream;
- TX is muted by the surrounding recovery procedure.

## Gate 11: soak and promotion

Minimum candidate campaign:

1. One-hour single-radio IP soak per radio.
2. One-hour two-radio parallel IP soak.
3. At least 1,000 START/STOP cycles spread across the supported rates.
4. Ten deliberate host interruptions.
5. Ten deliberate gadget restarts.
6. Sequential USB/IP parity capture on each radio.
7. Short V7 write/reopen with exact firmware and transport provenance.

Pass:

- zero unexplained control or frame loss;
- zero duplicated side effects;
- no unbounded RSS, socket, replay-cache, thread, or file-descriptor growth;
- every injected failure is counted and has the declared outcome;
- all radios return to standard IIO control and muted TX;
- rollback to the previous persistent firmware is demonstrated.

Only then create a persistent-QSPI canary. Do not overwrite or retag RC16.

## Required artifacts

Each hardware run should retain:

- source commit, Buildroot pin, firmware tag, DFU SHA-256, and device version;
- radio serials, IPs, physical paths, and FPGA/gadget identity;
- exact command and environment overrides;
- per-transition latency and counters;
- ladder JSON and pytest output;
- kernel UDP and interface counter deltas;
- sample-rate and TX-mute restoration evidence;
- failure-injection schedule and observed result;
- final pass/fail decision with the first failing gate.

Promotion is fail closed. A throughput improvement cannot compensate for a
lifecycle, ownership, metadata, or recovery failure.
