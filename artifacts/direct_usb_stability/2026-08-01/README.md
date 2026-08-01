# Direct-USB stability bring-up - 2026-08-01

This report records the first execution of the attached-radio pytest gates and
the finite-buffer firmware candidate described in
[`DIRECT_USB_STABILITY_PLAN.md`](../../../data_collection/rover/rover_v3.1/DIRECT_USB_STABILITY_PLAN.md).

## Radios

| Serial | Physical USB path | Historical note |
|---|---:|---|
| `104000bac4950008230026001b440a003a` | `1-1.1` | `.17`; candidate qualification radio |
| `1040007c4a94000211000b009186843ef2` | `1-1.2` | `.18`; control radio |

USB device addresses changed during reboots and were not used as identity.

Both radios were persistently updated, one at a time, with the published
fingerprint-v1 image using `pluto.frm` only. The updater wrote the Linux FIT to
`mtd3`; it did not write `boot.frm`, the firmware ZIP, `mtd0` or the U-Boot
environment. Both radios then passed a normal `device_reboot reset` and returned
with direct USB present, proving that the published image boots from QSPI.

Published persistent image:

- Release: `v0.38-plutoplus-spf-gain-rssi-fingerprint-v1`
- DFU SHA-256: `0a6a8939b31babed2ad7093d83941ebc809323d69804adcd8da5bcae0e48d3e9`
- Device firmware: `v0.38_plutoplus_with_timestamping-9-g7b02`
- Gadget build: `a1e6417d07188bd72be70692e28c5d6ae9a5ec62`

## Baseline and host limit

The first production-size run used 524,288 samples per channel, CS16 dual RX,
and protocol v2 gain/RSSI metadata.

Passed:

- identity and protocol-v2 capability query on both radios;
- ten repeated production-size one-frame START/STOP cycles per radio;
- exact IQ shape/type and non-zero channels;
- gain/RSSI validity and header CRC validation;
- protocol-v2 to V7 LMDB-Zarr write, close, reopen and field validation;
- simultaneous one-frame capture from both radios.

The host has `usbcore.usbfs_memory_mb=16`. An intentionally deeper test that
queued three 4,194,400-byte framed transfers on both radios failed during host
submission with `LIBUSB_ERROR_NO_MEM`: about 25 MiB was requested. No gadget
START or IQ transfer had occurred. Production queues one frame per radio (about
8 MiB total), which passed. Multi-frame ordering is therefore tested one radio
at a time, while simultaneous testing uses the production queue depth.

Ten-cycle two-radio host RSS grew by 20,512 KiB and remained below the declared
64 MiB gate.

After both radios returned to the published persistent image, the final opt-in
hardware suite passed all five tests: identity/capabilities, contiguous
multi-frame sequence, five production-sized lifecycle cycles per radio,
simultaneous two-radio capture, and a three-record-per-radio V7 close/reopen
round trip. The command selected both radios by serial and required an exact
count of two.

## Stability candidate

The candidate changes only two resource policies:

1. A finite request allocates `min(frame_count, 16)` gadget USB buffers. A
   production one-frame request now allocates one ~4 MiB buffer instead of
   sixteen (~64 MiB).
2. Gadget debug logging defaults off when `sdr_usb_gadget_debug` is absent.

It does not change the USB descriptors, protocol, metadata layout, capability
bits, IQ layout or standard USB-IIO function.

Candidate provenance:

- Gadget: `4872a1c3d67011858ba37a5db21a50591250d428`
- Buildroot: `680b1ad3761d2f00ceb81a40c88b1b23ce6e4956`
- Firmware: `cbf9fb8c90be74e7452ddea5f228731b7dc4f8fb`
- Candidate DFU SHA-256: `c867bbe107139eb6065b3a3282b76c38ad8bd0a5315505738c1e743c773e49bc`
- Candidate device firmware: `v0.38_plutoplus_with_timestamping-10-gcbf9`

Build/test results:

- Six gadget host-native tests passed, including the new buffer-policy test.
- The actual ARM daemon cross-compiled against Buildroot libaio/libiio.
- The complete `pluto.dfu` image built with one job.
- The candidate was RAM-loaded on `.17`; `.18` stayed on the persistent
  published image as a control.
- Candidate `.17` passed 50 production-sized one-frame cycles.
- Pluto used memory was 11,848 KiB before and 10,484 KiB after those cycles.
- No default `/var/log/sdr_usb_gadget.log` was created.
- A mixed candidate/control run passed 20 production-size cycles per radio,
  simultaneous dual-radio capture and five V7 records per radio.

The first candidate RAM-load attempt encountered the known USB-network
namespace discovery race before DFU download (`expected one USB-network
interface ... found []`). The loader's bounded second attempt succeeded. This
is recorded as a boot-loader robustness issue, not a capture or candidate
firmware failure.

## Production collector and provenance

The canonical Rover 3 V7 configuration was exercised through the real
`mavlink_radio_collection.py --fake-drone` path with both persistent radios:

| Run | Result |
|---|---|
| 10 records per receiver | Clean finalization and fresh-process validation |
| 100 records per receiver | Clean finalization; 200 total IQ frames validated |
| QSPI provenance smoke, 3 records per receiver | Clean finalization; manifest and both receiver groups record `qspi` and `firmware_verified=true` |

The 100-record artifact was
`/tmp/spf-stability-production100/rover_2026_08_01_05_28_10_nRX2_center_spacing0p043_tag_STABILITY100.zarr`
and occupied 356,593,728 bytes. Every signal frame had shape `(2, 524288)`,
finite, non-zero complex64 IQ, valid gain/RSSI metadata and the expected radio
identity. Median frame periods were 0.555 seconds (`.18`) and 0.551 seconds
(`.17`). The receiver spans were 59.319 and 59.125 seconds respectively.

During review, the production YAMLs were found to declare `boot-mode: ram`
while boot preparation actually defaulted to persistent QSPI. This did not
change the received samples but made the readiness/Zarr provenance false. The
three production V7 configs now declare `qspi`; the resolver accepts the
explicit `ram` and `qspi` modes; boot preparation follows the config unless
`SPF_PLUTO_RAM_LOAD` is explicitly set. A no-write hardware run confirmed both
active images matched, generated a `qspi` readiness manifest, and the QSPI
provenance smoke capture round-tripped that value into both receiver groups.

## Current disposition

- Both radios' persistent QSPI contains the published fingerprint-v1 image.
- `.18` currently runs that published persistent image after normal reset and
  passes the interruption/reclaim gate.
- `.17` suffered a real kernel-observed USB disconnect during the first
  two-radio interruption run (`usb 1-1.1: USB disconnect` at 05:59:23). It did
  not re-enumerate after a targeted USB hub-port cycle because its power is
  external. It requires an operator power cycle before the exact-count-two
  gate can be repeated.
- The candidate is not yet a release and must not be made persistent until its
  commits/image are published and the Rover config manifest is updated.

Remaining gates:

1. Repeat the passing interrupted-capture/reclaim gate with both radios after
   `.17` is externally power-cycled.
2. Repeat the longer lifecycle soak with both radios and inspect device/host
   memory and kernel log deltas.
3. Publish and pin the candidate, then repeat a cold-boot two-radio acceptance
   before considering persistent rollout.

## Interrupted-capture development

Interruption handling now has two layers of opt-in testing.

The deterministic subprocess tests cover SIGINT, SIGTERM and SIGKILL without
radio hardware. A complete record advances a persisted per-receiver count only
after its write finishes. SIGINT/SIGTERM cooperatively stop RX, preserve the
first error, mark the temporary store `incomplete`, close LMDB and exit
non-zero. SIGKILL cannot run cleanup, so the store deliberately remains
`in_progress`; it is still readable and is never promoted to a final `.zarr`.

The attached-radio test runs the ordinary production fake-drone collector,
waits for a configured committed-record boundary on every receiver, sends the
selected signal, validates the partial V7 store, and then claims every serial
for a fresh frame. It also rejects stale device mappings or readiness manifests
before starting, so USB address changes cannot be mistaken for a collector
failure.

One-radio result on `.18`:

- production frame size: 524,288 complex samples per channel;
- completed records before SIGTERM: at least two;
- pytest result: `1 passed` in 21.23 seconds;
- partial state: `incomplete`, `CaptureInterrupted`, SIGTERM recorded;
- final `.zarr`: absent;
- immediate direct-USB reclaim and one-frame receive: pass.

Focused regression result: 161 software tests passed across capture
finalization and the direct-USB collector, receiver and protocol modules. The
existing MAVLink fake-drone smoke also passed; ordinary hardware invocation
correctly skipped all six opt-in radio tests.

## P0 interruption matrix and writer-race fix

Commit `0c76e10971fec068d9dca9bc84068d22a45abe28` added the bounded
interruption and restart campaigns. Development deliberately preserved every
failed artifact instead of retrying it into the passing population.

The first 3,500-frame run exposed a silent V7 writer race at frame 1,016. Its
IQ payload was present, but `system_timestamp`, stream/sequence fields and all
gain/RSSI metadata retained their zero fill values. The Zarr scalar arrays use
one capture-wide chunk. Two worker threads could perform overlapping
read/modify/write assignments for different records of the same receiver; the
later whole-chunk write silently replaced the earlier update. No exception was
raised, so the old collector marked the store complete.

The fix gives each receiver one FIFO writer. Records for one radio are strictly
serialized, while independent receiver groups can still write in parallel.
This also makes the persisted committed-record count a contiguous prefix for
each receiver. A focused concurrency test fails with the old shared two-worker
executor and passes with the per-receiver executors. The failed artifact remains
at `/tmp/spf-p0-soak-20260801/session1-cycle2-retry` and is excluded from all
accepted duration.

The first abrupt-death test exposed a second P0: `SIGKILL` cannot send gadget
`STOP`, and a completed FunctionFS AIO write may remain queued on bulk-IN after
the host process releases the interface. A new process then received an
overflow or a short stale tail before its own frame. Opening a receiver now
claims the interface, sends an idempotent `STOP`, drains a bounded orphaned
backlog and resets the endpoint before the first `START`. It never allocates
from the gadget's theoretical UINT32-wide maximum; the drain is capped at 8
MiB, within the Pi's 16 MiB usbfs budget. Unit tests cover bounded drain and
fail-closed exhaustion, and the real `SIGKILL` test proves first-process
recovery.

The coherent hardware campaign is under
`/tmp/spf-p0-interrupt-campaign-rerun/20260801T064928Z_rover2`:

| Signal | Interrupt after | Expected store state | Exit | Result |
|---|---:|---|---:|---|
| `SIGTERM` | 2 committed records | `incomplete` / `CaptureInterrupted` | 143 | Pass; radio reclaimed |
| `SIGINT` | 10 committed records | `incomplete` / `CaptureInterrupted` | 130 | Pass; radio reclaimed |
| `SIGKILL` | 25 committed records | `in_progress`, no invented error | -9 | Pass; orphan drained and radio reclaimed |
| `SIGTERM` | 100 committed records | `incomplete` / `CaptureInterrupted` | 143 | Pass; radio reclaimed |

Every partial store remained `.zarr.tmp`, every safely committed prefix had
strictly increasing timestamps and valid gain/RSSI metadata, and all four
kernel-log deltas were empty. A clean 100-frame production V7 capture then
passed the full IQ/schema/metadata/provenance validator at 1.957 Hz median.

Focused checks after the fixes:

- 19 finalization/resource/restart tests passed;
- 47 direct-USB/recovery/finalization tests passed;
- 39 multi-Pluto manifest/restart/resource tests passed;
- the four-case attached-radio campaign and its final production capture
  passed.

The full local suite was intentionally not run because it has previously
exhausted this development Pi. CI remains responsible for the complete suite.

## Restart-separated one-hour production gate

The final automated run is under
`/tmp/spf-p0-restart-soak-final/20260801T065602Z_rover2`. It used the committed
Rover 2 production V7 YAML, fake drone, 524,288 samples per channel, protocol
v2, slow-attack AGC and receive-only operation. Each session restarted every
configured Pluto, required observed USB absence and fresh enumeration,
recreated/verified the readiness manifest, and then captured 3,500 frames.

| Session | USB address | Fingerprint session | Frames | Accepted span | Median rate | p99 interval | Anonymous RSS range | Kernel delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 44 -> 45 | `0701b186-aa4d-4342-a18d-4918c9418ead` | 3,500 | 1,802.903 s | 1.990 Hz | 0.597 s | 39.97 MiB | 0 bytes |
| 2 | 45 -> 46 | `697a9c34-dae4-429c-b0f4-6389365ed2d7` | 3,500 | 1,821.843 s | 1.941 Hz | 0.558 s | 55.25 MiB | 0 bytes |

Aggregate accepted capture time is **3,624.746 seconds (60m 24.746s)**. Both
strict validators passed all logical IQ, metadata, sequence and provenance
checks. Both sessions contain 3,500 unique finite stream IDs, valid gain/RSSI
metadata and the expected `(3500, 2, 524288)` complex64 shape. Stable identity
remained:

- serial `1040007c4a94000211000b009186843ef2` (`.18`);
- physical path `1-1.2`;
- stable fingerprint
  `854599ff8d81be79799ab0752e233cea0bc6f39f214406b66c7c7103efca70ae`.

Report SHA-256 values:

| Report | SHA-256 |
|---|---|
| Session 1 strict validation | `179c3da5c7b48debfeb87be065262b4420a99be9dd9cf19f40c307e6765ff4db` |
| Session 2 strict validation | `48caa227e0844c7a2eb5857c326d467f0b32ac6fe62252993134c0b19b944ba4` |
| Session 1 resources | `16ed5133bc9ed55676fd11acb42edcb8f5033abb9c92bc1972ab301bf4ebcbed` |
| Session 2 resources | `d8e34638ff71bd547f22ba6501967e76fe64027828d054777b8e2d78c7632277` |
| Aggregate | `0fc7116bf0ce29f17eec096f8e82579a4c975036e9f3b4ee6b3285700d01e243` |

The one remaining hardware gate is an exact-count-two rerun. `.17` is not USB
enumerated and requires an external power cycle; software cannot restore an
externally powered radio after its host link has disappeared. The one-radio
P0 and greater-than-one-hour restart gates pass, but this report does not claim
the unavailable second radio was tested.

The required post-soak quick test then ran without rebooting `.18`. Identity
and capabilities, contiguous three-frame ordering and 20 production-sized
START/STOP lifecycle cycles passed in 9.39 seconds. Host RSS grew 20,624 KiB,
below the 64 MiB bound, and the final channel RMS values were 13.65 and 13.39.
The simultaneous-radio case was the sole skip because the fixture required
exactly one attached radio for this run. Evidence is under
`/tmp/spf-p0-post-soak-quick-20260801`.

While `.17` remained unavailable, a deterministic two-receiver writer test
was added. A barrier proves both receiver groups can write concurrently, while
per-receiver active counts and recorded indices prove each radio remains
single-writer and FIFO. The focused finalization file passed all 11 tests. This
is software evidence for the dual-radio writer topology; it does not replace
the pending exact-count-two hardware campaign.

## Extended interruption-boundary campaign

The first unattended boundary sweep is preserved at
`/tmp/spf-interrupted-capture-soak-12h/20260801T084031Z`. Round 1 passed
`sigterm:1`, `sigint:8`, `sigkill:32`, `sigterm:128`, and its strict 100-frame
clean recovery. Round 2 then failed its `sigkill:192` harness gate after 90
seconds. The collector itself remained healthy at approximately 1.9 Hz and had
committed 137 records; no USB error occurred. The requested 192-frame boundary
cannot be reached in 90 seconds at that measured production rate, so this was
a test deadline defect rather than a radio or firmware failure.

Commit `0a3d221` derives the deadline from the requested boundary, preserving
the historical 90-second floor for early cases while allowing a conservative
1 Hz plus 30 seconds of setup overhead for later cases. Focused timing tests
cover the 192- and 256-frame boundaries. The exact real-radio red/green rerun
at `/tmp/spf-interrupt-192-redgreen-20260801` reached 192 committed frames,
sent `SIGKILL`, preserved the correct `in_progress` store, exited `-9` in
0.114 seconds, and reclaimed `.18`. The test passed in 119.05 seconds with a
222-second bound.

The corrected 12-hour campaign is isolated under
`/tmp/spf-interrupted-capture-soak-12h-v2/20260801T085617Z`. A separate
aggregate validator checks every ledger row against its individual report,
expected signal status/return code, committed receiver prefix, serial identity,
kernel delta and clean V7 recovery. It writes `aggregate-final.json` only after
the soak root itself has an exact `PASS` marker.

That run completed three full matrices (12 interruptions and 300 clean frames)
before its fourth boot-preparation call exposed another P0 race. USB-IIO,
direct USB and dual RX had passed, but the passive-fingerprint SSH step briefly
reported no serial-matched USB-network interface. `eth1` was present again by
the time the failure was inspected, and no USB disconnect occurred. The
fingerprint path first probes SSH in one temporary namespace, restores the
interface to the root namespace, and then opens a second namespace for the
actual fact read. Netlink can expose the restored link fractionally before
udev properties used for serial matching are visible.

Commit `0c964de` adds a bounded five-second wait for exactly one serial-matched
interface at namespace entry. Zero matches are treated as transient during the
bound; multiple matches remain an immediate identity failure. Two focused
tests cover recovery and ambiguity, and all 32 firmware/manifest tests pass.
The exact hardware preparation path then passed 10/10 consecutive manifest
builds and verifications under `/tmp/spf-network-restore-redgreen-20260801`.

The previously skipped fourth matrix passed end to end at
`/tmp/spf-round4-network-redgreen/20260801T092045Z_rover2`:

| Signal | Boundary | Store state | Exit/reclaim | Result |
|---|---:|---|---:|---|
| `SIGTERM` | 5 | `incomplete` | 0.717 s | Pass |
| `SIGKILL` | 40 | `in_progress` | 0.064 s | Pass |
| `SIGINT` | 160 | `incomplete` | 0.967 s | Pass |
| `SIGTERM` | 256 | `incomplete` | 0.917 s | Pass |

All kernel deltas were empty. Its post-fault 100-frame V7 capture passed at
1.964 Hz median with 100 unique streams and the same stable fingerprint. The
fresh 12-hour run incorporating both red/green fixes is under
`/tmp/spf-interrupted-capture-soak-12h-v3/20260801T092924Z`.

That v3 run completed three full matrices before preserving a round-4 failure:

- 12 interrupted captures passed (four each of SIGTERM, SIGINT and SIGKILL);
- 790 interrupted frames had a strictly valid committed prefix;
- three clean 100-frame V7 recovery captures passed;
- all kernel deltas were empty;
- 31 cgroup samples covered 903 seconds, with 585.81 MiB peak anonymous
  memory and 811.51 MiB minimum host-available memory.

The failure was not data corruption. The SIGKILL-at-40 store was readable and
correctly remained `in_progress`. Its immediate raw one-frame release probe
then received one transient libusb transfer error. That probe deliberately has
no PPlus `ReceiverConfig`, so its internal reconnect correctly failed closed
with `iio_rx_configuration: expected configured reconnect attestor`. The
production receiver does have the complete read-only IIO attestor.

The hardware harness now closes such a fail-closed raw probe and retries from
at most three wholly new caller-authorized probe sessions. It retries only the
specific missing-configuration-attestor result; identity, firmware or actual
configuration mismatches still fail immediately. Passing still requires a real
IQ frame, and the report records the number of sessions used. Two focused unit
tests cover the allowed retry and the non-retry identity case.

The exact SIGKILL-at-40 red/green is preserved at
`/tmp/spf-sigkill40-fresh-session-green-20260801`. It passed in 40.41 seconds,
exited `-9` in 0.064 seconds, retained 40 valid committed frames, reclaimed the
radio on the first fresh session and produced an empty kernel delta.

## One-hour checkpoint of the immutable v4 soak

The current 12-hour campaign runs from immutable source revision
`85d4e8d6a77b774d43b73e701cdcdc6e30c48a22` under
`/tmp/spf-interrupted-capture-soak-12h-v4/20260801T095933Z`. At the formal
3,611-second checkpoint it remained active and had completed eight whole
rounds:

- 32 interruptions: 12 SIGTERM, 10 SIGINT and 10 SIGKILL;
- 2,502 strictly committed frames in interrupted temporary stores;
- eight post-fault 100-frame V7 recovery captures, or 800 clean frames;
- maximum cooperative/forced signal exit time of 1.269 seconds;
- every raw radio release probe succeeded on its first fresh session;
- zero non-empty kernel-log deltas;
- peak cgroup anonymous memory of 581.79 MiB;
- minimum host-available memory of 587.28 MiB; and
- post-peak anonymous-memory recovery below 384 MiB in every round.

The current strict validator was also run independently over the first seven
complete recovery Zarrs. All 700 frames passed full-IQ shape, finite/nonzero,
nonconstant and RX1/RX2-distinctness checks, plus gain/RSSI metadata, sequence,
firmware provenance, hardware-fingerprint binding and serial/path identity.

The soak continues unattended for the full 12 hours. A detached validator at
revision `e8dc40bf5f47b8426f7468309126eb33739333eb` waits for campaign and
resource-monitor completion, then revalidates every clean recovery Zarr with
the current strict rules and reruns the per-round resource-recovery gate. It
writes `ENHANCED_PASS` only if both final validations pass.

This checkpoint covers `.18`, serial
`1040007c4a94000211000b009186843ef2`, at physical path `1-1.2`. The exact
two-radio gate remains blocked solely because `.17` is not physically
enumerated and requires an external power cycle.

## Operator-state and immutable recovery P0s

Production capture now writes `/home/pi/preflight/capture_status.json`
atomically at a bounded cadence. It contains the capture name and lifecycle
state, exact per-receiver committed counts, common progress, observed frame
rate, elapsed time, ETA, late-watchdog result and primary failure. A nonzero
collector exit updates the durable state, plays the failure tune three times
and leaves systemd failed. The normal completion state is published only after
all temporary artifacts have been renamed successfully.

`spf.scripts.recover_interrupted_v7` provides read-only-first partial recovery.
It independently finds each receiver's contiguous valid V7 prefix, selects the
aligned minimum, copies it into a new `recovered_incomplete` artifact, records
source path/status/hash and reason, and runs the full strict validator before
rename. It never modifies or directly promotes the source.

Real-artifact evidence:

| Source | Logical/allocated LMDB | Recovered | Result |
|---|---:|---:|---|
| Round-1 SIGTERM at 1 | shrunk / 3.7 MiB store | 1 frame | strict pass |
| Round-2 SIGKILL at 2 | 128 GiB / 5.6 MiB | 2 frames | strict pass in 6.83 s |

The SIGKILL source's logical size, allocated block count and mtime were
unchanged. A versioned sparse-extent SHA-256 hashes allocated content plus
logical layout without reading the 128-GiB hole. Synthetic tests cover safe
prefix truncation, duplicated-channel rejection, immutable source hashes and
separate recovered output.
