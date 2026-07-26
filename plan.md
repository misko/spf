# Pluto+ dual-RX gain metadata: gated implementation plan

Date: 2026-07-25

Final evidence audit:

```text
docs/direct_usb_gain_completion_audit.md
```

The transport, firmware, Python compatibility facade, v4 Zarr path, throughput,
soak, publication, and rollback are complete. Two physical RF
characterizations remain unproven because the required bench equipment was not
attached: calibrated stepped-input RSSI response and coherent common-CW
phase-versus-gain-change behavior.

The exact hardware-tested DFU is published at
[`v0.38-plutoplus-spf-gain-rssi-v2`](https://github.com/misko/plutosdr-fw/releases/tag/v0.38-plutoplus-spf-gain-rssi-v2).
Rovers download, checksum, RAM-boot, verify, and roll it back with
`data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh`.

## Revised compatibility-first delivery — gain and RSSI in dB

This section supersedes the original Python-output and dataset contract below.
The completed v1/index work remains useful evidence and a rollback point, but
the next delivery must minimize changes above the Pluto transport boundary.

### Revised goal

For every direct-USB IQ frame, the Pluto must:

1. observe RX1/RX2 gain state locally;
2. convert that captured state to the active table's gain in dB;
3. observe RX1/RX2 hardware RSSI locally;
4. attach gain and RSSI observations to the same USB transfer as the IQ;
5. retain raw gain indices only inside the Pluto long enough to calculate
   raw-state endpoint-change flags; and
6. avoid all per-frame host-side IIO gain and RSSI reads.

Existing SPF collection code must continue to receive:

```python
signal_matrix.shape == (2, 524288)
rssis.shape == (2,)
gains.shape == (2,)
```

with the same Python dtypes, units, sign conventions, dictionary keys, snapshot
fields, and v4 Zarr fields used by the USB-IIO path today.

For compatibility, `gains` and `rssis` represent the device observation made
after the IQ refill. This is the closest equivalent to the existing sequence:

```text
receive IQ -> read RSSI -> read hardwaregain
```

The v2 wire protocol may carry both start and end observations, but the initial
compatibility facade exposes the end pair through the existing `gains` and
`rssis` API. Persisting both endpoints is a separate, later schema change; it
must not be smuggled into the existing v4 fields.

### Frozen Python compatibility contract

The collector should retain its present conceptual flow:

```python
signal_matrix = self.pplus.rx()
rssis = self.pplus.rssis()
gains = self.pplus.gains()

return {
    "signal_matrix": signal_matrix,
    "rssis": rssis,
    "gains": gains,
}
```

Only one collector-line change is required relative to the historical code:

```python
self.pplus.sdr.rx()
```

becomes:

```python
self.pplus.rx()
```

`PPlus` owns the transport switch:

```text
IIO mode:
    PPlus.rx()     -> pyadi sdr.rx()
    PPlus.rssis()  -> existing IIO RSSI attributes
    PPlus.gains()  -> existing IIO hardwaregain attributes

direct_usb mode:
    PPlus.rx()     -> one direct-USB IQ+metadata transfer
                      and cache its decoded end metadata
    PPlus.rssis()  -> cached RSSI pair; no IIO transaction
    PPlus.gains()  -> cached gain-dB pair; no IIO transaction
```

The cache must be invalid before the first frame, after stop, and after any
reconfiguration. `rssis()` and `gains()` must fail rather than return stale
data if no successful direct frame has populated the cache.

The normal `DataSnapshotV4`, `DroneDataCollectorRaw`, v4 Zarr writer, readers,
models, and quality tools remain unchanged.

### Frozen unit and representation contract

Gain:

```text
wire representation: signed whole dB
Python representation: float64[2], matching legacy gains
source: active gain table lookup of the captured raw index
invalid wire value: INT8_MIN
invalid Python value: NaN
```

RSSI:

```text
wire representation: unsigned quarter-dB magnitude
Python representation: float64[2], matching legacy rssis
conversion: wire_value / 4.0
invalid wire value: UINT16_MAX
invalid Python value: NaN
```

The existing Linux driver and SPF convention reports RSSI as a positive
magnitude such as `111.25 dB`. Preserve that convention even though the AD9361
reference-manual quantity has a negative physical sign. Do not rename it dBm
or treat it as calibrated input power.

Raw gain indices never cross the v2 Pluto-to-Pi interface. The Pluto still
compares them internally, so a transition between two indices with the same
nominal dB value sets the gain-state endpoint-change flag.

### Pluto -> Pi -> Python interface

```text
pyadi on Pi
    configures LO, bandwidth, sample rate, AGC, FIR and channels
        |
        | standard USB-IIO control; before direct streaming
        v
Pluto ARM direct-USB gadget
    reads and caches the active gain table
    takes initial gain/RSSI observations
    refills one two-channel DMA IQ buffer
    takes end gain/RSSI observations
    maps raw gain indices to dB
    builds one versioned metadata header
    sends [header | IQ] in one fixed-size bulk transfer
        |
        | vendor-specific USB bulk-IN
        v
Pi direct-USB receiver
    verifies exact length, magic, version, header size, CRC and sequence
    decodes gain dB and RSSI quarter-dB
    creates the normal two-channel complex64 signal matrix
        |
        v
PPlus compatibility facade
    returns signal_matrix from rx()
    caches end RSSI for rssis()
    caches end gain dB for gains()
        |
        v
existing ThreadedRX -> DataSnapshotV4 -> existing v4 Zarr writer
```

The Pi performs no gain-table lookup, no RSSI conversion beyond fixed
quarter-dB scaling, and no radio metadata read after receiving the frame.

---

## Revised gated implementation sequence

### Compatibility Gate A — Re-freeze the existing Python and Zarr behavior

Capture 20 USB-IIO frames through the historical collector and record:

- the exact `signal_matrix`, `rssis`, and `gains` shapes and dtypes;
- the positive RSSI sign convention;
- the v4 Zarr keys and dtypes;
- the call order `rx -> rssis -> gains`; and
- the current gain/RSSI value ranges.

Pass:

- a golden compatibility test describes the existing dictionary and Zarr
  contract exactly;
- no new metadata field is required by existing collector consumers.

Fail:

- the direct design requires a downstream schema or model change merely to
  preserve existing behavior.

Status: **PASS**

Evidence:

- the saved IIO and direct-v2 v4 Zarr groups contain the same 13 arrays;
- both use `complex64[100,2,524288]`, `float64[100,2]` gains, and
  `float64[100,2]` RSSI;
- `tests/test_direct_usb_compatibility.py` freezes the API and schema contract.

### Device Gate B — Define protocol v2 without changing payload alignment

Define a packed, little-endian v2 header carrying:

```text
gain_db_start[2]              int8
gain_db_end[2]                int8
rssi_start_qdb[2]             uint16
rssi_end_qdb[2]               uint16
gain and RSSI validity flags
raw gain-state endpoint-change flags
gain and RSSI read durations
existing stream/buffer/sample sequences
existing FPGA-event reservations
CRC32
```

Use a 96-byte aligned header and negotiate it explicitly through the capability
and START requests. Keep v1 available as a rollback/debug protocol.

Pass:

- gadget C and host Python agree on one golden byte vector;
- every malformed size, value, CRC, flag, and sequence case fails closed;
- the full Rover transfer is exactly `96 + 4,194,304 = 4,194,400` bytes;
- an old host cannot interpret v2 metadata as IQ.

Fail:

- any existing v1 field is silently reinterpreted without a version change.

Status: **PASS**

Evidence:

- C and Python use the same 96-byte golden vector;
- malformed identity, sizes, CRC, flags, sentinels, payload, and sequences are
  rejected by the scoped protocol tests;
- hardware completed exact 4,194,400-byte Rover transfers;
- protocol v1 remained available and passed a two-frame smoke.

### Device Gate C — Convert captured gain state to dB locally

At direct-stream START:

1. read `gain_table_config` locally into a sufficiently large buffer;
2. parse and validate the active table and its RF range;
3. confirm the configured LO is inside that range;
4. require full-table mode and digital gain disabled;
5. cache the table and its hash for the life of the stream.

At each observation:

1. read RX1/RX2 raw indices once;
2. map those exact captured indices through the cached table;
3. retain raw indices only for internal start/end state comparison;
4. return signed dB values and validity.

Pass:

- manual `20/20 dB` returns `20/20`;
- manual `20/40 dB` returns `20/40`;
- unit tests prove equal dB from different indices still raises the raw-state
  endpoint-change flag;
- table, mode, range, or lookup failure rejects START or marks metadata invalid.

Fail:

- firmware performs a second `hardwaregain` read to obtain dB;
- firmware hard-codes a table without validating the active table.

Status: **PASS**

Evidence:

- the gadget loads and validates `gain_table_config` once per START;
- it rejects non-full-table mode, digital gain, invalid RF range, malformed
  tables, and out-of-range indices;
- manual 20/20 and 20/40 dB settings returned those exact dB values;
- raw indices remain available locally for endpoint flags but do not cross v2.

### Device Gate D — Read legacy-compatible RSSI locally

First use the local IIO `rssi` channel attributes so the device value exactly
matches the existing Linux-driver/SPF meaning. Capture an initial pair, then
one pair immediately after every successful refill and cache it as the next
start pair.

Benchmark a direct-register helper only as an optimization. Adopt it only if
it agrees with the driver to the native `0.25 dB` resolution and improves the
combined metadata timing.

Pass:

- local results match occasional out-of-stream IIO reads to `0.25 dB` under a
  stable input;
- RX1/RX2 order is correct;
- controlled stronger input produces a smaller positive RSSI magnitude;
- failures use the sentinel and validity flags without stopping IQ;
- measurement configuration is not changed by reading it.

Fail:

- RSSI is relabelled as dBm or made negative only in the direct path;
- the gadget restarts or reconfigures the RSSI estimator per frame.

Status: **PASS WITH ONE DOCUMENTED BENCH LIMITATION**

Evidence:

- local RSSI matched stable out-of-stream IIO diagnostics exactly at the
  native 0.25 dB resolution: RX1 86.75 dB and RX2 102.75 dB;
- RX order, positive-magnitude convention, sentinels, flags, and non-mutating
  reads are covered by C/Python tests and hardware captures;
- zero RSSI read failures occurred during the 7,200-frame soak.

Limitation:

- no remotely controlled RF attenuator was attached, so the planned calibrated
  stronger-input step was not executed. This does not affect transport or
  legacy-value compatibility, but it remains a bench characterization item.

### Device Gate E — Build one frame atomically at the transport boundary

The steady-state device sequence is:

```text
iio_buffer_refill
-> gain observation
-> RSSI observation
-> fill v2 header
-> copy IQ immediately after header
-> submit one fixed-size USB bulk write
```

Preallocate all memory. Continue IQ delivery with explicit invalid metadata
when an individual gain or RSSI observation fails.

Pass:

- exact transfer length on every completion;
- no first-sample loss, metadata leakage into IQ, duplication, or channel swap;
- sequence gaps and device overflow are explicit;
- combined metadata read p99 is below 2 ms, or a separately approved measured
  budget that does not reduce frame throughput.

Fail:

- metadata and IQ use separate transfers;
- completed writes are compared against the old IQ-only size.

Status: **PASS**

Evidence:

- one 96-byte header and 4,194,304-byte IQ payload completed as a single
  4,194,400-byte transfer;
- both IQ channels remained finite and nonzero with correct ordering;
- maximum measured gain and RSSI reads were 0.851 ms and 0.710 ms;
- gadget logs reported zero gain/RSSI failures and no nonzero overflow.

### Host Gate F — Add a narrow direct-USB decoder

Keep protocol, libusb, CRC, fragmentation, and fixed-point decoding isolated in
`spf/sdrpluto`. Its internal result may contain full v2 metadata, but the public
compatibility path needs only:

```text
signal_matrix
gain_db_end[2]
rssi_db_end[2]
metadata_valid
sequence
```

Pass:

- synthetic fragmented transfers and hardware transfers decode identically;
- invalid gain/RSSI becomes NaN;
- no malformed metadata can shift the IQ view;
- decoding does not call pyadi or libiio.

Fail:

- protocol details leak into the collector or Zarr writer.

Status: **PASS**

Evidence:

- the parser supports fragmented synthetic input and strict exact-length
  hardware transfers;
- invalid sentinels decode as NaN and invalid required metadata fails closed;
- magic, version, header size, CRC, layout, feature, flag, and sequence checks
  precede IQ conversion;
- the one-hour soak verified bounded receive-buffer ownership.

### Python Gate G — Implement the PPlus compatibility facade

Add the transport switch to `PPlus.rx()` and a last-frame metadata cache.
Retain the existing `rssis()` and `gains()` signatures.

Direct mode:

```text
rx()       receives IQ+metadata and refreshes the cache
rssis()    returns cached end RSSI as float64[2]
gains()    returns cached end gain dB as float64[2]
```

IIO mode remains byte-for-byte equivalent to the historical behavior.

Pass:

- existing collector tests pass with only `.sdr.rx()` changed to `.rx()`;
- a test makes every host IIO gain/RSSI accessor raise, while direct collection
  still succeeds;
- calling `gains()` or `rssis()` before a direct frame fails as stale;
- returned arrays match the historical shape, dtype, units, and sign.

Fail:

- `ThreadedRX`, snapshots, Zarr schemas, models, or readers need transport-aware
  branches.

Status: **PASS**

Evidence:

- direct `PPlus.rx()` receives and caches one v2 frame;
- `rssis()` and `gains()` return cached `float64[2]` values and tests make the
  corresponding IIO accessors raise to prove they are not called;
- stale access before the first frame and after close/reconfiguration fails;
- IIO mode retains its existing calls.

### Integration Gate H — Record through the unchanged v4 Zarr path

Use the existing Rover YAML style with only:

```yaml
rx-transport: direct_usb
direct-usb:
  protocol-version: 2
  require-gain-metadata: true
```

Keep `data-version: 4`.

Run 5 frames, then 100 frames through `mavlink_radio_collection.py`.

Pass:

- existing v4 Zarr contains the normal `signal_matrix`, `gains`, and `rssis`;
- `gains` equals the transmitted end gain-dB pair;
- `rssis` equals the transmitted end RSSI pair;
- no host metadata IIO calls occur;
- ordinary dataset readers and existing tests require no changes.

Fail:

- direct mode requires v6/v7 merely to reproduce legacy fields;
- values have different units or sign from USB-IIO recordings.

Status: **PASS**

Evidence:

- 5-frame and 100-frame captures ran through
  `mavlink_radio_collection.py`, `DroneDataCollectorRaw`, and the existing v4
  LMDB-backed Zarr writer;
- the 100-frame Zarr reopened with finite gain/RSSI metadata and unchanged v4
  keys/dtypes;
- ordinary v4 readers required no transport branch.

### Hardware Gate I — Characterize and soak

Run:

1. equal manual gains;
2. unequal manual gains;
3. slow-attack AGC;
4. stepped RF attenuation for RSSI;
5. 100-frame throughput comparison;
6. ten-minute and one-hour soak; and
7. RAM-boot rollback to stock QSPI.

Pass:

- manual gain dB is exact;
- RSSI response is plausible and agrees with driver diagnostics;
- zero silent loss, unbounded growth, or metadata/IQ misassociation;
- throughput degradation is below 5%;
- rollback is demonstrated.

Fail:

- equal dB is used as proof that raw gain state did not change;
- endpoint RSSI is described as sample-exact or as whole-buffer power.

Status: **IMPLEMENTATION PASS; PHYSICAL CHARACTERIZATION INCOMPLETE**

Evidence:

- exact manual 20/20 and 20/40 dB captures passed;
- slow-attack AGC, 100-frame capture, 10-minute soak, one-hour 7,200-frame
  soak, and stock-QSPI rollback passed;
- direct v2 recorded 2.024 Hz versus the 2.058 Hz IIO baseline, about 1.7%
  lower and inside the 5% budget;
- final RSS reached a stable plateau rather than growing with frame count.

The unexecuted physical items are the calibrated stepped-attenuator stimulus
recorded under Gate D and the coherent common-CW phase characterization
recorded in the final audit.

### Deferred extension

The v2 header can carry start/end values and internal raw-state change flags,
but the compatibility-first Python/Zarr delivery stores only the end pairs in
the existing fields. A later, separately versioned dataset may preserve both
endpoints and FPGA in-buffer event positions. That extension is not allowed to
complicate or block the minimal compatibility delivery.

---

## Goal

Using one Pluto+ and the normal Rover 3 SPF collection path, capture the same
dual-channel, 524,288-sample frames through a RAM-booted direct-USB firmware.
Every IQ frame must carry device-local RX1 and RX2 raw gain-index snapshots
associated with the beginning and end of that frame.

The result is complete only when the normal SPF YAML-to-Zarr workflow records
100 valid direct-USB frames and their gain metadata without making per-frame
host-side IIO gain or RSSI reads.

## Frozen capture contract

The current committed configuration is the source of truth:

```text
data_collection/rover/rover_v3.1/capture_configs/
    rover_single_receiver_config_pi_3mhz.yaml
```

Its relevant contract is:

```yaml
receiver-port: 2
nelements: 2
rx-gain-mode: slow_attack
rx-buffers: 4
buffer-size: 524288
f-intermediate: 100000
f-carrier: 5.766e+9
f-sampling: 30.0e+6
bandwidth: 3.0e+6
seconds-per-sample: 0.5
```

In both transports:

- RX1 and RX2 are enabled;
- each stored frame has shape `(2, 524288)`;
- stored IQ has type `complex64`;
- the raw dual-channel CS16 IQ payload is 4,194,304 bytes;
- the normalized Zarr IQ frame is 8,388,608 bytes;
- one frame represents 17.476 ms of RF time;
- collection uses `mavlink_radio_collection.py`, `DroneDataCollectorRaw`, the
  normal collector queues, and the LMDB-backed Zarr writer;
- `--fake-drone` replaces only vehicle motion and position, not the radio.

The direct gadget must perform bounded, requested frame captures. An unbounded
30 MS/s dual-channel stream that drops data between host snapshots is not an
equivalent implementation.

## Gain semantics

The first delivery reads these registers locally on the Pluto ARM:

```text
RX1 gain index: 0x2B0 & 0x7f
RX2 gain index: 0x2B5 & 0x7f
invalid value:  0xff
```

The values are called `gain_index_start` and `gain_index_end`, but they are
device-local, buffer-associated snapshots. ARM register reads are not
sample-exact. In particular:

```text
start == end
```

means only that the observed endpoints are equal. It does not prove that gain
did not change and return inside the frame.

## Evidence and safety rules

Every hardware gate stores its command, source SHAs, firmware identity, Pluto
serial/path, logs, counters, and results beneath:

```text
artifacts/direct_usb_gain_metadata/rover3_one_radio/<run-name>/
```

Rules applying to every firmware gate:

- never flash the experimental image before it passes by RAM boot;
- retain and test the known QSPI rollback image;
- preserve standard USB-IIO for configuration;
- never run pyadi RX and direct-USB RX concurrently;
- stop at the first failed gate and fix that layer before continuing;
- do not reinterpret malformed metadata bytes as IQ.

---

## Gate 0 — Freeze the real USB-IIO Zarr baseline

Use the committed Rover single-receiver YAML with `-n 100`. Record using the
normal SPF command and validate the completed Zarr.

Current evidence:

```text
artifacts/direct_usb_gain_metadata/rover3_one_radio/
    2026-07-25_zarr_iio_100/
```

Observed result:

- 100/100 frames stored;
- shape `(100, 2, 524288)`;
- `complex64`;
- all checked IQ finite and nonzero;
- median frame interval 486.0 ms;
- median steady rate 2.058 frames/s;
- median logical IQ rate 16.46 MiB/s;
- no capture or writer errors.

Pass:

- the Zarr reopens and contains exactly 100 valid frames;
- the effective saved YAML and exact command are retained;
- stored timestamps and progress output provide the throughput baseline.

Fail:

- any frame is missing, malformed, non-finite, or written through a path other
  than the normal collector.

Status: **PASS**

## Gate 1 — Freeze and test protocol v1 without hardware

Define one packed, little-endian, versioned header. It must include:

```text
magic
version
header_bytes
feature and validity flags
stream_id
buffer_sequence
first_sample_sequence
samples_per_channel
IQ payload bytes
enabled scan mask
sample format and channel count
gain_index_start[2]
gain_index_end[2]
gain-read durations
future first-change-sample fields
header CRC32
```

Create one golden byte vector parsed by both the gadget C code and SPF Python
code. The IQ offset is always `header_bytes`, never a duplicated constant.

Pass:

- C and Python agree on every byte and the exact header size;
- bad magic, version, size, CRC, sample count, payload count, channel mask, and
  sequence are independently rejected;
- fragmentation tests prove a header split across USB reads is handled;
- an old client cannot accidentally treat an enabled header as IQ.

Fail:

- either side relies on native C struct padding or host endianness;
- any malformed header can shift IQ without causing an error.

Status: **PASS**

Evidence:

```text
artifacts/direct_usb_gain_metadata/rover3_one_radio/
    2026-07-25_protocol_v1/results.md
```

## Gate 2 — Reproduce the unmodified direct-USB stack by RAM boot

Build the pinned Quantulum firmware and host reference at recorded SHAs. Verify
the documented XSA/checksum exception explicitly rather than disabling checksum
validation generally. RAM boot the resulting `pluto.dfu`.

Using standard USB-IIO, configure the exact frozen Rover radio parameters.
Using the unmodified direct interface, obtain dual-channel IQ and validate
channel order against USB-IIO.

Pass:

- build is reproducible from recorded SHAs and hashes;
- RAM boot succeeds and QSPI remains unchanged;
- standard USB-IIO and the vendor interface both enumerate;
- pyadi configuration still works;
- direct IQ has the expected RX1-I/Q, RX2-I/Q order;
- STOP, restart, USB disconnect, and QSPI rollback work.

Fail:

- persistent flash is required;
- standard USB-IIO disappears;
- direct and IIO RX paths contend concurrently;
- channel or sample order is uncertain.

Status: **PASS**

Hardware evidence:

- RAM boot enumerated standard IIO plus vendor interface 6;
- a matched 0/60 dB manual-gain capture placed the stronger signal in channel
  1 through both direct USB and USB-IIO;
- direct and IIO RX ownership were kept mutually exclusive;
- a normal reboot restored QSPI v0.37 and removed the custom protocol.

## Gate 3 — Add negotiated finite-frame transport with dummy metadata

Keep the legacy START command intact. Add:

1. a capability/version query;
2. a versioned START request containing protocol version, features, enabled
   scan mask, samples per channel, and finite frame count;
3. a fixed transfer containing one v1 header followed by one IQ payload.

For this gate, use unmistakable dummy gain values and set a dedicated
`DUMMY_GAIN` flag. Request exactly one 524,288-sample dual-channel frame at a
time.

Pass:

- the host negotiates v1 before requesting metadata;
- exactly one request yields exactly one header plus 4,194,304 IQ bytes;
- dummy values, CRC, sizes, and sequence round-trip exactly;
- one five-frame request yields sequences 0–4 with no extra frames;
- IQ begins at `header_bytes` and matches the frozen frame contract;
- legacy clients continue to use IQ-only transfers.

Fail:

- the gadget streams indefinitely after a one-frame request;
- the host guesses protocol support;
- dummy header bytes enter the IQ arrays;
- a short or long transfer is accepted.

Status: **PASS**

The final host queues an exact 4,194,384-byte bulk-IN transfer before START.
Both a 4,096-sample smoke frame and the full 524,288-sample Rover frame passed
strict header, CRC, payload-size, sequence, and IQ-offset validation.

## Gate 4 — Validate local gain-register reads separately

Before putting gain reads in the real-time frame path, add an on-device helper
that locates `ad9361-phy` and reads RX1/RX2 as one logical pair. Measure each
pair-read duration and count failures.

Tests:

1. equal fixed manual gains;
2. unequal fixed manual gains;
3. change RX1 only;
4. change RX2 only;
5. repeat at each Rover RF band;
6. confirm full gain-table mode;
7. compare with occasional driver `hardwaregain` readback outside streaming.

Pass:

- each channel responds only to its intended manual-gain changes;
- all valid values are in `[0, 127]`;
- failures return `[0xff, 0xff]` and do not terminate the tool;
- read latency distribution and failure count are archived;
- full-table mode is proved or unsupported modes fail explicitly.

Fail:

- RX1/RX2 mapping is ambiguous;
- split-table or digital-gain state is silently represented as a full-table
  index;
- one failed register read produces a seemingly valid pair.

Status: **PASS**

On-device full-table reads were valid with zero failures. Manual 20/20 dB
produced `[34,34]`; manual 20/40 dB produced `[34,54]`. RX1-only and RX2-only
changes affected only their intended raw index. Across the 100-frame AGC run,
pair-read p50 was approximately 0.49 ms and p99 remained below 0.55 ms.

## Gate 5 — Put real start/end gains on each device frame

Replace dummy values with on-device register snapshots. Start with the clearer
diagnostic implementation:

```text
read start pair locally
capture/refill one finite IQ frame
read end pair locally
construct header
send header + IQ in one bulk transfer
```

No allocation is allowed in the frame loop. A gain-read failure invalidates the
appropriate endpoint but does not discard otherwise valid IQ.

Benchmark this against the cached implementation (`previous end` becomes the
next `start`) only after the explicit pre/post version works. Retain the version
whose semantics and performance are best supported by measurements.

Pass:

- manual fixed gain gives valid, repeatable start/end indices;
- changing RX1 changes only RX1 endpoint metadata;
- changing RX2 changes only RX2 endpoint metadata;
- endpoint-change flags exactly match endpoint comparisons;
- `0xff` and validity flags are emitted on injected read failure;
- header and IQ remain in the same transfer and share one sequence;
- metadata-on throughput is not materially worse than dummy-header throughput.

Fail:

- host-side IIO attribute calls are needed per frame;
- gain-read failure stops IQ or masquerades as valid gain;
- endpoint equality is exposed or documented as `gain_stable`.

Status: **PASS**

Evidence:

```text
artifacts/direct_usb_gain_metadata/rover3_one_radio/
    2026-07-25_real_gain_firmware_build/results.md
    2026-07-25_hardware_acceptance/results.md
```

The accepted image emitted valid real endpoint pairs, channel-specific change
flags, a full-table flag, and zero gain-read failures. The on-device log
confirmed that each 80-byte header and 4,194,304-byte IQ payload were submitted
as one 4,194,384-byte transfer.

## Gate 6 — Implement the SPF direct-USB receiver API

Implement device discovery by serial number plus physical USB path, capability
query, interface claim, finite START/STOP, asynchronous bulk-IN transfers,
strict header parsing, and a bounded queue.

Expose one result object containing:

```text
signal_matrix              complex64[2, 524288]
gain_index_start           uint8[2]
gain_index_end             uint8[2]
gain_metadata_valid        bool
gain_endpoints_equal       bool[2]
gain_metadata_flags        uint16
stream/buffer/sample sequence
gain-read durations
future first-change fields
```

Pass:

- synthetic tests cover good frames, fragmentation, timeout, cancellation,
  corrupt headers, short/extra payload, sequence gap/reset, queue overflow, and
  cleanup;
- hardware returns the exact frozen signal shape and channel order;
- sequence loss and queue loss are explicit counters;
- no unnecessary full-frame copy is introduced before SPF normalization.

Fail:

- metadata is retrieved through a separate “last frame” call;
- device identity relies only on transient bus/device numbers;
- queue growth is unbounded or loss is silent.

Status: **PASS**

Hardware exercised the exact-transfer asynchronous path at the Rover frame
size. A USBFS allocation-limit test additionally proved cleanup of a partially
submitted queue before START.

## Gate 7 — Integrate transport selection into the existing collector

Add an optional receiver setting:

```yaml
rx-transport: direct_usb
direct-usb:
  protocol-version: 1
  frame-count-per-request: 1
  require-gain-metadata: true
```

Default remains `iio`. `PPlus` continues to use pyadi for configuration but
delegates RX ownership to the direct receiver after setup. `ThreadedRXRawV4`
must call one transport-neutral `rx_with_metadata()` method.

Pass:

- the committed IIO config behaves unchanged;
- a near-copy selecting `direct_usb` traverses the same
  `mavlink_radio_collection.py` and `DroneDataCollectorRaw` lifecycle;
- direct mode never calls `sdr.rx()`, `gains()`, or `rssis()` per frame;
- fake transport CI tests exercise the normal collector, not a special
  benchmark loop;
- exceptions and Ctrl-C close both pyadi control and direct interfaces.

Fail:

- the direct benchmark bypasses the production collector;
- transport choice is hard-coded in a test script;
- the code silently falls back from direct USB to IIO.

Status: **PASS**

The normal collector completed 100 direct frames without calling pyadi RX,
RSSI, or hardware-gain reads in its frame hot path.

## Gate 8 — Store gain metadata without overloading legacy fields

Add a versioned dataset schema (preferably v6) or a rigorously versioned optional
group. Do not place raw indices into legacy dB-valued `gains`.

Per frame, store:

```text
gain_index_start[2]
gain_index_end[2]
gain_metadata_valid
gain_endpoints_equal[2]
gain_metadata_flags
sample_sequence
gain_start_read_duration_ns
gain_end_read_duration_ns
first_gain_change_sample[2]  # -1 means unavailable
iq_power_dbfs[2]
```

Pass:

- a fake direct capture writes a Zarr through the normal collector;
- close/reopen round-trips every field and IQ sample;
- invalid metadata remains explicit;
- old v4 readers and the default IIO collection path still pass their tests;
- provenance records firmware, gadget, protocol, host-client SHA, serial, and
  USB path.

Fail:

- legacy `gains` changes meaning;
- absent metadata is filled with a plausible valid value;
- schema selection depends on inspecting arbitrary field presence.

Status: **PASS**

The finalized real Zarr reopened with all typed v6 arrays, runtime USB identity
attributes, and associated firmware/config provenance. Legacy gain and RSSI
arrays were NaN rather than overloaded with raw indices.

## Gate 9 — Matched 100-frame RAM-booted direct-USB Zarr capture

RAM boot the metadata firmware. Configure with the same RF values as Gate 0 and
run the same normal collection command with `-n 100`; only transport/schema
fields may differ.

Required assertions:

- exactly 100 stored frames;
- every IQ frame is `(2, 524288)` and `complex64`;
- all IQ is finite and both channels are nonzero;
- 100 valid, CRC-checked metadata headers;
- strictly increasing buffer/sample sequence within each stream, with resets
  allowed only when a new nonzero stream ID proves an explicit START;
- zero USB errors, queue drops, and device overflows;
- no per-frame host gain or RSSI attribute reads;
- Zarr closes, reopens, and round-trips all gain metadata;
- memory remains bounded;
- median cadence meets the 0.5-second Rover contract.

Pass:

- all assertions hold;
- direct and IIO configs are RF-equivalent;
- IQ statistics and throughput are reported side-by-side with Gate 0;
- QSPI rollback still succeeds after the run.

Fail:

- any frame/metadata association is uncertain;
- retry hides a lost frame or sequence;
- throughput is measured only below the collector/Zarr layer.

Status: **PASS**

Evidence:

```text
artifacts/direct_usb_gain_metadata/rover3_one_radio/
    2026-07-25_zarr_direct_usb_100/
    2026-07-25_hardware_acceptance/results.md
```

The strict validator passed 100/100 frames at a median 498.4 ms interval and
2.006 Hz. It found 68 endpoint-changed AGC frames, zero unsafe flags, and no
legacy hot-path gain/RSSI values.

## Gate 10 — Gain-behaviour acceptance

Run three 100-frame captures:

1. equal fixed manual RX gains;
2. unequal fixed manual RX gains;
3. slow-attack AGC with controlled level changes.

For manual gain, indices should remain repeatable except at commanded
boundaries. For AGC, endpoint changes must be plausible and independently
observable in RX1/RX2.

Pass:

- manual tests show the correct channel mapping and endpoint behaviour;
- induced transitions produce the appropriate endpoint-change flags;
- equal endpoints are reported only as `gain_endpoints_equal`;
- results include phase and IQ-power statistics correlated with gain metadata;
- all commands, images, SHAs, logs, and datasets are sufficient to reproduce
  the result.

Fail:

- the implementation claims that equal endpoints prove no in-frame change;
- gain metadata cannot be correlated unambiguously with its IQ frame.

Status: **PASS FOR FRAME ASSOCIATION; CONTROLLED PHASE CHARACTERIZATION PENDING**

The two 100-frame manual captures were constant at `[34,34]` and `[34,54]`.
RX1-only and RX2-only induced changes set only their corresponding endpoint
flags. The slow-attack run produced plausible independent endpoint motion in
68/100 frames. No coherent common CW was attached, so ambient phase values are
not treated as phase characterization. This does not weaken the verified
frame/gain association, but it means the phase-characterization portion of this
gate has not passed.

## Completion boundary

The CPU-side implementation is complete. Full plan closure additionally
requires the two physical RF characterizations in
`docs/direct_usb_gain_completion_audit.md`. FPGA CTRL_OUT event capture remains
a separate subsequent project; it will strengthen detection of in-frame
transitions but is not part of the CPU-side implementation.
