# Sample-associated gain observations: protocol v3

Status: implementation candidate; not yet promoted to rover production.

Protocol v3 extends the existing direct-USB radio frame with a bounded series
of RX1/RX2 gain observations. It preserves the v2 endpoint gain and RSSI fields
for existing V7 consumers. Production configurations remain on protocol v2
until the HDL, firmware, and two-radio acceptance gates below pass.

## What is measured

Each observation contains:

- RX1 and RX2 full-table gain indices;
- their locally resolved dB values;
- the FPGA RX sample counter immediately before and after the paired AD936x
  register reads;
- read duration and validity flags.

The inline 64-bit FPGA timestamp defines the exact first IQ sample in a frame.
An observation belongs to a frame when its counter interval overlaps the
frame's half-open sample interval:

```text
[observation.sample_before, observation.sample_after]
overlaps
[frame.first_sample, frame.first_sample + frame.samples_per_channel)
```

This is deliberately an interval, not a fabricated point sample. The ARM does
not know at which instant inside the two counter reads the AD936x gain register
value became effective. Equal observations also do not prove that AGC did not
change and return between reads.

## Expected resolution

The measured paired local gain read is approximately 0.49 ms median and
0.54 ms p99 on the current Pluto+ firmware. At 30 MS/s that spans roughly
14,700 to 16,200 IQ samples. Consequently:

- `32768` samples (1.092 ms) is the initial production candidate interval;
- a 524,288-sample frame nominally carries 16 observations;
- the default wire capacity is 32, leaving room for boundary-straddling reads;
- requesting `2048` samples (68.3 microseconds) is legal but cannot make the
  ARM/SPI read complete that quickly; the actual counter intervals remain the
  source of truth;
- 256 reliable observations in the largest frame require a later FPGA
  CTRL_OUT event implementation, not faster CPU polling.

## Wire layout

The v3 frame is transport-neutral:

```text
fixed v3 prefix
fixed-capacity gain-observation records
fixed-capacity future FPGA-event records
CRC-32
two-channel CS16 IQ payload
```

The header declares record sizes, counts, and capacities. Unused capacity is
zero-filled. The receiver rejects invalid magic/version/size/CRC, unordered or
off-frame observations, inconsistent overflow state, and sequence regressions.
The V7 Zarr writer stores counts plus bounded arrays, using explicit sentinel
padding. V1 and v2 decoding remain unchanged.

Direct USB carries one complete v3 frame in one bulk transfer. The direct-IP
implementation should fragment and reassemble the exact same frame bytes; it
must not define a second radio-metadata schema. A versioned UDP fragment header
needs 32-bit fragment indices/counts and byte offsets because a 4 MiB IQ frame
cannot fit the existing IP gadget's 8-bit packet counters.

## HDL alignment path

The timestamp design already has one 64-bit counter clocked by accepted RX
samples. Protocol v3 uses that same counter in both places:

1. its 64-bit value is inserted ahead of each IQ frame;
2. its low 32 bits are transferred coherently into the CPU clock domain;
3. the CPU reads them at the AD9361 ADC GPIO-status register (`0x800000B8`)
   immediately before and after each gain pair;
4. firmware extends the low word near the frame's exact 64-bit timestamp.

The 32-bit transfer uses a closed-loop multi-bit clock-domain crossing. Directly
synchronizing individual binary counter bits would permit torn values at carry
boundaries and is not acceptable.

## Promotion gates

| Gate | Pass | Fail action |
|---|---|---|
| Host protocol | Golden v3 frame, fragmentation, CRC, sequence, and wrap tests pass | Do not build firmware |
| Gadget native | C build and focused CTest suite pass | Fix before firmware integration |
| HDL CDC | Coherent-counter simulation passes; Vivado validates IP and block design | Do not create a DFU |
| RAM boot | Both radios enumerate standard IIO and direct USB; v2 still captures | Roll back RAM image |
| Counter identity | GPIO low word agrees with each inline timestamp modulo 2^32 and advances monotonically | Disable protocol v3 |
| Observation coverage | Every frame has at least one valid overlapping observation; no unexplained overflow | Fail the capture closed |
| IQ integrity | Channel order, payload length, first sample, and phase match v2 baseline | Reject candidate firmware |
| Throughput | 100-frame and soak captures have no sequence gaps, USB errors, or queue drops at production settings | Keep v2 in production |
| Zarr | Reopened V7 contains correct counts, sentinels, serial/firmware provenance, and readable IQ | Do not promote configs |
| Dual radio | Both radios capture concurrently through repeated start/stop and reboot trials | Do not deploy to rovers |

Only after all gates pass should rover configs select protocol v3. Persistent
QSPI flashing follows a successful RAM-boot campaign and retains the current
known-good image as the rollback artifact.
