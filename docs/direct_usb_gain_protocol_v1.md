# SPF direct-USB RX gain metadata protocol v1

Status: draft implementation specification
Byte order: little-endian
Header size: 80 bytes
IQ layout: two complex receivers, time-interleaved CS16

## Negotiation and finite capture

Legacy command `0x10` remains an IQ-only START and is not changed.

An SPF host first sends vendor/interface IN request `0x12`. A v1 gadget returns
this packed 32-byte capability record:

| Offset | Type | Field |
|---:|---|---|
| 0 | `uint32` | magic `0x50434753`, wire bytes `SGCP` |
| 4 | `uint16` | response bytes, 32 |
| 6 | `uint16` | minimum protocol version |
| 8 | `uint16` | maximum protocol version |
| 10 | `uint16` | reserved, zero |
| 12 | `uint32` | supported metadata features |
| 16 | `uint32` | maximum samples per channel |
| 20 | `uint32` | maximum finite frame count |
| 24 | `uint32` | capability flags |
| 28 | `uint32` | reserved, zero |

Capability bit 0 means finite RX is supported. Bit 1 means the bring-up
firmware deliberately returns dummy gain values.

After validating capabilities, the host sends vendor/interface OUT request
`0x13`, target RX, with this packed 32-byte body:

| Offset | Type | Field |
|---:|---|---|
| 0 | `uint32` | magic `0x31534753`, wire bytes `SGS1` |
| 4 | `uint16` | protocol version, 1 |
| 6 | `uint16` | request bytes, 32 |
| 8 | `uint32` | requested metadata features |
| 12 | `uint32` | enabled scan mask, `0x0f` |
| 16 | `uint32` | samples per channel |
| 20 | `uint32` | finite frame count, 1–16 |
| 24 | `uint32` | reserved, zero |
| 28 | `uint32` | reserved, zero |

Request `0x11`, target RX, stops and drains the stream. Every successful
versioned START creates a new nonzero stream ID. The gadget captures exactly
the requested number of buffers and then removes the IIO poll source; it does
not continue draining or dropping a 30 MS/s stream behind the host.

## Framing

Each RX frame is:

```text
80-byte metadata header
followed by iq_payload_bytes bytes of IQ
```

USB bulk completions are not frame boundaries. A host accumulates bytes until a
complete header and its declared payload are available. On any protocol error,
the host discards buffered bytes, stops the stream, and negotiates a new stream.
It must not scan forward for another magic value.

Protocol v1 permits exactly:

```text
enabled_scan_mask = 0x0000000f
channel_count = 2
sample_format = 1
iq_payload_bytes = samples_per_channel * 2 * 4
```

For every time sample, the payload contains:

```text
RX1 I int16
RX1 Q int16
RX2 I int16
RX2 Q int16
```

The hardware IQ-layout tests must confirm that IIO scan order matches this
contract before direct mode is enabled.

## Header

| Offset | Type | Field |
|---:|---|---|
| 0 | `uint32` | magic, `0x314d4753`, wire bytes `SGM1` |
| 4 | `uint16` | version, `1` |
| 6 | `uint16` | header bytes, `80` |
| 8 | `uint32` | negotiated feature bits |
| 12 | `uint32` | metadata/status flags |
| 16 | `uint64` | nonzero stream ID |
| 24 | `uint64` | buffer sequence |
| 32 | `uint64` | first transported sample sequence |
| 40 | `uint32` | samples per complex RX channel |
| 44 | `uint32` | IQ payload bytes |
| 48 | `uint32` | enabled IIO scan mask |
| 52 | `uint16` | sample-format/layout ID |
| 54 | `uint8` | complex RX channel count |
| 55 | `uint8` | RX1 gain start observation |
| 56 | `uint8` | RX2 gain start observation |
| 57 | `uint8` | RX1 gain end observation |
| 58 | `uint8` | RX2 gain end observation |
| 59 | `uint8` | reserved, zero |
| 60 | `uint32` | start-observation read duration, ns |
| 64 | `uint32` | end-observation read duration, ns |
| 68 | `uint32` | RX1 first FPGA change sample |
| 72 | `uint32` | RX2 first FPGA change sample |
| 76 | `uint32` | header CRC-32 |

The raw gain observations are full-table seven-bit indices. `0xff` means
invalid/unavailable. FPGA event position `0xffffffff` means unavailable or no
event as determined by the validity/change flags.

`stream_id` changes on every successful START. Buffer sequence begins at zero.
First-sample sequence is expressed in transported samples after any on-device
decimation. A new stream ID is required to reset either sequence.

## Features

| Bit | Meaning |
|---:|---|
| 0 | gain endpoint observations |
| 1 | header CRC-32 |
| 2 | first-sample sequence |
| 3 | FPGA gain events |

CRC is mandatory in protocol v1.

## Flags

| Bit | Meaning |
|---:|---|
| 0 | both start gain observations valid |
| 1 | both end gain observations valid |
| 2 | RX1 observed endpoints differ |
| 3 | RX2 observed endpoints differ |
| 4 | first-sample sequence valid |
| 5 | FPGA event fields valid |
| 6 | FPGA observed RX1 change in this payload |
| 7 | FPGA observed RX2 change in this payload |
| 8 | FPGA reports RX1 locked at end |
| 9 | FPGA reports RX2 locked at end |
| 10 | full gain-table mode confirmed |
| 11 | device IIO overflow observed |
| 12 | at least one local gain read failed |
| 13 | FPGA event capture overflowed |
| 14 | test-only dummy gain values; never application-valid |

Protocol v1 treats a register-read pair as one validity unit. A failed RX1 or
RX2 read invalidates both values for that observation.

All unassigned feature and flag bits are reserved and must be zero. The byte at
offset 59 is also reserved and must be zero. A host rejects unknown set bits or
a nonzero reserved byte, even when the CRC is valid.

The dummy-gain flag exists only for transport bring-up. A header carrying it may
be parsed and inspected, but its gain values must never be exposed as valid
radio metadata.

At a newly negotiated START, both buffer sequence and first-sample sequence
begin at zero. A sequence reset requires a new nonzero stream ID and a reset
parser; it is never inferred from a backwards value.

## CRC

The CRC is CRC-32/ISO-HDLC as implemented by `zlib.crc32`:

```text
polynomial = 0x04c11db7
reflected input and output
initial value = 0xffffffff
final XOR = 0xffffffff
```

It covers all 80 header bytes with bytes 76–79 set to zero. It does not cover
the IQ payload. USB transport integrity protects payload bytes; a future
protocol may negotiate a payload checksum if required.

## Gain interpretation

The ARM observations are associated with successive IIO refill completions.
They are not measurements of the literal first and last RF samples.

```text
start != end
    the two associated observations differ; mark the buffer phase-unsafe

start == end
    no endpoint difference was observed
```

Equality does not rule out an internal transition such as:

```text
42 -> 37 -> 42
```

Only valid FPGA event capture can make a stronger in-buffer statement, and its
marker-to-IQ timing must be bench-characterized.

## Golden header

The canonical Python test vector is:

```text
53474d3101005000
0700000017040000
f0debc9a78563412
0700000000000000
0004000000000000
0800000040000000
0f0000000100022a
2b292b00b0040000
14050000ffffffff
ffffffff796afe5d
```

The Pluto C serializer and host parser must reproduce this vector exactly.
