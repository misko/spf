# Direct-USB timestamp validation

Date: 2026-07-25
Firmware: `v0.38_plutoplus_with_timestamping-1-ga098-dirty`, RAM boot only
Host client: `pgreenland/SoapyPlutoSDR` at
`2bbf77152d4e6d30c6630807fdfc8a869a528cf3`

## Result

**PASS with a documented reference-client restart defect**

Timestamp reception works end-to-end over the custom direct-USB interface.
Fresh stream instances delivered valid, exactly spaced timestamps with
two-channel IQ. Reusing the same Soapy stream object across deactivate/activate
exposed a stale-queue discontinuity that our native client must not copy.

Nothing was persistently flashed. After testing, the Pluto rebooted back to
its original QSPI firmware `v0.37-dirty`.

## Important correction to the 2026-07-24 baseline

The reference Soapy branch requires the device argument `direct=1` to open the
custom USB gadget. The earlier generic Soapy rate tests omitted this argument,
so they exercised standard USB-IIO on the timestamp firmware rather than the
custom direct-USB data path.

This validation explicitly required and observed:

```text
[INFO] USB direct mode enabled!
```

## Configuration

```text
URI: usb:1.9.5
direct: 1
RX channels: 0,1
format: CS16
sample rate: 1,000,000 samples/s
timestamp_every: 4096 samples
stream MTU: 4096 samples/channel
```

For dual RX, the raw gadget buffer contains one 64-bit timestamp followed by
4096 time samples. The reference client removes the eight timestamp bytes,
reduces the raw buffer by the corresponding single dual-channel time sample,
and exposes an MTU of 4096 IQ samples per channel.

## Fresh-stream acceptance test

Two completely independent device/stream instances were tested. Each
instance received 200 buffers.

Both runs produced:

```text
successful buffers: 200/200
read errors: 0
missing SOAPY_SDR_HAS_TIME flags: 0
wrong buffer sizes: 0
non-monotonic timestamps: 0
unexpected timestamp deltas: 0
unique timestamp delta: 4,096,000 ns
```

Expected delta:

```text
4096 samples / 1,000,000 samples/s = 4.096 ms
```

The second independent instance also passed, demonstrating clean close,
reopen, START, and STOP operation.

## Same-object restart defect

Reusing a single Soapy stream object with:

```text
activate -> receive -> deactivate -> activate -> receive
```

does not reliably preserve continuity.

Observed as an unprivileged process:

```text
run 1: all 200 deltas exactly 4,096,000 ns
run 2: one delta of 258,048,000 ns
```

Observed with realtime-priority permission:

```text
run 1: all 250 deltas exactly 4,096,000 ns
run 2: first in-stream delta was 274,432,000 ns
```

This is not caused by failure to assign realtime priority. Source inspection
shows:

- `_stop()` joins the USB thread but does not clear its receive queue;
- the next thread deliberately drops 32 initial raw buffers;
- the consumer can therefore receive a stale queued buffer followed by data
  from the new stream generation.

The timestamp made this loss visible. A native SPF client must:

1. clear completed and queued buffers during STOP;
2. tag buffers with a stream generation;
3. reject buffers from an earlier generation;
4. allow timestamp/sequence reset only at an explicit restart boundary;
5. report all other discontinuities.

## Direct transport throughput observation

With `direct=1`, the generic two-channel CS16 test requested 1 MS/s and
observed approximately:

```text
0.776 MS/s
6.21 MB/s
```

This host therefore does not quite sustain 1 MS/s dual-channel direct USB in
the reference client. The timestamp correctness test still passed for finite
captures, but production throughput remains a separate open gate.

## Gate decision

The timestamp protocol and extraction path are proven sufficiently to begin
the versioned dummy gain-metadata framing work.

The same-object restart defect is a host-client requirement, not a reason to
modify timestamp semantics or begin FPGA work.
