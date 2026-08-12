# PlutoPlus SPF libiio frame-metadata v5

Date: 2026-08-12

## Summary

This release carries an IIO RX frame together with its capture index, hardware
sample sequence/time anchor, gain at frame start and end, in-frame gain
observations, and RSSI at frame start and end. The ordinary libiio buffer API
and IQ byte layout remain unchanged. The radio uses libiio/iiOD 0.25; the
supported patched host libraries are the pinned 0.25 and 0.26 lines.

Capture remains request-driven and bounded by the configured IIO kernel-buffer
count. A slow host does not cause iiOD to build an unbounded userspace or TCP
history. Once the finite kernel-buffer inventory has been consumed, the radio
does not request another frame until the host requests one. If the requested RF
rate exceeds transport capacity, intermediate radio time is skipped and the
next returned frame exposes that fact through its increasing capture and sample
sequences; stale frames are not accumulated to fill host memory.

## USB and IIO IP/TCP benchmark

The release benchmark used both radios, two kernel buffers, 262144 samples per
channel, two warm-up frames, and 12 timed dual-channel CS16 frames per cell.
Every rate was run in ordinary and metadata mode over USB and standard libiio
`ip:`/TCP. The complete matrix was repeated with host libiio 0.25 and 0.26.

The values below are conservative minima across both radios and both host
versions. Payload rate is IQ only (8 bytes per dual-channel sample); coverage
is captured samples divided by the advancing FPGA sample span. `yes` requires
at least 98% coverage and host delivery of at least 90% of the configured RF
rate.

| RF rate | Required IQ | USB metadata | USB coverage | USB continuous | TCP metadata | TCP coverage | TCP continuous |
|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 1.0 MS/s | 8 MB/s | 7.988 MB/s | 100.0% | yes | 7.979 MB/s | 100.0% | yes |
| 1.5 MS/s | 12 MB/s | 11.979 MB/s | 100.0% | yes | 11.967 MB/s | 100.0% | yes |
| 2.0 MS/s | 16 MB/s | 15.954 MB/s | 100.0% | yes | 15.990 MB/s | 100.0% | yes |
| 2.5 MS/s | 20 MB/s | 13.294 MB/s | 66.7% | no | 19.982 MB/s | 100.0% | yes |
| 3.0 MS/s | 24 MB/s | 15.933 MB/s | 70.6% | no | 23.948 MB/s | 100.0% | yes |
| 5.0 MS/s | 40 MB/s | 20.701 MB/s | 52.2% | no | 28.430 MB/s | 75.0% | no |
| 10 MS/s | 80 MB/s | 21.515 MB/s | 28.6% | no | 38.832 MB/s | 52.2% | no |
| 20 MS/s | 160 MB/s | 21.139 MB/s | 14.3% | no | 40.503 MB/s | 27.3% | no |
| 30 MS/s | 240 MB/s | 21.453 MB/s | 9.8% | no | 41.301 MB/s | 18.5% | no |

The qualified continuous metadata limits are therefore 2 MS/s over USB and
3 MS/s over IP/TCP for this hardware and network. At 30 MS/s, ordinary IIO
delivered 22.650--22.789 MB/s over USB and 46.967--49.246 MB/s over TCP;
metadata IIO delivered 21.453--21.783 MB/s and 41.301--42.156 MB/s,
respectively. Host 0.25 and 0.26 behavior was effectively indistinguishable.

These are finite-request capture rates, not a claim that an overloaded link
preserves every continuously produced sample. Applications that require a
gapless stream must configure an RF rate at or below the continuous boundary.

## Large-frame AGC metadata gate

Both radios captured 524288-sample frames at 3 MS/s while a cabled, split,
30 dB attenuated TX2 tone changed between -30 and -60 dB every 70 ms. RX1 and
RX2 used independent slow-attack AGC loops. Every accepted frame had valid gain
and RSSI metadata, ordered sample-associated gain observations, zero clipping,
and no overflow or metadata-read failure flags.

For both host versions, seven of eight frames per radio contained at least a
2 dB gain change on both channels. The maximum RX1/RX2 gain disagreement was
1 dB on radio A and 2 dB on radio B. Frame-start and frame-end gain snapshots
matched the nearest gain-history observations exactly in the recorded runs.
No negative gain sentinel or one-channel-only result was accepted.

## Broader hardware qualification

Before the final release-name rebuild, the identical data path was persistently
flashed to both radios and passed 144 fresh RF sessions and 576 metadata frames
over USB and TCP while LO and RF bandwidth were repeatedly changed. The matrix
covered 868, 915, 1280, 2412, 4000, and 5804 MHz and 0.8, 1.5, and 3.0 MHz
bandwidths with host 0.25 and 0.26. Both radios retained the firmware across
QSPI reboot. Slow-host tests from 30 seconds through five minutes confirmed
that stale data remained bounded by the requested one or four kernel buffers,
with stable iiOD RSS and no growing TCP queue.

The final v5 artifact is a release-identity rebuild from the same pinned radio
components. It is separately confirmation-tested on both radios before
publication; the final artifact identity and checksums are published with the
GitHub release.

## Reproducible focused tests

The new tests are opt-in and are not collected as ordinary hardware-free unit
tests:

```text
pytest tests/radio_hardware/test_iio_metadata_agc_hardware.py \
  --radio-hardware --radio-expected-count=2 --radio-tx-loopback \
  --radio-tx-loopback-attenuation-db=30 --radio-report-dir=<dir>

pytest tests/radio_hardware/test_iio_transport_benchmark_hardware.py \
  --radio-hardware --radio-expected-count=2 --radio-report-dir=<dir>
```

The machine-readable reports are in `libiio_v5_host025/` and
`libiio_v5_host026/` beside this file. The full pytest suite was deliberately
not run on the Pi.

## Known limitation

The PlutoPlus DDS can occasionally be silent on its first arm after a retune.
The RF tests detect this before accepting a frame and repeat the documented
TX-quadrature calibration/arm. This is a TX setup behavior, not an RX metadata
or TCP framing failure; RX-only operation is unaffected.
