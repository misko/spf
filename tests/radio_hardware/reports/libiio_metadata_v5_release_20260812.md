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
| 1.0 MS/s | 8 MB/s | 7.990 MB/s | 100.0% | yes | 7.992 MB/s | 100.0% | yes |
| 1.5 MS/s | 12 MB/s | 11.977 MB/s | 100.0% | yes | 11.950 MB/s | 100.0% | yes |
| 2.0 MS/s | 16 MB/s | 15.956 MB/s | 100.0% | yes | 16.000 MB/s | 100.0% | yes |
| 2.5 MS/s | 20 MB/s | 13.311 MB/s | 70.6% | no | 19.990 MB/s | 100.0% | yes |
| 3.0 MS/s | 24 MB/s | 15.937 MB/s | 70.6% | no | 23.880 MB/s | 100.0% | yes |
| 5.0 MS/s | 40 MB/s | 20.419 MB/s | 52.2% | no | 28.376 MB/s | 75.0% | no |
| 10 MS/s | 80 MB/s | 21.293 MB/s | 27.9% | no | 38.313 MB/s | 50.0% | no |
| 20 MS/s | 160 MB/s | 21.565 MB/s | 14.5% | no | 40.313 MB/s | 27.3% | no |
| 30 MS/s | 240 MB/s | 21.446 MB/s | 9.8% | no | 40.814 MB/s | 18.2% | no |

The qualified continuous metadata limits are therefore 2 MS/s over USB and
3 MS/s over IP/TCP for this hardware and network. At 30 MS/s, ordinary IIO
delivered 22.472--22.715 MB/s over USB and 46.942--49.629 MB/s over TCP;
metadata IIO delivered 21.446--21.800 MB/s and 40.814--41.635 MB/s,
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

Seven or eight of eight frames per radio and host version contained at least a
2 dB gain change on both channels. The maximum RX1/RX2 gain disagreement was
1 dB on radio A and 3 dB on radio B. Frame-start and frame-end gain snapshots
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
components. Its source commit is
`d7c87a9a28094ee6f0b23cb47df9ff737b5a69d8`; its DFU SHA-256 is
`948b46506febacb087f3955be86015e074f8c0e3370a9dfc6a942e735d97f882`.
Both radios were persistently flashed with these bytes, ran both complete host
matrices, rebooted from QSPI, retained the exact v5 identity, and passed the
post-reboot USB/TCP metadata smoke.

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

The exact-release machine-readable reports are in
`libiio_v5_release_host025/`, `libiio_v5_release_host026/`, and
`libiio_v5_release_post_reboot/` beside this file. Their SHA-256 values are:

- host 0.25 AGC: `a0d8bc1e5d4d75b93cdff13de6783690341910a46d281bd323e2bda378999f7c`;
- host 0.25 rate matrix: `91c3998f98c3f7fe109cc67e2feb68c6bf9d2c577dfa50b00f72596569bae9e2`;
- host 0.26 AGC: `7b99bf77f1f623e68010f78ec4266d78ddc073f1a8775672d9e3daa34f26659b`;
- host 0.26 rate matrix: `3d99b2150295e40c04edeba549721eb7bc0b9bcb119ad78767c2e3f9454c6376`;
- post-reboot smoke: `0745e118c2fb5b25f2c08fd8ba3c91034261b657307b1e2388f7b87c6b58cae7`.

The full pytest suite was deliberately not run on the Pi.

## Known limitation

The PlutoPlus DDS can occasionally be silent on its first arm after a retune.
The RF tests detect this before accepting a frame and repeat the documented
TX-quadrature calibration/arm. This is a TX setup behavior, not an RX metadata
or TCP framing failure; RX-only operation is unaffected.

After a firmware reboot the AD9361 TX hardware-gain attributes initially read
-10 dB, while all DDS enables remain off. The hardware tests immediately apply
and verify the fail-closed -80 dB TX mute. This predates the metadata extension,
but applications that may later enable DDS should explicitly set TX gain first.
