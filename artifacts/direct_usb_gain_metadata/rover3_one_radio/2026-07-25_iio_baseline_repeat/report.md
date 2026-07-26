# Rover 3 one-radio USB-IIO baseline

Date: 2026-07-25
Result: **PASS**

Command:

```sh
/home/pi/spf-virtualenv/bin/python \
  -m spf.sdrpluto.benchmark_rover_frame \
  --config data_collection/rover/rover_v3.1/capture_configs/rover3_one_radio_benchmark_iio.yaml \
  --uri usb:1.10.5 \
  --output-dir artifacts/direct_usb_gain_metadata/rover3_one_radio/2026-07-25_iio_baseline_repeat
```

Device:

```text
Pluto serial: 104000f6ad020002fdff3a00bba2f096a1
firmware: v0.37-dirty
model: Analog Devices PlutoSDR Rev.C (Z7010-AD9361)
transport: standard USB-IIO
```

Frame contract:

```text
channels: 2
samples/channel: 524,288
normalized shape: (2, 524288)
normalized dtype: complex64
raw CS16 payload: 4,194,304 bytes
normalized IQ: 8,388,608 bytes
snapshot interval: 0.5 s
```

Acceptance:

```text
warm-up frames: 25
measured frames: 100
valid frames: 100/100
validation failures: 0
deadline misses: 0
```

RX call duration:

```text
p50: 253.885 ms
p90: 264.639 ms
p99: 304.284 ms
max: 325.459 ms
```

Raw CS16-equivalent acquisition throughput:

```text
p50: 15.755 MiB/s (16.521 MB/s)
p90: 16.289 MiB/s
p99: 16.332 MiB/s
min: 12.290 MiB/s
max: 16.341 MiB/s
```

Unpaced frame rate:

```text
p50: 3.939 frames/s
p99: 4.083 frames/s
min: 3.073 frames/s
```

Per-channel effective drain rate:

```text
p50: 2.065 MS/s
p99: 2.141 MS/s
```

The AD9361 still samples each snapshot at 30 MS/s. The value above is the
host-observed rate for draining a 524,288-sample/channel frame through
USB-IIO, not the ADC clock.

Deadline margin:

```text
p50: 246.115 ms
p99: 255.075 ms
worst: 174.541 ms
```

Separate control-attribute timing:

```text
gain pair p50: 1.625 ms
gain pair p99: 2.049 ms
RSSI pair p50: 1.328 ms
RSSI pair p99: 1.538 ms
```

The gain and RSSI reads were performed after the measured frame series and are
not included in RX call duration.

The preceding independent baseline run also passed 100/100 frames. This repeat
therefore verifies close/reopen lifecycle as well as the enhanced speed
reporting.
