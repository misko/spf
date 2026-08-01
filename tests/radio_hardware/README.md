# Attached-radio pytest gates

These tests claim real Pluto direct-USB interfaces and are always opt-in. They
do not transmit. An ordinary `pytest` invocation collects and skips them.

Quick production-sized two-radio gate:

```bash
pytest tests/radio_hardware \
  --radio-hardware \
  --radio-expected-count=2 \
  --radio-samples=524288 \
  --radio-cycles=20 \
  --radio-report-dir=/tmp/spf-radio-report
```

Fast developer smoke with smaller frames:

```bash
pytest tests/radio_hardware \
  --radio-hardware \
  --radio-expected-count=2 \
  --radio-samples=16384 \
  --radio-cycles=3
```

Repeat `--radio-serial=SERIAL` to select exact devices. The expected count is
checked after serial selection and is exact, so a missing or unexpected radio
is reported before streaming begins.

`--radio-zarr` and `--radio-soak` are reserved as additional explicit gates;
they are never implied by `--radio-hardware`.

Add a short hardware-backed protocol-v2 to V7 Zarr round trip:

```bash
pytest tests/radio_hardware \
  --radio-hardware \
  --radio-zarr \
  --radio-expected-count=2 \
  --radio-zarr-frames=3
```

This surgical Zarr test is not a substitute for the production-YAML,
fake-drone capture in the Rover pre-field checklist; that remains the final
collector acceptance gate.

Exercise graceful SIGTERM against the real production collector, verify its
partial LMDB-Zarr, and immediately reclaim both radios:

```bash
pytest tests/radio_hardware/test_interrupted_collection_hardware.py \
  --radio-hardware \
  --radio-interrupt \
  --radio-expected-count=2 \
  --radio-capture-config=data_collection/rover/rover_v3.1/capture_configs/rover3_production_v7.yaml \
  --radio-device-mapping=/home/pi/device_mapping \
  --radio-ready-manifest=/run/spf/direct_usb_ready.json
```

The subprocess is terminated only after every configured receiver has at
least two fully committed records. Passing requires an `incomplete` temporary
store with `CaptureInterrupted`, monotonically safe progress counts, no final
`.zarr`, and a successful new direct-USB request on every serial.
