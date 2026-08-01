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
