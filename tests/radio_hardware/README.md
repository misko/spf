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

Exercise a single graceful SIGTERM against the real production collector,
verify its partial LMDB-Zarr, and immediately reclaim both radios:

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

The signal and interruption point can be selected explicitly:

```bash
pytest tests/radio_hardware/test_interrupted_collection_hardware.py \
  --radio-hardware --radio-interrupt \
  --radio-interrupt-signal=sigkill \
  --radio-interrupt-min-records=25 \
  --radio-expected-count=2 \
  --radio-capture-config=data_collection/rover/rover_v3.1/capture_configs/rover3_production_v7.yaml \
  --radio-device-mapping=/home/pi/device_mapping \
  --radio-ready-manifest=/run/spf/direct_usb_ready.json
```

`SIGINT` and `SIGTERM` must finalize a readable `incomplete` store and exit
with the conventional signal status. `SIGKILL` cannot run cleanup, so its
store must remain `in_progress`; it must never be promoted or represented as
complete. All modes validate every safely committed prefix and then reclaim
each radio immediately.

Recreate the Rover 3 software-visible dual-timeout signature by suspending the
collector for 12.5 seconds, then resuming it:

```bash
pytest tests/radio_hardware/test_interrupted_collection_hardware.py \
  --radio-hardware --radio-interrupt \
  --radio-interrupt-signal=sigstop \
  --radio-interrupt-min-records=25 \
  --radio-expected-count=2 \
  --radio-capture-config=data_collection/rover/rover_v3.1/capture_configs/rover3_production_v7.yaml \
  --radio-device-mapping=/home/pi/device_mapping \
  --radio-ready-manifest=/run/spf/direct_usb_ready.json
```

Passing means the resumed process records one owned incident, enters the
ordinary bounded failure exit, leaves a readable `incomplete` Zarr, and
releases both radios. This reproduces the deadline behavior; it does not claim
that host scheduling was the physical cause of the August 1 incident.

For the reproducible pre-field matrix, use:

```bash
data_collection/rover/rover_v3.1/run_interrupted_capture_campaign.sh
```

By default this interrupts production V7 collection with `SIGTERM` after 2
records, `SIGINT` after 10, `SIGKILL` after 25, and `SIGTERM` after 100. It
preserves each partial store and report, rejects new kernel USB errors, and
finishes with a strict 100-record production capture. Override the matrix with
`SPF_INTERRUPT_CASES`, for example `SPF_INTERRUPT_CASES='sigkill:2 sigterm:50'`.

For an unattended, bounded repetition of the complete campaign, use:

```bash
SPF_INTERRUPT_SOAK_SECONDS=43200 \
  data_collection/rover/rover_v3.1/run_interrupted_capture_soak.sh
```

The soak rotates four deterministic early/middle/late signal matrices and runs
the campaign's clean V7 recovery capture after every matrix. It stops on the
first failed case, when free space drops below 25 GiB, at the wall-clock
deadline, or when its printed `STOP` file is created. Each completed round has
an independent artifact directory and the root `rounds.tsv` is append-only,
so evidence before a failure remains usable. The runner is receive-only because
all underlying collectors use `--fake-drone`.

Every case preserves `dmesg-before.txt`, `dmesg-after.txt`, `dmesg-delta.txt`
and `case-status.env`, even when pytest fails. The status file records the
pytest exit status, kernel-snapshot status and kernel-USB-error decision. This
keeps the original test exit code while retaining the evidence needed to tell
a collector failure from a host USB failure.

After the soak finishes, independently audit every ledger row, signal report,
committed prefix, kernel delta, clean-recovery identity, and V7 validation with:

```bash
python -m spf.scripts.validate_interruption_soak SOAK_ROOT \
  --expected-receivers 2 --require-complete \
  --output SOAK_ROOT/aggregate.json
```

Omit `--require-complete` to audit only fully completed rounds while a soak is
still running. In-progress rounds are deliberately absent from `rounds.tsv`.
The aggregate audit requires every expected serial's post-interruption IQ probe
to succeed within the bounded one-to-three fresh-session policy.

Resource safety is audited independently:

```bash
python -m spf.scripts.validate_soak_resources SOAK_ROOT/resources.csv \
  --rounds SOAK_ROOT/rounds.tsv \
  --maximum-anon-mib 1024 \
  --minimum-available-mib 256 \
  --recovery-anon-mib 384 \
  --output SOAK_ROOT/resources-final.json
```

Passing requires bounded anonymous memory and host-available memory across the
whole run, plus a sample below the recovery threshold inside every completed
round after that round's peak. A low-memory sample before the peak or in a
different round cannot hide failure to reclaim lifecycle allocations.
