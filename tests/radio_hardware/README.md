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

The default direct-USB gates cover both legacy finite capture and the bounded
rolling path used by SPF session groups. The rolling two-radio test keeps one
production-sized transfer resident per radio, validates contiguous sequences,
and therefore stays within the common 16 MiB usbfs memory budget.

Deliberately kill only the vendor gadget daemon and require the on-radio
supervisor to rebind the USB composite while standard USB-IIO returns:

```bash
sudo env PYTHONPATH="$PWD" python -m pytest \
  tests/radio_hardware/test_gadget_supervisor_hardware.py \
  --radio-hardware --radio-crash-recovery --radio-expected-count=2 \
  --radio-samples=524288 --radio-report-dir=/tmp/spf-radio-report
```

This gate changes the USB device address and process nonce by design. It does
not reboot the Pluto, enable TX, or write QSPI.

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

Protocol-v3 gain-series firmware has a separate fail-closed gate. It validates
the FPGA sample counter, sample-bracketed gain observations, IQ alignment, and
continuous frame sequences on every selected USB radio:

```bash
pytest tests/radio_hardware/test_gain_series_v3_hardware.py \
  --radio-hardware \
  --radio-gain-series-v3 \
  --radio-expected-count=2 \
  --radio-samples=524288 \
  --radio-frames-per-request=3 \
  --radio-gain-observation-interval=2048 \
  --radio-gain-observation-capacity=256 \
  --radio-report-dir=/tmp/spf-radio-report
```

The direct-IP parity test is independently opt-in. Select one Pluto with a
unique reachable IP address and add:

```bash
--radio-direct-ip --radio-direct-ip-host=192.168.1.163
```

These flags must only be used after RAM-booting a protocol-v3 candidate. The
currently promoted protocol-v2 image is expected to reject the test.

Run the complete ordered, receive-only candidate campaign with:

```bash
tests/radio_hardware/run_gain_series_v3_candidate.sh /path/to/candidate-pluto.dfu
```

Pass the uniquely addressed Pluto as a second argument to include direct-IP
parity, for example `192.168.1.163`. The runner verifies the image checksum,
records a protocol-v2 baseline, checks persistent 2R2T configuration, RAM-loads
the exact attached radio count, re-runs v2 compatibility, then runs v3 USB,
production-sized V7 Zarr, optional IP, and final identity gates. It never
writes QSPI or enables TX unless the explicit attenuated-loopback option below
is supplied. A failure leaves the volatile candidate running for inspection
and prints the explicit rollback command.

### Explicit TX2 loopback release gate

The default campaign remains receive-only. When **every selected radio** has
TX2 connected through at least 30 dB of physical attenuation and a splitter to
that same radio's RX1 and RX2, add the explicit TX gate:

```bash
tests/radio_hardware/run_gain_series_v3_candidate.sh \
  --with-tx-loopback \
  --loopback-attenuation-db=30 \
  /path/to/candidate-pluto.dfu \
  192.168.1.163
```

`--radio-hardware` alone can never enable TX. The TX test additionally requires
`--radio-tx-loopback` and a declared attenuation of at least 30 dB. It tests one
radio at a time using only TX2 and the FPGA DDS, with TX1 held at -80 dB. The
gate measures a muted baseline, verifies the +100 kHz tone on RX1 and RX2,
checks known unequal manual gains against every protocol-v3 observation, steps
the tone level under slow-attack AGC, and verifies that direct streaming did
not change LO, sample rate, bandwidth, or gain-control mode.

The complete runner performs this mandatory TX gate on three independent
volatile FPGA boots before any candidate receive/IP/Zarr gates. Override the
count only when diagnosing a candidate with `SPF_V3_TX_BOOT_EPOCHS`; release
promotion requires at least three. Each epoch has separate RAM-load, pytest,
and explicit TX-mute artifacts.

Both the pytest fixture and the outer campaign runner disable DDS, clear TX
channels, destroy the TX buffer, and verify TX1/TX2 at -80 dB on exit. A failed
mute is a campaign failure. The TX gate still never writes QSPI.

To run only the TX stage after a protocol-v3 candidate is already in RAM:

```bash
pytest -q tests/radio_hardware/test_gain_series_v3_tx_loopback_hardware.py \
  --radio-hardware --radio-gain-series-v3 --radio-tx-loopback \
  --radio-tx-loopback-attenuation-db=30 \
  --radio-expected-count=2 \
  --radio-gain-observation-interval=2048 \
  --radio-gain-observation-capacity=256 \
  --radio-report-dir=/tmp/spf-radio-tx-report
```

Add `--radio-zarr --radio-zarr-frames=3` to the USB command to write and
reopen a hardware-backed V7 store. That gate verifies observation counts,
sample-counter bounds, explicit sentinel padding, serial/USB identity, FPGA
DNA, gadget build ID, IQ shape, and protocol/transport provenance.

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
