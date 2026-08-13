# Calibrating a new Pluto radio

This is the end-to-end operator checklist for adding one or more physical
PlutoPlus-compatible radios to the dual-RX phase-calibration corpus. It covers
the safe hardware setup, direct-USB firmware preparation, V7 dataset
collection, storage layout, resumability, and stage-by-stage pass/fail rules.

Use [MODEL_FITTING_AND_EVALUATION.md](MODEL_FITTING_AND_EVALUATION.md) after
collection. The package architecture and protocol details remain in
[README.md](README.md).

## Safety and invariants

- The loopback fixture is, independently for each radio:

  ```text
  TX2 -> 30 dB attenuator -> two-way splitter -> RX1 and RX2
  ```

- Never connect TX2 directly to an RX input.
- The calibration program activates only one radio's TX2 at a time.
- The published SPF image is loaded into RAM. The loader does not write QSPI;
  a radio power cycle returns to the installed QSPI firmware.
- Persistent 2R2T provisioning is a separate, explicit operation. Inspect its
  dry run before authorizing `--apply`.
- Never reuse an output root for a new physical-radio cohort. A restart of the
  same run may resume the same root, but a new cohort or configuration gets a
  new run name.
- Keep the repository clean when collecting. The dataset records the software
  Git SHA and whether the checkout was dirty.
- Do not run ordinary pyadi receive streaming while the direct-USB receiver
  owns RX DMA.

Run all commands from the repository root:

```bash
cd /home/pi/spf
```

The commands below assume the project environment is
`/home/pi/spf-virtualenv`. If it is not already active:

```bash
source /home/pi/spf-virtualenv/bin/activate
```

## 1. Choose the run and check capacity

Choose a unique, descriptive run ID before touching the radios:

```bash
RUN_ID=survey_cross_band_YYYYMMDD_RADIOS_5_6_v1
RUN_ROOT="artifacts/dual_rx_gain_frequency/${RUN_ID}"
test ! -e "$RUN_ROOT"
df -h artifacts
git status --short --branch
git rev-parse HEAD
```

The committed designs are:

| Design | Frames per radio | Purpose | Measured or expected wall time |
| --- | ---: | --- | --- |
| `pilot_cross_band.yaml` | 324 | Fast fixture, firmware, metadata, and cross-band qualification | About 3 minutes per radio plus setup |
| `frequency_scout_cross_band.yaml` | 1,269 | Broad 47-frequency behaviour with a 3-by-3 gain grid | About 9–12 minutes per radio plus block setup |
| `survey_cross_band.yaml` | 10,404 | Dense 12-frequency, 17-by-17 gain surface, three epochs | About 1.5 hours per radio; about 3–3.5 hours for two radios |

Each V7 frame contains two channels of 65,536 `complex64` samples, or about
1 MiB of IQ before store overhead. Budget at least 11 GiB per radio for a dense
run. For two dense radios, start with at least 30 GiB free.

Pass:

- the run root does not exist;
- the checkout and intended Git SHA are recorded; and
- there is enough free disk space.

Fail:

- the root belongs to a different cohort or configuration;
- the checkout has unexplained changes; or
- free space is marginal.

## 2. Attach, inventory, and identify the radios

Power down or disconnect the old radios, transfer the complete attenuator and
splitter fixtures, attach the new radios, and wait for USB enumeration.

Read-only inventory:

```bash
lsusb -d 0456:b673
lsusb -t
bash data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh discover-count
```

If a stale boot-readiness manifest exists from the previous radios, remove it
before any new capture:

```bash
sudo rm -f /run/spf/direct_usb_ready.json
```

Do not use a USB bus/device address as the radio identity. It can change after
RAM loading. The V7 pipeline uses the Pluto serial as the durable identity and
also records the physical USB path, firmware provenance, and passive hardware
fingerprint.

If the radio has a historical label, former IP address, or other operator note,
add it to [RADIO_NOTES.md](RADIO_NOTES.md) only after reading the immutable
serial from the sole attached radio. Historical IP addresses are provenance,
not connection targets. Add the resulting pilot and dense dataset roots to the
same registry row after collection.

Pass:

- the attached count is exactly the intended cohort size;
- every fixture includes the 30 dB attenuator; and
- no capture process has an open radio handle.

Fail:

- an unexpected Pluto is attached;
- a fixture is uncertain; or
- a stale readiness manifest is still present.

## 3. Verify or provision persistent 2R2T state

Normal startup only verifies persistent AD9361/2R2T configuration. It never
rewrites U-Boot. First run the provisioning wrapper in dry-run mode:

```bash
sudo bash \
  data_collection/rover/rover_v3.1/check_and_set_pluto.sh \
  --dry-run 2
```

Replace `2` with the exact attached count. The required persistent values are:

```text
attr_name=compatible
attr_val=ad9361
compatible=ad9361
mode=2r2t
```

If and only if the dry run reports that a radio needs provisioning, preserve
the generated U-Boot backup and explicitly authorize the one-time change:

```bash
sudo bash \
  data_collection/rover/rover_v3.1/check_and_set_pluto.sh \
  --apply 2
```

Then rerun the dry-run command. The loader's default approved persistent QSPI
identity is `v0.37-dirty`; do not override that gate casually.

Pass:

- the final dry run verifies every radio as AD9361-compatible and 2R2T; and
- actual dual-RX operation is verified by the script.

Fail:

- the installed QSPI identity is not approved;
- backups cannot be written; or
- the final verification still reports a mismatch.

Do not proceed by manually editing U-Boot variables to bypass a failure.

## 4. RAM-load and verify the direct-USB firmware

The firmware identity is pinned in every calibration YAML. The current
published image is:

```text
release: v0.38-plutoplus-spf-gain-rssi-fingerprint-v2
asset:   plutoplus-spf-direct-usb-gain-rssi-fingerprint-v2-pluto.dfu
SHA256:  5f8220bc3a9c23b891ad8a19e52eeb24ecfcd24b2ae5923a1e50e450f49a802d
mode:    RAM only
```

For the standard two-radio calibration bench, the recommended entry point
automates preparation, per-serial probing, V7 capture, and stored-IQ
validation:

```bash
python -m spf.calibrations.dual_rx_gain_frequency automate \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml \
  --output \
    artifacts/dual_rx_gain_frequency/pilot_cross_band_UNIQUE_RUN_ID \
  --expected-radios 2
```

This is the normal new-radio command. It fails before mutation if the
calibration and Rover preparation configs pin different firmware. It then
checks persistent 2R2T state, RAM-loads and verifies every attached radio,
regenerates device mapping and the readiness/fingerprint manifest, probes TX2
on every serial, collects one serial-specific Zarr, and recomputes validation
metrics from stored IQ. It writes `automation_plan.json` and
`automation_result.json` at the run root.

An interrupted run can be resumed only with the exact same command plus
`--resume`. The stored automation plan must match the configs, firmware,
expected radio count, and serial cohort. A new physical cohort or independent
repeat always gets a new output root.

The lower-level preparation commands below remain available for diagnosis.

For a configured Rover with a valid `/home/pi/rover_id`, the canonical
one-command preparation is:

```bash
sudo bash \
  data_collection/rover/rover_v3.1/prepare_direct_usb_boot.sh
```

That command invalidates stale readiness, resolves the Rover capture config,
checks the exact radio count and persistent state, RAM-loads every attached
radio, regenerates `/home/pi/device_mapping`, verifies the direct interface,
and writes `/run/spf/direct_usb_ready.json`.

For a calibration bench without `/home/pi/rover_id`, use the same primitives
explicitly. For two radios:

```bash
LOADER=data_collection/rover/rover_v3.1/load_direct_usb_firmware.sh
sudo bash "$LOADER" check-config-all 2
sudo bash "$LOADER" load-all 2
sudo bash "$LOADER" verify-all 2
```

`load-all` resolves duplicate Pluto USB-network addresses in isolated network
namespaces and addresses radios by serial and physical path. Do not attempt to
SSH to two radios through the default `192.168.2.1` namespace yourself.

Pass:

- all expected radios return after RAM loading;
- standard USB-IIO remains present;
- vendor-specific direct-USB interface 6 is present; and
- both `iiod` and `sdr_usb_gadget` run on every radio.

Fail:

- the image checksum differs;
- a radio disappears or changes physical-port identity;
- only part of the cohort verifies; or
- either required USB function is absent.

To roll back the RAM image, power-cycle the radios or use the loader's
`rollback-all N` command. Do not flash an experimental image into QSPI.

## 5. Generate the post-firmware mapping and readiness manifest

USB device addresses change during RAM loading, so generate the mapping only
after every radio has returned:

```bash
bash data_collection/rover/rover_v3.1/device_mapping.sh \
  > /tmp/device_mapping.new
cat /tmp/device_mapping.new
sudo install -o pi -g pi -m 0644 \
  /tmp/device_mapping.new /home/pi/device_mapping
```

On a configured Rover, `prepare_direct_usb_boot.sh` already wrote the manifest.
On a two-radio calibration bench, use the calibration firmware-compatible
Rover 1 V7 config to fingerprint the final post-firmware state:

```bash
sudo env PYTHONPATH=/home/pi/spf \
  /home/pi/spf-virtualenv/bin/python3 \
  -m spf.scripts.pluto_ready_manifest \
  write \
  --rover-id 1 \
  --config \
    /home/pi/spf/data_collection/rover/rover_v3.1/capture_configs/rover1_production_v7.yaml \
  --output /run/spf/direct_usb_ready.json \
  --device-mapping /home/pi/device_mapping
```

Verify and inspect it:

```bash
sudo env PYTHONPATH=/home/pi/spf \
  /home/pi/spf-virtualenv/bin/python3 \
  -m spf.scripts.pluto_ready_manifest \
  verify \
  --rover-id 1 \
  --config \
    /home/pi/spf/data_collection/rover/rover_v3.1/capture_configs/rover1_production_v7.yaml \
  --output /run/spf/direct_usb_ready.json \
  --device-mapping /home/pi/device_mapping

python3 -m json.tool /run/spf/direct_usb_ready.json
```

The selected calibration YAML and readiness manifest must describe the same
firmware image. The runner checks that relationship again and writes each
radio's serial, USB path, image hash, firmware and gadget SHAs, capabilities,
and passive fingerprint into its V7 dataset.

Pass:

- mapping has one unique row per radio;
- the manifest verifies after the final RAM enumeration; and
- its serials are exactly the intended cohort.

Fail:

- the manifest names a removed radio;
- serials or physical paths are duplicated; or
- the firmware/fingerprint verification fails.

## 6. Render the schedule and probe every serial

Render the deterministic pilot schedule without transmitting:

```bash
python -m spf.calibrations.dual_rx_gain_frequency schedule \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml
```

Read the serials from the verified manifest:

```bash
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("/run/spf/direct_usb_ready.json").read_text())
for radio in manifest["radios"]:
    print(radio["serial"])
PY
```

Qualify TX2-off versus TX2-on loopback dominance once for each serial:

```bash
python -m spf.calibrations.dual_rx_gain_frequency probe \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml \
  --serial SERIAL \
  --output "/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/probes/dual_rx_gain_frequency_probe_SERIAL.json"
```

The production path is TX2, represented by `--tx-channel 1` if the diagnostic
override is specified. A probe failure often means the TX path did not arm,
the fixture is wrong, or the received peak is not at the expected DDS offset.
Do not treat an arbitrary strong signal as a passing loopback.

Pass:

- every serial passes the probe;
- the received tone is at the configured `+100 kHz` offset; and
- TX-on dominates matched TX-off by the configured threshold.

Fail:

- any radio fails independently;
- the peak is at the wrong frequency; or
- a radio sees another fixture's transmitter.

## 7. Collect the cross-band pilot

Use a new pilot root and let `run` take its serial list from the verified
readiness manifest:

```bash
PILOT_ID=pilot_cross_band_YYYYMMDD_RADIOS_5_6_v1
PILOT_ROOT="artifacts/dual_rx_gain_frequency/${PILOT_ID}"
test ! -e "$PILOT_ROOT"

python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml \
  --output "$PILOT_ROOT" \
  --expected-radios 2
```

The runner performs an RF-DC calibration and direct-RX/DDS handoff preflight
at each radio/frequency block. It validates gain and RSSI metadata for every
frame and stores failed quality observations rather than hiding them.

The output is one dataset per physical serial:

```text
artifacts/dual_rx_gain_frequency/PILOT_ID/
├── run_result.json
├── SERIAL_A/
│   ├── calibration.v7.zarr/
│   ├── observations.jsonl
│   ├── preflight.jsonl
│   └── preflight_failures.jsonl  # only when failures occurred
└── SERIAL_B/
    └── ...
```

The output root is resumable only for the same config, serial, USB path, and
firmware signature. If interrupted, rerun the identical command. Completed
frames are not rewritten. Do not resume after moving radios between physical
ports or changing the configuration.

Pass:

- `run_result.json` reports every scheduled frame stored for every serial;
- sequence, firmware, serial, hardware fingerprint, and endpoint metadata
  checks pass; and
- no unexplained preflight failure remains.

Fail:

- frames are missing;
- any metadata byte enters IQ;
- the expected TX tone is absent; or
- the run mixes identities, ports, firmware, or configurations.

## 8. Validate the pilot from stored IQ

Validation recomputes the signal-quality metrics from the Zarr IQ by default:

```bash
for serial_dir in "$PILOT_ROOT"/*/; do
  dataset="${serial_dir}calibration.v7.zarr"
  test -d "$dataset" || continue
  serial="$(basename "$serial_dir")"
  python -m spf.calibrations.dual_rx_gain_frequency validate \
    --config \
      spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml \
    --dataset "$dataset" \
    --serial "$serial" \
    --output "${serial_dir}validation.json"
done
```

Interpret results in two layers:

1. Capture integrity is mandatory: scheduled shape, V7 schema, firmware and
   serial provenance, gain/RSSI metadata, sequence continuity, and stored
   versus recomputed IQ metrics must agree.
2. Phase-cell quality is deliberately conservative. Extreme asymmetric gain
   pairs may be weak or clip one channel; those frames remain recorded and can
   make the aggregate status `fail_quality`. That is not by itself a transport
   or firmware failure. The affected cells remain unsupported and fail closed.

Pass:

- structural and metadata validation pass; and
- quality rejections have explicit, physically plausible reason masks.

Fail:

- stored and recomputed IQ metrics disagree;
- provenance is missing or inconsistent; or
- ordinary moderate-gain cells fail without a understood cause.

## 9. Collect the dense survey

Only proceed after both radios pass the pilot review:

```bash
DENSE_ID=survey_cross_band_YYYYMMDD_RADIOS_5_6_v1
DENSE_ROOT="artifacts/dual_rx_gain_frequency/${DENSE_ID}"
test ! -e "$DENSE_ROOT"
df -h artifacts

python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/survey_cross_band.yaml \
  --output "$DENSE_ROOT" \
  --expected-radios 2
```

Every configured frequency gets the complete 17-by-17 ordered RX1-by-RX2 gain
grid in three separately randomized epochs. The repetitions are separated in
time; they are not three adjacent captures of the same cell.

Monitor free space and progress without modifying the Zarr. If interrupted,
rerun the identical command with the same root to resume. A repeat intended to
measure drift must use a new run ID so it cannot overwrite or resume the first
survey.

Pass:

- 10,404 frames are stored per radio;
- all 36 frequency/epoch blocks are complete per radio;
- loss, retries, and quality exclusions are explicit; and
- each radio has its own serial-named V7 store.

Fail:

- a partial root is renamed and treated as complete;
- two independent repeats share an output root; or
- a physical radio is counted twice because it has two datasets.

## 10. Preserve and hand off the results

Large capture evidence remains local and is intentionally ignored by Git:

```text
artifacts/dual_rx_gain_frequency/RUN_ID/SERIAL/calibration.v7.zarr
```

Per-radio sidecars in the same serial directory include:

- `observations.jsonl`: frame coordinates and quality results;
- `preflight.jsonl`: selected direct-RX/DDS handoff and tone qualification;
- `preflight_failures.jsonl`: attempted failures, if any;
- `validation.json`: strict stored-IQ validation;
- `model.json`: fitted per-radio correction model; and
- `analysis/`: generated report and diagnostic plots.

Commit only reviewed, reproducible summaries and figures under:

```text
spf/calibrations/dual_rx_gain_frequency/reports/REPORT_NAME/
```

Before disconnecting the cohort, record:

- run root and config path;
- serial-to-fixture and physical-port mapping;
- repository SHA and clean/dirty state;
- firmware image SHA;
- per-radio stored/scheduled counts;
- validation integrity result and quality-valid count;
- any retry, preflight, or hardware anomaly; and
- whether a second independent drift run is planned.

Then follow [MODEL_FITTING_AND_EVALUATION.md](MODEL_FITTING_AND_EVALUATION.md).

## Compact pass/fail checklist

| Stage | Pass | Stop and investigate |
| --- | --- | --- |
| Fixture | TX2 through 30 dB and splitter to RX1/RX2 | Missing/uncertain attenuation or cross-wired radio |
| Inventory | Exact intended count and identities | Extra/missing radio or active capture process |
| Persistent state | Final dry run verifies AD9361 and 2R2T | QSPI identity or U-Boot verification mismatch |
| RAM firmware | All radios verify IIO plus direct USB | Checksum, enumeration, interface, or daemon failure |
| Readiness | Fresh mapping and manifest match cohort | Stale serial/path, firmware, or fingerprint |
| Probe | Correct +100 kHz TX2 tone dominates TX-off | Wrong/missing tone or fixture cross-talk |
| Pilot capture | All 324 frames per radio stored | Missing frames, sequence, metadata, or provenance error |
| Pilot review | Integrity passes; exclusions explained | Recomputed mismatch or unexplained moderate-cell failures |
| Dense capture | All 10,404 frames per radio stored | Partial/mixed run or insufficient storage |
| Handoff | Paths, SHAs, counts, and anomalies recorded | Dataset cannot be tied to one physical radio and firmware |
