# Dual-RX gain/frequency calibration

This package measures the phase relationship between PlutoPlus RX1 and RX2 for
manual RX-gain pairs and RF frequencies. It uses the SPF direct-USB protocol-v2
firmware, writes one data-version-7 Zarr per radio serial, validates every frame
from its stored IQ, and fits a circular additive gain model.

## Start here

- [Calibrating a new radio](NEW_RADIO_CALIBRATION.md) is the operational
  checklist: attach and identify radios, establish 2R2T state, RAM-load and
  verify the firmware, collect pilot and dense V7 datasets, and locate every
  output.
- [Fitting and evaluating models](MODEL_FITTING_AND_EVALUATION.md) is the
  analysis checklist: validate stored IQ, fit each physical radio, compare the
  model ladder without data leakage, evaluate low-cost onboarding, and publish
  reproducible reports.
- [Power-cycle reproducibility](POWER_CYCLE_REPRODUCIBILITY.md) is the compact
  before/after test for deciding whether a stored calibration survives cold
  power removal or requires session anchors.
- [Physical radio notes](RADIO_NOTES.md) maps immutable Pluto serials to
  historical human labels such as former IP addresses and the calibration
  dataset roots collected from that physical radio.
- [Rover v3.1 pre-field checklist](../../../data_collection/rover/rover_v3.1/PRE_FIELD_CHECKLIST.md)
  covers the wider Rover, fake-drone, SITL, and field-deployment checks.

New calibration stores also require the post-firmware passive hardware
fingerprint from `/run/spf/direct_usb_ready.json`. Historical stores can be
augmented without rewriting arrays using the dry-run-first procedure in
[`HARDWARE_FINGERPRINT.md`](../../../data_collection/rover/rover_v3.1/HARDWARE_FINGERPRINT.md).

The [reports index](reports/README.md) combines the current model errors,
per-radio RF-DC findings, calibration status, and correction recommendations
for previously calibrated and new radios.

The physical path on each radio is:

```text
TX2 -> 30 dB attenuator -> two-way splitter -> RX1 and RX2
```

Only one radio transmits at a time. Every frequency/gain Cartesian scan is
repeated in three separately randomized epochs. A frame is usable for phase
only when its gain/RSSI metadata, tone level, clipping, cross-channel
coherence, and segmented phase-stability checks pass. Failed frames remain in
the dataset with an explicit quality-reason mask.

The calibration tone comes from TX2's FPGA DDS, so TX DMA never competes with
the direct-USB receive path. Only one Pluto control/direct context is open at a
time. At every radio/frequency block, the runner negotiates the two observed
Pluto+ RX-DMA handoff sequences (direct start, then post-arm IIO prime) and
accepts one only after a direct-USB reference-gain tone preflight passes. The
selected handoff is logged with the preflight. There are no host-side IIO gain
or RSSI reads in the frame loop, and TX attenuation changes in place without
restarting the DDS.

## Commands

First RAM-load and verify every attached Pluto using the Rover v3.1 preparation
script so `/run/spf/direct_usb_ready.json` exists. Then qualify each serial:

```bash
python -m spf.calibrations.dual_rx_gain_frequency probe \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --serial SERIAL
```

Run the pilot for both radios from the verified manifest:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --output artifacts/dual_rx_gain_frequency/pilot
```

For broad-band model discovery, qualify the three AD9361 gain-table regions
before launching a large Cartesian scan:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml \
  --output artifacts/dual_rx_gain_frequency/pilot_cross_band_RUN_NAME
```

The cross-band pilot covers 868/915 MHz, both sides of the 1.3 GHz table
boundary, SPF's 2.412/2.467 GHz frequencies, both sides of the 4.0 GHz table
boundary, and four 5.8 GHz anchors. If its stored-IQ validation passes, map
frequency behaviour more densely before choosing the final gain grid:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/frequency_scout_cross_band.yaml \
  --output artifacts/dual_rx_gain_frequency/frequency_scout_RUN_NAME
```

The frequency scout covers 47 points from 433 MHz through 5.9 GHz, including
tight spacing around the 1.3 and 4.0 GHz gain-table boundaries. At every point
it captures the complete `[-1, 26, 62]` RX1-by-RX2 grid in three separated
epochs. Its TX reference level uses the safe headroom measured by the pilot so
moderate asymmetric cells remain measurable without approaching the clipping
guard.

After interpreting the scout, run or refine the 17-gain stage-focused
Cartesian design:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/survey_cross_band.yaml \
  --output artifacts/dual_rx_gain_frequency/survey_cross_band_RUN_NAME
```

Both frequency-block order and ordered gain-pair order are independently
deterministic-randomized in each of three epochs. This is not random sparse
sampling: every configured frequency receives the complete configured
RX1-by-RX2 Cartesian grid, which keeps missing-cell and repeatability tests
well-defined.

To test the more economical additive-gain hypothesis at every integer gain,
use the committed 2.4 GHz additive-cross design:

```bash
python -m spf.calibrations.dual_rx_gain_frequency schedule \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/integer_gain_cross_2p4.yaml

python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/integer_gain_cross_2p4.yaml \
  --output artifacts/dual_rx_gain_frequency/integer_gain_cross_2p4_RUN_NAME
```

At each of 2.412 and 2.467 GHz, it measures all integer gains from -1 through
62 dB against a 26 dB reference on each receiver. It also records 48 explicit
off-axis pairs. Those pairs are excluded from fitting and therefore directly
test whether the two one-dimensional gain curves predict a previously unseen
RX1/RX2 combination:

```bash
python -m spf.calibrations.dual_rx_gain_frequency fit-additive-cross \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/integer_gain_cross_2p4.yaml \
  --dataset artifacts/dual_rx_gain_frequency/RUN/SERIAL/calibration.v7.zarr \
  --output-dir artifacts/dual_rx_gain_frequency/RUN/SERIAL/additive_cross
```

After fitting every radio, compare whether the gain-dependent curve transfers
while retaining a radio/frequency-specific intercept:

```bash
python -m spf.calibrations.dual_rx_gain_frequency compare-additive-cross \
  --analysis artifacts/dual_rx_gain_frequency/RUN/SERIAL_A/additive_cross/analysis.json \
  --analysis artifacts/dual_rx_gain_frequency/RUN/SERIAL_B/additive_cross/analysis.json \
  --output-dir artifacts/dual_rx_gain_frequency/RUN/comparison
```

The per-radio report also evaluates the old 17-gain stage grid against the
same held-out cells using linear interpolation and nearest-neighbour snapping.
This makes any off-grid policy an evidence-based result of the dense integer
experiment rather than an assumption made by a model consumer.

The historical integer experiment deliberately covered the cross-band common
range `-1..62` dB. For a complete model of the AD9361 1.3–4.0 GHz full gain
table, use the separate all-gain design:

```bash
python -m spf.calibrations.dual_rx_gain_frequency schedule \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/all_gain_cross_2p4.yaml

python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/all_gain_cross_2p4.yaml \
  --output artifacts/dual_rx_gain_frequency/all_gain_cross_2p4_RUN_NAME
```

This design measures every integer dB state from `-3` through `71` on both
receiver axes at exactly 2.412 and 2.467 GHz. It contains 149 additive-axis
training pairs and 56 off-axis held-out pairs per frequency, repeated in three
separated epochs: 1,230 frames per physical radio.

The companion
`configs/historical_exact_lo_cross_2p4.yaml` repeats the same experiment at
the intended historical wall-array centres 2.411950 and 2.467100 GHz. The
runtime exporter accepts repeated analysis/validation arguments to merge
disjoint exact-frequency runs for the same serial and gain axis. See
[`reports/HISTORICAL_EXACT_LO_CROSS_2P4_20260729.md`](reports/HISTORICAL_EXACT_LO_CROSS_2P4_20260729.md).

The fitted correction is conditioned on the realized per-frame RX1/RX2 gains,
not on the gain controller that selected them. Manual, slow-attack,
fast-attack, and hybrid control therefore do not receive separate model
families. A frame remains usable only when its direct-USB endpoint metadata is
valid and the reported dB values are supported by the exact-LO model.

The ordinary Cartesian schedule remains the default. An additive-cross
configuration is explicit in the YAML and carries its reference gain and
held-out pairs in the dataset provenance/signature, so it cannot be resumed
against a different experimental design.

The committed dense design contains 10,404 frames per radio: 12 frequencies,
289 ordered gain pairs per frequency, and three separated epochs. Its
adaptive-TX reference is 0 dB, the safe level established by the broad scout.
TX is then attenuated according to the stronger receive channel. Consequently,
highly asymmetric pairs can still leave the lower-gain channel below the
quality threshold; those frames are retained and fail closed rather than
being assigned a phase correction.

The frequency list consists of representative calibration anchors. A fitted
anchor is not automatically a correction for a nearby LO value. In
particular, committed configurations also contain exact centres such as
2.457 GHz, 2.4671 GHz, 5.770 GHz, and 5.839 GHz that are not all members of
the dense design. The model's exact-frequency support rule still applies:
either standardize deployment on a calibrated centre or acquire an additional
dense block at the exact requested centre.

Validate and fit each serial-specific dataset:

```bash
python -m spf.calibrations.dual_rx_gain_frequency validate \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --dataset artifacts/dual_rx_gain_frequency/pilot/SERIAL/calibration.v7.zarr \
  --serial SERIAL \
  --output artifacts/dual_rx_gain_frequency/pilot/SERIAL/validation.json

python -m spf.calibrations.dual_rx_gain_frequency fit \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --dataset artifacts/dual_rx_gain_frequency/pilot/SERIAL/calibration.v7.zarr \
  --output artifacts/dual_rx_gain_frequency/pilot/SERIAL/model.json

python -m spf.calibrations.dual_rx_gain_frequency report \
  --validation artifacts/dual_rx_gain_frequency/pilot/SERIAL/validation.json \
  --model artifacts/dual_rx_gain_frequency/pilot/SERIAL/model.json \
  --output-dir artifacts/dual_rx_gain_frequency/pilot/SERIAL/analysis
```

The generated `REPORT.md` embeds one four-panel diagnostic for every fitted
frequency. It compares passing three-epoch cell means with the final additive
fit while sweeping RX2 at three representative fixed RX1 gains and,
symmetrically, RX1 at three fixed RX2 gains. It also shows observed versus
fitted phase and cell-mean residual versus gain mismatch. Failed or unsupported
cells are omitted, and phase is plotted on the circular branch nearest the
fitted frequency intercept. The report links the corresponding coverage and
residual heatmaps as additional views.

Compare the fitted baselines of two physical radios without interpreting a
linear phase slope as literal PCB trace length:

```bash
python -m spf.calibrations.dual_rx_gain_frequency compare-radios \
  --model-a artifacts/dual_rx_gain_frequency/RUN/SERIAL_A/model.json \
  --model-b artifacts/dual_rx_gain_frequency/RUN/SERIAL_B/model.json \
  --output-dir artifacts/dual_rx_gain_frequency/RUN/cross_radio
```

To reproduce the stricter comparison of constant, gain-difference, ordered
stage-boundary, and categorical models across every serial in one run:

```bash
python -m spf.calibrations.dual_rx_gain_frequency compare-models \
  --config spf/calibrations/dual_rx_gain_frequency/configs/coarse_5ghz.yaml \
  --artifact-root artifacts/dual_rx_gain_frequency/RUN_NAME \
  --output-dir spf/calibrations/dual_rx_gain_frequency/reports/RUN_NAME
```

For the completed dense run, compare radio-specific and strict-universal
models across frequency, gain, and gain pair; then separately test whether a
differential path-delay model predicts an omitted frequency:

```bash
python -m spf.calibrations.dual_rx_gain_frequency model-matrix \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/survey_cross_band.yaml \
  --artifact-root \
    artifacts/dual_rx_gain_frequency/survey_cross_band_20260727_v1 \
  --output-dir \
    artifacts/dual_rx_gain_frequency/survey_cross_band_20260727_v1/model_matrix
```

This command uses three distinct tests:

- leave one randomized epoch out, for correction of a measured
  frequency/gain cell;
- leave one frequency out, for models that claim frequency
  generalization; and
- leave one radio out, for strict universal models with no radio-specific
  baseline adjustment.

Datasets from separate capture roots can be combined without copying or
symlinking the LMDB stores by repeating `--dataset` instead of supplying
`--artifact-root`:

```bash
python -m spf.calibrations.dual_rx_gain_frequency model-matrix \
  --config spf/calibrations/dual_rx_gain_frequency/configs/survey_cross_band.yaml \
  --dataset artifacts/RUN_A/SERIAL_A/calibration.v7.zarr \
  --dataset artifacts/RUN_A/SERIAL_B/calibration.v7.zarr \
  --dataset artifacts/RUN_B/SERIAL_C/calibration.v7.zarr \
  --dataset artifacts/RUN_B/SERIAL_D/calibration.v7.zarr \
  --output-dir spf/calibrations/dual_rx_gain_frequency/reports/OUTPUT
```

To evaluate whether a universal gain LUT can onboard an unseen physical radio
with only one or two scalar phase values, use the same one-dataset-per-radio
inputs with `low-cost-calibration`. Independent repeat datasets may be added
with `--repeat-dataset`; repeats measure temporal drift and are never counted
as additional radios:

```bash
python -m spf.calibrations.dual_rx_gain_frequency low-cost-calibration \
  --config spf/calibrations/dual_rx_gain_frequency/configs/survey_cross_band.yaml \
  --dataset artifacts/RUN_A/SERIAL_A/calibration.v7.zarr \
  --dataset artifacts/RUN_A/SERIAL_B/calibration.v7.zarr \
  --dataset artifacts/RUN_B/SERIAL_C/calibration.v7.zarr \
  --dataset artifacts/RUN_B/SERIAL_D/calibration.v7.zarr \
  --repeat-dataset artifacts/RUN_C/SERIAL_C/calibration.v7.zarr \
  --repeat-dataset artifacts/RUN_C/SERIAL_D/calibration.v7.zarr \
  --output-dir spf/calibrations/dual_rx_gain_frequency/reports/OUTPUT
```

The gain-dependent delay model is identifiable only as differential RX1−RX2
delay. Its separate RX1/RX2 lookup terms are relative contributions under the
chosen reference constraints, not absolute branch delays or literal PCB trace
lengths.

This read-only command selects only completely captured epoch/frequency blocks,
hashes every V7 scalar array used by the analysis, and writes:

- `comparative_analysis.json`, containing held-out cell, held-out quadrant,
  drift/order, cross-frequency, and cross-radio results;
- `calibrations/SERIAL.json`, containing compact stage-boundary and exact
  categorical coefficients plus the fail-closed list of production-supported
  ordered pairs; and
- `REPORT.md`, containing model-error tables and recommendations for known and
  previously unseen radios.

The committed report for the paused 2026-07-27 run is in
`reports/coarse_5ghz_20260727_dds_v1/`. Its source IQ remains under
`artifacts/` and is intentionally gitignored because of its size. A matching
scalar-input SHA-256 in the regenerated report proves that the same model
inputs were used; it does not replace full-IQ validation.

When diagnosing an AD9361 DC-correction failure, capture a read-only snapshot
of both RF-input correction banks:

```bash
python -m spf.calibrations.dual_rx_gain_frequency dc-registers \
  --serial PLUTO_SERIAL \
  --output rf_dc_registers.json
```

This command does not change gain, LO, tracking, calibration, or streaming
state. It decodes all four packed 10-bit RX1/RX2 I/Q correction words and
flags the documented stuck value `0x200`.

For the RX2 high-gain DC/rail diagnostic, use a new output directory and one
radio serial. This command opens a fresh context for every gain/state point,
never arms TX2 in the off condition, and stores full IQ plus protocol-v2
metadata and RF-DC correction words:

```bash
python -m spf.calibrations.dual_rx_gain_frequency diagnose-rx2-dc \
  --config spf/calibrations/dual_rx_gain_frequency/configs/coarse_5ghz.yaml \
  --serial PLUTO_SERIAL \
  --frequency-hz 5866000000 \
  --gain-rx1-db 26 \
  --gain-rx2-db 45 \
  --gain-rx2-db 48 \
  --gain-rx2-db 50 \
  --gain-rx2-db 51 \
  --gain-rx2-db 52 \
  --gain-rx2-db 55 \
  --gain-rx2-db 60 \
  --gain-rx2-db 62 \
  --frames-per-state 3 \
  --output artifacts/dual_rx_gain_frequency/rx2_dc_diagnostic/RUN_NAME
```

The diagnostic directory is intentionally separate from the exhaustive V7
artifact. It contains `manifest.json`, one `.npy` full-IQ file per frame,
`records.jsonl`, and a matched on/off `summary.json`. It must never be pointed
at an existing directory, so it cannot overwrite or resume the exhaustive
checkpoint.

If the diagnostic finds saturated RF-DC correction words, the supported Linux
RF-only initialization can be run with TX stopped and its selected LUT entries
recorded before and after:

```bash
python -m spf.calibrations.dual_rx_gain_frequency recover-rf-dc \
  --config spf/calibrations/dual_rx_gain_frequency/configs/coarse_5ghz.yaml \
  --serial PLUTO_SERIAL \
  --frequency-hz 5866000000 \
  --gain-rx1-db 26 \
  --gain-rx2-db 45 \
  --gain-rx2-db 48 \
  --gain-rx2-db 50 \
  --gain-rx2-db 51 \
  --gain-rx2-db 52 \
  --gain-rx2-db 55 \
  --gain-rx2-db 60 \
  --gain-rx2-db 62 \
  --output artifacts/dual_rx_gain_frequency/rx2_dc_diagnostic/RUN_NAME_recovery.json
```

This explicitly invokes the driver’s `calib_mode=rf_dc_offs`; it does not claim
to rerun the separate BB-DC initialization. ADI’s complete recovery procedure
requires both initializations with the receive input isolated, so the RF-only
operation must be followed by a new matched TX-off/TX-on diagnostic and must
not be treated as a successful recovery merely because the command returned.

Turn the matched before/recovery/after artifacts into a deterministic report:

```bash
python -m spf.calibrations.dual_rx_gain_frequency report-rf-dc \
  --before artifacts/dual_rx_gain_frequency/rx2_dc_diagnostic/BEFORE \
  --recovery artifacts/dual_rx_gain_frequency/rx2_dc_diagnostic/RECOVERY.json \
  --after artifacts/dual_rx_gain_frequency/rx2_dc_diagnostic/AFTER \
  --output-dir \
    spf/calibrations/dual_rx_gain_frequency/reports/RF_DC_REPORT_NAME
```

The report command verifies the serial, frequency, gain grid, and completion
state. It hashes every evidence file, including all full-IQ frames, and writes
`evidence.json` plus `REPORT.md`. This lets a committed report identify its
large gitignored source evidence exactly.

Before TX2 is armed for every radio/frequency block, normal pilot and exhaustive
collection now invokes the same driver-supported RF-DC initialization and then
requires the direct-USB tone preflight to pass. The preparation policy is part
of the calibration configuration signature. Consequently, an older partial
run that predates this policy cannot be silently resumed as a new run.

After the pilot establishes usable signal levels, replace `pilot_5ghz.yaml`
with `coarse_5ghz.yaml` to enumerate all 73 manual gain states on both
receivers at the four coarse frequencies. Runs are resumable: the completion
bit is committed only after IQ, V7 metadata, coordinates, and quality metrics.
Initial and resumed capture use the same asynchronous LMDB write mode, with a
durable sync after every radio/frequency block and on clean shutdown.

## Model selection

The operational candidate is fitted independently for each radio and RF
frequency:

```text
angle(RX1) - angle(RX2)
    = frequency intercept + RX1_effect(gain1) + RX2_effect(gain2) + residual
```

The `compare-models` command also evaluates a compact ordered stage model. Its
candidate boundaries are derived reproducibly from the high-band full gain
table in `drivers/iio/adc/ad9361.c` at Linux commit
`d798b0d821b85ebd51ecffbfa68d8e4d69b77132`: start a new segment when the
LNA/mixer byte begins a plateau of at least three requested gain states. This
yields `-6, 6, 16, 23, 26, 41 dB`; the final 52–62 dB one-index-per-dB mixer
ramp is represented by the linear term. Because this compact basis was
developed while inspecting epoch 0, its apparent parsimony must be confirmed on
the untouched repeat epochs before it is selected for deployment.

The fit command evaluates predictions only on an epoch excluded from training.
It also performs paired, identical-mask comparisons against:

- a model that knows only `gain1 - gain2`;
- an additive model plus one residual for every ordered gain pair;
- one set of gain curves shared by all RF frequencies; and
- independent per-frequency baselines versus a constant phase plus a linear
  frequency slope (an effective differential-delay description); and
- the additive model adjusted from one equal-gain anchor frame in each held-out
  frequency/epoch.

Differences smaller than the declared 0.1° held-out MAE margin are treated as
practically equivalent and select the predeclared simpler operational choice.
The report also repeats additive cross-validation with both channels above
-10, 0, and 10 dB tone SNR. These are confidence/coverage diagnostics, not
permission to use a frame that failed any base quality check.

`predict_phase_offset()` accepts only an exact fitted frequency and an ordered
gain pair that passed the repeatability criterion. It raises instead of
interpolating an unsupported point, even when both individual gains were
observed elsewhere in the surface. With the recorded convention, apply a valid
prediction as:

```text
corrected_phase = wrap(measured_angle_RX1_minus_RX2 - predicted_offset)
```

## Pass/fail interpretation

- Structural validation must pass: all scheduled frames, exact V7 shape,
  direct-USB protocol v2, verified firmware/serial provenance, valid endpoint
  gain and RSSI metadata, and recomputed IQ metrics matching stored values.
- Each gain/frequency cell passes when at least two of its three epochs pass
  frame quality and repeat phase circular standard deviation is at most 5°.
- A failed cell is not assigned a phase correction. This is expected where one
  channel is too weak or either ADC clips.
- Model effects are emitted only for gain states observed in that receiver
  role. Missing effects are JSON `null`; they must never be interpreted as
  zero phase correction.
- A correction is available only for the exact radio serial and calibrated RF
  frequency, with non-null RX1/RX2 effects and a live frame that passes gain
  metadata and signal-quality checks. Never extrapolate across an unsupported
  gain, frequency, clipping region, or weak-signal region.
- The ordered gain pair matters. A model based only on gain difference is
  retained as a measured baseline, not assumed correct.
- The fitted baseline follows `phase = constant - 2*pi*frequency*delay`.
  Its reported delay and free-space-equivalent path describe the measured
  electrical group delay. They do not identify literal cable length: cables,
  splitter paths, PCB traces, analogue filters, and retune/calibration state
  can all contribute. The report gives the linear-fit residual and a paired
  held-out comparison against independent frequency baselines.
