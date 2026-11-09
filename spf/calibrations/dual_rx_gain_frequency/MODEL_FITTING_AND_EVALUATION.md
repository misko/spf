# Fitting and evaluating dual-RX phase models

This guide starts with completed V7 calibration datasets and ends with
reproducible per-radio models, cross-radio comparisons, low-cost onboarding
results, plots, and a deployment recommendation. For collection and storage,
start with [NEW_RADIO_CALIBRATION.md](NEW_RADIO_CALIBRATION.md).

Run commands from the repository root in the SPF virtual environment:

```bash
cd /home/pi/spf
source /home/pi/spf-virtualenv/bin/activate
```

Use one primary dense dataset per physical serial in any multi-radio analysis.
An independent repeat of the same serial measures temporal drift; it is not an
additional radio.

## 1. Freeze the analysis inputs

Record the exact inputs before fitting:

```bash
CONFIG=spf/calibrations/dual_rx_gain_frequency/configs/survey_cross_band.yaml
RUN_ROOT=artifacts/dual_rx_gain_frequency/survey_cross_band_RUN_ID
git status --short --branch
git rev-parse HEAD
find "$RUN_ROOT" -mindepth 2 -maxdepth 2 \
  -type d -name calibration.v7.zarr -print | sort
```

Check that:

- each path belongs to a different physical serial;
- every dataset used with this `CONFIG` was collected with the same signed
  calibration configuration;
- primary and repeat datasets are labeled separately; and
- generated results will go to a new output directory.

Do not infer radio identity from directory order, USB address, or a handwritten
label. Use the serial and passive fingerprint stored in V7.

## 2. Validate each dataset from stored IQ

Run strict validation without `--no-recompute-iq`:

```bash
SERIAL=PLUTO_SERIAL
SERIAL_ROOT="$RUN_ROOT/$SERIAL"

python -m spf.calibrations.dual_rx_gain_frequency validate \
  --config "$CONFIG" \
  --dataset "$SERIAL_ROOT/calibration.v7.zarr" \
  --serial "$SERIAL" \
  --output "$SERIAL_ROOT/validation.json"
```

Repeat for every serial. Validation has two distinct meanings:

- Capture integrity covers scheduled shape, V7 schema, direct-USB protocol,
  serial and firmware provenance, gain/RSSI endpoint metadata, sequence, and
  agreement between stored and recomputed IQ metrics. These checks must pass.
- Phase-cell quality covers tone SNR and level, clipping, coherence,
  within-frame stability, and repeatability. Extreme asymmetric pairs may
  legitimately fail and remain in the store with an explicit reason mask.

An aggregate `fail_quality` does not automatically mean capture corruption.
It means one or more scheduled cells are not safe correction points. Never
turn a failed or missing cell into a zero-valued correction.

Pass:

- capture integrity passes;
- complete frame/block counts match the design; and
- quality failures are explicit and plausible.

Fail:

- stored IQ and scalar metrics disagree;
- firmware, serial, or fingerprint provenance is inconsistent;
- the dataset is structurally incomplete but presented as complete; or
- moderate equal/similar-gain cells fail without explanation.

## 3. Fit and report one physical radio

The operational per-radio candidate is an additive circular lookup model at
each measured frequency:

```text
predicted phase(radio, frequency, gain1, gain2)
    = wrap(
        frequency intercept
        + RX1 effect(frequency, gain1)
        + RX2 effect(frequency, gain2)
      )
```

The reference constraints make the lookup terms identifiable, but an
individual RX1 or RX2 term is not an absolute physical path delay. The ordered
receiver roles matter.

Fit one serial:

```bash
python -m spf.calibrations.dual_rx_gain_frequency fit \
  --config "$CONFIG" \
  --dataset "$SERIAL_ROOT/calibration.v7.zarr" \
  --output "$SERIAL_ROOT/model.json"
```

Generate its report and plots:

```bash
python -m spf.calibrations.dual_rx_gain_frequency report \
  --validation "$SERIAL_ROOT/validation.json" \
  --model "$SERIAL_ROOT/model.json" \
  --output-dir "$SERIAL_ROOT/analysis"
```

The generated `analysis/REPORT.md` includes, for every fitted frequency:

- an RX2 sweep at three fixed RX1 gains;
- an RX1 sweep at three fixed RX2 gains;
- observed versus fitted phase;
- residual versus gain mismatch; and
- linked coverage and residual heatmaps.

Review, rather than merely generate, the plots. Look for phase wrapping
mistakes, systematic residual structure, isolated stage-boundary errors,
weak-signal regions, clipped regions, and unsupported holes.

## 4. Interpret the held-out evaluation

The fit excludes an epoch from training when evaluating known cells. Report at
least:

- circular mean absolute error (MAE);
- circular root mean square error (RMSE);
- 95th-percentile absolute error (P95);
- maximum absolute error;
- supported-cell coverage; and
- quality-valid frame and cell counts.

The current four-radio dense reference achieved approximately 0.90 degrees
overall held-out MAE and 3.07 degrees P95 for the per-frequency additive
per-radio model. Individual-radio MAE ranged from 0.83 to 0.97 degrees and P95
from 2.92 to 3.22 degrees. Treat these as measured references, not permanent
acceptance constants.

A useful provisional review gate for newly calibrated radios is:

- structural validation passes;
- known-cell held-out coverage is 100 percent of the model's declared
  production-supported cells;
- held-out MAE is at most 1.5 degrees;
- held-out P95 is at most 5 degrees; and
- outliers and failed cells have been inspected.

This gate is a review recommendation, not an enforcement rule in the CLI.
Change it only with recorded evidence and update the report when doing so.

## 5. Compare two radios

Compare two individually fitted radio baselines:

```bash
python -m spf.calibrations.dual_rx_gain_frequency compare-radios \
  --model-a artifacts/dual_rx_gain_frequency/RUN_A/SERIAL_A/model.json \
  --model-b artifacts/dual_rx_gain_frequency/RUN_B/SERIAL_B/model.json \
  --output-dir artifacts/dual_rx_gain_frequency/radio_A_vs_B
```

This tests whether a linear phase-versus-frequency description is supported.
Its fitted differential delay is an effective electrical group delay, not a
literal PCB trace-length measurement. Splitter/cable paths, analogue filters,
PCB traces, calibration state, and retune state can all contribute.

## 6. Evaluate the full model ladder across radios

Use `model-matrix` for the main cross-radio comparison. Supply exactly one
primary dense dataset per physical radio:

```bash
MATRIX_OUT=spf/calibrations/dual_rx_gain_frequency/reports/six_radio_dense_RUN_ID

python -m spf.calibrations.dual_rx_gain_frequency model-matrix \
  --config "$CONFIG" \
  --dataset \
    /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/survey_cross_band_20260727_v1/104000707f0700120f001a0095f2dbee49/calibration.v7.zarr \
  --dataset \
    /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/survey_cross_band_20260727_v1/104000f6ad020002fdff3a00bba2f096a1/calibration.v7.zarr \
  --dataset \
    /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/survey_cross_band_20260728_new_radios_v1/104000b299050013f4ff0700255e35222f/calibration.v7.zarr \
  --dataset \
    /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/survey_cross_band_20260728_new_radios_v1/104473b80a16000de6ff2000f8a6beca79/calibration.v7.zarr \
  --dataset \
    artifacts/dual_rx_gain_frequency/NEW_RUN/SERIAL_E/calibration.v7.zarr \
  --dataset \
    artifacts/dual_rx_gain_frequency/NEW_RUN/SERIAL_F/calibration.v7.zarr \
  --output-dir "$MATRIX_OUT"
```

The command evaluates radio-specific and universal constant, gain, frequency,
categorical lookup, and differential-delay formulations. Its tests answer
different deployment questions:

| Evaluation | What is held out | What it measures |
| --- | --- | --- |
| Leave one epoch out | one randomized repeat of known frequency/gain cells | Repeatability and correction of already calibrated cells |
| Leave one frequency out | one complete RF frequency | Whether a model truly predicts an unmeasured frequency |
| Leave one radio out | one physical serial | Whether a universal model transfers to an unseen radio |

Do not compare headline errors from different masks as if they were paired.
Prefer paired evaluation on identical valid observations, and report coverage
beside error. A lower error obtained by silently dropping hard cells is not a
better correction model.

Frequency-specific lookup models cannot predict an unseen frequency. Their
leave-one-frequency result is unsupported by design, not zero error. Likewise,
a radio-specific model does not become universal merely because multiple
radio-specific fits have similar shapes.

## 7. Evaluate one- or two-value onboarding

Use `low-cost-calibration` to test whether a universal gain LUT can be adapted
to an unseen radio with only one or two scalar measurements:

```bash
LOW_COST_OUT=spf/calibrations/dual_rx_gain_frequency/reports/low_cost_six_radio_RUN_ID

python -m spf.calibrations.dual_rx_gain_frequency low-cost-calibration \
  --config "$CONFIG" \
  --dataset artifacts/dual_rx_gain_frequency/RUN_A/SERIAL_A/calibration.v7.zarr \
  --dataset artifacts/dual_rx_gain_frequency/RUN_A/SERIAL_B/calibration.v7.zarr \
  --dataset artifacts/dual_rx_gain_frequency/RUN_B/SERIAL_C/calibration.v7.zarr \
  --dataset artifacts/dual_rx_gain_frequency/RUN_B/SERIAL_D/calibration.v7.zarr \
  --dataset artifacts/dual_rx_gain_frequency/NEW_RUN/SERIAL_E/calibration.v7.zarr \
  --dataset artifacts/dual_rx_gain_frequency/NEW_RUN/SERIAL_F/calibration.v7.zarr \
  --repeat-dataset \
    artifacts/dual_rx_gain_frequency/REPEAT_RUN/SERIAL_E/calibration.v7.zarr \
  --repeat-dataset \
    artifacts/dual_rx_gain_frequency/REPEAT_RUN/SERIAL_F/calibration.v7.zarr \
  --output-dir "$LOW_COST_OUT"
```

Omit `--repeat-dataset` until an independent repeat exists. Repeats are used
only to quantify temporal drift. They must never be entered as primary
datasets or counted as extra physical radios.

The four-radio result found that a universal LUT plus one exact-frequency
baseline per new radio had about 3.39 degrees leave-one-radio-out MAE. That is
materially worse than a dense per-radio fit, so the report must state the
accuracy/collection-time tradeoff rather than treating low-cost onboarding as
equivalent.

## 8. Avoid data leakage

The following rules are required for a meaningful evaluation:

- Keep complete randomized epochs together when training or testing a known
  cell.
- Keep the entire frequency out when claiming unseen-frequency performance.
- Keep every dataset and repeat from one physical serial out when claiming
  unseen-radio performance.
- Do not tune gain-stage boundaries or quality thresholds on a held-out fold
  and then report that fold as untouched.
- Do not select a model on test performance and reuse the same score as its
  final unbiased estimate.
- Do not let the same physical radio appear under two labels because a USB
  address or run directory changed.
- Report unsupported predictions as unsupported, never as zero residual.

The deterministic schedule separates the three repetitions over time to make
epoch holdout meaningful. Preserve the stored epoch labels.

## 9. Select a correction and fail closed

For the current per-radio additive lookup model, a prediction is valid only
when all of the following match:

- exact Pluto serial;
- exact measured RF frequency;
- exact ordered RX1/RX2 gain pair;
- non-null fitted RX1 and RX2 effects;
- cell repeatability criterion;
- live gain/RSSI metadata validity; and
- live signal-quality requirements.

Apply the repository's phase convention as:

```text
corrected phase
    = wrap(measured angle(RX1) - angle(RX2) - predicted offset)
```

Do not interpolate to an unmeasured frequency unless a separately evaluated
frequency-generalizing model has passed its leave-one-frequency test. Do not
use one radio's intercept for an unseen serial unless the chosen universal or
low-cost method has passed leave-one-radio-out evaluation at an acceptable
error.

When two candidate models differ by less than the declared practical margin,
prefer the simpler predeclared model. Inspect the residual plots before
concluding that a lower aggregate MAE explains gain-stage behaviour.

## 10. Publish a reproducible report

Raw Zarr stores under `artifacts/dual_rx_gain_frequency/` are intentionally
Git-ignored. Commit the compact evidence needed to reproduce and review the
conclusion under:

```text
spf/calibrations/dual_rx_gain_frequency/reports/REPORT_NAME/
```

A complete report handoff should include:

- Markdown report with the model equations written as portable text;
- exact dataset paths and physical serials;
- config path and config/input hashes;
- SPF Git SHA and dirty state;
- validation and model JSON;
- summary CSV or JSON tables;
- plotted data versus fit and residual distributions;
- held-out split definition, error metrics, and coverage;
- explicit unsupported regions;
- comparison with the prior accepted model; and
- recommendation for seen and unseen radios.

The large IQ does not belong in Git, so input hashes identify the analyzed
scalar evidence but do not replace retaining and backing up the source Zarr.

## Reproducibility checklist

| Check | Required evidence |
| --- | --- |
| Physical identity | One unique stored serial and fingerprint per primary dataset |
| Firmware identity | Release, image SHA, firmware SHA, and gadget SHA in V7 |
| Software identity | SPF Git SHA and clean/dirty flag |
| Dataset integrity | Strict validation output with IQ recomputation |
| Collection completeness | Scheduled/stored frames and complete block counts |
| Quality policy | Configured thresholds and per-reason exclusion counts |
| Fit inputs | Exact config and dataset paths plus hashes |
| Evaluation split | Epoch, frequency, or radio holdout stated explicitly |
| Error and support | Circular MAE/RMSE/P95/max plus coverage |
| Visual review | Data-versus-fit, residual, and coverage plots |
| Deployment scope | Exact serial/frequency/gain support and fail-closed behavior |
| Independent drift | Repeat dataset identified separately, never as another radio |
