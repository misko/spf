# Runtime phase-offset models

This directory contains deployable, serial-specific phase-offset models
exported from the reproducible dense calibration model matrix.

The phase convention is:

```text
RX1 phase minus RX2 phase
```

The model predicts the systematic phase offset in radians. Correct a measured
phase by subtracting the prediction and wrapping to `[-pi, pi)`.

## Recommended model

Use `frequency_specific_additive_gain_per_radio` for known frequencies and
ordered gain pairs. It is the parsimonious high-accuracy model:

```text
offset(f, g1, g2) =
    intercept[f] + RX1_effect[f, g1] + RX2_effect[f, g2]
```

The other per-radio model families are retained for reproducible comparisons,
diagnostics, and deployment trade-off tests. `registry.json` lists every model
and available physical serial.

## Python

```python
from spf.calibrations.models import load_model

model = load_model(
    "frequency_specific_additive_gain_per_radio",
    "104000707f0700120f001a0095f2dbee49",
)

offset_rad = model.predict_phase_offset(
    frequency_hz=2_412_000_000,
    gain_rx1_db=26,
    gain_rx2_db=41,
)

corrected_rad = model.correct_measured_phase(
    measured_phase_rad=0.4,
    frequency_hz=2_412_000_000,
    gain_rx1_db=26,
    gain_rx2_db=41,
)
```

Strict support checking is enabled by default. The exact frequency and ordered
gain pair must have passed the source dataset's three-epoch cell-quality gate.
Unsupported cells raise `UnsupportedPhaseModelInput`; they never silently
interpolate or return zero.

## Command line

```bash
python -m spf.calibrations.models predict \
  --model frequency_specific_additive_gain_per_radio \
  --serial 104000707f0700120f001a0095f2dbee49 \
  --frequency-hz 2412000000 \
  --gain-rx1-db 26 \
  --gain-rx2-db 41
```

## Re-export

The committed configs are generated from a model-matrix JSON:

```bash
python -m spf.calibrations.models export \
  --matrix spf/calibrations/dual_rx_gain_frequency/reports/six_radio_dense_20260729_v1/model_matrix.json \
  --output spf/calibrations/models
```

Each model config records the source matrix hash, dataset analysis-input hash,
firmware provenance, named coefficients, and a hash-pinned serial-specific
support profile.
