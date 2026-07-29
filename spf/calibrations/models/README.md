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

## Complete 2.4 GHz integer-gain model

For the two radios historically labelled `.17` and `.18`, use
`complete_2p4_shared_gain_lut_per_radio`. It covers every ordered pair of
integer RX gains from -3 through 71 dB at these exact integer-Hz LOs:

```text
2411950000
2412000000
2467000000
2467100000
```

```python
model = load_model(
    "complete_2p4_shared_gain_lut_per_radio",
    "104000bac4950008230026001b440a003a",
)
```

This family is fitted from both complete one-dimensional receiver axes and
validated on off-axis pairs excluded from fitting. Its strict support is the
validated cartesian model domain, not a claim that all 5,625 ordered pairs per
frequency were directly captured. It rejects other gains and frequencies.
The gain controller mode is not an input: AGC and manual captures use the same
model when they report the same realized dB states.

Historical float32 fields may expose `2467100000` as `2467099904`, or
`2411950000` as `2411950080`. Prefer the original integer configuration value.
If only the float32 representation is available, opt in explicitly:

```python
offset_rad = model.predict_phase_offset(
    frequency_hz=2_467_099_904,
    gain_rx1_db=26,
    gain_rx2_db=41,
    allow_float32_frequency_alias=True,
)
```

This only recognizes the exact float32 representation of a fitted LO. It does
not enable a frequency tolerance, nearest-frequency selection, or
interpolation. The default remains fail closed.

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
