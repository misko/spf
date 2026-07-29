# Complete 2.4 GHz gain model

Date: 2026-07-29

## Outcome

A complete integer-gain model was collected, validated, fitted, and exported
for the two attached radios historically labelled `.17` and `.18`.

The model does not distinguish AGC from manual gain. Its inputs are:

```text
physical Pluto serial
exact LO frequency
realized RX1 gain in dB
realized RX2 gain in dB
```

Its output is the systematic RX1-minus-RX2 phase offset:

```text
offset(r, f, g1, g2) = C(r, f) + H(r, f, g1) - H(r, f, g2)
```

The gain domain is every integer from -3 through 71 dB. The frequency domain
is exactly 2412 and 2467 MHz. The implementation deliberately refuses other
frequencies.

## Experimental design

At each frequency and epoch, the capture measured both complete gain axes
around a 26 dB reference:

```text
(g, 26) for every g in [-3, 71]
(26, g) for every g in [-3, 71]
```

This is 149 fitted pairs per frequency. A further 56 off-axis ordered pairs
were excluded from fitting and used to test whether the additive
factorization predicts unseen RX1/RX2 combinations. Three separately
randomized epochs were collected.

| Quantity | Per radio | Both radios |
|---|---:|---:|
| Frequencies | 2 | 2 |
| Integer gain states | 75 | 75 |
| Training cells | 298 | 596 |
| Held-out cells | 112 | 224 |
| Frames | 1,230 | 2,460 |

Every gain value was therefore observed in both receiver roles at both
frequencies. Once the additive form passed off-axis tests, its mathematical
domain became all 75 × 75 ordered gain pairs at each exact frequency: 11,250
predictions per radio.

## Capture integrity and quality

| Radio | Frames complete | Quality-valid frames | Training cells passing | Held-out cells passing |
|---|---:|---:|---:|---:|
| `1040007c4a94000211000b009186843ef2` (`.18`) | 1,230 / 1,230 | 1,206 | 298 / 298 | 104 / 112 |
| `104000bac4950008230026001b440a003a` (`.17`) | 1,230 / 1,230 | 1,206 | 298 / 298 | 104 / 112 |

All frames had valid protocol-v2 gain/RSSI metadata, verified firmware
provenance, and the expected batched discard/capture sequence. Stored-IQ
recomputation completed for both datasets.

The same eight held-out cells failed on each radio: `(-3, 71)`, `(71, -3)`,
`(-2, 63)`, and `(63, -2)` at both LOs. These deliberately extreme
65–74 dB channel imbalances put the weak-channel tone below the phase-quality
threshold. They do not leave an unmeasured gain-axis parameter: -3, -2, 63,
and 71 dB all passed when measured on each receiver against the 26 dB
reference. The validator continues to report the overall datasets as
`fail_quality`; the export gate separately requires all 298 fitted-axis cells
to pass and held-out p95 error to remain below 5 degrees.

## Held-out model accuracy

The exported runtime model uses the parsimonious antisymmetric gain curve
`H(g1) - H(g2)`.

| Radio | Held-out frames | MAE | p95 | Maximum |
|---|---:|---:|---:|---:|
| `.18` | 312 | 1.95° | 3.38° | 4.70° |
| `.17` | 312 | 2.03° | 3.55° | 7.07° |

The maximum is descriptive; the export pass/fail gate is based on p95 and is
5 degrees.

The gain shape is highly consistent between these radios:

| Frequency | Curve correlation | RMS radio difference | Maximum radio difference |
|---:|---:|---:|---:|
| 2412 MHz | 0.9979 | 0.48° | 1.69° |
| 2467 MHz | 0.9991 | 0.43° | 0.97° |

A curve shared between the two radios, while retaining each radio/frequency
intercept, scored 1.95° MAE and 3.12° p95 on held-out cell means. The committed
runtime models conservatively retain a fitted gain curve for each serial.

## Runtime use

```python
from spf.calibrations.models import load_model

model = load_model(
    "complete_2p4_shared_gain_lut_per_radio",
    "104000bac4950008230026001b440a003a",
)

offset_rad = model.predict_phase_offset(
    frequency_hz=2_412_000_000,
    gain_rx1_db=27,
    gain_rx2_db=44,
)

corrected_rad = model.correct_measured_phase(
    measured_phase_rad=0.4,
    frequency_hz=2_412_000_000,
    gain_rx1_db=27,
    gain_rx2_db=44,
)
```

This API is identical whether the realized gain values came from AGC or were
set manually. Production use still requires valid, buffer-associated gain
metadata. Equal endpoint gains are not proof that no transition occurred
inside a buffer.

## Capture throughput

The original finite-stream path issued one direct-USB request for the
discarded frame and another for the retained frame. The optimized path asks
for both frames in one finite request and converts only the retained frame.

| Measurement | Original path | Batched path |
|---|---:|---:|
| Mean observation time | 441.7 ms | 275.8 ms |
| Median observation time | 433.8 ms | 272.9 ms |
| p95 observation time | 478.4 ms | 279.7 ms |
| Observed throughput | about 2.1 frames/s | 3.30 frames/s |

This reduced per-frame time by 37.6%. The complete 2,460-frame capture
finished in about 12 minutes 25 seconds. Profiling showed that the bottleneck
was the serialized finite USB lifecycle plus tone/phase analysis, not overall
Pi CPU capacity.

## Reproduction

Collect:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config spf/calibrations/dual_rx_gain_frequency/configs/all_gain_cross_2p4.yaml \
  --output artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN
```

Validate and fit each serial:

```bash
python -m spf.calibrations.dual_rx_gain_frequency validate \
  --config spf/calibrations/dual_rx_gain_frequency/configs/all_gain_cross_2p4.yaml \
  --dataset artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/calibration.v7.zarr \
  --serial SERIAL \
  --output artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/validation.json

python -m spf.calibrations.dual_rx_gain_frequency fit-additive-cross \
  --config spf/calibrations/dual_rx_gain_frequency/configs/all_gain_cross_2p4.yaml \
  --dataset artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/calibration.v7.zarr \
  --output-dir artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/additive_cross
```

Export only after those gates:

```bash
python -m spf.calibrations.dual_rx_gain_frequency export-complete-2p4 \
  --analysis artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/additive_cross/analysis.json \
  --validation artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/validation.json \
  --output-root spf/calibrations/models
```

## Reproducibility hashes

| Input | SHA-256 |
|---|---|
| `all_gain_cross_2p4.yaml` | `4872c405e816689e6a0a91f5319a56a6dedb239965ddc7aa39f7bad8b65a3d6a` |
| `.18` analysis | `d9b28600bc606545d42193dd11eb136c823a8c55b4c3c09a6744624ce3061bde` |
| `.18` validation | `efe13d6ce220d89c44bdb1b11151ce91d29f752bd4d9ea36f39022a99e21d217` |
| `.17` analysis | `e5e1a24a204da8f72422c6291707d8aac1bc191e70ab3d8d0f70e2fed5697c31` |
| `.17` validation | `43cbc30385f83a695825a3f4395603457c72b7ec6d5b3627ba7c6d6cc2e725a1` |
| Two-radio comparison | `af82386c3ef309dbb844e067e86377b7f20c5ff47fb70e1844e18d800810741e` |

The large source datasets remain under the gitignored `artifacts/` tree. The
small exported model and support files contain their source hashes.
