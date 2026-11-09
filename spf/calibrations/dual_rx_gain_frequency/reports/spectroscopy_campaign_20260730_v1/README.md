# A–G spectroscopy campaign

This directory contains the committed, reviewable outputs from the
2026-07-30 controlled two-radio spectroscopy campaign.

- [Final campaign analysis](campaign/REPORT.md) covers acquisition integrity,
  active gain-table bytes, the RX1 pad and jumper, shared ripple-delay
  components, restored-baseline and hot-repeat stability, crossed TX levels,
  low-gain overlap, schedule-order hysteresis, and the model ladder.
- [Stage A model ladder](model_matrix_A/MODEL_MATRIX_REPORT.md) evaluates the
  400–5900 MHz spectroscopy baseline.
- [Stage F model ladder](model_matrix_F/MODEL_MATRIX_REPORT.md) evaluates the
  extended low-gain and TIA-boundary measurements.

The treated radio was serial
`104000bac4950008230026001b440a003a` (historical label `.17`). The unchanged
control was `1040007c4a94000211000b009186843ef2` (historical label `.18`).
Only the treated radio's RX1 path was modified:

- Stage B used the available nominal 2 + 3 + 6 dB pad stack (11 dB total).
- Stage C replaced the pads with a nominal, uncharacterized 30 cm jumper.
- Stage D restored the original harness.

The large V7/LMDB captures remain outside Git. Their scalar inputs, model
inputs, hashes, fitted results, and figures are recorded in `analysis.json`
and the two `model_matrix.json` files.

The main result is deliberately qualified. The dominant 2.548 ns
(381.9 mm one-way free-space-equivalent) ripple component falls by 81.5% on
the padded RX1 while the three untouched arms retain a median 98.6% of their
baseline amplitude. However, restoring the original harness did not restore
the treated radio's high-band Stage A phase state. The pad result therefore
supports an external-path mechanism, but connector re-mating or a persistent
RX1 state transition prevents a clean pad-only causal claim.

Recreate the campaign analysis with:

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_analysis \
  --campaign-root /home/pi/spf-campaigns/spectroscopy_20260730_full_r2 \
  --treated-serial 104000bac4950008230026001b440a003a \
  --control-serial 1040007c4a94000211000b009186843ef2 \
  --prior-calibration-root /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/overnight_wide_integer_gain_cross_20260730_special_17_18_v1 \
  --output-dir /home/pi/spf-campaigns/spectroscopy_20260730_full_r2/analysis/campaign
```
