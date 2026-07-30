# A–G spectroscopy campaign

This directory contains the committed, reviewable outputs from the
2026-07-30 controlled two-radio spectroscopy campaign.

- [Campaign analysis](campaign/REPORT.md) covers the RX1 pad, RX1 jumper,
  restored-baseline, TX-level, low-gain, and hot-repeat comparisons.
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

Recreate the campaign analysis with:

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_analysis \
  --campaign-root /home/pi/spf-campaigns/spectroscopy_20260730_full_r2 \
  --treated-serial 104000bac4950008230026001b440a003a \
  --control-serial 1040007c4a94000211000b009186843ef2 \
  --output-dir /home/pi/spf-campaigns/spectroscopy_20260730_full_r2/analysis/campaign
```
