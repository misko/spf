# Cross-band scout plotted reports

These reports are generated from the completed three-epoch V7 cross-band
scout. Each radio report embeds:

- 47 four-panel per-frequency data-versus-fit figures;
- four paginated fitted-gain-effect overviews; and
- one fitted frequency-baseline/effective-delay figure.

The four-panel figures show RX2 swept at fixed RX1 gains of -1, 26, and 62 dB,
the symmetric RX1 sweep, observed passing cell means versus the final additive
fit, and cell-mean residual versus gain mismatch. Failed or unsupported cells
are absent rather than interpolated.

- [Radio A — `104000f6ad020002fdff3a00bba2f096a1`](104000f6ad020002fdff3a00bba2f096a1/REPORT.md)
- [Radio B — `104000707f0700120f001a0095f2dbee49`](104000707f0700120f001a0095f2dbee49/REPORT.md)

The large source V7 datasets remain gitignored under:

```text
artifacts/dual_rx_gain_frequency/frequency_scout_cross_band_20260727_v1
```

Regenerate the reports with the commands documented in
[`../../README.md`](../../README.md).
