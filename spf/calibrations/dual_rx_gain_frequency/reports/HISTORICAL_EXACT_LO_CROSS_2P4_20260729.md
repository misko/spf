# Exact historical 2.4 GHz LO calibration

Date: 2026-07-29

## Outcome

The two radios historically labelled `.17` and `.18` were calibrated at the
exact intended wall-array LOs:

```text
2411950000 Hz
2467100000 Hz
```

Both captures completed, every fitted-axis cell passed, and the two
serial-specific runtime models now cover all four exact 2.4 GHz centres:

```text
2411950000, 2412000000, 2467000000, 2467100000
```

The original 2412 and 2467 MHz coefficients remain available; nearby
frequencies were not aliased or overwritten.

## Why the historical values looked different

Some historical array/tensor views report:

| Reported integer | Intended configured LO | Difference |
|---:|---:|---:|
| 2,411,950,080 Hz | 2,411,950,000 Hz | +80 Hz |
| 2,467,099,904 Hz | 2,467,100,000 Hz | -96 Hz |

These are exactly the float32 representations of the intended integer-Hz
values. They are representation artifacts, not evidence that the synthesizer
was deliberately configured 80 or 96 Hz away.

The runtime API still requires exact support by default. A caller that only
has an integerized float32 field may explicitly request float32 alias
recovery. This option only recognizes the exact float32 representation of a
fitted LO; it is not interpolation and does not make 2467.0 MHz transferable
to 2467.1 MHz.

## Experimental design

The experiment used the same complete additive-cross design as the earlier
2412/2467 MHz run:

```text
(g, 26) for every integer g from -3 through 71 dB
(26, g) for every integer g from -3 through 71 dB
56 off-axis ordered gain pairs held out from fitting
3 separately randomized epochs
```

| Quantity | Per radio | Both radios |
|---|---:|---:|
| Frequencies | 2 | 2 |
| Integer gain states | 75 | 75 |
| Training cells | 298 | 596 |
| Held-out cells | 112 | 224 |
| Frames | 1,230 | 2,460 |

The hardware path was TX2 through the existing 30 dB attenuator and splitter
to RX1/RX2. Capture used V7, protocol-v2 gain/RSSI metadata, a 30 MS/s sample
rate, 3 MHz RF bandwidth, 65,536 samples per frame, and FPGA DDS at a 100 kHz
offset. The run completed in about 12 minutes 39 seconds.

## Capture and validation

| Radio | Frames complete | Quality-valid | Training cells passing | Held-out cells passing |
|---|---:|---:|---:|---:|
| `.17` (`104000bac4950008230026001b440a003a`) | 1,230 / 1,230 | 1,206 | 298 / 298 | 104 / 112 |
| `.18` (`1040007c4a94000211000b009186843ef2`) | 1,230 / 1,230 | 1,206 | 298 / 298 | 104 / 112 |

The eight failed held-out cells per radio are the same deliberately extreme
weak-channel cases observed in the earlier complete run. The validator
therefore honestly labels each complete dataset `fail_quality`. Runtime export
uses a narrower model acceptance gate: every fitted-axis cell must pass, and
each frequency's off-axis held-out p95 must be below 5 degrees. Both radios
pass that model gate.

## Held-out accuracy

The exported model is:

```text
offset(r, f, g1, g2) = C(r, f) + H(r, f, g1) - H(r, f, g2)
```

| Radio | Shared-curve held-out frames | MAE | p95 | Maximum |
|---|---:|---:|---:|---:|
| `.17` | 312 | 1.96° | 3.48° | 6.32° |
| `.18` | 312 | 1.92° | 3.19° | 5.10° |

By exact frequency:

| Frequency | `.17` MAE / p95 | `.18` MAE / p95 |
|---:|---:|---:|
| 2411.950 MHz | 2.46° / 4.80° | 2.33° / 3.60° |
| 2467.100 MHz | 1.47° / 2.56° | 1.51° / 2.46° |

The 2467.100 MHz calibration is the stronger external-validation target
identified by the model consumer. Applying the correction to the independent
98-capture wall set remains a consumer-side validation, not part of this
bench result.

## Runtime usage

Use the intended integer LO when it is available:

```python
model = load_model(
    "complete_2p4_shared_gain_lut_per_radio",
    "104000bac4950008230026001b440a003a",
)

offset_rad = model.predict_phase_offset(
    frequency_hz=2_467_100_000,
    gain_rx1_db=26,
    gain_rx2_db=41,
)
```

Recover a historical float32 representation only when necessary:

```python
offset_rad = model.predict_phase_offset(
    frequency_hz=2_467_099_904,
    gain_rx1_db=26,
    gain_rx2_db=41,
    allow_float32_frequency_alias=True,
)
```

Without that explicit option, the second call fails closed.

## Reproduction

Collect the exact-frequency run:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config spf/calibrations/dual_rx_gain_frequency/configs/historical_exact_lo_cross_2p4.yaml \
  --output artifacts/dual_rx_gain_frequency/HISTORICAL_EXACT_RUN
```

Validate and fit each serial using the same `validate` and
`fit-additive-cross` commands documented in
[`ALL_GAIN_CROSS_2P4_20260729.md`](ALL_GAIN_CROSS_2P4_20260729.md), substituting
this configuration and run directory.

Merge the old and new disjoint exact-frequency fits:

```bash
python -m spf.calibrations.dual_rx_gain_frequency export-complete-2p4 \
  --analysis artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/additive_cross/analysis.json \
  --validation artifacts/dual_rx_gain_frequency/ALL_GAIN_RUN/SERIAL/validation.json \
  --analysis artifacts/dual_rx_gain_frequency/HISTORICAL_EXACT_RUN/SERIAL/additive_cross/analysis.json \
  --validation artifacts/dual_rx_gain_frequency/HISTORICAL_EXACT_RUN/SERIAL/validation.json \
  --output-root spf/calibrations/models
```

The exporter requires one physical serial, identical contiguous gain axes,
identical phase conventions, and disjoint fitted frequencies.

## Reproducibility hashes

| Input | SHA-256 |
|---|---|
| `historical_exact_lo_cross_2p4.yaml` | `cf1e8bb91a9c86d618c4ee1aeb1138acc3170d0f9e11154ea22d9da3dc1c52a8` |
| `.17` analysis | `c5dd314d181c24ee6f22d1c8e9f1830d67f71ac2240cc58b95c8516323604800` |
| `.17` validation | `0f8bc09283ba54554ff2bb20456522d3af9d1804b6970e9eca403b16269fa2bb` |
| `.18` analysis | `9edd1f9914ba6dd1dfcc964512e85e8072e21f6205da589c56384205be0e5d30` |
| `.18` validation | `d21c3a416edcaaf7a28effe78716c5bf6693206b3e919cda7f2f6934b6befbb9` |
| Two-radio comparison | `c4a64aa13a73c22722dcbccff6034e6d8657011dad50621821c4da62c1b58fff` |

Large source datasets remain under the gitignored `artifacts/` tree. The
committed model and support JSON files record the source paths and hashes.
