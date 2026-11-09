# New-radio cross-band pilot, 2026-07-28

This report set records the first bounded V7 calibration of the two replacement
Pluto radios:

- `104473b80a16000de6ff2000f8a6beca79`
- `104000b299050013f4ff0700255e35222f`

The physical path on each board was:

```text
TX2 -> 30 dB attenuator -> two-way splitter -> RX1 and RX2
```

The source configuration is
[`pilot_cross_band.yaml`](../../configs/pilot_cross_band.yaml): 12 RF
frequencies, the complete `[-1, 26, 62]` dB RX1-by-RX2 grid, and three
separately randomized epochs. The run completed all 648 scheduled frames in
8 minutes 5 seconds.

## Firmware isolation and TX qualification

The radios were fully USB-power-cycled before this run. Their persistent
`v0.37-dirty` firmware and `ad9361`/`2r2t` U-Boot configuration were verified
before loading the direct-USB image.

An otherwise identical standard-IIO TX2/RX probe was run at 868 MHz before and
after RAM loading:

| Serial | Firmware state | RX1/RX2 on-off delta | Result |
|---|---|---:|---|
| `…beca79` | persistent v0.37 | 76.65 / 70.57 dB | pass |
| `…5222f` | persistent v0.37 | 70.90 / 70.73 dB | pass |
| `…beca79` | modified RAM image, before direct RX | 73.72 / 70.05 dB | pass |
| `…5222f` | modified RAM image, before direct RX | 2.95 / -3.54 dB | first arm failed |

Four fresh standard-IIO retries on `…5222f` then passed three times and failed
once. This establishes intermittent TX/DDS arming, not a permanently unusable
TX2 path or a deterministic incompatibility with the modified image.

After the power cycle and RAM load, both radios passed the committed
direct-USB observed-tone probe on the first attempt at both 868 MHz and
5.866 GHz. During the calibration itself, all 72 radio/frequency/epoch
preflights passed on attempt 1. No failed preflight or runtime error was
admitted to either dataset. The evidence is in each radio's
[`preflight.jsonl`](104473b80a16000de6ff2000f8a6beca79/preflight.jsonl).

The earlier all-attempt failure was therefore consistent with stale runtime
state cleared by a complete USB power cycle. The isolated one-in-four failure
after RAM boot means the exact low-level DDS-arm cause is not yet proven. The
observed-tone preflight remains mandatory; successful register readback alone
is not enough.

## Dataset and metadata integrity

Both V7 calibration stores contain 324/324 completed frames and are readable
by the offline validator. Every frame contains valid protocol-v2 RX1/RX2 gain
dB and RSSI start/end metadata. All observed gain endpoints were equal.

| Serial | Materialized-array SHA-256 | Stable hardware fingerprint |
|---|---|---|
| `…beca79` | `42faa026ec87ebfea3ef26b6f3863b26b7770b01d60cbdba081232f751987aea` | `f6ab18698d1f1f6d056bfc2959a0c95889ae6dfeca8e2da4dc90d90542828ca8` |
| `…5222f` | `7bfde2e42cf049d6178d1592a855804f3da0c6dfb5ed6d3c2a1aad751096e6d0` | `05ca9fe0df8b2603df09dc0ba7c758b923c730a6d1a00809e64d46eb98f4bba5` |

The materialized-array hash covers every stored Zarr array schema and written
chunk while excluding mutable attributes. It deliberately does not synthesize
the unwritten portion of the sparse preallocated arrays.

Acquisition provenance:

- SPF acquisition SHA: `dea791bb8693a43f2eed2979f2ece8af75b90373`
- firmware release:
  `v0.38-plutoplus-spf-gain-rssi-fingerprint-v1`
- firmware image SHA-256:
  `0a6a8939b31babed2ad7093d83941ebc809323d69804adcd8da5bcae0e48d3e9`
- firmware Git SHA: `7b02276519a802aed83d47b6672c46e578ce4de0`
- gadget Git SHA: `a1e6417d07188bd72be70692e28c5d6ae9a5ec62`
- fingerprint session:
  `0bcacbff-3add-4326-bd1a-46472f5aa2d7`

The raw LMDB/Zarr stores remain gitignored under:

```text
/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/pilot_cross_band_20260728_new_radios_v2
```

## Quality coverage

Both strict validators report `fail_quality`, but this is not a missing-frame,
metadata, transport, or TX-preflight failure. The declared criterion requires
at least two phase-valid epochs in every one of the 108 frequency/gain cells.
The common tone cannot simultaneously remain comfortably measurable in the
low-gain receiver and below clipping in the high-gain receiver at a 63 dB gain
mismatch.

| Gain mismatch | `…beca79` passing cells | `…5222f` passing cells |
|---:|---:|---:|
| 0 dB | 36/36 | 36/36 |
| 27 dB | 24/24 | 24/24 |
| 36 dB | 14/24 | 23/24 |
| 63 dB | 0/24 | 0/24 |

Consequently, equal-gain and moderate-mismatch calibration is strong, while
63 dB mismatch is explicitly unsupported by this experiment. It must not be
filled by interpolation or treated as a measured correction.

## Fitted-model findings

| Serial | Quality-valid frames | Additive CV MAE | CV RMSE | CV p95 |
|---|---:|---:|---:|---:|
| `…beca79` | 221/324 | 1.45° | 2.06° | 4.41° |
| `…5222f` | 249/324 | 1.03° | 1.49° | 3.48° |

For both radios, the parsimonious supported model is:

```text
phase = intercept[radio, frequency]
      + RX1_effect[radio, frequency, gain1]
      + RX2_effect[radio, frequency, gain2]
```

A gain-pair interaction table does not materially improve held-out error.
Sharing gain curves across all frequencies is worse, and replacing the
per-frequency intercept with one constant-plus-delay slope is much worse.

The boards are externally compatible but not phase-interchangeable. The
cross-radio phase difference is strongly band-dependent. One global
differential-delay fit leaves 14.77° MAE and 44.13° p95 residual, so literal
PCB path length is not a sufficient explanation. Gain tables, analogue
filtering, LO/calibration state, and board-specific RF implementation remain
material.

Reports:

- [`…beca79`](104473b80a16000de6ff2000f8a6beca79/REPORT.md)
- [`…5222f`](104000b299050013f4ff0700255e35222f/REPORT.md)
- [cross-radio comparison](comparison/CROSS_RADIO_REPORT.md)

## Reproduction

After RAM-loading and writing a fresh readiness manifest:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml \
  --output /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/pilot_cross_band_20260728_new_radios_v2
```

Then run `validate`, `fit`, and `report` for each serial as documented in
[`../../README.md`](../../README.md), followed by `compare-radios` using the
two generated `model.json` files.

