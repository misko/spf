# Integer-gain additive-cross experiment at 2.4 GHz

Date: 2026-07-29

## Question

Can the dual-RX phase correction be represented parsimoniously as:

```text
phase(radio, frequency, gain_rx1, gain_rx2)
  = intercept(radio, frequency)
  + H(frequency, gain_rx1)
  - H(frequency, gain_rx2)
  + residual
```

and can that model predict ordered gain pairs that were not used for fitting?

This experiment assumes that an AGC-reported hardware gain in dB refers to the
same AD9361 gain state as the equal manual-gain value. It does not replay
historical capture timing.

## Reproduction

The experiment definition is
[`integer_gain_cross_2p4.yaml`](../configs/integer_gain_cross_2p4.yaml).
It captures every integer gain from -1 through 62 dB at 2.412 and 2.467 GHz.
At each frequency and epoch:

- 127 axis pairs train the model: every `(gain, 26)` and `(26, gain)` pair,
  with `(26, 26)` once;
- 48 off-axis pairs are excluded from fitting and used only for evaluation;
- all pairs are deterministic-randomized; and
- the complete design is repeated in three separated epochs.

That is 175 pairs per frequency and 1,050 frames per physical radio.

The exact collection command was:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/integer_gain_cross_2p4.yaml \
  --output \
    artifacts/dual_rx_gain_frequency/integer_gain_cross_2p4_20260729_special_17_18_v1 \
  --serial 104000bac4950008230026001b440a003a \
  --serial 1040007c4a94000211000b009186843ef2 \
  --ready-manifest /run/spf/direct_usb_ready.json
```

The schedule signature was
`ef76b41b65e2a117e612ddfe75316e9d60560387d5ebe01e37faa18d29c99a44`.
The local artifact root is 437 MiB and contains the two V7 Zarrs, observation
logs, strict validation JSON, per-radio analysis JSON/Markdown, and a
cross-radio comparison bundle. The raw artifact is intentionally not added to
Git.

### Input provenance

| Input | SHA-256 |
|---|---|
| Experiment YAML | `edcb4fdff2ffee633bed391250a715f8b612ebf4ed530503b23d2c63cac43709` |
| `.17` analysis-input arrays and acquisition attributes | `c63ba5e0c4ee0c13c2352aa918c710ced959992d1d5bb6bf8cd40f1034bfe7c6` |
| `.18` analysis-input arrays and acquisition attributes | `8e0feb957172426d7e69cf3c69fcefbf6969ce6d371746e5bf9dabfef23f17b1` |
| `.17` additive-cross analysis JSON | `d7b9d822ce334c82b3516078db3d1dd42137ea295c946b4f9b8c75f5584f0019` |
| `.18` additive-cross analysis JSON | `41ff78bb7453d2d1f45c19d75bbf8d4d6d01dd31fdc7f0707af5d7335200731d` |
| Cross-radio comparison JSON | `5e5e6aa6001d479d9ac9b2f6a4fc74586652f6624fbcc680350cea26313aceae` |

The analysis-input digests hash the coordinate, completion, quality, phase,
and timestamp arrays plus acquisition provenance attributes. They do not hash
the 128 GiB logical LMDB map byte-for-byte.

## Acquisition gates

| Gate | Pass condition | Result |
|---|---|---|
| Firmware | Both radios verified against the config’s RAM image and protocol v2 | Pass |
| Schedule | 1,050 unique frames/radio; 127 training + 48 held-out pairs/frequency | Pass |
| Completion | 2,100/2,100 total frames | Pass |
| Metadata | Exact requested start/end gains and safe direct-USB flags | Pass |
| IQ re-analysis | Stored phase, tone, coherence, clipping, and reasons reproduce | Pass |
| Cell quality | At least 2/3 valid repeats and repeat circular std ≤ 5° | 350/350 cells on each radio |
| Capture errors | No terminal measurement errors | Pass; zero errors |

The capture took 16 minutes 26 seconds. Every one of the 2,100 frames was
quality-valid.

## Held-out prediction

These metrics use the 48 off-axis pairs that were never used to estimate the
gain curves. Frame metrics contain 288 observations per radio: 48 pairs × 2
frequencies × 3 epochs.

| Radio | Model | Held-out MAE | Held-out p95 | Maximum |
|---|---|---:|---:|---:|
| `…440a003a` (.17) | Independent RX1/RX2 curves | 1.48° | 2.73° | 6.09° |
| `…440a003a` (.17) | Shared `H(g1) - H(g2)` | 1.41° | 2.65° | 4.47° |
| `…86843ef2` (.18) | Independent RX1/RX2 curves | 1.48° | 2.94° | 4.33° |
| `…86843ef2` (.18) | Shared `H(g1) - H(g2)` | 1.25° | 2.49° | 5.35° |

The shared antisymmetric curve is at least as good as independently fitted RX1
and RX2 curves. This supports the parsimonious `H(g1) - H(g2)` formulation.

## The gain curve is stepped, not smooth

The dominant adjacent integer-gain transitions reproduce on both physical
radios:

| Gain transition | Typical signed phase step | Interpretation |
|---:|---:|---|
| 2→3 dB | +3.2° to +3.4° | Repeatable stage transition |
| 14→15 dB | +4.1° to +4.2° | Repeatable stage transition |
| 24→25 dB | +2.5° to +2.9° | Repeatable stage transition |
| 31→32 dB | −2.6° to −4.4° | Repeatable stage transition |
| 49→50 dB | −14.3° to −16.7° | Dominant transition |

The old stage-focused grid sampled 41 and 52 dB but not 49 and 50 dB. It
therefore made the dominant discontinuity look like a gradual change across
an 11 dB interval. This is the main reason naive interpolation fails.

## Dense integer lookup versus the old 17-gain grid

The following comparison uses held-out ordered-pair cell means. Sparse methods
see only the old 17 gains:

```text
[-1, 0, 3, 5, 6, 15, 16, 17, 23, 25, 26, 27, 33, 34, 41, 52, 62]
```

| Radio/frequency | Dense integer MAE / p95 | Sparse linear MAE / p95 | Sparse nearest MAE / p95 |
|---|---:|---:|---:|
| `.17`, 2412 MHz | 1.73° / 2.61° | 3.03° / 9.78° | 3.80° / 16.82° |
| `.17`, 2467 MHz | 0.97° / 1.76° | 2.47° / 9.39° | 3.08° / 14.77° |
| `.18`, 2412 MHz | 1.54° / 2.28° | 2.92° / 9.96° | 3.58° / 16.85° |
| `.18`, 2467 MHz | 0.87° / 1.46° | 2.46° / 9.86° | 2.92° / 14.17° |

Linear interpolation over the old grid is not a safe general policy. It
smooths across real gain-table discontinuities. Unbounded nearest-neighbour
snapping is worse.

For the held-out cells in this experiment, nearest snapping no more than 2 dB
has MAE 1.65–2.14° and p95 3.65–4.57°. At a 3 dB maximum snap, MAE jumps to
4.88–5.76° and p95 to 13.60–14.39°. Therefore, if an old sparse artifact must
be used temporarily:

- exact lookup remains preferred;
- `nearest_within_2_db` is a defensible preliminary fallback;
- requests more than 2 dB from a measured value should fail closed; and
- linear interpolation must not cross a known step.

This fallback is based on two radios and 48 held-out pairs/frequency. The dense
integer table removes the need to snap integer-valued AD9361 gains.

## Transfer between radios

The gain-dependent shape is substantially more transferable than the absolute
intercept:

| Frequency | Cross-radio curve correlation | Curve RMS difference | Maximum difference |
|---:|---:|---:|---:|
| 2412 MHz | 0.9982 | 0.41° | 1.79° |
| 2467 MHz | 0.9979 | 0.46° | 2.29° |

With a separate intercept for each radio and exact frequency:

- one radio-shared curve per frequency predicts held-out cell means at 1.27°
  MAE and 2.36° p95;
- one gain curve shared across both tested 2.4 GHz frequencies gives 1.39° MAE
  and 2.81° p95.

The safer operational model is therefore:

```text
radio/frequency-specific intercept + frequency-specific shared integer H(g)
```

A low-cost onboarding calibration can measure the radio’s intercept at
`(26, 26)` for each exact operating frequency, while reusing the shared dense
gain curve. A future experiment must validate that proposal on radios excluded
from constructing the shared curve.

## Recommendation

1. Publish exact integer-gain tables for 2.412 and 2.467 GHz; do not interpolate
   the old 17-point table into the 49→50 dB discontinuity.
2. Keep the radio/frequency intercept separate from the gain-dependent term.
3. Use the shared `H(g1) - H(g2)` formulation; it is simpler and performed at
   least as well on genuine held-out pairs.
4. For an unseen radio, acquire one `(26, 26)` intercept per exact frequency
   and apply the shared integer curve, then validate on a small off-axis spot
   check before production.
5. Continue to treat endpoint gain metadata honestly: equality at the two
   observed endpoints is not proof that AGC did not transition inside a frame.

The unexplained difference between the current loopback intercepts and some
historical wall-corpus absolute offsets is outside this gain-curve experiment.
The physical arms may be identical; that discrepancy should remain an open
controlled-validation question rather than being attributed to the fixture
without evidence.
