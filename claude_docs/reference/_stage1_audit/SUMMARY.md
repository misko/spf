# Stage-1 training audit — synthesis & debate resolution

End-to-end read-only audit of the exact stage-1 path
`python spf/scripts/train_single_point.py -c model_configs/jul8_single_3p7_thin_noblade_jun26_2026.yaml`
(model `beamformer` = `SinglePointWithBeamformer`), traced segmentation → dataset → model → loss.
Three independent agents audited one layer each; the cross-layer claims below were then
**personally re-verified against source** (the "debate"). **No code was changed.**

Layer reports: [`layer1_data_segmentation.md`](layer1_data_segmentation.md) ·
[`layer2_model.md`](layer2_model.md) · [`layer3_training.md`](layer3_training.md).

## Verdict

**The stage-1 path is internally consistent and correct to run.** All three boundary contracts
match, the two hard shape contracts are verified, and the two relevant scheduler/keys hazards
(#1 cosine, #36 KeyError) do **not** apply to this config. No new blocking bug was found. The
findings below are clarifications and non-blocking caveats.

## The reconciled data flow (verified)

`n_radios: 2` ⇒ the dataset loads **both receivers** (`paired = n_radios>1`,
`train_single_point.py:83`) and the collate flattens them, so the effective batch is
**B = 256×2 = 512 rows**, S=1, n_windows=256, nthetas=65, all float32. The single model processes
each radio-row independently.

```
batch (B=512, S=1):                         AllWindowsStatsNet (conv_type "1d"):
  windowed_beamformer (B,1,256,65) f32 ─┐     concat → (B, 68, 256)         [3 stats + 65 θ = 68 ✓ src:786-788]
  all_windows_stats   (B,1,3,256)  f32 ─┘     Conv1d(68→64)×8 → outputs 12
  gains (B,1,2), rx_wavelength_spacing,      ┌ 12 window features
    rx_lo, vehicle_type, sdr_device_type ────┤ + 6 scalars (gains2+spacing1+freq1+veh1+sdr1)
    (each (B,1))                             └ = 18  → FFNN Linear(18→512) ✓ [src:437-454,919-935]
  y_rad,y_phi,craft_y_rad (B,1); empirical (B,1,65)
                                             → {"single": (B,1,65)}  .abs() → L1-norm over dim=2
                                             → mse_loss vs target_from_scatter(y_rad)  (both sum to 1)
```

**Hard contracts verified directly:** `input_channels = 3 + nthetas(65) = 68`
(`single_point_networks.py:786-788`); FFNN `in_features = 6 + 12 = 18` matching `Linear(18,512)`.
The dead config-only nets (2d/cross/transformer/1d2d) are **not** engaged (`conv_type` defaults `"1d"`).

## What is actually back-propagated

Exactly **one term: `single_loss`** = `mse_loss(output["single"], target)` (SSE over 65 bins, both
sides L1-normed). Every auxiliary term (phi, paired, multipaired, tx_pos, direct) is **off** — the
`beamformer` model emits only `"single"`. Four `no_grad` diagnostics are logged but never enter the
backward. `amp:false`+fp32 ⇒ GradScaler/autocast are no-ops. Scheduler = `StepLR(step_size=6,
gamma=0.5)`, stepped per epoch.

## Symmetry handling (verified, and it's deliberate)

The single 2-element model is trained to be ambiguous in exactly the ways the hardware is, via
**two complementary mechanisms that do not double-fold**:

- **Left/right** → data augmentation `flip: true`: 50% of samples negate θ/φ and flip the
  windowed-beamformer/mean-phase consistently (`render_session`, every batch key flipped together;
  the one inconsistent field `abs_signal_and_phase_diff` is unreachable — signal is skipped).
- **Front/back** → baked into the **target**: `target_from_scatter(single=True)` averages the
  target with its `sign(θ)·π − θ` reflection (`train_single_point.py:563-574`), so the supervised
  distribution is bimodal where the array can't disambiguate.

These are different reflections (negate vs π-fold) applied at different stages (input aug vs target),
so they compose rather than cancel. `double_flip:false` for stage 1.

## Debate resolutions (claims I re-verified myself)

| Claim (origin) | Resolution |
|---|---|
| 68-channel & 18-dim contracts (L2) | ✅ **Confirmed** at `single_point_networks.py:786-788` and the FFNN `Linear(18,512)`. |
| B=512 / single trains paired (L1) | ✅ **Confirmed** — `paired = n_radios>1` (`train_single_point.py:83`), n_radios=2. |
| Continuous target is analytic; `scatter_k:21` unused (L3) | ✅ **Confirmed** (`:553-558`); kernel blur is only in the binned path (`:592-595`). Corrected `latest_models.md`. |
| `empirical` in batch despite `empirical_input:false` (L1/L3) | ✅ **Confirmed benign** — used only for `.shape[:2]`/`.device` in `target_from_scatter` (`:543-556`); not a model feature. The `torch.zeros(...)` at `:543` is a dead (discarded) allocation. |
| #34 in-place mutation (L2) | ✅ The `weighted_windows_stats` half does **not** fire (`phase_input:false`); the **`vehicle_type` view-mutation DOES** fire each step (`input_dropout 0.1>0`). Benign here (single consumer) — but L3 must not re-read `batch["vehicle_type"]` after forward (it doesn't). |
| #45 segmentation drops abutting signal (L1) | ✅ Does **not** affect this run — the model consumes the raw windowed-beamformer (delay-and-sum over all 256 windows), never the segmentation mask; #45 only perturbs `mean_phase`/`weighted_*`, which aren't model inputs here. |
| #1 cosine crash / #36 KeyError (L3) | ✅ **Neither applies** — no `scheduler:` key ⇒ default `"step"`; the config sets every flag `global_config_to_keys_used` reads. |
| **F5: val split includes out-of-distribution BladeRF (L1)** | ⚠️ **Overstated / not supported.** The used val split `apr17_val_nosig_noroverbounce.txt` has **0 "blade" filenames** and is **rover (PlutoSDR) missions** — in-distribution for the noblade train. The only train/val filter asymmetry (train `noblade`, val not) appears immaterial because the val data is already all-Pluto. *(Not device-type-checked inside each zarr, but the filename evidence contradicts the original claim.)* |

## Non-blocking caveats worth knowing

- ⚠️ **`scatter_k: 21` is inert** for this run (continuous path) — harmless but misleading in the config.
- ⚠️ **`vehicle_type` batch is mutated in place** every training step (KNOWN_ISSUES #34) — latent, benign here.
- ⚠️ **Reproducibility:** `numpy` RNG is not seeded by `load_seed` (only torch + python `random`);
  any numpy-based randomness in the data path is not pinned by `seed: 10`.
- ⚠️ **Operational:** the run aborts (`sys.exit(1)`) if the output dir
  `checkpoints/jun26_2026/single_3p7_thin_noblade` already exists without `--resume`. (It doesn't yet.)
- ⚠️ **Scale:** `normalize_windowed_beamformer:false` feeds raw f16 power maps scaled only by a
  hardcoded `/500`; the 3 `all_windows_stats` channels are normalized but the 65 beamformer
  channels are not — intended, but a scale-sensitivity surface.
- Doc nit: production segmentation `trim=20.0` (an older doc prose said 10, which is the unit-test value).

## Cross-checked docs
`segmentation.md`, `v5spfdataset.md`, `single_point_networks.md`, `train_single_point.md` verified
accurate (one staleness nit each, captured in the layer reports). `latest_models.md` exists at
`claude_docs/04_training_inference/latest_models.md` (the agents looked under `reference/spf/` and
didn't find it) and was corrected re: `scatter_k`.
