# Adversarial Verification — `train_single_point.md`

Verified against `spf/scripts/train_single_point.py`, `spf/scripts/train_utils.py`,
`spf/model_training_and_inference/models/single_point_networks.py`, and `model_configs/*.yaml`.
Read-only. Default posture: skepticism.

## (a) Priority claims table

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | `load_optimizer` cosine branch is `pass` → `train_scheduler` unbound → UnboundLocalError | **CONFIRMED** | `train_single_point.py:389-393` cosine body is just `pass`; `:397 return optimizer, train_scheduler` with `train_scheduler` only bound in the `step` branch (`:384`). |
| 1b | Reachable via checked-in config `dec3_…_bigger_4x.yaml` with `scheduler: cosine` | **CONFIRMED** (line off by one) | Config exists; `scheduler: cosine` is at **line 75** (doc says `:75` in body but the table/§ also writes `:74-75`/`:75` — actual is line 75; `scheduler_step: 20` is at 74). Grep confirms it is the **only** config setting cosine. |
| 1c | Default scheduler is `step` (`train_utils.py:33`) | **CONFIRMED** | `train_utils.py:33 get_key_or_set_default(config,"optim/scheduler","step")`. |
| 2 | Backward loss = single(1.0)+paired(1.0)+multipaired(1.0)+phi(0.01, output_phi only)+tx_pos(0.1, pred_xy only)+direct(1/30, direct_loss only) | **CONFIRMED** | single `:868/871`; phi gate `:873`, weight `:877` `loss += 0.01 *`; paired `:911/912`; multipaired `:938/939`; tx_pos gate `:952`, `:959 loss += ...*0.1`; direct gate `:961`, `:971-972 loss += ...["multipaired_direct_loss"]/30` w/ `# TODO constant to keep losses balanced`. All weights/gates exact. |
| 3 | Both committed `multipaired*` configs set `direct_loss: False` so /30 never fires | **CONFIRMED** | `multi_16x16_rand_randsnap_lrstep1_a.yaml:81 direct_loss: False`; `multi_16x16_rand_traj_randsnap_lrstep1_b.yaml:85 direct_loss: False`. These are the only two configs naming `multipairedbeamformer`/`trajmultipairedbeamformer`. |
| 4 | `compute_loss`'s `training` param passed everywhere but never read in body | **CONFIRMED** | Param declared `:745`; no reference anywhere in `:746-994`. Passed at `:1018` (val, `training=False`) and `:1317` (train, `training=True`). Dead. |
| 5a | `global_config_to_keys_used` KeyErrors on non-defaulted `beamformer_input`/`phase_input` | **CONFIRMED** | `train_utils.py:95 global_config["beamformer_input"]`, `:104 global_config["phase_input"]` (no `.get`). `load_defaults` (`:24-64`) sets `windowed_beamformer_input` (`:36`) but NOT `beamformer_input`/`phase_input`. A config omitting them KeyErrors. (Real configs set them, e.g. cosine config lines 28/33, so latent not live.) |
| 5b | `load_dataloaders` 4-way split has no final `else` → unbound `train_paths`/`val_paths` for single-`.txt` no-val non-train_on_val | **REFUTED** | There **is** a final `else` at `:158-164` setting both `val_paths` and `train_paths` (holdout split). The cited scenario (one `.txt`, val None, not train_on_val) fails branches (a)`:141`, (b)`:150`, (c)`:154` and falls into the `else`, which binds both. No UnboundLocalError path exists. The doc's headline "no final `else`" is factually wrong. |

## (b) Ordinary-claim spot-check (hit rate)

| Doc claim | Verdict |
|---|---|
| `scaler.scale(loss_d["loss"]).backward()` at `:1323` | HIT |
| `loss_d["loss"]` assembled/returned at `:991` | HIT |
| `load_model` dispatch table lines `:514-523`, bare `raise ValueError` `:525` | HIT |
| `target_from_scatter` asserts `y_rad∈[−π,π]` `:542`, odd `target_ntheta` `:548` | HIT |
| `:543` allocates `torch.zeros` and discards it (dead) | HIT |
| wrap-fold reshape `(B,S,3,n).sum(axis=2)` `:561` | HIT (`effective_n=2*padding+n=3n`) |
| `mse_loss` `:608-609`, `discrete_loss` `:612-614` | HIT |
| `load_config_from_fn` `:1035`, `load_model(...)` `:1123`, `load_dataloaders(...)` `:1165` | HIT |
| `load_seed` seeds torch+random only, not numpy `:45-47` | HIT |
| `scheduler.step()` once per epoch `:1360`; `get_last_lr()` every step `:1333` | HIT |
| `sys.exit(1)` if output dir exists `:1068-1073` | HIT |
| `save_model` copies `empirical_data_fn`→`empirical_dist.pkl` `:444-446` | HIT |
| `load_defaults` sets `output_ntheta=nthetas` `:57-61`, `val_snapshots_adjacent_stride` `:52-56` | HIT |
| `losses["epoch"]` = `step/len(train_dataloader)` `:1337` (vs int in val) | HIT |

Spot-check hit rate: **14/14 ≈ 100%**. Priority-claim accuracy: 6 of 7 dimensions correct; 1 refuted (5b); 1 minor line-number imprecision (1b).

## (c) Missed / misclassified

1. **5b is a false positive** — the doc invented an `UnboundLocalError` gap that does not exist; the `else` at `:158` covers it. Misclassified as a real issue (and surfaced in the Open Questions/TODO as needing a maintainer). Significant: it's a fabricated defect.
2. **Operator-precedence at `:901`** (`if paired and "paired" in output or "multipaired" in output`) parses as `(paired and "paired" in output) or ("multipaired" in output)`. Doc did not flag it. Not a live bug (multipaired models always populate `output["paired"]` — `single_point_networks.py:1140` reads it, inner paired net emits it; verified), but `paired_loss`/`:908 output["paired"]` access for a multipaired model is precedence-dependent. Minor missed observation, correctly benign.
3. **1b line imprecision** — `scheduler: cosine` is line 75 (doc body says `:75`, acceptable; priority prompt's "~:74-75" matches). `scheduler_step: 20` at 74. Trivial.
4. Doc correctly notes `paired_loss` also fires for multipaired models (row 31 note) — consistent with `single_point_networks.py:1140`.

## (d) Grade

**A−**

Rationale: The two headline bugs (cosine UnboundLocalError + its reachable config; non-defaulted-flag KeyError) are real and precisely evidenced; loss-assembly weights/gates and the dead `training` param are exactly right; ordinary spot-check is a clean sweep. The one substantive error is the **fabricated `load_dataloaders` `else` gap (5b)** — a confidently-stated, repeated false positive that a maintainer would waste time chasing. That, plus a missed (benign) precedence subtlety, keeps it just below A.
