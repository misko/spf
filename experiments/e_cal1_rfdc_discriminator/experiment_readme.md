# E-CAL1 — RF-DC vs RF-state discriminator

**Status:** ✅ **arm 1 RUN 2026-08-07 — H₀ upheld, RF-DC injects no resolvable phase**
(+0.069° ± 0.077 against the 2.664° mixer step). Result, deviations and the four
runbook defects found in execution:
[`reports/e_cal1_rfdc_20260807_v1/REPORT.md`](../../spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_rfdc_20260807_v1/REPORT.md).
Arm 2 was **unblocked in code on 2026-08-07** and is ready but unrun — see
[Arm 2](#arm-2--unblocked-2026-08-07-not-yet-run) for what changed and why it is
low priority.
**Bench time (actual):** ~25 min of capture for both radios, plus audits, pilot and validation.
**Queue entry:** [`docs/future_experiments.md` → E-CAL1](../../docs/future_experiments.md)

> **Before re-running this, read §6 of the report.** The step-2 audit command below
> points at `configs/spectroscopy_campaign.yaml`, which pins the *campaign* firmware
> and therefore fails closed against this experiment's v3 firmware; and running that
> audit under `sudo` makes the session directory root-owned, which then breaks the
> capture in step 3.

---

## 1. Purpose

The gain-state phase model attributes phase steps to the AD9361's **RF gain
state** — specifically to the mixer word, which moves `H` by a median 2.664° per
1 dB step against 0.343° for a baseband-LPF-only step.

That attribution is **confounded**. Gain-table byte 2 bit 5 is `RF_DC_CAL`, and
it is set on exactly the rows that begin a new LNA/mixer/TIA state. So "the LMT
words changed" and "the RF-DC correction was re-run" happen together in nearly
every capture, and every claim of the form *"the mixer word moves the phase"*
must currently be read as *"the RF-state transition, **including any RF-DC
correction it triggers**"*.

That hedge propagates into `docs/learnings.md` L10, the source report's §3.3 and
§7 ledger, and the model package README. This experiment removes it.

**What is already known.** The high table has two rows where `RF_DC_CAL` toggles
while the LMT words stay frozen. Row 11 (−3 dB) was sampled by accident inside
the excluded `F_unsupported_negative_gain_attempt_20260730` stage, and bounds an
RF-DC-only step at **≲0.7°** (n=24, median 0.722°) against a **4.364°** LMT step
at the same LOs. But at n=4 rising edges against a ~0.5° per-step floor it cannot
reach a 0.35° decision rule, and **the second, higher-SNR edge has never been
sampled**: 8 and 9 dB appear at no high-band LO in any stage of either campaign.

## 2. Hypothesis

**H₀ (the model's current assumption):** the RF-DC calibration machinery
contributes no resolvable phase of its own. A 1 dB step that toggles `RF_DC_CAL`
with the LMT words frozen will move `H` by **≤ 0.35°** — at or below the
measurement noise floor.

**H₁:** the RF-DC machinery injects phase independently. The same step will move
`H` by an amount comparable to the 2.664° median mixer step, in which case the
model needs an `RF_DC_CAL`-indexed term and every "mixer word" attribution in the
corpus is partly misattributed.

The discriminating row is **row 23 (+9 dB)** of the high table, verified against
the committed audited tables:

```
  dB  row  LNA  MIX  TIA  LPF  RF_DC_CAL
   5   19    0    1    0   12      0        <- reference
   8   22    0    2    0   10      0
   9   23    0    2    0   11      1        <- ONLY RF_DC_CAL and LPF move
  10   24    0    2    0   12      0
```

Across 8 → 9 → 10 dB the LMT words are frozen at `(LNA 0, MIX 2, TIA 0)`. The
only other thing moving is the baseband LPF word, whose contribution is
independently measured at 0.343° — the noise floor. So any large step at 9 dB is
the RF-DC machinery.

## 3. Approach

Additive-cross schedule about a **5 dB reference**, gains {5, 8, 9, 10}, at three
high-band LOs, 25 separately randomized epochs.

Per epoch, compute the model-free symmetric response

```text
H(f, g) = [ D(g, 5) − D(5, g) ] / 2      where D = phase − equal-gain anchor
```

then take the across-epoch mean and standard error of `|ΔH|` for the 8→9 and
9→10 steps. Compare against the **LPF-only floor derived from this same
dataset** — not imported from the campaign, which was a different session.

**Why 5 dB as the reference:** it sits below the whole {8,9,10} triple, so every
measured step is an increase from the reference and no cell crosses an LNA or
mixer boundary.

**Why 25 epochs:** the source documents disagree on the per-step standard error
at 3 epochs — the report's §3.3 gives 0.355–0.368°, the queue entry gives
0.54–0.81° with no derivation. 25 epochs reaches the 0.35° gate under either
reading (0.124° optimistic, 0.280° pessimistic) and costs 8 minutes, so there is
no reason to economise. **Reconcile that discrepancy when writing up.**

## 4. Hardware setup

### 4.1 Radios

| | |
|---|---|
| Count | **2 × PlutoSDR (Pluto+)** |
| R17 | `104000bac4950008230026001b440a003a` — historical `.17` |
| R18 | `1040007c4a94000211000b009186843ef2` — historical `.18` |
| Role | both measured identically; there is **no** treated/control split in this experiment |
| Provisioning | 2R2T, direct-USB gain/RSSI firmware, RAM-loaded, QSPI boot mode |
| Firmware release | `v0.38-plutoplus-spf-gain-rssi-fingerprint-v3` |
| Image SHA-256 | `86f2115eb344efcbd3d59af02caf80d396291cb9e20dcb01651cacf7e0334191` |
| TX | only one radio's TX2 is active at a time — enforced by the runner |

### 4.2 Physical schematic

The loopback fixture is **independent for each radio**. Two identical chains, no
RF path between the radios:

```text
+------------------ PLUTO A : R17  (104000bac495...440a003a) -------------------+
|                                                                               |
|   TX2 o---[a]--->  [ 30 dB attenuator ]  ---[b]--->  [ two-way splitter ]      |
|                                                          |         |          |
|                                                         [c]       [d]         |
|                                                          |         |          |
|   RX1 o<-------------------------------------------------+         |          |
|   RX2 o<-----------------------------------------------------------+          |
|                                                                               |
|   USB o=== direct-USB control + RX DMA ===> host                              |
+-------------------------------------------------------------------------------+

+------------------ PLUTO B : R18  (1040007c4a94...86843ef2) -------------------+
|                                                                               |
|   TX2 o---[e]--->  [ 30 dB attenuator ]  ---[f]--->  [ two-way splitter ]      |
|                                                          |         |          |
|                                                         [g]       [h]         |
|                                                          |         |          |
|   RX1 o<-------------------------------------------------+         |          |
|   RX2 o<-----------------------------------------------------------+          |
|                                                                               |
|   USB o=== direct-USB control + RX DMA ===> host                              |
+-------------------------------------------------------------------------------+

            NEVER connect TX2 directly to an RX input.
            The 30 dB attenuator is mandatory on every fixture.
```

The measured quantity is `angle(RX1) − angle(RX2)` **within one radio**, which is
why each radio needs its own splitter and its own source. The two radios are
independent measurements of the same universal effect.

### 4.3 Passive parts and adapters

Two identical sets, one per radio. Record the actual parts used — the campaign
convention is that anything not written down did not happen.

| Ref | Item | Requirement | Part / serial (record before run) |
|---|---|---|---|
| — | Attenuator | 30 dB, DC–6 GHz, SMA | |
| — | Splitter | two-way, DC–6 GHz, SMA, matched output arms | |
| a, e | Cable TX2 → attenuator | SMA, phase-stable, ≤6 GHz | |
| b, f | Cable/adapter attenuator → splitter | SMA, keep short | |
| c, g | Cable splitter → RX1 | SMA, **matched pair with d/h** | |
| d, h | Cable splitter → RX2 | SMA, **matched pair with c/g** | |
| — | USB cable | direct-USB, per radio | |

**Connector discipline.** This must be a single unchanged-harness session. Do
not re-mate any RX connector between the pre-run audit and the post-run audit.
Record torque if a torque wrench is used. The A→D result in the source campaign
showed a connector re-mate moving the >4 GHz band by 12–34°, which is far larger
than the effect being measured here.

## 5. Software setup

| | |
|---|---|
| Repo | `/home/pi/spf` on the capture host |
| Environment | `/home/pi/spf-virtualenv` |
| Config | [`spf/calibrations/dual_rx_gain_frequency/configs/e_cal1_rfdc_discriminator.yaml`](../../spf/calibrations/dual_rx_gain_frequency/configs/e_cal1_rfdc_discriminator.yaml) |
| Sample rate | 30 MS/s, 3 MHz RF bandwidth, 65536-sample buffers |
| Tone | 100 kHz offset, FPGA DDS on TX2 (TX DMA never competes with direct-USB RX) |
| Schedule | additive cross, 7 unique gain pairs, 3 LOs, 25 epochs = **525 frames per radio** |

The repository must be **clean** when capturing — the dataset records the Git SHA
and whether the checkout was dirty.

### 5.1 Commands

```bash
cd /home/pi/spf && source /home/pi/spf-virtualenv/bin/activate
```

**Step 1 — pilot.** Copy the config with `repetitions: 1` and run it first.
This is a low-gain experiment and the tone must still clear `min-tone-dbfs` at
5 dB RX gain:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config /tmp/e_cal1_pilot.yaml \
  --output artifacts/dual_rx_gain_frequency/e_cal1_pilot_SESSION
```

**Step 2 — gain-table audit, before the run.** The whole experiment is keyed on
row 23 of the high table; verify it is the table you think it is:

```bash
sudo -E /home/pi/spf-virtualenv/bin/python \
  -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  audit --output artifacts/dual_rx_gain_frequency/e_cal1_SESSION/gain_table_audit.json
```

Expected high-table byte SHA-256:
`90d34d61e8612277529dccfc3323f6c684c2bc36b7670dff078e009eb84a1143`

**Step 3 — the run** (~8 min per radio):

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config spf/calibrations/dual_rx_gain_frequency/configs/e_cal1_rfdc_discriminator.yaml \
  --output artifacts/dual_rx_gain_frequency/e_cal1_SESSION
```

**Step 4 — gain-table audit, after the run**, to `gain_table_audit_final.json`.
Tables must be byte-identical to step 2.

**Step 5 — validate** each dataset from stored IQ, without `--no-recompute-iq`.

## 6. Outputs

Everything below must exist before the experiment is considered complete.

### 6.1 Raw capture (gitignored)

```
artifacts/dual_rx_gain_frequency/e_cal1_SESSION/
├── gain_table_audit.json           # pre-run, both radios, 3 tables each
├── gain_table_audit_final.json     # post-run, must match byte-for-byte
├── <serial_R17>/calibration.v7.zarr
├── <serial_R18>/calibration.v7.zarr
└── <serial>/validation.json        # strict validation, per radio
```

**Acceptance gates on the capture:**

| Gate | Requirement |
|---|---|
| Completeness | 525/525 scheduled frames per radio |
| Quality | ≥ 20 of 25 epochs quality-valid per cell (`min-quality-valid-per-cell: 20`) |
| Tone level | every cell clears `min-tone-dbfs: -75`, `max-clipping-fraction: 0` |
| Gain tables | pre-run and post-run audits byte-identical, and matching the committed hashes |
| Harness | no connector operation between the two audits |
| Provenance | firmware release, image SHA, firmware SHA, gadget SHA, SPF Git SHA + clean flag all recorded in V7 |

### 6.2 Committed analysis

```
spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_rfdc_20260808_v1/
├── REPORT.md
├── results.json
├── inputs_manifest.json            # SHA-256 of every input and output
└── figures/*.png                   # optional
```

`results.json` must contain, per radio and pooled:

| Field | Meaning |
|---|---|
| `rfdc_step_deg` | median and mean \|ΔH\| for the 8→9 dB step (the `RF_DC_CAL` rising edge) |
| `rfdc_step_sem_deg` | standard error across the 25 epochs — **the headline number** |
| `lpf_only_floor_deg` | \|ΔH\| for steps where only the LPF word moves, **from this dataset** |
| `n_epochs_valid` | per step, after quality masking |
| `lo_hz` | per-LO breakdown, all three |
| `cluster_ci` | cluster-bootstrap 95% CI over (radio, LO) clusters |
| `mann_whitney_p` | RF-DC-only vs LPF-only |

`REPORT.md` must state the **sem beside every estimate** so the power is
auditable, and must state explicitly whether the 0.35° gate was reached.

### 6.3 Downstream updates the result requires

- `docs/learnings.md` — update L10's RF-DC paragraph in the **same change**.
- `spf/calibrations/dual_rx_gain_frequency/reports/gain_state_phase_model_20260802_v1/REPORT.md`
  — §3.3, §6.2 and the §7 ledger row all carry the hedge; a pointer to the new
  report, not a rewrite of the committed result.
- `spf/calibrations/gain_state_phase_model_v1/README.md` §3.6.
- `docs/future_experiments.md` — mark E-CAL1 completed with the outcome.

## 6bis. RESULTS — arm 1, run 2026-08-07

Session `e_cal1_20260807`, SPF `3fe21e7` (clean). Full report, code and input
hashes:
[`reports/e_cal1_rfdc_20260807_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_rfdc_20260807_v1/REPORT.md).

**H₀ upheld — the RF-DC machinery injects no resolvable phase.**

| Quantity | Result |
|---|---|
| RF-DC contribution at row 23 | **+0.069° ± 0.077** (sem) |
| Cluster-robust 95% CI | **[−0.168°, +0.392°]** over 6 (radio, LO) clusters |
| Compared against | **2.664°** median mixer step → ≈7× below H₁ at the CI's upper edge |
| Decision-rule branch reached | **"≤ 0.35° with sem < 0.35°"** — no resolvable phase |
| Frames | 1050/1050 (525 per radio), 21/21 cells both radios |

The estimator is the second difference `H(9) − mid[H(8), H(10)]`. Because the LPF
word steps linearly 10 → 11 → 12 across 8 → 9 → 10 while the LMT words stay
frozen, this is algebraically the pre-registered "8→9 step against the LPF-only
floor from this dataset" comparison — formed pairwise inside each epoch, so the
per-cell common-mode noise cancels instead of being carried. The unpaired form
agrees: +0.147° ± 0.135 against the paired +0.118° ± 0.094 (all cells).

**The cleanest evidence needs no modelling.** 8→9 and 9→10 are both 1 dB steps
with the LMT words frozen, so they carry identical noise. The step that *raises*
`RF_DC_CAL` (median |ΔH| **0.320°**) is *smaller* than the one that *lowers* it
(**0.446°**). A flag injecting phase cannot produce that ordering.

**All gates passed:** tables byte-identical pre/post (high = `90d34d61…`, and
identical under the new v3 firmware), strict validation from stored IQ, harness
unchanged, provenance recorded with `dirty = False`.

**Do not quote `mann_whitney_rfdc_vs_lpf` (p = 0.010) as an effect.** It compares
a 1 dB step against a *halved* 2 dB step, which shrinks the LPF noise by two
(std 1.786° raw → 0.893° halved, against 1.337° for 8→9). It detects a difference
in spread, not location. §4.2 of the report explains it.

**One cell excluded, on a pre-registered basis.** R18 @ 5100 MHz kept only 15 of
25 epochs and had std 2.76° — the cell the step-1 pilot had already flagged as
weakest. It fails `min-quality-valid-per-cell: 20`. It is not an RF-DC effect:
the LPF-only step is elevated in lockstep there (0.914°), so the excursion tracks
cell SNR, not the flag.

**Four runbook defects were found in execution** — see §6 of the report. Two of
them will block anyone who re-runs this; they are summarised in the warning at
the top of this file.

## 7. Decision rule

Pre-registered. Do not renegotiate after seeing the data.
**Outcome: row 1 — see [§6bis](#6bis-results--arm-1-run-2026-08-07).**

| Measured 8→9 dB \|ΔH\| | Conclusion |
|---|---|
| **≤ 0.35°** with sem < 0.35° | The RF-DC machinery contributes no resolvable phase. The attribution closes to the LNA/mixer/TIA network. Delete the hedge everywhere. |
| **comparable to 2.664°** | The RF-DC machinery injects phase on its own. The model needs an `RF_DC_CAL`-indexed term, and existing mixer-step magnitudes are partly misattributed. |
| **between**, or sem ≥ 0.35° | Inconclusive. Report the sem and the achieved power; do not claim either branch. |

The comparison must be against the **LPF-only floor from this dataset**. Deriving
the floor from the campaign would import a different session, different harness
state and different dates.

## 8. Risks

| Risk | Why it matters | Check |
|---|---|---|
| **Low-gain SNR** | Runs at 5–10 dB RX where every prior campaign used 26 dB. The `F_neg` stage failed outright in low/middle bands for this class of reason. | The step-1 pilot. Inspect tone dBfs and SNR at the 5 dB cells before committing 25 epochs. |
| **RF-DC re-runs mid-schedule** | `rf-dc-calibration-policy` is `before_each_frequency_block`, so an RF-DC calibration fires at every frequency block regardless of gain. | This is uniform across all cells and cancels in `ΔH` between adjacent gains, but state it in the report rather than assuming it away. |
| **Gain-table drift** | The entire design depends on row 23 having `RF_DC_CAL = 1` and frozen LMT words. | Pre- and post-run audits, compared byte-for-byte against the committed hashes. |
| **Connector movement** | A re-mate can move the >4 GHz band by 12–34°, dwarfing the ~0.35° effect. | Single unchanged-harness session; no operations between audits. |
| **Assuming the LPF is silent** | The 8→9 step also moves the LPF word. | The LPF-only floor is measured in the same dataset and subtracted, not assumed. |

## Arm 2 — unblocked 2026-08-07, not yet run

**The code blocker is gone.** It was: `rf_dc_offset_tracking_en` was read-only
(`dc_offset.py` only snapshotted it) and `rf_dc_calibration_policy` was
hard-restricted to `before_each_frequency_block` by `config.py`. Both are fixed:

| Piece | Where |
|---|---|
| `rf_dc_offset_tracking_en` config knob, tri-state (`null` = leave driver default) | `config.py` |
| `never` added to `RF_DC_CALIBRATION_POLICIES` — still fail-closed on anything else | `config.py` |
| `apply_rf_dc_offset_tracking()` — writes both channels, **reads back, raises on mismatch**; re-asserted after `run_rf_dc_calibration()` and `configure_frequency()` | `hardware.py` |
| `rf_dc_offset_tracking_en_requested` / `_observed` recorded in V7 | `dataset.py`, `runner.py` |
| Arm-2 config, arm-1's schedule byte-for-byte | `configs/e_cal1_arm2_rfdc_tracking.yaml` |
| 12 unit tests | `tests/test_dual_rx_gain_frequency_rf_dc_tracking.py` |
| 4 hardware tests, `--radio-hardware --radio-rf-dc-tracking` | `tests/radio_hardware/test_rf_dc_tracking_hardware.py` |

The readback is the load-bearing part. The AD9361 driver can accept an attribute
write without applying it, and it re-asserts tracking across `calib_mode` writes
and LO retunes. An unverified write would let arm 2 report "disabling tracking
changed nothing" when tracking was never disabled — **a false null
indistinguishable from a real one**, on precisely the question arm 2 exists to
answer. The hardware tests were run against both radios on 2026-08-07 and pass:
the write reaches silicon on this firmware and survives both hazards.

Adding the knob does **not** change any existing config's run signature — a
config that leaves it unset is omitted from `as_json()`, verified byte-identical
against unmodified HEAD, so every pre-existing dataset still resumes.

**Before running arm 2, read this.** Arm 1 measured the *total* RF-DC
contribution at +0.069° ± 0.077. Arm 2 partitions a quantity already
indistinguishable from zero, so it can essentially only return "also zero", and
it **cannot** discriminate between "the tracking loop is quiet" and "this harness
cannot see RF-DC effects at all". Only a **positive control** — inject a known
perturbation, confirm the pipeline recovers it at the expected magnitude — can
close that. Sequence the positive control first; run arm 2 to close the
mechanism for its own sake, not for the primary question.

Also still unsampled: the row-11 (−3 dB) edge, which still rests on the ≲0.7°
`F_neg` bound.
