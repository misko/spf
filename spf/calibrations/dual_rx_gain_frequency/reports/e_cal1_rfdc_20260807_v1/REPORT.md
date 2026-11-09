# E-CAL1 arm 1 — the RF-DC machinery injects no resolvable phase

**Session:** `e_cal1_20260807` · captured 2026-08-07 · SPF `3fe21e7`, clean checkout
**Design:** [`experiments/e_cal1_rfdc_discriminator/experiment_readme.md`](../../../../../experiments/e_cal1_rfdc_discriminator/experiment_readme.md)
**Raw capture:** `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_cal1_20260807/` (gitignored)

---

## 1. Result

**H₀ holds. The pre-registered "no resolvable phase" branch is reached.**

The `RF_DC_CAL` flag on high-table row 23, measured with the LMT words frozen and
the baseband-LPF ramp differenced out, moves `H` by

> **+0.069° ± 0.077 (sem)**, cluster-robust 95% CI **[−0.168°, +0.392°]**

against the **2.664°** median mixer step that motivated the question. That is a
factor of **≈ 7 below** the H₁ prediction even at the CI's upper edge, and
statistically indistinguishable from zero (t = 0.90).

**The hedge can be deleted.** Finding 2 of `docs/learnings.md` L10 no longer has to
be read as "the RF-state transition, *including any RF-DC correction it triggers*".
The attribution closes to the LNA/mixer/TIA network.

| Decision rule (pre-registered) | Measured | Reached? |
|---|---|---|
| ≤ 0.35° with sem < 0.35° → no resolvable phase | 0.069°, sem 0.077° | **yes** |
| comparable to 2.664° → RF-DC injects phase | — | no |
| between, or sem ≥ 0.35° → inconclusive | — | no |

## 2. What was measured

Additive cross about a 5 dB reference, gains {5, 8, 9, 10}, three high-band LOs
(4001 / 5100 / 5766 MHz), 25 separately randomized epochs, two radios —
**525/525 frames per radio, 1050 total**.

The discriminating structure, **verified on this hardware in this session** (both
radios, pre- and post-run):

```
  dB  row  LNA  MIX  TIA  LPF  RF_DC_CAL   digital gain
   5   19    0    1    0   12      0             0        <- reference (mixer differs)
   8   22    0    2    0   10      0             0
   9   23    0    2    0   11      1             0        <- the discriminating edge
  10   24    0    2    0   12      0             0
```

Across 8 → 9 → 10 the LMT words are frozen at (LNA 0, MIX 2, TIA 0) and the LPF
word steps linearly 10 → 11 → 12. Only `RF_DC_CAL` breaks that linearity.

## 3. Method

Per epoch and LO, model-free, using the campaign's own convention
(`spectroscopy_analysis.py`):

```text
D(g1, g2) = phase(g1, g2) − phase(5, 5)        [same epoch, same LO]
H(g)      = wrap( [D(g, 5) − D(5, g)] / 2 )
```

The pre-registration compares the 8→9 step against an LPF-only floor measured in
this same dataset. Both are available here — 8→9 is a 1 dB step with the LMT
frozen, and 8→10 is a 2 dB **LPF-only** step (`RF_DC_CAL` 0→0) — so the floor is
measured, never imported from the campaign.

That contrast is formed **pairwise within each epoch** as a second difference:

```text
excess = (H9 − H8) − (H10 − H8)/2 = H9 − mid[H8, H10]
```

which is algebraically the pre-registered comparison, with the LPF ramp cancelled
exactly and the per-cell common-mode noise cancelled rather than carried. Under H₀
it is zero; under H₁ it is the full RF-DC injection. The unpaired form of the same
contrast agrees: **+0.147° ± 0.135** vs the paired **+0.118° ± 0.094** (all cells).

## 4. Results

RF-DC excess `H9 − mid[H8, H10]`, degrees, per (radio, LO) cell:

| Radio | LO (MHz) | n epochs | signed mean | sem | median \|·\| |
|---|---|---|---|---|---|
| R17 `0a003a` | 4001 | 25 | +0.134 | 0.036 | 0.125 |
| R17 `0a003a` | 5100 | 25 | +0.020 | 0.055 | 0.234 |
| R17 `0a003a` | 5766 | 25 | −0.438 | 0.161 | 0.683 |
| R18 `843ef2` | 4001 | 25 | +0.015 | 0.054 | 0.109 |
| R18 `843ef2` | 5100 | **15** | +0.528 | 0.608 | 1.667 |
| R18 `843ef2` | 5766 | 25 | +0.613 | 0.312 | 1.318 |

| Estimate | n | signed mean | sem |
|---|---|---|---|
| All cells | 140 | +0.118° | 0.094° |
| **Quality-restricted** (cells with ≥ 20 valid epochs) | 125 | **+0.069°** | **0.077°** |

Cluster-robust 95% CI over the six (radio, LO) clusters: **[−0.168°, +0.392°]**.

**The raw steps, for the record** (pooled, signed mean ± sem):

| Step | Words moving | signed mean | sem | median \|ΔH\| |
|---|---|---|---|---|
| 8→9 | LPF +1, `RF_DC_CAL` 0→1 | +0.343° | 0.112° | 0.320° |
| 9→10 | LPF +1, `RF_DC_CAL` 1→0 | +0.098° | 0.136° | 0.446° |
| 8→10 (per 1 dB) | LPF +2 only | +0.197° | 0.074° | 0.229° |

The two like-for-like 1 dB steps settle it without any modelling: the step that
**raises** `RF_DC_CAL` (median 0.320°) is *smaller* than the step that **lowers**
it (0.446°). A flag that injected phase could not produce that ordering.

### 4.1 One cell is noise-dominated, and it is not an RF-DC effect

R18 @ 5100 MHz retains only 15 of 25 epochs for this statistic and has std 2.76°.
That is the same cell the pilot flagged as weakest (rx1 tone −50.8 dBFS, SNR
7.6 dB — the lowest in the pilot matrix; the full run reaches −54.24 dBFS there,
still 21 dB clear of the −75 gate, so this is a *relative* SNR deficit, not a
gate failure). It is excluded by the **pre-registered**
`min-quality-valid-per-cell: 20` rule, not post hoc.

Decisively, the elevation there is not `RF_DC_CAL`-specific: in that same cell the
**LPF-only** step is elevated in lockstep (0.914°, against 0.100° and −0.013° in the
two clean R17 cells). Wherever the RF-DC step looks large, the LPF-only step is large
too — the excursions track cell SNR, not the flag.

### 4.2 The Mann-Whitney p = 0.010 is an artifact — do not report it as an effect

`results.json` carries `mann_whitney_rfdc_vs_lpf` p = 0.0100 comparing |8→9| against
|8→10 per dB|. **This is a normalization artifact.** Dividing the 2 dB LPF step by
two shrinks its noise by two: std 1.786° raw → 0.893° halved, against 1.337° for
8→9. The LPF distribution is artificially tightened, so the test detects a
difference in spread, not in location. The raw 8→10 step is in fact *noisier* than
8→9. The paired second difference is the correct test, and it is null.

## 5. Acceptance gates

| Gate | Requirement | Result |
|---|---|---|
| Completeness | 525/525 frames per radio | **pass** — 1050/1050 |
| Quality | ≥ 20 of 25 epochs valid per cell | **pass** — 21/21 cells both radios |
| Tone level | ≥ −75 dBFS, clipping 0 | **pass** — run range −54.24 … −26.67 dBFS, clipping 0 |
| Gain tables | pre/post audits identical, match committed hashes | **pass** — all 6 tables byte-identical, high = `90d34d61…` |
| Harness | no connector operation between audits | **pass** — unchanged throughout |
| Provenance | firmware, image SHA, git SHA + clean flag in V7 | **pass** — `3fe21e7`, dirty = False |
| Validation | strict, recomputed from stored IQ | **pass** — both radios |

Frame-level quality: R17 525/525 valid; R18 513/525 (10 `cross_channel_coherence_low`,
8 `rx1_tone_snr_low`, 7 `within_capture_phase_unstable`; counts overlap per frame),
all cells still clearing the ≥ 20-epoch gate.

Note the per-cell gate is evaluated on single cells, while a per-epoch `H` needs the
forward cell, the reverse cell and the anchor all valid in the *same* epoch. That
intersection is why cells show 15–25 usable epochs rather than 20–25.

## 6. Deviations from the runbook

Four, all recorded here rather than silently absorbed:

1. **The audit manifest had to be re-pinned.** `experiment_readme.md` step 2 runs the
   audit against `configs/spectroscopy_campaign.yaml`, whose `gain-table-audit` block
   pins the *campaign* firmware (`7b7fb140…`). E-CAL1 runs on
   `v0.38-…-v3` (`f53dd006…`), so that audit fails closed on a firmware mismatch.
   `e_cal1_gain_table_audit.yaml` here is that manifest with only the three firmware
   fields re-pinned and the base-config path absolutised — **every band expectation,
   including the `90d34d61…` high-table hash, is unchanged**. The tables hash
   identically under the new firmware, which is a stronger result than the runbook
   asked for: the gain tables survived the firmware change untouched.
2. **The audit runs as root and poisons the session directory.** `sudo -E … audit
   --output <session>/gain_table_audit.json` creates the session directory owned by
   root, and the subsequent non-root `run` then dies with `PermissionError`. The
   directory needs `chown` between the two steps, or the audit needs to write
   elsewhere. Ordering the run before the pre-audit would break the
   no-operations-between-audits rule, so `chown` is the fix.
3. **The readiness manifest was stale.** The radios had been re-cabled for this
   fixture and re-enumerated (USB address 59/60 → 61/67), so the V7 fingerprint gate
   refused to write. `/home/pi/device_mapping` and `/run/spf/direct_usb_ready.json`
   were regenerated; the radios' stable fingerprints are identical to the previous
   manifest, so these are the same physical units, re-blessed at the current attachment.
4. **The pilot needs `min-quality-valid-per-cell` relaxed.** `repetitions: 1` fails
   config validation against the gate of 20 ("minimum valid count must fit within the
   epochs"). `e_cal1_pilot.yaml` here sets it to 1; it is a go/no-go SNR probe, not a
   gated dataset.

Item 4's pilot is worth keeping in the runbook regardless: it correctly predicted
which cell would go bad.

## 7. What this does not settle

- **Arm 2 was blocked when this ran, and remains unrun.** Arm 1 shows the RF-DC
  machinery injects no resolvable phase *as it is configured here*. It does not
  identify which part of that machinery is quiet, which is what arm 2's
  `rf_dc_offset_tracking_en = 0` A/B would answer. At capture time that attribute
  was read-only in `dc_offset.py`, and `rf_dc_calibration_policy` was
  hard-restricted to `before_each_frequency_block` by `config.py`.

  > **Addendum, later on 2026-08-07:** that code blocker has since been removed —
  > `rf_dc_offset_tracking_en` is now a tri-state config knob with a
  > readback-verified write path, `never` joins the calibration-policy enum, both
  > requested and observed states are recorded in V7, and
  > `configs/e_cal1_arm2_rfdc_tracking.yaml` carries arm 1's schedule
  > byte-for-byte. Hardware tests confirm the write reaches silicon on this
  > firmware. **The measurement reported here is unaffected**: it was captured
  > before that change, with the tracking loop at the driver default, and its run
  > signature is unchanged by the new field (a config that leaves the knob unset
  > omits it from `as_json()`). Arm 2 is still unrun, and the case for running a
  > positive control ahead of it stands.
- **`rf-dc-calibration-policy: before_each_frequency_block` was active throughout**,
  so an RF-DC calibration fired at every frequency block regardless of gain. It is
  uniform across the {8, 9, 10} cells within a block and cancels in the second
  difference — but it is a property of the schedule, stated rather than assumed away.
- **Two radios, one harness topology, high band only.** Row 11 (−3 dB) — the other
  `RF_DC_CAL` edge with frozen LMT words — was not resampled; the ≲0.7° bound from
  the excluded `F_neg` stage still stands as the only evidence there, and this result
  is consistent with it.
- **Demonstrated sensitivity — CLOSED 2026-08-07 by E-CAL5**
  (`../e_cal5_positive_control_20260807_v1/`). This report's null rested on a noise
  floor rather than on a demonstration that the chain can see an effect of the size H₁
  predicted. E-CAL5 measured a known 1 dB MIXER step on the same harness with this same
  estimator at **7.434° ± 0.097** against a **0.440°/dB** floor — **16.9×**, resolved in
  all six (radio, LO) cells. Against this report's sem of 0.077°, an H₁-sized 2.664°
  effect would have appeared at **34.5σ**. **The null above is informative about the
  physics: the RF-DC machinery is quiet, not invisible to this measurement.**
- **The 25-epoch power question is settled empirically.** The source documents
  disagreed (§3.3 of the 2026-08-02 report: 0.355–0.368°; `future_experiments.md`
  E-CAL1: 0.54–0.81°). Measured here: per-cell sem on the second difference is
  0.036–0.312° in the five well-conditioned cells — the optimistic reading was right,
  and 25 epochs was more than enough.

## 8. Reproduce

```bash
python analyze.py /mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_cal1_20260807 results.json
```

`inputs_manifest.json` carries SHA-256 for every input and output. Dataset hashes use
the campaign's `scalar_input_sha256` (hash of the decoded scalar arrays) because
`data.mdb` is a 128 GiB sparse LMDB map whose raw bytes are not a meaningful digest.
