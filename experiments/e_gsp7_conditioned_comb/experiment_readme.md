# E-GSP7 — does a conditioning-chosen sparse comb calibrate?

**Status:** pre-registered 2026-08-07, capture pending in the same session.
**Est. bench time:** ~80–120 min for both radios (3,330 frames).
**Queue entry:** derived from the E-GSC decision ledger row *"Is a sparse protocol
now recommendable? **Not yet** — it needs a prospective sparse capture with the
comb chosen by conditioning"*
([`gain_state_computational_20260807_v1`](../../spf/calibrations/dual_rx_gain_frequency/reports/gain_state_computational_20260807_v1/REPORT.md) §3.4).

**The combs below were chosen before any of this data existed.** That is the whole
point; see [§7](#7-pre-registration-integrity).

---

## 1. Purpose

Calibration cost is the project's main operational lever: the committed model was
fitted from a 111–113 LO comb, and a ten-LO protocol would make per-session,
per-harness calibration practical.

E-CAL3 tested that prospectively and it **failed**: a fresh `L26` refit from ten
uniformly spaced LOs scored **11.61° MAE** against a **9.06°** anchor-only
baseline — worse than no model. E-GSC3 then diagnosed *why*, and the diagnosis was
not "ten points is too few":

> The uniform 600 MHz spacing **aliased the two ripple delays onto each other**.
> `Δ(τ₁−τ₂) = 0.984` cycles; condition number **17.92** against a **2.35** median
> over random 10-LO combs — worse-conditioned than **1,999 of 2,000** of them.

E-GSC also showed `N* = 16` with free delays and `N* = 8` with delays frozen, and
that frozen-delay fitting recovers 73.4% of the dense improvement at `N = 10`.
**All of that is retrospective subsampling of an already-dense capture.** Nothing
has ever captured a sparse comb prospectively and calibrated from it.

## 2. Hypothesis

**H₁ (the E-GSC diagnosis is right and actionable):** a ten-LO comb *chosen by
conditioning* will calibrate successfully — beating the anchor-only baseline
decisively — while a ten-LO *uniform* comb captured in the same session, on the
same harness, in the same hours, will fail as E-CAL3 did.

**H₀:** ten LOs is simply too few regardless of placement, and both fail. The
conditioning diagnosis would then be a correct description of E-CAL3's design
defect but not a route to sparse calibration.

The conditioning numbers make this sharp. Over the 111-LO candidate pool
(400–5900 MHz at 50 MHz), scored with the committed `ripple_conditioning` at the
fleet delays `τ = (2.56, 0.92) ns`:

| Comb | N | Condition number |
|---|---|---|
| **Chosen by conditioning** (pre-registered below) | 10 | **1.0899** |
| **The actual E-CAL3 comb** (primary control) | 10 | **17.9208** |
| A `linspace` uniform comb (secondary control) | 10 | 21.7782 |
| Random, median of 2000 draws | 10 | 2.3218 |
| Random, best of 2000 draws | 10 | 1.2524 |
| **Dense reference (all 111 LOs)** | 111 | **1.0308** |
| Chosen by conditioning | 16 | 1.0463 |
| Uniform | 16 | 1.1332 |

Two things follow, and both shape the design:

1. The chosen 10-LO comb sits within **6%** of the *full 111-LO* comb, and beat
   **all 2000** random draws. The uniform 10-LO comb is **20× worse**.
2. **At N = 16 the uniform comb is already well-conditioned** (1.13), so comb
   choice barely matters there. **The decisive test is at N = 10** — exactly the
   point count at which E-CAL3 failed.

## 3. Pre-registered combs

Chosen by `select_comb.py` (committed here) minimising `ripple_conditioning`,
constrained to ≥3 LOs per AD9361 gain-table band, seed `20260807`. Exact values in
[`comb_selection.json`](comb_selection.json).

**Primary — conditioning-chosen, N = 10 (MHz):**

```
900, 1050, 1200, 1350, 1750, 3050, 3800, 4500, 5200, 5900
```

**Control — the E-CAL3 comb, N = 10 (MHz)**, captured in the same session so the
comparison is controlled:

```
400, 1000, 1600, 2200, 2800, 3400, 4100, 4700, 5300, 5900
```

> **Pre-registration amendment, 2026-08-07, made before any data existed** (capture
> still running, no analysis performed). The control above is now the **actual**
> E-CAL3 comb, taken from `gsc_common.PREREG_10_MHZ`, replacing a `linspace`
> reconstruction of it that I had written here first
> (`400, 1000, 1600, 2250, 2850, 3450, 4050, 4700, 5300, 5900`).
>
> The real comb scores **17.9208**, reproducing the E-GSC report's quoted 17.92
> exactly and confirming this `ripple_conditioning` implementation matches theirs.
> My reconstruction scored 21.7782 — i.e. *worse* conditioned — so this amendment
> makes the control **more faithful and harder to beat**, not easier. Both combs
> lie inside the captured 111-LO grid and both will be scored; the real E-CAL3
> comb is the primary control.

**Secondary — conditioning-chosen, N = 16 (MHz):**

```
750, 800, 1050, 1600, 2600, 2650, 2850, 2900, 3250, 3450, 3550, 3650, 3700, 4150, 4450, 5500
```

## 4. Approach

Capture the **dense 111-LO stage-A design** in one unchanged-harness session, then
fit only on the pre-registered sub-combs and score on the complement.

Capturing densely and fitting sparsely is deliberate: it makes the training comb
the *only* difference between the primary and control arms, so the comparison is
not confounded by session, harness, temperature or time-of-day — which is exactly
what E-CAL3 could not control for. **What it does not test is cross-session
transfer**; see [§8](#8-risks-and-what-this-cannot-settle).

Per comb: fit `L26`, then score on the ~101 LOs not used for that fit, against

- the **anchor-only baseline** on the same held-out LOs (the number to beat), and
- the **committed `L26` coefficients** scored with no refit, which measures
  cross-session transfer on this fresh capture and is directly comparable to the
  4.79–4.80° prospective figure.

Also fit with the ripple **delays frozen** at fleet values, since E-GSC puts
`N* = 8` in that regime and predicts 73.4% recovery at N = 10.

## 5. Hardware setup

**Unchanged from the E-CAL1 arm 1 / arm 2 session — no connector has been touched.**
That continuity is a deliberate asset: an A→D connector re-mate moved the >4 GHz
band by 12–34° in the source campaign, far more than the effects here.

| | |
|---|---|
| Count | 2 × PlutoSDR (Pluto+) |
| R17 | `104000bac4950008230026001b440a003a` |
| R18 | `1040007c4a94000211000b009186843ef2` |
| Firmware | `v0.38-plutoplus-spf-gain-rssi-fingerprint-v3`, QSPI, 2R2T, direct-USB |

```text
+---------------- PLUTO A : R17 ----------------+   (identical, independent chain for R18)
|  TX2 o--->[ 30 dB attenuator ]--->[ splitter ]  |
|                                     |      |    |
|  RX1 o<-----------------------------+      |    |
|  RX2 o<------------------------------------+    |
|  USB o=== direct-USB control + RX DMA ===> host |
+-------------------------------------------------+
        NEVER connect TX2 directly to an RX input.
```

## 6. Software setup

| | |
|---|---|
| Config | [`configs/e_gsp7_conditioned_comb.yaml`](../../spf/calibrations/dual_rx_gain_frequency/configs/e_gsp7_conditioned_comb.yaml) |
| Design | stage-A: additive cross about 26 dB, gains {5, 26, 45}, 3 epochs |
| Frequencies | 400–5900 MHz at 50 MHz — **111 LOs** |
| Volume | 5 pairs × 111 LOs × 3 epochs = **1,665 frames per radio; 3,330 total** |

The gain set, reference gain, epoch count and acquisition contract are copied from
`spectroscopy_campaign_base.yaml` unchanged, so this capture is directly
comparable to the stage-A data `L26` was fitted on. **Only the frequency comb and
the seed differ.**

```bash
# pre-run audit, then capture, then post-run audit, then validate
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config spf/calibrations/dual_rx_gain_frequency/configs/e_gsp7_conditioned_comb.yaml \
  --output artifacts/dual_rx_gain_frequency/e_gsp7_20260807
```

## 7. Pre-registration integrity

`select_comb.py`, `comb_selection.json` and this file are **committed before the
first frame is captured**. The combs are a function of the candidate grid and the
fleet delays only — no measurement enters the choice. Anyone can re-run
`select_comb.py` and reproduce them from the committed seed.

**Do not re-select a comb after seeing the data.** If the primary comb fails, that
is the result.

## 8. Risks and what this cannot settle

| Risk / limit | Why it matters | Handling |
|---|---|---|
| **Same-session evaluation** | Held-out LOs share session, harness and thermal state with the training LOs, so this is *not* a cross-session transfer test. Session drift is real — a 1356-parameter LUT degrades 0.62° → 2.74° across a 12-hour boundary. | Stated as a limit, not designed away. The committed-coefficient score in the same run gives the cross-session number alongside it. |
| **Objective is the ripple basis only** | `ripple_conditioning` scores the two ripple delay pairs, not the full design matrix. A comb can be ripple-optimal and still poor for `H(state)`. | The ≥3-LOs-per-band constraint, and the dense capture means a post-hoc full-design analysis is possible without another capture. |
| **Frozen-delay arm assumes fleet delays** | τ₂ did **not** replicate on the wide survey's clustered comb; only τ₁ (2.54–2.56 ns) did. | Report free-delay and frozen-delay fits separately; do not merge them into one headline. |
| **Low-gain SNR at 5 dB, high band** | Weak cells cost epochs, as R18 @ 5100 MHz did in both E-CAL1 arms. | Stage A ran this exact gain set successfully (18,195/19,836 quality-valid). Quality gates unchanged. |
| **Band coverage** | Extrapolating across a gain-table band is a known failure (L10). | ≥3 LOs per band enforced in the selection. |

## 9. Decision rule

Pre-registered. Do not renegotiate after seeing the data.

| Result | Conclusion |
|---|---|
| **Chosen-10 beats the anchor-only baseline** *and* **uniform-10 does not** | The E-GSC aliasing diagnosis is confirmed prospectively and **comb conditioning is the actionable lever**. A sparse protocol becomes recommendable pending a cross-session repeat. |
| **Both beat the baseline** | E-CAL3's failure was not purely spacing; something session-specific contributed. Report both and do not credit conditioning alone. |
| **Neither beats the baseline** | Ten LOs is too few regardless of placement (H₀). The conditioning diagnosis stands as a description of E-CAL3's defect but not as a route to sparse calibration. Escalate to the N = 16 comb. |
| **Chosen-10 fails but chosen-16 succeeds** | The lever is real but `N* = 16` is the true floor; recommend 16, not 10. |

Report the held-out MAE **and** its spread for every arm, and state explicitly
that evaluation is within-session.
