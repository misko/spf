# E-GSP7 — results (2026-08-07)

**Full report, code hashes and per-arm detail:**
[`reports/e_gsp7_conditioned_comb_20260807_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/e_gsp7_conditioned_comb_20260807_v1/REPORT.md)
· raw capture `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_gsp7_20260807/` (gitignored)

Session `e_gsp7_20260807`, SPF `70d84b8` clean, 111 LOs, **3,330/3,330 frames**,
one unchanged-harness session. Combs pre-registered before the first frame.

---

## Answer

**A ten-LO comb calibrates — but only as a conjunction.** It must be *chosen by
conditioning* **and** fitted with the *ripple delays frozen* at fleet values.
Drop either and it is worse than applying no model at all.

| Arm | cond | free-delay | frozen-delay | baseline |
|---|---|---|---|---|
| **chosen-10** | 1.09 | 8.352 ✗ | **4.950 ✓ (1.51×)** | 7.476 |
| **ecal3-10** *(control)* | 17.92 | 9.369 ✗ | **23.503 ✗✗ (0.31×)** | 7.280 |
| linspace-10 | 21.78 | 46.116 ✗ | 17.557 ✗ | 7.344 |
| **chosen-16** | 1.05 | **5.573 ✓** | 5.285 ✓ | 7.717 |
| committed coefficients, **no refit** | — | **3.863 (1.93×)** | — | 7.451 |

Held-out MAE (deg) on the 95–101 LOs each arm did not train on. **Every arm ran at
100% state coverage**, verified before capture, so nothing here is a coverage
artifact — the only difference between chosen-10 and ecal3-10 is where the LOs sit.

## The operational conclusion

**Do not refit sparsely by default — use the committed coefficients.** Scored on
chosen-10's *own* held-out set (identical 101 LOs, same 7.476° baseline, n = 3030),
the committed `l26_pooled_v1` gives **3.818°** against chosen-10-frozen's **4.950°** —
better by **1.13° MAE**, with no refit at all. And the logic is self-undermining: if
fleet delays are trusted enough to freeze, the committed coefficients are trusted
enough to use. A sparse refit earns its keep only where a genuinely *local* fit is
required.

## The four prospective confirmations

| Prediction | Source | Measured here |
|---|---|---|
| E-CAL3's ten-LO failure | E-CAL3 (1.281× worse than baseline) | **1.287×** worse |
| 73.4% recovery with frozen delays at N=10 | E-GSC2/3, retrospective | **70.4%** |
| `N* = 16` with free delays | E-GSC2 | confirmed; only chosen-16 recovers τ = (2.50, 0.98) vs fleet (2.56, 0.92) |
| "~1.9× on transfer" | E-GSC4 / L10 | **1.93×** |

## The unpredicted finding

**Freezing the delays on a badly-conditioned comb is actively dangerous.** It helps
the good comb (8.35 → 4.95) and wrecks the aliased one (9.37 → **23.50**). With
delays free the fit escapes by moving τ somewhere less collinear — visible in the
(4.15, 0.16) and (2.44, 4.90) fits. Freezing removes that escape route.
**Never freeze the delays without first checking the comb's conditioning.**

## Decision rule — which branch

| Regime | Branch reached |
|---|---|
| Frozen-delay, N=10 | *"chosen beats baseline, control does not"* → **conditioning is the actionable lever** |
| Free-delay, N=10 | *"neither beats baseline"* → ten LOs too few regardless of placement |
| Free-delay, N=16 | *"chosen-16 succeeds"* → `N* = 16` is the floor when delays are fitted |

## Caveats

- **Evaluation is within-session**, as pre-registered: training and held-out LOs
  share session, harness and thermal state. This establishes identifiability and
  conditioning, **not** session-to-session robustness. The committed-coefficient
  row is the only genuinely cross-session number.
- **R18 failed the strict gate** — 5 of 555 cells above the 5° across-epoch
  circular-std threshold, all 5100–5350 MHz, every frame quality-valid. One is the
  5200 MHz anchor cell, and **5200 MHz is in the chosen-10 comb** — which works
  *against* chosen-10, so its 4.950 is if anything pessimistic.
- The objective was `ripple_conditioning` (the two-delay ripple basis), not the
  full design matrix.
- Two radios, one harness topology.
