# E-CAL1 — results, both arms (2026-08-07)

**Full reports and code hashes:**
[arm 1](../../spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_rfdc_20260807_v1/REPORT.md) ·
[arm 2](../../spf/calibrations/dual_rx_gain_frequency/reports/e_cal1_arm2_rfdc_tracking_20260807_v1/REPORT.md)
· sensitivity closed by
[E-CAL5](../e_cal5_positive_control/RESULTS.md)

Both arms ran on one unchanged harness, 1,050/1,050 frames each, arm 2's schedule
copied from arm 1 byte-for-byte.

---

## Answer: the RF-DC machinery injects no resolvable phase

| | RF-DC excess `H(9) − mid[H(8), H(10)]` | 95% CI |
|---|---|---|
| **Arm 1** — tracking ON | **+0.069° ± 0.077** | [−0.168, +0.392] |
| **Arm 2** — tracking OFF (verified by chip readback) | **+0.019° ± 0.082** | [−0.258, +0.250] |
| Arm 1 − arm 2 | +0.050° ± 0.113 (t = 0.44) | no detectable difference |

Against the **2.664°** median mixer step that H₁ predicted — a factor of ~7 below
it even at arm 1's CI upper edge. Pre-registered branch reached:
*"≤ 0.35° with sem < 0.35° → the RF-DC machinery contributes no resolvable phase."*

**And the null is informative.** E-CAL5 measured a known mixer step on the same
harness at 7.434° ± 0.097 against a 0.440°/dB floor, so an H₁-sized 2.664° effect
would have appeared at **34.5σ** (arm 1) / **32.4σ** (arm 2).

## The cleanest evidence needs no modelling

8→9 and 9→10 are both 1 dB steps with the LMT words frozen, so identical noise.
The step that **raises** `RF_DC_CAL` (median \|ΔH\| **0.320°**) is *smaller* than
the one that **lowers** it (**0.446°**). A flag injecting phase cannot produce that
ordering.

## What it changes

The hedge is retired everywhere: `docs/learnings.md` L10 finding 2, the source
report's §3.3/§6.2/§7 ledger, and the model package README. **Every "the mixer word
moves the phase" statement is now a plain RF-state statement**, and the attribution
closes to the LNA/mixer/TIA network.

## Do not quote

`mann_whitney_rfdc_vs_lpf` (p = 0.010 arm 1, 0.0073 arm 2) is a **normalization
artifact** — it compares a 1 dB step against a *halved* 2 dB step, so it detects a
difference in spread, not location. The raw 8→10 step is in fact noisier than 8→9.

## Caveats and gate status

- **Arm 2 does not pass the strict gate on R18**: one cell of 21 (5100 MHz,
  g=(10,5)) at 5.15° against a 5.00° circstd threshold, on 23/25 epochs. Same
  low-SNR cell arm 1's pilot flagged before either capture. Arm 1 passed both radios.
- **One cell excluded in arm 1** on the pre-registered ≥20-epoch rule (R18 @
  5100 MHz, 15/25). Not an RF-DC effect — the LPF-only step is elevated in lockstep
  there, so it tracks cell SNR, not the flag.
- Row 11 (−3 dB), the other `RF_DC_CAL` edge with frozen LMT words, was **not
  resampled**; the ≲0.7° `F_neg` bound remains the only evidence there.
- The one-shot `calib_mode = rf_dc_offs` calibration fired at every frequency block
  in **both** arms, so it is common-mode and cancels in the second difference
  rather than being tested. A `never`-policy arm is now possible in code, unrun.
- Two radios, one harness topology, high band only.

## Four runbook defects found while executing

Recorded in arm 1's report §6; two will block anyone re-running this — the audit
command points at a manifest pinning the *campaign* firmware (fails closed against
this experiment's), and running that audit under `sudo` leaves the session
directory root-owned, which then breaks the capture.
