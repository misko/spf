"""Proof-of-concept walkthrough of the gain-state phase model.

Run it::

    python -m spf.calibrations.gain_state_phase_model_v1.demo

No campaign data and no network access needed -- everything it uses is committed
in this directory. It walks through the six things an integrator has to get
right, in the order they matter:

    1. decode a requested gain to its audited hardware state
    2. predict the gain-dependent residual D
    3. apply the full two-layer correction with a measured session anchor
    4. see the correction refuse an uncalibrated state instead of guessing
    5. see the rule-5 RF-state guard suppress a known-harmful correction
    6. see what the model costs you when the gain pair is badly mismatched
"""

from __future__ import annotations

import math

from .gain_tables import band_for_lo, default_tables
from .model import GainStatePhaseModel, UnsupportedGainState

RULE = "=" * 78


def hdr(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def main() -> None:
    tables = default_tables()
    model = GainStatePhaseModel.load_named("l26_pooled_v1")

    print(f"model            : {model.name}")
    print(f"ripple delays    : {[round(t * 1e9, 3) for t in model.tau_seconds]} ns")
    print(f"term families    : {model.families_used}")
    print(f"fitted on        : {model.provenance.get('n_rows')} rows, "
          f"{model.provenance.get('n_los')} LOs, "
          f"{len(model.provenance.get('gains_db', []))} distinct requested gains")
    print(f"columns / rank   : {model.provenance.get('n_columns')} / "
          f"{model.provenance.get('rank')}   "
          f"(rank < columns: individual coefficients are NOT identified)")

    # ---------------------------------------------------------------- 1
    hdr(1, "Decode a requested gain to its audited hardware state")
    print("The same requested dB is a DIFFERENT hardware state in each band.")
    print("This is read off the chip, not fitted.\n")
    print(f"  {'LO':>10}  {'band':>7}  {'dB':>4}  {'row':>4}  "
          f"{'LNA':>4} {'MIX':>4} {'TIA':>4} {'LPF':>4}  RF_DC")
    for lo in (900e6, 2_412e6, 5_100e6):
        for g in (5, 26, 45):
            st = tables.state_for_lo(lo, g)
            print(f"  {lo/1e6:>8.0f}M  {band_for_lo(lo):>7}  {g:>4}  {st.row:>4}  "
                  f"{st.lna:>4} {st.mixer:>4} {st.tia:>4} {st.lpf:>4}  {st.rf_dc_cal}")
    print("\n  -> at 26 dB the high band already sits at LNA 2 while low/middle are")
    print("     still at LNA 0. A model keyed on requested dB cannot represent that.")

    # ---------------------------------------------------------------- 2
    hdr(2, "Predict the gain-dependent residual D")
    lo = 2_412_000_000
    print(f"LO = {lo/1e6:.0f} MHz ({band_for_lo(lo)} band), RX2 held at the 26 dB reference\n")
    print(f"  {'RX1':>4} {'RX2':>4}  {'D (deg)':>9}   state RX1 -> RX2")
    for g1 in (5, 26, 45):
        p = model.predict(lo, g1, 26)
        s1, s2 = p.state_rx1, p.state_rx2
        note = "" if p.supported else f"  [{p.reason}]"
        val = f"{p.residual_deg:>+9.3f}" if p.supported else f"{'refused':>9}"
        print(f"  {g1:>4} {26:>4}  {val}   "
              f"({s1.lna},{s1.mixer},{s1.tia},{s1.lpf}) -> "
              f"({s2.lna},{s2.mixer},{s2.tia},{s2.lpf}){note}")

    # ---------------------------------------------------------------- 3
    hdr(3, "The full two-layer correction")
    print("corrected = wrap( measured_RX1_minus_RX2  -  session anchor  -  D )\n")
    print("The anchor is a MEASUREMENT of the equal-gain cell at this exact LO,")
    print("taken this session on this serial. It is never transferred across a")
    print("re-mate, harness change, radio swap or unvalidated boot.\n")
    anchor = math.radians(95.4802)   # the measured 2467.1 MHz wall-array anchor
    measured = math.radians(112.0)
    corrected = model.correct_measured_phase(measured, anchor, lo, 45, 26)
    d = model.predict_residual_rad(lo, 45, 26)
    print(f"  measured RX1-RX2   {math.degrees(measured):>+9.3f} deg")
    print(f"  session anchor     {math.degrees(anchor):>+9.3f} deg   (measured, radio-specific)")
    print(f"  model residual D   {math.degrees(d):>+9.3f} deg   (universal, 38 coefficients)")
    print(f"  {'-' * 40}")
    print(f"  corrected          {math.degrees(corrected):>+9.3f} deg")

    # ---------------------------------------------------------------- 4
    hdr(4, "Fail closed on an uncalibrated hardware state")
    print("LNA index 1 was never measured at any frequency in the source campaign.")
    print("The gains that would visit it are 31-32 dB (low), 30-31 (middle),")
    print("23-25 (high). The model refuses them; it does not interpolate.\n")
    for lo_hz, g in ((2_412e6, 30), (2_412e6, 45), (5_100e6, 24)):
        # guard off here so the status column reports support, not the guard
        p = model.predict(lo_hz, int(g), 26, apply_rf_state_guard=False)
        mark = "ok" if p.supported else "REFUSED"
        print(f"  {lo_hz/1e6:>6.0f} MHz  RX1={g:>3} dB  {mark:>8}  {p.reason}")
    try:
        model.predict_residual_rad(2_412_000_000, 30, 26)
    except UnsupportedGainState as exc:
        print(f"\n  raising API: UnsupportedGainState({exc})")

    n_ok = len(model.supported_gains_db(2_412_000_000))
    lo_db, hi_db = tables.gain_range_db("middle")
    print(f"\n  supported requested gains at 2412 MHz: {n_ok} of "
          f"{hi_db - lo_db + 1} in the table")

    # ---------------------------------------------------------------- 5
    hdr(5, "The rule-5 RF-state guard")
    lo5 = 5_100_000_000
    print(f"At {lo5/1e6:.0f} MHz the whole 27-40 dB span shares the RF words (LNA 2, MIX 4,")
    print("TIA 1) -- only the baseband LPF word moves. The source experiment")
    print("measures no phase there, so the fitted LPF differences are noise.\n")
    on = model.predict(lo5, 40, 27, apply_rf_state_guard=True)
    off = model.predict(lo5, 40, 27, apply_rf_state_guard=False)
    print(f"  guard ON  (default) : D = {on.residual_deg:+.4f} deg   {on.reason}")
    print(f"  guard OFF           : D = {off.residual_deg:+.4f} deg   would be injected")
    print("\n  Across the pooled set this guard is worth a lot: on the 672 frozen-RF")
    print("  unequal-gain cells, the unguarded model injects a mean 1.362 deg and")
    print("  makes 81.4% of them worse.\n")
    for name in ("l30_pooled_v1", "l31_pooled_v1"):
        m = GainStatePhaseModel.load_named(name)
        v = m.predict(lo5, 40, 27, apply_rf_state_guard=False)
        print(f"  {name:<16} D = {v.residual_deg:+.4f} deg  (neutral by construction; "
              f"no LPF term)")

    # ---------------------------------------------------------------- 6
    hdr(6, "What the correction is worth")
    print("D is the error a per-session anchor alone LEAVES BEHIND. The bigger")
    print("|D| is, the more the gain-state model is doing for you.\n")
    print(f"  {'LO':>10}  {'RX1':>4} {'RX2':>4}  {'|D| (deg)':>10}")
    worst = []
    for lo_hz in (433e6, 900e6, 2_412e6, 3_500e6, 5_100e6, 5_800e6):
        for g1, g2 in ((45, 5), (45, 26), (5, 26)):
            p = model.predict(lo_hz, g1, g2)
            if p.supported and not p.guarded:
                worst.append((abs(p.residual_deg), lo_hz, g1, g2))
    for mag, lo_hz, g1, g2 in sorted(worst, reverse=True)[:8]:
        print(f"  {lo_hz/1e6:>8.0f}M  {g1:>4} {g2:>4}  {mag:>10.3f}")
    print("\n  For reference: with a per-frequency equal-gain anchor already applied,")
    print("  changing the gain pair still costs 6.65 deg MAE / 18.4 deg P95 /")
    print("  41.6 deg max if you do nothing. L26 takes that to 2.26 deg MAE at an")
    print("  unmeasured frequency and 2.22 deg at an unmeasured radio -- but those")
    print("  are dense-comb CROSS-VALIDATION figures. Prospectively, on a fresh")
    print("  103-LO capture, the committed coefficients score 4.8 deg against a")
    print("  9.06 deg anchor-only baseline. Treat 4.8 deg as the number to plan on.")
    print(f"\n{RULE}\nSee README.md for the physical backing, the full performance")
    print(f"tables, the limitations, and the queued follow-up experiments.\n{RULE}")


if __name__ == "__main__":
    main()
