# E-AGC1 results — sessions 1 and 2 (complete except O-3)

**Runs:** R17 2026-08-10 22:43–22:47 UTC · R18 2026-08-10 23:00–23:02 UTC.
**Radios:** R17 `104000bac4950008230026001b440a003a` (USB `1-1.1`) and
R18 `1040007c4a94000211000b009186843ef2` (USB `1-1.2`).
**Firmware:** stock, `device-fw v0.38-plutoplus-spf-gain-series-v4-rc16-7-g1f3fe`
(the volatile RC17 candidate), kernel `5.15.0-gd798b0d821b8`, on both units.
**Both:** RX LO 868 MHz, both RX in `manual` gain, ENSM `fdd`, gain index 44 (41 dB),
TX muted at −80 dB throughout.
**Raw:** `artifacts/agc_pin_bringup/20260810_r17_v1/` and
`artifacts/agc_pin_bringup/20260811_r18_v1/` (gitignored).

No FPGA change, no bitstream, no RAM boot, no QSPI write. **~4 minutes of measurement
per radio** against a ~1–1.5 h budget (§5.5) — the budget assumed manual sequencing and
scripting collapsed it.

**All six measurements have now run.** Steps 1–4 and 6 ran on **both radios**
(session 1, no RF). Step 5 (H3–H5, detectors) ran on **R17** with the tone on
(session 2, 2026-08-11) — see §"Step 5".

**Three of the four open items E-AGC1 was written to close are answered:** O-5 (pin
map), O-1 (ENSM) and O-2 (hold band). **O-3 (blank timing) remains open** and, as
pre-declared, is not closeable from userspace.

## Summary

| Hypothesis | Outcome | Artifact |
|---|---|---|
| **H1 — pin mapping is the identity** | ✅ **PASS**, **40/40 trials across two radios**, gate met, radios agree exactly | [`pin_map.json`](pin_map.json) |
| **H2 — step size** | ✅ **PASS** on both radios at both programmed values | [`step_size.json`](step_size.json) |
| **H6 — ENSM dependence** | ⚠️ **edges are NOT honoured outside RX**, both radios — the contract-changing outcome | [`ensm_result.json`](ensm_result.json) |
| *(unplanned)* armed writes | ⚠️ software gain writes silently ignored while armed, both radios | [`armed_write_ab.json`](armed_write_ab.json) |
| **H3 — detector page** | ✅ **PASS** for all 7 bits that could be provoked; **zero cross-channel leakage** | [`detector_map.json`](detector_map.json) |
| **Hold band (O-2)** | ✅ **22 dB** on both arms — the oscillation rule did **not** fire | [`hold_band.json`](hold_band.json) |
| Threshold sweep | ✅ `0x114` identified at **0.5 dB/LSB**, 6 monotonic points; `0x107`/`0x108` do **not** resolve at this drive level | [`threshold_sweep.json`](threshold_sweep.json) |
| **H4 — latch/blank** | ⚠️ **NOT RESOLVED** — no blank seen at 322 µs granularity; O-3 stays open | [`latch_trace.json`](latch_trace.json) |
| **H5 — low-power period** | ⚠️ **NOT RESOLVED** — the bit is a stable level, no dither to time | [`lp_period.json`](lp_period.json) |
| Restore proof | ✅ 47/47 values match baseline on **each** radio | [`restore_proof.json`](restore_proof.json) |

Because both units reproduce every result identically, the single-unit caveat is gone.
This is a part-family property, not a per-board one.

## H1 — pin mapping confirmed on the part, twice

The schematic-plus-constraints inference is now a measurement. On **both** radios every
pin moved its own channel in the predicted direction on **5/5** trials, and **the other
channel never moved once** in 20 trials per radio:

| GPIO | EMIO | `CTRL_IN` | Own-channel delta ×5 | Other channel | R17 | R18 |
|---|---:|---|---|---|---|---|
| 968 | 8 | 0 → RX1 increase | +2, +2, +2, +2, +2 | 0, 0, 0, 0, 0 | PASS | PASS |
| 969 | 9 | 1 → RX1 decrease | −2, −2, −2, −2, −2 | 0, 0, 0, 0, 0 | PASS | PASS |
| 970 | 10 | 2 → RX2 increase | +2, +2, +2, +2, +2 | 0, 0, 0, 0, 0 | PASS | PASS |
| 971 | 11 | 3 → RX2 decrease | −2, −2, −2, −2, −2 | 0, 0, 0, 0, 0 | PASS | PASS |

The two radios' delta sequences are identical, cell for cell.

**Consequence:** the §7 row "H1 passes → close O-5; the contract's §3 table stands". No
transposition, so the RTL work is not blocked. This was the highest-consequence single
outcome in the experiment and it is clean on both units.

## H2 — an edge moves the index by exactly the programmed step

As shipped both radios have `0x0FC = 0x0FE = 0x23` → both step fields hold
`value − 1 = 1`, a programmed step of **2**, and all 20 H1 edges per radio moved 2.

Programming step 1 — `0x0FC = 0x03`, `0x0FE = 0x03`, with the Peak Overload Wait Time in
`0x0FE[4:0]` **preserved at 3** by read-modify-write — gave exactly ±1 per edge on both:

| Radio | up ×5 | down ×5 | PWOT preserved |
|---|---|---|---|
| R17 | +1, +1, +1, +1, +1 | −1, −1, −1, −1, −1 | yes |
| R18 | +1, +1, +1, +1, +1 | −1, −1, −1, −1, −1 | yes |

## H6 — `CTRL_IN` edges are not honoured outside RX

The open question is answered on both units, and it is the outcome that changes the
contract.

| ENSM state | Edges per radio | R17 delta | R18 delta | Honoured |
|---|---:|---|---|---|
| `fdd` (baseline, RX active) | 1 | +2 | +2 | **yes** |
| `alert` | 3 | 0, 0, 0 | 0, 0, 0 | **no** |
| `sleep` | 3 | 0, 0, 0 | 0, 0, 0 | **no** |
| `wait` | — | — | — | not reachable |

After every state each radio was returned to `fdd` and re-pulsed as a control: **delta =
+2 every time, on both units**, so the nulls are a real state dependence and not an
unresponsive part.

**Consequence:** the §7 row "**H6: edges NOT honoured outside RX** → the enable sequence
in §11 must guarantee RX is active before arming, and must handle an ENSM transition
while armed. A real change to the contract."

Two incidental findings, both on both radios:

- **`wait` is advertised but not reachable.** `ensm_mode_available` lists
  `sleep wait alert fdd pinctrl pinctrl_fdd_indep`; writing `wait` returns success and
  lands in `alert`. Recorded as not tested rather than as a null.
- `pinctrl` and `pinctrl_fdd_indep` were deliberately **not** tested: both hand ENSM
  state to external pins, which would confound H6 with a second pin-control surface.

## Unplanned: while armed, the pins own the gain index exclusively

Found because the between-pin `hardwaregain` reset in the H1 script did not take effect.
Isolated as a clean A/B, and it reproduces on both radios:

| State | Action | Index before | R17 after | R18 after | Write took effect |
|---|---|---:|---:|---:|---|
| **disarmed** (`0x0FB[1:0]=0`) | write `hardwaregain` 35 dB | 44 | **38** | **38** | yes |
| **armed** (`0x0FB[1:0]=3`) | write `hardwaregain` 35 dB | 44 | **44** | **44** | **no** |
| **armed** | one `CTRL_IN0` edge | 44 | **46** | **46** | — (control) |

The write **returns success** (`rc = 0`) and the subsequent `hardwaregain` readback
reports `41.000000 dB` — the pin-controlled index, not the requested value. So it fails
silently in both directions. The pin edge in the same armed state proves the part was
responsive, so this is the write being dropped rather than a dead radio.

**Consequences:**

- The tandem-AGC enable sequence must treat arming as **taking gain ownership away from
  software**. Any host-side gain write during tandem operation is a silent no-op.
- For SPF specifically: `radio.set_gains()` would silently no-op while tandem is armed.
  The calibration path fails closed on this by luck rather than design —
  `_validate_frame_gain` (`runner.py:93-108`) compares the frame's gain metadata against
  the request and raises, so a capture would abort rather than record mislabelled data.
  Any *other* code path that writes gain without verifying readback would not notice.
- This belongs in the design contract next to H6: both are enable-sequence facts.

## Step 5 — detectors (session 2, R17, 2026-08-11, tone on)

**Source:** the radio's own TX2 `fpga_dds` tone at +100 kHz, through the 30 dB pad and the
tee. At TX2 full scale (`hardwaregain 0 dB`) the RX ports sit near **−57 dBm**, so the
+2.5 dBm RF-pin limit was never approached; the level was walked instead with **RX gain**,
which moves the ADC-referred level without changing input power at all.

**Quiescent pattern with no signal: `10001011`** (CTRL_OUT7…0). The two low-power bits
(7 and 3) asserted, which is exactly right for no signal — so "any bit high" is not a
usable safety limit, and RSSI was used to control the ramp instead.

### H3 — detector map, by differential attribution

Rather than disconnect a splitter leg, each arm's gain was swept 10→72 dB in 1 dB steps
with the other arm fixed at 41 dB. **Bits that move belong to the swept arm** — an
attribution test that does not depend on the harness having port-to-port isolation, which
the bare tee does not have.

| Transition | CH1 (RX1 swept) | CH2 (RX2 swept) |
|---|---:|---:|
| low power **de**-asserts | 22 dB | 22 dB |
| small ADC overload asserts | 44 dB | 44 dB |
| large ADC overload asserts | 45 dB | 45 dB |
| large LMT overload asserts | *not provoked* | 72 dB |

**Zero cross-channel leakage in either sweep.** Sweeping RX1 moved only CH1 bits;
sweeping RX2 moved only CH2 bits. The predicted `CTRL_OUT` map is confirmed for all
**7 of 8** bits that could be provoked. CH1's large-LMT bit was never provoked: LMT
overload needs real input power, and the pad plus tee split cannot deliver it.

### Hold band — open item O-2, closed

**22 dB on both arms** (low-power de-assert at 22 dB → small-ADC assert at 44 dB). Because
this is a *difference* of two gain settings on one arm, the harness insertion loss
cancels, so the tee does not compromise it.

**The §7 decision rule does not fire.** It warned that a hold band "narrower than 1 dB"
would make the policy oscillate as designed, and called that "the most likely single
change to come out of the run". At 22 dB the band is 22× wider than that threshold — no
hysteresis rework is needed on this account.

### Threshold sweep

| Register | Predicted | Points | Result |
|---|---|---:|---|
| `0x114` | low-power threshold | 6 | ✅ **identified**: monotonic, **0.5 dB/LSB** — 0x10→38 dB, 0x20→30, 0x30→22, 0x40→14 |
| `0x107` | ADC small overload | 6 | ❌ edge pinned at 44 dB for 0x1b–0x3b; bit never asserted at 0x43 |
| `0x108` | ADC large overload | 6 | ❌ edge pinned at 46 dB across all six values |

All writes were read-modify-write preserving bit 7, and every register was restored.

The ADC-threshold nulls are a **result, not a failure**: at this drive level the converter
goes from clear to saturated within about **1 dB** of gain (small at 44, large at 45), so
the ADC's own saturation — not the programmed threshold — is the binding constraint, and
the threshold only becomes binding at the top of its range. Practical consequence for the
contract: the low-power trip point is cleanly programmable over ~30 dB, while the ADC
overload bits carry almost no graded information at this operating point.

`0x104`/`0x105` (LMT thresholds) were **not** swept: CH1's LMT bit could not be provoked
at all and CH2's only at the very top of the gain range, so the sweep would have had at
most one observable point.

### H4 (latch) and H5 (period) — not resolved, as pre-declared

Both were flagged resolution-limited in §2 before the run, and that held — **worse** than
budgeted. Reading a GPIO *value* file costs **322 µs**, not the 134 µs measured on a plain
sysfs attribute, against a predicted 256–410 µs period.

- **H4.** CH1 was driven into large-ADC overload (bits `00110100` at RX1 = 52 dB) and the
  gain stepped down with a `CTRL_IN1` edge — a shell-builtin GPIO write, chosen precisely
  because a 67 ms `iio_attr` write cannot act inside a sub-millisecond blank. The index
  moved 55 → 53, so the step landed. **No 1→0→1 excursion was observed.** That bounds any
  blank at **under 322 µs** but neither confirms nor refutes the latch. **O-3 stays open.**
- **H5.** The low-power bit is a **stable level** on both sides of a sharp 22 dB
  threshold: 200/200 high at 20–21 dB, 0/200 high at 22–24 dB, **zero transitions
  anywhere**. With no dither there is no interval to time. Compounding the sampler limit,
  RX gain is quantised to 1 dB, likely too coarse to park the comparator at its trip
  point.

Both belong to the FPGA stage, alongside the minimum-pulse-width question §8 already
defers there. **Do not re-attempt them from userspace.**

One incidental H4 note worth keeping: the repeat traces from k≥3 read all-zero because pin
control was still armed, so the `iio_attr` re-arm of RX1 to 52 dB before each repeat was
**silently ignored** — the armed-write finding above, reproducing unprompted in a
different context — and the index walked down 2 per repeat until it fell below the
overload point. Only the first two repeats test what was intended.

### An observation consistent with the tee's lack of isolation

At RX1 = 52 dB with RX2 held at 41 dB, **CH2's large-LMT bit asserted** — yet in the
phase-C sweep, with RX2 fixed at that same 41 dB, that bit stayed low throughout. The
difference is RX1's gain state. With ~0 dB port-to-port isolation, changing RX1's gain
changes its input impedance and therefore the level delivered to RX2, which is exactly the
`g1 → RX2` coupling path predicted for a tee in `docs/learnings.md`. Recorded as an
observation, not a measurement: it is a single incidental data point, and a real divider
would settle it.

### Restore

Every phase restored and verified: `0x035` back to `0x00`, all five threshold registers
back to their originals, `0x0FB` back to `0x08`, TX2 back to −80 dB, the DDS tone
disabled, RX gains back to 41/41, index back to 44/44, and all twelve GPIO lines
(`CTRL_OUT` 960–967 and `CTRL_IN` 968–971) released — `/sys/kernel/debug/gpio` back to
exactly the original five claimed lines. See [`step5_restore_proof.json`](step5_restore_proof.json).

One incident is recorded there rather than hidden: an early attempt at the phase-C sweep
was killed by a host-side 2-minute tool timeout, and because the trap did not cover
`SIGHUP` the radio was left with the tone on, `0x035 = 0x03` and `CTRL_OUT` exported. It
was explicitly restored and verified before continuing, and `HUP` was added to the trap
for every later phase. **No `CTRL_IN` pin was armed at any point during that window**, so
no gain-control hazard existed.

## The harness question, and which sessions it touches

Recorded because it will be asked. On 2026-08-10 it was established that the dual-RX
bench "splitter" is a bare SMA tee with ~0 dB port-to-port isolation, which puts the
arm-specific residual `A` from the phase campaigns in doubt (see the harness entry in
`docs/learnings.md`). **None of it reaches the session-1 results** (H1, H2, H6, the
armed-write finding), for three independent reasons:

1. **There was no RF in the harness.** On both radios TX1 and TX2 were at −80 dB
   (muted) in step 1 and still at −80 dB in the step-8 re-read, and the 47-value
   baseline match proves they never moved in between. The tee was splitting nothing.
2. **Nothing analogue was observed.** The only observables in H1, H2 and H6 were
   `iio_reg 0x2B0` and `iio_reg 0x2B5` — the gain-index registers, read over SPI. No
   phase, magnitude, RSSI or IQ appears anywhere in the run.
3. **The causality measured was digital**: GPIO edge → gain-index register. That path
   does not traverse the RF chain.

This is also true by construction rather than by luck: in manual gain mode the index is
set by the pins irrespective of signal, and H6's nulls arise because the RX path is not
clocked in `alert`/`sleep` — a state fact, not a level fact.

**Session 2 (step 5) did use RF**, and its exposure is exactly as §4.2 of the plan
scoped it in advance:

- the **detector map** is harness-independent — attribution came from a differential
  gain sweep with zero cross-channel leakage, not from absolute levels;
- the **hold band** is a level *difference* on one arm, so insertion loss cancels;
- **H4/H5** are timing and logic, and failed on sampler resolution, not on the harness;
- the one genuinely harness-referenced quantity — **absolute input power** — is reported
  only as TX `hardwaregain` settings plus a recorded harness description, never as dBm at
  the RF port. CH1's unprovoked LMT bit is a direct consequence of that limit.

The single observation that *is* affected is the incidental CH2 LMT assert noted above,
and it is labelled as an observation rather than a result.

**R18's role as E-GSP2's control radio is preserved.** That designation is about its RF
harness never being disturbed; this session touched no connector, ran with TX muted, and
restored every register exactly.

## Restore

Performed per §5.4 on each radio — disarm before releasing the pins, then restore gain,
then unexport. Verified by re-running the full baseline collection and diffing:

- `0x0FB` returned to `0x08` exactly, including bit 3, after every armed phase;
- `0x0FC` / `0x0FE` returned to `0x23` / `0x23`, PWOT intact;
- both gain indices back to 44 (41 dB), both modes `manual`, ENSM `fdd`;
- all four CTRL_IN lines unexported; `/sys/kernel/debug/gpio` back to exactly the
  original five claimed lines (921, 952, 973, 974, 977);
- **all 47 recorded values match the step-1 baseline on each radio** (timestamps and
  uptime excluded).

## Method notes worth keeping

- Every phase that armed `0x0FB` did so by read-modify-write inside a script with an
  `EXIT`/`INT`/`TERM` trap that disarms, so an error or a dropped SSH session still
  releases the part. `0x0FB` bit 3 is set on both builds, so a bare `0x03` write would
  have cleared a live bit — the §8 risk row is real, not theoretical.
- The pins were exported, driven low and **verified low** before anything was armed,
  with the gain indices confirmed unchanged across that step — an unarmed pin cannot
  move gain, which is what makes that order safe.
- `iio_attr` needs `-i` for RX channels. Without it the tool matches the output channel
  of the same name and returns the TX gain (−80 dB here), not RX.
- Command cost measured 67 ms per `iio_reg`/`iio_attr` invocation, so a 20-trial sweep
  is ~6 s. Nothing here is throughput-limited.
- The GPIO base is 906 on both units, so `CTRL_IN` is 968–971 and `CTRL_OUT` is
  960–967, and nothing in the shipped stack claims EMIO 0–11 on either.

## Open

1. **O-3 (blank timing) — the one measurement this method cannot make.** H4 and H5 were
   pre-declared resolution-limited in §2 and that prediction held, worse than budgeted:
   reading a GPIO *value* file costs **322 µs**, not the 134 µs measured on a plain
   sysfs attribute, against a predicted 256–410 µs period. This needs the FPGA stage,
   alongside the minimum-pulse-width question §8 already defers there. Do not re-attempt
   it from userspace.
2. **`TANDEM_AGC_V1_DESIGN.md` is still not in the repository**, so O-5, O-1 and O-2
   cannot be marked closed and §7's references to the contract's §3/§11 cannot be
   amended. The measurements do not expire; the traceability is what is blocked.
3. **CH1's large-LMT bit was never provoked**, because the 30 dB pad plus the tee split
   leaves the RX ports near −57 dBm at TX full scale. Reaching LMT overload needs more
   input power — removing the pad, or a stronger source. Not required by any open item.
4. **Step 5 on R18**, if a per-part detector-threshold comparison is ever wanted. The
   plan notes detector thresholds are per-part while the open items are not, so this is
   optional.
