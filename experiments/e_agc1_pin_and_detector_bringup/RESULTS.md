# E-AGC1 results — session 1, no-RF measurements, both radios

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

**Steps 1, 2, 3, 4 and 6 plus the §5.4 restore are complete on both radios. Step 5
(H3–H5, detectors) has not been run** — it needs the RF level sweep and is session 2.

## Summary

| Hypothesis | Outcome | Artifact |
|---|---|---|
| **H1 — pin mapping is the identity** | ✅ **PASS**, **40/40 trials across two radios**, gate met, radios agree exactly | [`pin_map.json`](pin_map.json) |
| **H2 — step size** | ✅ **PASS** on both radios at both programmed values | [`step_size.json`](step_size.json) |
| **H6 — ENSM dependence** | ⚠️ **edges are NOT honoured outside RX**, both radios — the contract-changing outcome | [`ensm_result.json`](ensm_result.json) |
| *(unplanned)* armed writes | ⚠️ software gain writes silently ignored while armed, both radios | [`armed_write_ab.json`](armed_write_ab.json) |
| H3–H5 — detectors | not run (session 2) | — |
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

## Why the harness question does not apply to this session

Recorded because it will be asked. On 2026-08-10 it was established that the dual-RX
bench "splitter" is a bare SMA tee with ~0 dB port-to-port isolation, which puts the
arm-specific residual `A` from the phase campaigns in doubt (see the harness entry in
`docs/learnings.md`). **None of it reaches these results**, for three independent
reasons:

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

The RF-dependent half of E-AGC1 is **step 5, which was not run.** Its sensitivity is
scoped in §4.2 of the plan: the detector map, H4 and H5 are harness-independent, the
hold band is a level *difference* and so cancels a constant offset, and only absolute
input-level claims are harness-referenced.

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

1. **Step 5 (H3–H5)** — detector map, threshold sweep, hold band, latch trace, low-power
   period. Needs the RF level sweep; ~2–2.5 h. H4/H5 are pre-declared resolution-limited
   (§2), so **O-3 may not be closeable from userspace at all** and may need the FPGA
   stage regardless. On the tee, isolate one channel physically (§4.2).
2. **`TANDEM_AGC_V1_DESIGN.md` is still not in the repository**, so O-1 and O-5 cannot be
   marked closed and §7's references to the contract's §3/§11 cannot be amended. The
   measurements above do not expire; the traceability is what is blocked.

Of the four open items E-AGC1 was written to close, **O-5 (pin mapping) and O-1 (ENSM)
are answered**; O-2 (hold band) and O-3 (blank timing) remain with step 5.
