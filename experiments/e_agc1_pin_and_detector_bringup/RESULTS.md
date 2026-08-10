# E-AGC1 results — session 1, no-RF measurements

**Run:** 2026-08-10, 22:43–22:47 UTC. **Radio:** R17
`104000bac4950008230026001b440a003a`, USB `1-1.1`.
**Firmware:** stock, `device-fw v0.38-plutoplus-spf-gain-series-v4-rc16-7-g1f3fe`
(the volatile RC17 candidate), kernel `5.15.0-gd798b0d821b8`.
**RX LO** 868 MHz, sample rate 3 MS/s, both RX in `manual` gain, ENSM `fdd`.
**Raw:** `artifacts/agc_pin_bringup/20260810_r17_v1/` (gitignored).

No FPGA change, no bitstream, no RAM boot, no QSPI write. Total elapsed **4 minutes**
of measurement against a ~1–1.5 h budget (§5.5) — the budget was dominated by expected
manual sequencing, and scripting it collapsed that.

**Steps 1, 2, 3, 4 and 6 plus the §5.4 restore are complete. Step 5 (H3–H5,
detectors) was not run** — it needs the RF level sweep and is session 2.

## Summary

| Hypothesis | Outcome | Artifact |
|---|---|---|
| **H1 — pin mapping is the identity** | ✅ **PASS**, 20/20 trials, gate met | [`pin_map.json`](pin_map.json) |
| **H2 — step size** | ✅ **PASS** at both programmed values | [`step_size.json`](step_size.json) |
| **H6 — ENSM dependence** | ⚠️ **edges are NOT honoured outside RX** — the contract-changing outcome | [`ensm_result.json`](ensm_result.json) |
| *(unplanned)* armed writes | ⚠️ software gain writes silently ignored while armed | [`armed_write_ab.json`](armed_write_ab.json) |
| H3–H5 — detectors | not run (session 2) | — |
| Restore proof | ✅ all 47 recorded values match baseline exactly | `step8_restore_verify_r17.json` |

## H1 — pin mapping confirmed on the part

The schematic-plus-constraints inference is now a measurement. Every pin moved its own
channel in the predicted direction on **5/5** trials, and **the other channel never
moved once** in 20 trials:

| GPIO | EMIO | `CTRL_IN` | Own-channel delta ×5 | Other channel | Verdict |
|---|---:|---|---|---|---|
| 968 | 8 | 0 → RX1 increase | +2, +2, +2, +2, +2 | 0, 0, 0, 0, 0 | PASS |
| 969 | 9 | 1 → RX1 decrease | −2, −2, −2, −2, −2 | 0, 0, 0, 0, 0 | PASS |
| 970 | 10 | 2 → RX2 increase | +2, +2, +2, +2, +2 | 0, 0, 0, 0, 0 | PASS |
| 971 | 11 | 3 → RX2 decrease | −2, −2, −2, −2, −2 | 0, 0, 0, 0, 0 | PASS |

**Consequence:** the §7 row "H1 passes → close O-5; the contract's §3 table stands".
No transposition, so the RTL work is not blocked. This was the highest-consequence
single outcome in the experiment and it is clean.

## H2 — an edge moves the index by exactly the programmed step

As shipped, `0x0FC = 0x0FE = 0x23` → both step fields hold `value − 1 = 1`, a
programmed step of **2**. All 20 H1 edges moved the index by exactly 2.

Programming step 1 — `0x0FC = 0x03`, `0x0FE = 0x03`, with the Peak Overload Wait Time
in `0x0FE[4:0]` **preserved at 3** by read-modify-write — gave exactly ±1 per edge:

- up ×5: `+1, +1, +1, +1, +1`
- down ×5: `−1, −1, −1, −1, −1`

## H6 — `CTRL_IN` edges are not honoured outside RX

The open question is answered, and it is the outcome that changes the contract.

| ENSM state | Edges | Index delta | Honoured |
|---|---:|---|---|
| `fdd` (baseline, RX active) | 1 | +2 | **yes** |
| `alert` | 3 | 0, 0, 0 | **no** |
| `sleep` | 3 | 0, 0, 0 | **no** |
| `wait` | — | — | not reachable (see below) |

After every state the radio was returned to `fdd` and re-pulsed as a control: **delta
= +2 every time**, so the nulls are a real state dependence, not an unresponsive part.

**Consequence:** the §7 row "**H6: edges NOT honoured outside RX** → the enable
sequence in §11 must guarantee RX is active before arming, and must handle an ENSM
transition while armed. A real change to the contract."

Two incidental findings:

- **`wait` is advertised but not reachable.** `ensm_mode_available` lists
  `sleep wait alert fdd pinctrl pinctrl_fdd_indep`; writing `wait` returns success and
  lands in `alert`. So `wait` was recorded as not tested rather than as a null.
- `pinctrl` and `pinctrl_fdd_indep` were deliberately **not** tested: both hand ENSM
  state to external pins, which would confound H6 with a second pin-control surface.

## Unplanned: while armed, the pins own the gain index exclusively

Found because the between-pin `hardwaregain` reset in the H1 script did not take
effect. Isolated as a clean A/B on the same radio in the same session:

| State | Action | Index before | Index after | Write took effect |
|---|---|---:|---:|---|
| **disarmed** (`0x0FB[1:0]=0`) | write `hardwaregain` 35 dB | 44 | **38** | yes |
| **armed** (`0x0FB[1:0]=3`) | write `hardwaregain` 35 dB | 44 | **44** | **no** |
| **armed** | one `CTRL_IN0` edge | 44 | **46** | — (control) |

The write **returns success** (`rc = 0`) and the subsequent `hardwaregain` readback
reports `41.000000 dB` — the pin-controlled index, not the requested value. So it fails
silently in both directions. The pin edge in the same armed state proves the part was
responsive, so this is the write being dropped rather than a dead radio.

**Consequences:**

- The tandem-AGC enable sequence must treat arming as **taking gain ownership away
  from software**. Any host-side gain write during tandem operation is a silent no-op.
- For SPF specifically: `radio.set_gains()` would silently no-op while tandem is armed.
  The calibration path fails closed on this by luck rather than design —
  `_validate_frame_gain` (`runner.py:93-108`) compares the frame's gain metadata against
  the request and raises, so a capture would abort rather than record mislabelled data.
  Any *other* code path that writes gain without verifying readback would not notice.
- This belongs in the design contract next to H6: both are enable-sequence facts.

## Restore

Performed per §5.4 — disarm before releasing the pins, then restore gain, then
unexport. Verified by re-running the full baseline collection and diffing:

- `0x0FB` returned to `0x08` exactly, including bit 3, after every armed phase;
- `0x0FC` / `0x0FE` returned to `0x23` / `0x23`, PWOT intact;
- both gain indices back to 44 (41 dB), both modes `manual`, ENSM `fdd`;
- all four CTRL_IN lines unexported; `/sys/kernel/debug/gpio` back to exactly the
  original five claimed lines (921, 952, 973, 974, 977);
- **all 47 recorded values match the step-1 baseline** (timestamps and uptime excluded).

## Method notes worth keeping

- Every phase that armed `0x0FB` did so by read-modify-write inside a script with an
  `EXIT`/`INT`/`TERM` trap that disarms, so an error or a dropped SSH session still
  releases the part. `0x0FB` bit 3 is set on this build, so a bare `0x03` write would
  have cleared a live bit — the §8 risk row is real, not theoretical.
- The pins were exported, driven low and **verified low** before anything was armed,
  with the gain indices confirmed unchanged across that step — an unarmed pin cannot
  move gain, which is what makes that order safe.
- `iio_attr` needs `-i` for RX channels. Without it the tool matches the output channel
  of the same name and returns the TX gain (−80 dB here), not RX.
- Command cost measured 67 ms per `iio_reg`/`iio_attr` invocation, so a 20-trial sweep
  is ~6 s. Nothing here is throughput-limited.

## Open

1. **Step 5 (H3–H5)** — detector map, threshold sweep, hold band. Needs the RF level
   sweep; ~2–2.5 h. H4/H5 are pre-declared resolution-limited (§2).
2. **`TANDEM_AGC_V1_DESIGN.md` is still not in the repository**, so O-1/O-2/O-3/O-5
   cannot be marked closed and §7's references to the contract's §3/§11 cannot be
   amended. The measurements above do not expire; the traceability is what is blocked.
3. **Second radio.** Everything here is per-part-family rather than per-unit, but H1 and
   H6 on R18 would cost ~4 minutes and remove the single-unit caveat.
