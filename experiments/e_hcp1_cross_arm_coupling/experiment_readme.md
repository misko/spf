# E-HCP1 — how much does the harness couple one RX arm into the other?

**Status:** designed and run 2026-08-11 (R17). One session, ~3 minutes.
**Cost:** no new parts, no harness change, no firmware change.
**Answers:** whether the bare-tee harness can account for the arm-specific residual `A`,
which is the doubt the 2026-08-10 harness finding raised over the whole dual-RX phase
programme.

---

## 1. Purpose

Every dual-RX gain-phase result was captured on what the docs call a "splitter" but is in
fact a bare SMA tee (`docs/learnings.md`, harness entry 2026-08-10). A tee's two output
ports are the same electrical node, so port-to-port isolation is ~0 dB, where a Wilkinson
divider would give ~20 dB.

The concern is specific. A change in one arm's gain state changes its input impedance,
and with no isolation that can change what the *other* arm receives — a `g1 → RX2` path
that would masquerade as the arm-specific residual `A`. Three published observations fit a
harness origin: `A` concentrates above 4 GHz, it is unit-specific (cross-radio ρ only
+0.50 / +0.59 / −0.23), and connector work once drove one radio's high-band mean `|A|`
from 3.49° to 29.41° without recovering.

**What this experiment adds that E-GSP2 cannot yet:** E-GSP2 is the definitive
tee-versus-divider A/B and needs parts we do not have. This measures the coupling's
*magnitude* directly on the harness in hand, which is enough to say whether the tee is a
plausible explanation for `A` at all.

### 1.1 Why the differential phase does not simply cancel it

Worth stating, because it nearly does. With a tee both RX ports share one node, so a
change in the junction voltage shifts both arms equally and **cancels exactly** in
`D = angle(RX1) − angle(RX2)`. That common-mode part is harmless.

What does not cancel is the arm-specific path: a wave reflecting off RX1 travels back to
the junction and enters RX2 down *its own* cable. That term depends on `g1` and on the two
cables separately, so it survives the subtraction. It is the only mechanism by which the
tee can imitate `A`, and it is what this experiment bounds.

## 2. Hypothesis

**H1 (the tee is not the explanation).** Cross-arm coupling is small and **frequency-flat**,
while `A` rises about 5× from the low band to the high band (0.73 → 3.72° mean). If
coupling does not concentrate where `A` concentrates, the tee cannot be `A`'s main source.

**H0 (the tee is implicated).** Coupling grows above 4 GHz, tracking `A`'s own band
profile. Then a large share of the published `A` is harness, `E-GSC6` must not run on this
bench, and every per-band `A` needs restating.

## 3. Approach

Hold one arm's gain fixed and sweep the other's, at LOs spanning all three bands.

**Observable:** the *fixed* arm's RSSI. RSSI is input-referred, so on an isolated harness a
fixed arm reads constant no matter what the other arm does. Any movement is coupling.

**Why an amplitude measurement bounds phase.** A reflection adds `ε·e^{jθ}` to the fixed
arm's signal; amplitude moves as `ε·cosθ` and phase as `ε·sinθ`. One frequency cannot
separate them — a quadrature-dominated coupling would show little amplitude change. But
θ rotates with frequency (the ripple period is ~392 MHz), so across 12 LOs spanning
5.5 GHz θ takes many values. If `ε` were large, *some* LO would show large amplitude
coupling. A flat, small amplitude bound across the sweep therefore bounds `ε` itself, and
so bounds phase.

Controls: both directions (sweep RX1 watching RX2, and the reverse), and the swept arm's
own RSSI recorded at every point so the source level and SNR are known per LO.

## 4. Hardware setup

### 4.1 Radios

One Pluto+: R17 `104000bac4950008230026001b440a003a`, USB `1-1.1`. Resolve by serial.

### 4.2 Physical schematic

The bench exactly as it stands — no change, which is the point:

```
   ┌───────────── PLUTO R17 ─────────────┐
   │  TX2 ──► 30 dB attenuator           │
   │             │                       │
   │        ┌────┴────┐                  │
   │        │ SMA TEE │  ~0 dB isolation │
   │        └──┬───┬──┘  (the thing      │
   │           │   │      under test)    │
   │        ┌──┴─┐ ┌┴───┐                │
   │        │RX1 │ │RX2 │                │
   │        └────┘ └────┘                │
   └─────────────────────────────────────┘
```

Tone: the radio's own TX2 `fpga_dds` at +100 kHz, TX2 `hardwaregain 0 dB` (full scale).
At full scale the RX ports sit 54–78 dB below ADC full scale depending on LO, far inside
the AD9361's +2.5 dBm pin limit.

### 4.3 Passive parts

30 dB attenuator, one SMA tee, two coax cables. Record any change — the residual under
test is a property of this exact assembly.

## 5. Software setup

Stock RC17, `device-fw v0.38-plutoplus-spf-gain-series-v4-rc16-7-g1f3fe`. Userspace only:
`iio_attr` for LO, gains and RSSI; sysfs for the DDS enable. No register writes at all
beyond the DDS tone enable, and no GPIO.

Script: [`scripts/hcp1_coupling.sh`](scripts/hcp1_coupling.sh), driven by
`agc1_mkjson.py` from the E-AGC1 scripts directory.

- 12 LOs: 433 / 700 / 1000 / 1300 (low), 1500 / 2400 / 3200 / 4000 (middle),
  4300 / 5000 / 5500 / 5900 MHz (high)
- fixed arm at 41 dB; swept arm at 20 / 30 / 40 / 50 / 60 / 70 dB
- both sweep directions per LO → 24 coupling figures
- LO, TX gain, DDS state and both RX gains restored from an `EXIT INT TERM HUP` trap

## 6. Outputs

`/mnt/qnap01/mouse9911/spf/calibration_data/raw/harness_coupling/<run>/` for raw (gitignored); [`RESULTS.md`](RESULTS.md) and
`coupling.json` committed.

| Artifact | Content | Acceptance gate |
|---|---|---|
| `coupling.json` | per LO and direction: fixed-arm RSSI spread, swept-arm signal level, SNR | all 12 LOs, both directions, source level recorded per LO |
| `RESULTS.md` | per-band coupling against `A`'s own band profile | states the band comparison explicitly, not pooled |
| restore proof | LO, TX gain, DDS raw, both RX gains back to entry values | all match |

## 7. Decision rule

Pre-registered, judged on the **band profile** rather than the absolute number.

| Result | Reading | Consequence |
|---|---|---|
| coupling frequency-flat and ≲1.5 dB | H1 — the tee is not `A`'s source | `A` stays a device-plus-assembly property; E-GSC6 may run on this bench with the caveat recorded; E-GSP2 stays worth doing as the definitive A/B but drops in urgency |
| coupling rises above 4 GHz, tracking `A` | H0 — the tee is implicated | hold E-GSC6 for a divider; restate every per-band `A`; E-GSP2 becomes blocking |
| coupling ≳3 dB anywhere | the harness dominates | stop quoting `A` as a device property until re-measured on a divider |

## 8. Risks

| Risk | Why it matters | Check |
|---|---|---|
| **Source too weak at high LO** | Pluto TX rolls off above 4 GHz and the 30 dB pad plus tee split costs more; a weak tone makes the fixed-arm reading noise-dominated and a null meaningless | record the swept arm's own signal level and SNR at every LO and report them beside the coupling figure. Do not read a high-band null as isolation if SNR is marginal |
| **RSSI quantisation** | RSSI steps in 0.25 dB, so sub-0.25 dB coupling is invisible and a 1-LSB reading is not a measurement | report the quantum; treat ≤0.25 dB as "at the floor" rather than as a value |
| **Amplitude-only measurement** | a purely quadrature coupling moves phase with almost no amplitude change, so one LO cannot bound phase | sweep 12 LOs across 5.5 GHz so the reflection phase rotates — §3 |
| **Common-mode cancellation misread as absence** | the junction-voltage shift *does* cancel in `D`, so a naive phase measurement would understate the risk | measure per-arm RSSI, not `D` — §1.1 |
| **Single radio, single harness** | `A` is unit-specific, so one assembly is one assembly | scope the result to R17's harness explicitly; R18 is a cheap repeat |
| **Not the definitive test** | only a divider A/B separates harness from part | say so; this bounds plausibility, it does not replace E-GSP2 |
