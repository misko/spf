# E-HCP1 results — the tee is not the source of `A`

**Run:** 2026-08-11, R17 `104000bac4950008230026001b440a003a`, ~3 minutes.
**Firmware:** stock RC17 `…-rc16-7-g1f3fe`. **Harness:** unchanged —
TX2 → 30 dB pad → bare SMA tee → RX1/RX2.
**Raw:** `artifacts/harness_coupling/20260811_r17_v1/` (gitignored) ·
**Committed:** [`coupling.json`](coupling.json)

## Answer: coupling is small and **frequency-flat**, while `A` is not

24 coupling figures — 12 LOs × both sweep directions:

| Band | Coupling max | Coupling median | Published `A` mean | Source level |
|---|---:|---:|---:|---:|
| low ≤1300 MHz | 1.00 dB | **0.75 dB** | 0.73° | 54–59 dB below FS |
| middle 1301–4000 | 0.50 dB | **0.38 dB** | 1.24° | 61–69 dB |
| high >4000 MHz | 1.25 dB | **0.50 dB** | **3.72°** | 70–78 dB |

**Overall: max 1.25 dB, median 0.50 dB, n = 24.**

`A` rises about **5×** from the low band to the high band. Coupling does not — it is flat
to within the measurement floor, and its *median* is actually lowest in the middle band and
highest in the low band, the opposite of `A`'s profile. **Whatever concentrates `A` above
4 GHz, it is not this tee.**

### What the amplitude bound says about phase

A reflection adds `ε·e^{jθ}` to the fixed arm's signal: amplitude moves as `ε·cosθ`, phase
as `ε·sinθ`. A single LO cannot separate them — a quadrature-dominated coupling would show
almost no amplitude change. The frequency sweep is what closes that gap: the reflection
phase rotates with a ~392 MHz period, so across 12 LOs spanning 5.5 GHz θ takes many
values, and if `ε` were large some LO would have shown it.

The largest amplitude excursion anywhere is 1.25 dB, giving **ε ≲ 0.155** and a worst-case
phase coupling of **≲8.9°**, with the typical case at the 0.50 dB median corresponding to
about **3°**. That is an upper bound, not a measurement.

## Decision-rule outcome

The pre-registered rule fires on the **first** row: *"coupling frequency-flat and ≲1.5 dB →
H1 — the tee is not `A`'s source."* Consequences as written:

- **`A` stays a device-plus-assembly property.** The 2026-08-10 harness entry's stronger
  reading — that `A` may be largely harness — is not supported. It is corrected in
  `docs/learnings.md` rather than left standing.
- **E-GSC6 may run on this bench**, with the bound recorded. Its target quantity `C(g,g)`
  is expected to be a few degrees or less from the harness, against per-band `A` of
  0.73 / 1.24 / 3.72°. In the **low band** those are the same size, so a low-band `C(g,g)`
  near 1° should not be over-read; in the high band the harness term is comfortably below
  `A` and the experiment is clean.
- **E-GSP2 stays worth doing** as the definitive tee-versus-divider A/B, but it is no
  longer urgent and no longer blocks anything.

## What this does *not* settle

- **Amplitude only.** Phase is bounded through the frequency argument above, not measured.
  A divider A/B remains the definitive test.
- **The high band has the weakest evidence** — exactly where it matters most. The source is
  ~20 dB weaker above 4 GHz (Pluto TX rolloff plus pad and split), so high-band SNR is
  19–27 dB against 37–43 dB in the low band, and the 0.25 dB RSSI quantum is a larger share
  of each reading. The 1.25 dB point at 5000 MHz sits at the lowest SNR of the whole sweep
  (18.8 dB) and should be treated as the noisiest, not the most alarming.
- **One radio, one assembly.** `A` is unit-specific (cross-radio ρ +0.50 / +0.59 / −0.23),
  so this scopes to R17's harness. R18 is a ~3-minute repeat.
- **It does not explain the 2026-07-30 connector incident**, where one radio's high-band
  mean `|A|` went 3.49° → 29.41° and never recovered. A 1.25 dB coupling bound cannot
  produce 29°, so that remains unexplained and remains the strongest single piece of
  evidence that *something* in the harness can dominate. A degraded connector is not the
  same failure mode as finite isolation.

## Next, cheaply

Swapping the 30 dB pad for ~10 dB would recover ~20 dB of source level and firm up the
high band — **the same swap that would provoke E-AGC1's one unprovoked detector bit
(CH1 large-LMT)**. Do not remove the pad entirely: at TX full scale with no pad the RX
ports would see roughly +4 dBm, over the AD9361's +2.5 dBm limit.
