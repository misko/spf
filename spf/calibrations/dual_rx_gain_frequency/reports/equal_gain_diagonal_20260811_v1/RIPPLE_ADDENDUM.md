# E-GSC6 addendum — the diagonal's frequency structure reproduces the campaign's harness delays

Derived from the same 2026-08-11 capture; no new data. Motivated by E-CAL4 and E-GSP1/2,
whose shared hypothesis is that the gain-dependent phase is a **standing wave**: an LNA state
change alters `Γ_RX`, and against a mismatched source the round trip contributes a phase
periodic in frequency with period `1/τ`.

## Method, and the filter that makes it honest

For each (band, audited LNA state), take the mean `D(g,g)` at each LO and fit
`a·cos(2πfτ) + b·sin(2πfτ) + c` over a 4,000-point delay grid from 0.1 to 8 ns.

**A naive version of this is meaningless**, and the first pass proved it: with 8 unevenly
spaced LOs per band and a free delay, one fit returned τ = 0.100 ns — the grid edge — with a
726° amplitude. Fitting three parameters plus a frequency to 8 points will always find
something.

So each band's **resolvable delay window** is computed from the LO set actually used and
enforced:

| Band | span | max gap | resolvable τ |
|---|---:|---:|---|
| low | 867 MHz | 150 MHz | 1.15 – 3.33 ns |
| middle | 2650 MHz | 400 MHz | 0.38 – 1.25 ns |
| high | 1800 MHz | 300 MHz | 0.56 – 1.67 ns |

Lower bound: at least one full cycle must fit across the span (`τ > 1/span`). Upper bound: a
period shorter than twice the largest gap aliases (`τ < 1/(2·max_gap)`). **Any best fit
outside its window is discarded as unresolvable — 17 of 24 were.**

## Survivors

| Radio | Band | LNA | fitted τ | period | amplitude | R² |
|---|---|---:|---:|---:|---:|---:|
| R17 | low | 0 | 2.577 ns | 388 MHz | 0.26° | 0.637 |
| R17 | low | 1 | **2.546 ns** | **393 MHz** | 4.24° | **0.944** |
| R17 | low | 2 | **2.508 ns** | **399 MHz** | 5.11° | **0.942** |
| R17 | high | 3 | 1.299 ns | 770 MHz | 70.22° | 0.862 |
| R18 | middle | 0 | 0.959 ns | 1042 MHz | 0.31° | 0.834 |
| R18 | middle | 2 | 1.076 ns | 930 MHz | 0.81° | 0.835 |
| R18 | high | 1 | 1.489 ns | 672 MHz | 4.84° | 0.721 |

**The campaign's independently fitted harness delays are 2.54 ns (392.5 MHz) and
0.88–0.92 ns (~1.1 GHz).**

- **R17's low band reproduces 2.54 ns three times over** — LNA states 0, 1 and 2 give 2.577,
  2.546 and 2.508 ns, all within 0.04 ns of the published value, with R² 0.94 on the two
  states that carry real amplitude. These data never entered that fit.
- **R18's middle band lands in the second family**: 0.959 and 1.076 ns against 0.88–0.92 ns.
  Same neighbourhood, less exact.

## What this does and does not establish

**Supports** the standing-wave mechanism that E-GSP1/E-GSP2 and the whole frequency half of
the gain-state model rest on. Reproducing an independently fitted delay from held-out
diagonal cells, in three LNA states at once, is not something noise does.

**Does not answer E-CAL4.** E-CAL4 asks whether the *arm asymmetry* is a cable-length
difference, and its design requires a VNA-characterised jumper inserted on one arm with an
ABABA sequence plus an RX1↔RX2 cable swap. This addendum fits the frequency structure of
`D(g,g)`; it does not compare arms and it changes no path length. It supplies a prior, not a
substitute.

**Underpowered by design.** 8 LOs per band was chosen so band *means* would average over the
392.5 MHz ripple — the requirement for E-GSC6's threshold. That is a weaker requirement than
resolving a delay, which is why two thirds of the fits had to be thrown away. A proper delay
measurement wants the denser comb E-GSP2 specifies.

**The two high-band survivors sit at the top of their window** (1.299 and 1.489 ns against a
1.67 ns ceiling) and should be treated as the least trustworthy rows here.
