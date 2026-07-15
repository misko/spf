# LCSC code decisions — power_board v4.5 BOM (2026-07-14)

Spec-confirmation pass over the JLC stock proposals: every code below was
checked against live JLC attributes (voltage / dielectric / tolerance /
brand / package) and the design intent in the BOM comments +
P1_DETAIL_DESIGN.md. 92/96 lines filled; 4 left blank (see bottom).

## Deviations from the naive string-match proposals (would have been wrong)

| Line | Proposal rejected | Filled instead | Why |
|---|---|---|---|
| 10u 50V X7R (x6, buck in) | C77100 (25V!) | **C77102** GRM32ER71H106KA12L | exact Murata 50V X7R, 226k stock |
| 47u 10V X7R (x8, buck out) | C77101 (16V X5R) | **C84494** GRM32ER71A476KE15L | exact Murata 10V X7R design part |
| 100u hybrid 25V (CE1) | C2749 (THT radial!) | **C454289** EEHZA1V101P | Panasonic hybrid 35V, 8x10.2 SMD matches CP_Elec_8x10 |
| 220u poly 25mR | C2063 (THT radial 35V) | **C128504** 6SVPC220MV | Panasonic polymer 6.3V 6.3x6 SMD; **ESR 15mR vs 25mR design — see note** |
| 100n 0402 (x11) | C1525 (16V, basic) | **C131394** Yageo 50V X7R | C1/C2/C15 sit on ~13V nodes; 16V margin too thin |
| ATtiny816-SSN | C3093796 (a Renesas flyback!) | — blank | all SOIC-20 ATtiny816 variants 0 stock |
| CSD18543Q3A (x6) | C49108752 (ElecSuper clone) | **C840100** | TI original, 1247 stock |
| LM74800-Q1 | C129328 (= DRV8835 motor driver!) | **C3215600** LM74800QDRRRQ1 | correct automotive ideal-diode ctrl, WSON-12 3x3 |
| XT60_BATT | C5334533 (XT60H-M cable housing) | **C98732** XT60PW-M | PCB horizontal mount |
| USBLC6-2SC6 (D2,D3,D8) | C2687116 (UMW clone) | **C7519** | ST original |
| 15A ATO holder | C310991 (1206 chip fuse!) | **C207061** 178.6165.0002 | Littelfuse blade holder, THT |
| green status LED | C9900270145 (consigned, 0 stock) | **C2297** KT-0805G | basic tier, 3.4M stock |
| 2A polyfuse | C2319 (a PH connector!) | **C20812** SMD1812P200TF16 | RUILON 1812 2A hold / 16V |
| pi-filter 1u (L4) | no match | **C2838328** BMRA000606301R0MA1 | exact Chilisin series from footprint; Isat 22A |
| AUX 5V 2A (J6) | no match | **C474881** KF301-5.0-2P | 5.0mm-pitch MKDS-compatible clone |
| PANEL_SW / AUX_CTL (J2,J8) | garbage | **C225118** CJT A1257WV-S-2P | GH-clone vertical; JST BM02B top-entry all 0-stock |
| Pi USB headers (J9,J10,J14) | shield-frame garbage | **C705184** M3025V-1x4P | vertical THT 2.54 header |
| UPDI (J11) | consigned 0-stock | **C161691** BM03B-GHS-TBT(LF)(SN) | JST original |
| PI GPIO (J7) | RP2040 (!) | **C265111** SM08B-GHS-TB(LF)(SN) | JST original |

## Flagged for design review (deliberate deviations, all minor)

1. **220u polymer ESR**: design says 25 mΩ (comp ESR zero ~29 kHz);
   Panasonic 6SVPC220MV is 15 mΩ max → zero moves toward ~48 kHz. Type-III
   comp tolerates it, but re-check phase margin at P2 or bench (Bode).
2. **2m shunt is 2 W** (LR2512-22R002F4), name says 3 W. Actual dissipation
   at 13 A is 0.34 W — 6x margin. Accepted.
3. **CE1 hybrid is 35 V** (EEHZA1V101P) vs 25 V called out — strictly
   better; only 25 in stock (need 1). Order-day recheck.
4. **GH 2P connectors are CJT clones** (A1257WV) — verify land pattern +
   latch orientation in the JLC assembly preview against the BM02B footprint.
5. **1x4 headers (C705184)**: naming says vertical THT; confirm in preview.
6. Generic-brand TVS (SMBJ16A/5.0A) accepted — commodity clamps, specs match.

## Left blank (4 lines)

- **ATtiny816-SSN (U3)** — every SOIC-20 variant at JLC is 0-stock. Options:
  order-day recheck, consign from LCSC/Digikey, or hand-place.
- **USB-A x3 (J4,J5,J13)** — CNCTech THT part not in JLC catalog;
  hand-solder after assembly (also true for any consigned substitutes).

## Stock watchlist (thin stock at decision time, order-day recheck)

- C454289 (CE1 hybrid): 25
- C17700181 (MWSA1005S-3R3MT): 318
- C2838328 (L4 1uH): 159
- C485912 (LM5145): 584
- C226807 (16k5): 7.1k — fine, listed for completeness

Codes live in `fab/bom_jlc.csv`; regeneration preserves them via the
Comment+Footprint carry-over. Verify gate: run
`python3 ~/.claude/skills/jlcpcb-fab/scripts/jlc_stock_check.py fab/bom_jlc.csv --min-stock 5`
on order day.

## Resolutions (second pass, same day — flagged items closed)

1. **220u polymer ESR — RESOLVED: keep C128504, no comp change.** The output
   bank is 8x47uF X7R (~200uF effective at 5V) in parallel with the polymer;
   above ~20kHz the ceramics dominate the aggregate impedance, so the
   polymer's ESR-zero shift (29k->48kHz) contributes far less phase change
   at the 60kHz crossover than the single-cap model suggests (worst case
   ~13 deg, realistically a few degrees). Contingency if bring-up shows
   ringing on load steps: change RC2 (RA6/RB6) 4k64 -> 2k74, which moves
   the comp's cancel pole from 28.6kHz onto the actual ~48kHz zero. One
   resistor, no layout change.
2. **Shunt 2W — CLOSED**: 0.34W worst case at 13A, 6x margin.
3. **CE1 hybrid — CLOSED**: 35V EEHZA1V101P is strictly better than the 25V
   callout; same 8x10.2 can. Stock contingency: LCSC has C454289 in stock
   (consign), Mouser carries the ZA series (EEH-ZA1V101P). Do NOT
   substitute the ZC-series 6.3mm cans — land pattern mismatch.
4. **GH 2P clones — RESOLVED: use the CJT clone (C225118).** JST original
   BM02B-GHS-TBT is unobtainable everywhere near-term: Mouser 0 (backorder
   Aug/Sep 2026), DigiKey 0 (20-week lead), LCSC 0. CJT A1257WV is a
   dimensional GH clone; verify in the assembly preview. If JST-original
   matters for harness retention long-term, backorder and rework later.
5. **U3 MPN correction**: "ATtiny816-SSN" (BOM comment) is not an orderable
   MPN — the 20-pin wide-body SOIC is **ATTINY816-SN** (tube) /
   **ATTINY816-SNR** (reel). This typo is why JLC searches looked dead.
   Fix the value string in generate_schematic.py at the next regeneration.

## External sourcing for the 4 blank lines (checked live 2026-07-14)

| Part | Source | Stock | Price@1 | Note |
|---|---|---|---|---|
| ATTINY816-SN (U3) | DigiKey 9947564 | 93 | $1.11 | tube; -SNR reel variant also exists; Mouser lists it too |
| 1001-011-01101 USB-A x3 | DigiKey 3064739 | 17,260 | $0.86 | CNC Tech is not a Mouser brand; hand-solder after JLC assembly |
| BM02B-GHS-TBT (if JST orig wanted) | Mouser/DigiKey | 0 / backorder | $0.32 | clone C225118 assembles now |
| ATO 15A blade fuse (F1 cartridge) | any auto store / Mouser 576-0287015.PXCN | commodity | — | holder C207061 is on the JLC BOM; the fuse itself is user-supplied |

## U3 substitution (third pass): ATtiny1616-SN — C614136

ATTINY816-SN/-SNR remain 0-stock at JLC. Filled **C614136 ATtiny1616-SN**
(15 stock): same tinyAVR 1-series, identical SOIC-20 pinout, peripheral
and register compatible — a strict superset (16KB/2KB vs 8KB/512B).
Deep-stock alternate: C145558 ATTINY1616-SFR (313, 125C grade, 16MHz max —
fine at 3.3V). Follow-ups when convenient:
- firmware: compile target -mmcu=attiny1616 (no code change expected;
  verify UPDI programming + fuse config on first board)
- generate_schematic.py U3 value at next regen:
  "ATtiny1616-SN (SOIC-20, 816-compatible)" — also fixes the "-SSN" typo
- BOM Comment still says "ATtiny816-SSN" until that regen; the LCSC code
  is what JLC assembles

## Fourth pass: BOM merged one-line-per-part + R28 NTC correction

JLC's uploader warned "multiple lines matched to same part" — the BOM
grouped by value-comment, so e.g. 11 different "100n <role>" lines all
carried C131394. Export script now merges lines by (LCSC, footprint):
96 -> 58 lines, zero duplicate codes, 132 refs unchanged.

The merge display exposed a REAL spec-pass error: R28 "10k NTC 3380K" is
a THERMISTOR (3380K = beta), not a resistor — it had been coded C60490
(plain 10k) and would have shipped a resistor as the temperature sensor.
Corrected to **C77131 Murata NCP15XH103F03RC** (0402 NTC 10k ±1% B3380,
160k stock). Lesson: first-token value matching ("10k ...") cannot see
part-CLASS differences; any comment naming a special class (NTC, fuse,
ferrite) needs a class-aware look.
