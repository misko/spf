# Resume state — 2026-07-30

Two independent threads ran in this session. Thread A is a new hardware project (design only,
nothing built). Thread B is a read-only CI investigation (nothing changed).

---

# Thread A — Starlink Ku receive rig

**Goal:** receive and characterize the Starlink Ku-band downlink, primarily the channel centered at
**11.325 GHz**, using PlutoPlus SDRs. Nothing built yet; this is a design + shopping state.

## Signal facts (from published literature, verified this session)

- Ku user downlink spans **10.7–12.7 GHz**: 8 channels, 250 MHz spacing, **240 MHz occupied** each.
- Channel centers: 10.825, 11.075, 11.325, 11.575, 11.825, 12.075, 12.325, 12.575 GHz.
- **11.325 GHz − 9750 MHz LO = 1575 MHz IF** (coincides with GPS L1 — handy, everything tunes it).
- OFDM: **1024 subcarriers, 234.375 kHz spacing**, 4.4 µs symbol, 0.133 µs cyclic prefix,
  **302 symbols/frame, 750 Hz frame rate**. Frame starts with time-domain PSS, then a 4QAM SSS,
  ending in a coda symbol. Only **4QAM and 16QAM** observed.
- Doppler swing ≈ **±260 kHz** at 11.325 GHz for a 550 km orbit.

Sources: Humphreys et al., *Signal Structure of the Starlink Ku-Band Downlink*
(arxiv.org/pdf/2210.11578) and *Timing Properties…* (arxiv.org/pdf/2501.05302).

## What is achievable

- **Decodable:** PHY structure — frame timing, PSS/SSS lock, Doppler, beam activity statistics,
  MCS classification, satellite ID via TLE matching.
- **Not decodable:** user payload. Encrypted, and the framing/FEC/interleaver/scrambler are
  unpublished. The wall is the undocumented stack, not cryptography.
- **Bandwidth math:** N Plutos × ~56 MHz. 6 units at 40 MHz spacing = 240 MHz with 16 MHz seams.
  With realistic ~45 MHz usable flat bandwidth, **5 units is marginal, 6 is comfortable**.
- Erasure budget: at 3 units (70% coverage) a rate-1/2 code is theoretically recoverable, rate-8/9
  is not. At 6 units the erasure problem disappears entirely.

## Hardware already owned

| Part | What it is | Role |
|---|---|---|
| `Z4BT-2R15G+` | Bias-tee/**diplexer**, 950–2150 MHz RF + 10 MHz REF port + DC, **2 A / 48 V**, all **SMA female** | The bias tee. Better than needed. Its REF port could later feed an external-ref LNB up the same coax. |
| `ZAPD-2DC-S+` | 2-way DC-pass splitter, 950–2150 MHz, ≥20 dB isolation | Stage-2 splitter, or comb injection |
| `ZSC-3-1+` | 3-way, **1–200 MHz**, BNC | **Reference distribution only — will NOT pass 1.5 GHz** |
| `BLK-89-S+` ×4 | DC block, 0.1 MHz–8 GHz, SMA | Insurance on Pluto inputs (need 6, have 4) |
| `VAT-15A+` ×3, `VAT-20A+` ×4 | Fixed attenuators | For **impedance match**, not level |
| GEOSATpro **UL1PLL** | Universal PLL LNB, LO 9750/10600, 0.5 dB NF, 60 dB gain, **500 kHz stability** | **Use this one** |
| 2× generic "Universal Single" LNBs | LO 9750/10600 | Work, but no published stability |
| GEOSATpro **SL1PLL v2** | **STANDARD** LNB, LO **10750**, input **11.7–12.2 GHz** | ⚠️ **Cannot receive 11.325 GHz.** Can reach the 11.825 and 12.075 GHz channels only. |

**LNB line item is $0** — the UL1PLL is sufficient.

## Key engineering conclusions (do not re-derive)

1. **Split loss is irrelevant.** LNB gain 55–60 dB at ~0.5 dB NF means a 6-way split costs
   **0.002 dB** of system noise figure. There is **~32 dB of margin** between the LNB's amplified
   noise at each Pluto and the Pluto's own floor. You could throw away 20 dB anywhere and not care.
   **Never add an amplifier or a "powered splitter."**
2. **No attenuator is needed for level.** Integrated power at each Pluto ≈ −46 dBm. Pads are still
   worth using — for **impedance match** (a pad raises return loss by 2× its value, which is what
   makes a cheap splitter behave like a lab part). Use **matched values across all six**.
3. **The LNB's LO error is common-mode.** One LNB feeds all receivers, so its drift and phase noise
   appear identically on every slice and cancel in cross-receiver processing. **A cheap LNB is
   correct, not a compromise.** This is why the EOL Bullseye doesn't matter.
4. **A shared 40 MHz reference is mandatory** — proven by this repo's own bench data.
   `docs/learnings.md` **L9**: two Plutos on independent crystals wander **2.6 rad of relative LO
   phase within one 8 ms buffer**, vs 0.007–0.12 rad intra-pair. 25 ppm crystals slip ~1400 samples
   inside a 0.5 s burst at 56 MSPS. **Not fixable in software.**
5. ⚠️ **The naive 40 MHz tiling violates `docs/learnings.md` L7 "Never 0-center."** Each Pluto's LO
   would land at its own slice center = IF at DC, where LO leakage, the quadrature image, the 1/f
   skirt and the DC tracking notch all live. L4 documents per-channel unlogged DC tracking loops
   producing differential drift that **destroyed a prior dataset**. For a wideband tile you cannot
   move DC out of band, so: **disable BBDC/RFDC tracking, record the static spur, remove in post**,
   and retile so each unit's DC lands in a seam its neighbour covers. **This retiling is not yet done.**
6. **L6**: gain-change phase inversion was fixed in code (commit `9d00b7b`). Run **fixed manual
   gain, AGC off** — a gain-index change causes a phase discontinuity.
7. **The comb generator can be cut.** The LNB's own amplified thermal noise is a fully-correlated,
   always-on wideband reference present in every seam — 16 MHz × 1 s = **72 dB of correlation gain**,
   works with the dish disconnected. Residual gap: per-slice interior frequency response, closed
   once offline by stepping one radio against a parked reference.
8. **The GPSDO can be cut.** You need the six clocks **identical, not accurate**. GPS buys absolute
   frequency and 1PPS; neither is used here. A $2 TCXO + $6 fanout does the load-bearing job.
9. **PlutoPlus sustains ~28 Msps over GbE, not 56.** Six Plutos never covered 240 MHz continuously —
   the design was always implicitly **burst-to-DDR**. Confirm 0.5 s at 56 MSPS (112 MB) fits a
   one-shot buffer before designing around it.
10. **No single box under ~$10k does 240 MHz.** USRP X310 tops out at 160 MHz and fails at $15,870.
    **RFSoC 4x2 is $2,499 and does it — but sales are gated to accredited universities/research
    institutes.** ← **Check whether this applies; it changes the whole calculus.**

## Antenna sizing

Size against Dishy (~33–35 dBi), not against the beacon. A bare LNB feedhorn is fine for the
narrowband beacon but **15–20 dB short for 240 MHz OFDM**.

- **60 cm offset dish ≈ 35 dBi**, ~3.1° beamwidth
- LEO at 550 km overhead → ~0.79°/s → **~4 s transit at zenith, longer at low elevation**
- Bursts are 0.5 s → **8× margin, so a FIXED dish works. No rotator, no tracking.**

Free used DirecTV/DISH dishes are abundant, but their feed arms use **proprietary LNB carriers**
while a universal LNB has a 40 mm collar — budget a bracket, or buy the Digiwave DWD60T ($78.70,
hvtg.com/products/dwd60t) which includes a compatible mount.

## Staged build plan (agreed shape)

The front end never changes; each stage only adds things after the bias tee.

**Stage 1 — first light, ~$60.** Dish + UL1PLL → RG6 → `Z4BT` → 1 Pluto.
Gets: beacon at 1575 MHz, Doppler curves, TLE-matched satellite ID, 750 Hz frame lock,
beam activity statistics. **Most of the science.**

**Stage 2 — proves coherence, ~$30.** Adds the owned `ZAPD-2DC-S+` (2-way) + a 40 MHz clock board
feeding 2 Plutos. **This is the real go/no-go gate.** Don't buy stage-3 parts until two units
stitch cleanly.

**Stage 3 — full 240 MHz.** 6-way splitter + 4 more Plutos.

## To buy

### Stage 1
| Item | ~Price | Source |
|---|---|---|
| RG6 quad-shield 50 ft, F-male both ends, **3 GHz rated** | $15–25 | Amazon (PHAT SATELLITE B01GBVYAZC — weather-sealed, preferred) |
| Dish 500+ reflector + feed arm | $13.99 | solidsignal.com ES183446 |
| 12 V or 13.8 V regulated supply, ≥300 mA | ~$10 | any (**avoid ~15 V** — band-select threshold zone) |
| **SMA-male → bare-wire pigtail** (for `Z4BT`'s SMA DC port) | ~$9 | Amazon XMSJSIY B0C6ZTJN5N |
| **F-female → SMA-male adapter** | $12.79 | showmecables.com SKU 3879 |
| SMA male–male cable, short | ~$8 | any |

Prefer a **bench supply** if available: set 13.5 V, current-limit 300 mA. The current reading is a
free diagnostic — **~150 mA means the LNB is alive and the coax is good**; 0 = open, high = short.

### Stage 2
| Item | ~Price | Source |
|---|---|---|
| 40 MHz TCXO ±2 ppm `NSC5318C-40M` | $2.03 | DigiKey |
| 1:6 LVCMOS fanout `CDCLVC1106PW` | $6.08 | DigiKey (thin stock — `PWR` reel is deeper) |
| U.FL pigtails, 10-pack | $22.00 | rfbat.com |
| Small 2-layer JLCPCB board | ~$10 | — |

PlutoPlus ext ref = **U.FL, 40 MHz** (works 20–80 MHz), max ~3 Vpp Rev C, works below 100 mV.
Enable with `fw_setenv refclk_source external` and `fw_setenv ad936x_ext_refclk_override "<40000000>"`.
Fixed skew between the six copies does **not** matter (it calibrates out) — only common lock does.

### Stage 3 — 6-way splitter, decision open
| Option | Price | Note |
|---|---|---|
| **Blonder Tongue LPD-6** | **$14.95** | 75 Ω F, 10–2150 MHz. **20 dB isolation / 14.5 dB IL at 1450–2150** — the only 75 Ω 6-way with a published isolation spec. Markertek. |
| **Mini-Circuits ZB6PD-2-S+** | **$224** | 50 Ω SMA, **800–2000 MHz**, DC-pass, in stock. Band sits mid-passband. |
| INSTOCK PD1160 | $139.92 | 50 Ω SMA, 698–2700 MHz, 18 dB isolation |

⚠️ `ZN6PD1-63-S+` ($417) was an earlier mistaken recommendation — 2× the price and wideband
(typically worse isolation). `ZN6PD1-63W+` **does not exist**.

The $132 delta between LPD-6 and ZB6PD buys elimination of ~12 F interfaces. An F center pin *is*
the cable conductor, untorqued — a marginal one produces a phase **jump** of degrees-to-tens-of-
degrees at a random time, **indistinguishable in post-analysis from a real channel event**. That is
the nastiest day-5 failure mode on a 7-day unattended run.

## Connector notes

- `Z4BT` ports are **all SMA female**, including the DC port — that's the non-obvious bit.
- Only **one** F-to-SMA transition exists in stage 1, at the `Z4BT` input.
- 75 Ω → 50 Ω mismatch costs **0.18 dB**. **Do not buy impedance-matching adapters** — plain
  between-series adapters are correct here.
- **22 kHz tone: leave OFF.** 11.325 GHz is low band. The tone is a **band switch, not a boost** —
  it never gives more bandwidth at once, it slides the window 850 MHz up. Accidental activation is
  a **silent** failure; add a per-burst in-band-power check.
- **13 V vs 18 V is moot** against a circular signal — both give the same 3 dB down and both
  polarizations summed. Use 13 V.
- Optional $0 upgrade later: the dielectric-plate mod (olegkutkov.me/2020/09/20/reworking-linear-
  polarization-satellite-lnb-into-a-circular-polarized/) converts circular→linear in the feedhorn,
  recovering the 3 dB **and** making 13/18 V a meaningful RHCP/LHCP selector.

## Open decisions

1. ⚠️ **Retile to respect L7** — current 40 MHz plan is 0-center on every unit. Must be resolved
   before stage 3.
2. **Check RFSoC 4x2 academic eligibility** ($2,499, does 240 MHz in one box). Would replace the
   whole six-Pluto + clock-distribution architecture.
3. **6-way splitter:** $15 LPD-6 vs $224 ZB6PD-2-S+.
4. **Consider the DF/interferometer variant** — same six radios on **six antenna elements, one
   sub-band**, instead of tiling the band. The literature flags satellite-assignment ambiguity as an
   open problem ("Starlink synchronization sequences are not unique to each satellite… must be
   solved combinatorially"); AoA resolves it **directly**, and it reuses existing SPF phased-array
   calibration tooling instead of new stitching code. Not cheaper — possibly better science.
5. **Co-located Starlink terminal pulling a video stream** to guarantee downlink activity. This is
   the published method and will drive data yield harder than any hardware choice here. Free.

## Storage / ingest (not yet sized)

Continuous is impossible: 6 × 56 MSPS × 4 B = 1.34 GB/s = **813 TB over 7 days**.
Triggered bursts: 0.5 s × 6 radios = **672 MB per burst** → ~6.8 TB at 60 s cadence, 1.4 TB at 300 s.
**Cadence is limited by drain, not capture** — 112 MB/radio over GbE at ~110 MB/s real ≈ 6 s through
one NIC. Also missing from any BOM so far: powered USB hub with real current headroom (~0.5 A per
Pluto), multiple USB host controllers, NVMe, and a UPS.

---

# Thread B — CI root cause analysis (read-only, complete)

**Pulled `main` fast-forward `4ebf744` → `287c83a`.** Working tree had no tracked modifications.
**No code was changed.**

## Finding

CI red for **26 consecutive runs since 2026-07-28T01:22:49Z**. One test fails of 559; lint and the
`build` job are green.

```
tests/test_mavlink_radio_collect.py::test_mavlink_radio_collect_v7_iio_schema
RuntimeError: None: V7 capture requires boot-verified firmware
```

**Root cause:** commit **`b4e00f3`** ("Add passive Pluto hardware fingerprints to V7",
2026-07-28 01:21:24Z) added a firmware-verification guard at `spf/data_collector.py:679-690` that
runs unconditionally for every receiver identity, **with no exemption for simulated receivers**.
The V7 end-to-end test runs against `fake:` URIs.

Bisection is unambiguous:

```bash
git merge-base --is-ancestor b4e00f3 59f81db9   # last green → NO
git merge-base --is-ancestor b4e00f3 dea791bb   # first red  → YES
```

First red run began **85 s** after that commit was authored. The failing test is older (added
2026-07-26, `205d0e9`) and was passing — the test didn't change, the contract under it did.

## Why it cannot self-heal

`firmware_verified` is written only by `_capture_firmware_provenance()`. Three independent blockers:

1. `tests/test_config.yaml` has no `pluto-firmware` block → provenance returns `None`.
2. No ready-manifest at `/run/spf/direct_usb_ready.json` in CI.
3. **Fake receivers have `serial=None` by construction** (`sdr_controller.py:699-707`) and the
   provenance path requires a truthy serial. **Structural, not configurable** — this is why the
   error message begins `None:`.

The guard's own tests pass (`test_sdr_identity_provenance.py`, `test_direct_usb_receiver.py`) because
they build identities directly and never traverse the fake path.

## Not yet decided / not implemented

- **Recommended fix:** exempt simulated receivers in the guard, keyed on something explicit rather
  than inferred, plus a regression test asserting the guard still fires for a non-fake receiver
  lacking provenance.
- **Arguably better:** block the coercion in `spf/capture_schema.py:52`, where
  `normalize_capture_config` **forces** `rx-transport: direct_usb` onto every V7 receiver including
  `fake:` ones. That coercion is what drags simulated devices into the direct-USB provenance path.
- A second guard (`hardware_fingerprint_v1`, line 684-690) would fail identically; currently shadowed.

## Secondary findings

- `pytest` job runs ~19 min; the failing test sits at 49% of the suite. ≈**8 hours** of self-hosted
  runner burned on this streak.
- All four GitHub Actions (`actions/checkout@v3`, `actions/setup-python@v3`,
  `docker/login-action@v3`, `docker/build-push-action@v5`) are on deprecated Node 20, force-run on
  Node 24. Not causal — a scheduled future break.

## Caveats

- **Not reproduced locally** — this machine has no `pytest` or `zarr` and no project virtualenv was
  found. Installing deps was out of scope for a read-only investigation. Static trace is complete and
  the exception is in the CI log, so confidence is high, but a local run would be conclusive.
- CI logs older than ~3 days have aged out. Same-cause across the full 26-run streak is inferred
  from identical job/step failure plus an unchanged code path, not read from every log.

Full report was written to the session scratchpad (**ephemeral**) — its content is reproduced above.
