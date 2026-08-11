# E-LNK1 — results (2026-08-07)

**Radio:** R18 `1040007c4a94000211000b009186843ef2` · **Raw data:**
[`reports/e_lnk1_transport_20260807_v1/`](../../spf/calibrations/dual_rx_gain_frequency/reports/e_lnk1_transport_20260807_v1/throughput_sweep.json)
· 132 cells (11 rates × 4 arms × 3 reps), arms interleaved within each rate.

---

## Answer: Ethernet holds up **exactly** as well as USB — and no better

Both hit the **same ~23 MB/s wall**, so the bottleneck is not the link.

**Tuned comparison — 30 MS/s requested, 524288-sample buffers, 3 reps:**

| Arm | Sustained | = MS/s |
|---|---|---|
| **iio-eth** (real Gigabit RJ45) | **23.27 ± 0.21 MB/s** | 2.909 |
| **direct-usb** (SPF bulk, USB 2.0) | **23.33 ± 1.86 MB/s** | 2.916 |
| iio-rndis (IP over the USB gadget) | 13.99 ± 0.09 MB/s | 1.749 |

Ethernet and direct-USB are **statistically indistinguishable**. Two entirely
different media and two different protocols landing on the same ceiling means the
limit sits **upstream of the transport** — the radio's Zynq/DMA path or the host —
not the wire. Gigabit Ethernet is delivering **~20% of its wire speed**, so there
is no headroom being lost to the cable.

## The trap this design was built to avoid

At SPF's **production** buffer size (65536), Ethernet *looks* 28% faster:

| Requested | direct-usb | iio-usb | iio-rndis | **iio-eth** |
|---|---|---|---|---|
| 0.521 MS/s | 0.488 (94%) | 0.520 (100%) | 0.521 (100%) | **0.521 (100%)** |
| 1 MS/s | 0.869 (87%) | 0.999 (100%) | 1.001 (100%) | **1.001 (100%)** |
| 2 MS/s | 1.505 (75%) | 1.860 (93%) | 1.810 (90%) | **2.000 (100%)** |
| 5 MS/s | 2.304 (46%) | 1.836 (37%) | 1.808 (36%) | **2.882 (58%)** |
| 10 MS/s | 2.297 (23%) | 1.849 (18%) | 1.819 (18%) | **2.896 (29%)** |
| 15 MS/s | 2.314 (15%) | 1.854 (12%) | 1.822 (12%) | **2.925 (19%)** |
| 20 MS/s | 2.324 (12%) | 1.838 (9%) | 1.802 (9%) | **2.913 (15%)** |
| 30 MS/s | 2.332 (8%) | 1.831 (6%) | 1.811 (6%) | **2.942 (10%)** |

Ceilings at 65536: eth **24.2**, direct-usb **18.9**, iio-usb **15.0**,
rndis **14.6** MB/s.

**That 28% is a buffer-tuning artifact, not a property of the medium.** Raise the
buffer and direct-USB catches up completely (18.2 → 25.3 MB/s). A two-arm test at
the production buffer size would have concluded "Ethernet is 28% faster, switch to
Ethernet" — and been wrong about the reason and about the fix.

## What the four arms separate

- **Protocol, same wire:** direct-usb 18.9 vs iio-usb 15.0 MB/s → SPF's bulk
  protocol is **~26% more efficient** than libiio's USB backend.
- **Medium, same protocol:** iio-eth 24.2 vs iio-rndis 14.6 MB/s → **+66%**, and
  it confirms RNDIS is *not* Ethernet: it performs like USB because it **is** USB,
  riding the same cable.
- **Low rates invert.** Below ~1 MS/s direct-usb is *worse* (87–94% vs 100%),
  because it pays a START/STOP per 16-frame request (`max_finite_frames = 16`).
  That fixed cost dominates when there is little data, and amortises away above
  ~2 MS/s where it then wins.

## Two hard limits found

**1. The radio caps at 30 MS/s.** All 36 failures were `OSError(22)` at 40/50/61.44
MS/s, on *every* arm — a radio-side limit, not a transport one. Confirmed
independently: `sampling_frequency` maxes at 30000000, in both 1R and 2R.

**2. No transport can stream the production rate.** SPF configures 30 MS/s = 240
MB/s with 2 RX channels; the best any transport delivers is ~23 MB/s, about
**10%**. **SPF's finite-buffer capture model is therefore a requirement, not a
convenience** — this is currently implicit in the code and worth stating.

## Recommendation

**Do not switch to Ethernet for throughput.** It buys nothing once buffers are
tuned, and it would cost the entire V7 metadata contract: direct-USB carries
`GAIN_ENDPOINT_SNAPSHOTS`, `RSSI_ENDPOINT_SNAPSHOTS`, `SAMPLE_SEQUENCE` and
`HEADER_CRC32`; the libiio paths carry **none** of them, and every V7 frame
asserts that metadata is valid.

Ethernet remains attractive for *reach* — long cable runs, remote radios — not for
speed. If throughput is the goal, the lever is **buffer size** (65536 → 524288 is
worth ~+39% on direct-USB), and beyond that the bottleneck is the radio.

## Caveats

- Single radio, single host. Do not generalise to the fleet.
- Throughput only. The pre-registered **phase-agreement** check (metric 5) and the
  CPU-load metric were **not run**; a transport that matched on throughput could
  still corrupt phase, and that remains untested.
- The Ethernet arm ran on a **DHCP lease** (`192.168.1.174`), not a static address;
  the lease does not survive a reboot. Nothing on the radio was made persistent.
- Buffer-size sensitivity was measured at 30 MS/s only.

---

## Metric 5 (H3) — phase agreement across transports, 2026-08-11

**Radio:** R18 · **Committed:** `reports/e_lnk1_phase_20260811_v1/phase_agreement.json` ·
**Script:** [`scripts/lnk1_phase.py`](scripts/lnk1_phase.py)

### First pass, two arms at full drive: PASS

12 quality-valid captures, arms interleaved within each repetition so fixture drift cannot
alias onto arm:

| Arm | n | Circular mean | Within-arm spread |
|---|---:|---:|---:|
| `iio-usb` (`usb:1.86.5`) | 6 | −3.27° | 1.34° |
| `iio-eth` (`ip:192.168.1.175`) | 6 | −3.18° | 1.33° |

**Max between-arm difference: 0.089°, against 1.34° within-arm repeatability** — the arm
effect is **15× smaller than the fixture's own noise**. The §8 rule *"any arm shows a phase
difference beyond fixture repeatability → that transport is disqualified"* does not fire.
Combined with the throughput half, Gigabit Ethernet is neither faster nor phase-worse than
USB: it is simply equivalent.

### Two design points that decide whether this means anything

1. **The RX setup is copied from the calibration path attribute-for-attribute**
   (`hardware.py:120-141`), including the RX1/RX2 phase-inversion debug attribute *and*
   register 0x22 bit 6, `quadrature_tracking_en` on both channels, and
   `set_kernel_buffers_count(1)`. That fix changes measured phase directly, so an arm which
   skipped it would look like transport corruption when it is really a config difference.
2. **3 MS/s, deliberately well below the ~2.9 MS/s wall** every arm hits. Testing phase near
   the wall would let drops differ per arm and alias onto phase; at 3 MS/s every arm streams
   contiguously, isolating phase from throughput.

### A false FAIL, and what caused it

The first run reported *"FAIL — a transport shifts phase beyond fixture repeatability"* from
a between-arm difference of 0.97°. It was wrong, and the reason is worth recording: only 2–3
of 5 captures per arm were quality-valid, and on the invalid ones the analyzer had locked
onto **noise at 75–122 kHz instead of the tone at 100021 Hz** — `rx1_tone_too_weak`,
`cross_channel_coherence_low`, `within_capture_phase_unstable`. The TX tone was not on.

Cause: the harness muted TX after every measurement, so each arm re-armed the DDS and raced
its own settling. Fix: **arm the source once and leave it on for the whole run**, with each
arm applying RX configuration only. That is also better hygiene — the fixture is then
bit-identical across arms rather than rebuilt per measurement. After the fix, 12/12 captures
were valid.

The lesson generalises: a verdict computed from whatever survived a quality gate is not a
verdict. Check the *valid fraction* before reading the number.

### Final answer, three arms including the production path: PASS

Adding the direct-USB arm needed a matched probe config
(`configs/e_lnk1_metric5_probe.yaml` — 3 MS/s, 868 MHz, 41 dB, 65536 buffers, +100 kHz,
identical transient and segment counts) and one drive change, explained below. All 18
captures quality-valid:

| Arm | n | Circular mean | Within-arm spread |
|---|---:|---:|---:|
| `iio-eth` (`ip:192.168.1.175`) | 6 | −2.895° | 1.515° |
| `iio-usb` (`usb:1.86.5`) | 6 | −3.415° | 1.097° |
| **`direct-usb`** (production path) | 6 | **−3.222°** | 1.284° |

**Max between-arm difference 0.519° against 1.515° worst within-arm repeatability — a ratio
of 0.34.** All three transports agree to about a third of the fixture's own noise. The §8
disqualification rule does not fire for any of them. This matters most for `direct-usb`,
since every calibration dataset and every rover capture comes through it.

Committed: `reports/e_lnk1_phase_20260811_v1/three_arm_phase.json`.

### A second finding: the paths disagree on absolute level by ~6 dB

Not what the experiment was looking for, and worth acting on. Same physical signal, same
nominal drive, same analyzer with the same `adc_full_scale`:

| | libiio | direct-usb | offset |
|---|---:|---:|---:|
| `tone_dbfs` mean | −16.58 (n=24) | −10.63 (n=12) | **+5.95 dB** |
| `tone_snr_db` | ~34.1 | ~34.1 | — |
| phase | agrees | agrees | — |

SNR and phase agree; only the **absolute level** differs, so this is a full-scale or
sample-scaling difference between the two paths, not a signal difference.

It has a practical consequence. At full drive the direct-USB path reported −0.7 dBFS and
failed its own `rx*_tone_too_strong` gate, while libiio reported −7.4 dBFS for the same
signal and passed. So **the quality gate that decides which frames enter calibration trips
~6 dB earlier on direct-USB than a libiio-based measurement predicts.** Anyone setting drive
level from libiio numbers will be ~6 dB hot on the production path. The three-arm comparison
above was therefore run at −10 dB, where both paths sit inside their quality windows — a
valid-to-valid comparison rather than a comparison of survivors.

Which path is *correct* in absolute terms is not established here and is worth a follow-up:
it decides whether the calibration corpus has been running closer to clipping than intended.

### Still outstanding on this metric

- **`iio-rndis` — blocked.** Both USB-gadget interfaces answer on `192.168.2.1`, and
  `iio_info -s` attributes *both* contexts to R17's serial, so the arm cannot be pointed at
  R18 reliably. This is the duplicate-IP hazard; it needs the netns isolation §7 points at
  (`spf/scripts/pluto_multi_firmware.py`). Low priority: the throughput half already showed
  RNDIS at ~40% of the other arms, so it is not an SPF candidate.
- **Metric 4 (host CPU %, link bytes/s)** not collected — the least decision-relevant of
  the five.
