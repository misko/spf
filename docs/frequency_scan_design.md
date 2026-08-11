# Fast N-frequency power scanning on the Pluto+

**Question.** Scan N frequencies, each with a specified bandwidth, dwelling Y ms, and
return received power per frequency as fast as possible.

**Status:** design, 2026-08-11. Every number below is measured on R18
(`1040007c4a94…`, RC17 `…-rc16-7-g1f3fe`) over USB, not taken from a datasheet.
Benchmarks: [`scripts/scan_bench.py`](../spf/scripts/scan_bench.py),
prototype: [`scripts/scan_proto.py`](../spf/scripts/scan_proto.py).

---

## 1. The measurements that decide the design

All on a **persistent** libiio context. The 67 ms/call figure recorded elsewhere in
`learnings.md` is process-spawn cost — a scanner must never spawn a process per operation.

| Primitive | p50 | p95 |
|---|---:|---:|
| `rssi` read | **0.54 ms** | 0.57 ms |
| `frequency` write (LO retune) | **1.28 ms** | 2.28 ms |
| `fastlock_recall` | **0.65 ms** | 0.72 ms |
| `fastlock_load` (16 bytes from host) | 1.18 ms | 1.24 ms |
| `rf_bandwidth` write, **value changed** | **14.34 ms** | 16.95 ms |
| `rf_bandwidth` write, value unchanged (no-op) | 0.46 ms | 0.50 ms |

Derived, end to end:

| Quantity | Measured |
|---|---|
| **Per-frequency floor** (LO write + 1 RSSI read) | **1.88 ms → ~505 frequencies/s** |
| Flat from N=10 to N=200? | yes — 1.976 / 1.881 / 1.867 ms |
| Depends on frequency spacing? | **no** (0.1 MHz and 5 MHz steps identical) |
| Depends on sample rate? | **no** (2.75 / 2.68 / 2.52 ms at 3 / 10 / 30 MS/s) |
| Dwell Y ≤ ~1.9 ms | **free** — Y=0 and Y=1 ms both give 1.88 ms/freq |
| Dwell Y = 5 ms | 5.14 ms/freq → 191 freq/s |
| AD9361 hardware LO+RSSI settle | **below the 1.9 ms measurement floor** |
| `rf_bandwidth` settable to | **56 MHz** |
| Sample rate | 30 MS/s confirmed; 61.44 MS/s rejected (`EINVAL`) |
| Fastlock profiles | **8**, all readable via `fastlock_save`, writable via `fastlock_load` |

### 1.1 The two facts that matter most

**(a) A host-driven scan is bound by control-path latency, not by the radio.** The 1.88 ms
floor decomposes exactly as `1.28` (LO write) + `0.57` (one RSSI read) = 1.85 ms, and the
measured minimum was 1.96 ms. Raising the sample rate 10× changed nothing, and frequency
spacing changed nothing. **The AD9361 is settled before the first RSSI read completes** —
its true settle is below what this measurement can see. Every microsecond saved must come
from removing USB round-trips, not from tuning the radio.

**(b) Changing bandwidth per frequency is catastrophic.** `rf_bandwidth` triggers an
AD9361 baseband filter re-calibration costing **14.34 ms** — 7.6× the entire per-frequency
budget. For N=100 with varying bandwidths that is 1.62 s instead of 188 ms, an 8.6×
penalty for one attribute.

## 1.2 Speed model for N frequencies at Y ms each

Validated to within ±3% against measurement at Y = 0, 0.5, 1, 2, 3, 5 and 10 ms
(`spf/scripts/scan_bench.py`):

```
T_point = T_hop + T_read + ceil(max(0, Y - T_hop - T_read) / T_read) * T_read
T_total = N * T_point

T_read = 0.544 ms      one RSSI read (the USB round-trip floor is 0.498 ms)
T_hop  = 1.278 ms      full retune   -> 549 pts/s at small Y
       = 0.637 ms      fastlock recall (<=8 profiles) -> 847 pts/s at small Y
```

Useful simplification: **`T_total ≈ N × max(T_hop + T_read, Y + T_read)`**.

| N | Y=0.1 ms | Y=1 ms | Y=5 ms | Y=20 ms |
|---|---|---|---|---|
| 8 | 15 ms / **9 ms** | 15 ms / **9 ms** | 41 ms | 163 ms |
| 50 | 91 ms / **59 ms** | 91 ms / **59 ms** | 254 ms | 1.02 s |
| 200 | 364 ms / **236 ms** | 364 ms / **236 ms** | 1.02 s | 4.06 s |
| 1000 | 1.82 s / **1.18 s** | 1.82 s / **1.18 s** | 5.09 s | 20.3 s |

*(retune / fastlock where they differ)*

### The answer depends almost entirely on Y

Overhead against the ideal `N × Y` — the observation time you actually asked for:

| Y | retune pts/s | fastlock pts/s | ideal | wasted |
|---:|---:|---:|---:|---:|
| 0.1 ms | 549 | 847 | 10000 | **1722%** |
| 1 ms | 549 | 847 | 1000 | **82%** |
| 2 ms | 423 | 441 | 500 | 18% |
| 5 ms | 197 | 181 | 200 | **2%** |
| 20 ms | 49 | 49 | 50 | **2%** |

**For Y ≥ 5 ms the host-driven scan is already within a few percent of optimal.** There is
nothing to win and no reason to build firmware — you are simply integrating for as long as
you asked to. Between 1 and 2 ms the overhead is 18–82% and fastlock recovers most of it.
Below 1 ms the overhead dominates by 2–20× and only the on-device path helps.

**Fastlock is only worth the complexity for Y < ~2 ms.** Above that the two converge (at
Y=5 ms fastlock is marginally *worse*, 181 vs 197 pts/s, because the dwell rounds up to a
whole number of RSSI reads). Don't carry 8-profile bookkeeping for a long-dwell scan.

### On-device projection

Removing the two USB round-trips per point leaves the radio-work component measured from
the host — 780 µs for a full retune, 139 µs for a fastlock recall — plus an SPI RSSI read
assumed at ~50 µs. **That assumption is unverified and needs on-device timing.**

| Y | on-device retune | on-device fastlock | gain vs best host |
|---:|---:|---:|---:|
| 0.1 ms | 1205 pts/s | **5263 pts/s** | 6.2× |
| 0.5 ms | 1205 | 1818 | 2.1× |
| 1 ms | 952 | 952 | 1.1× |
| 5 ms | 198 | 198 | 1.1× |

Note what this says: on-device, the **full retune caps at ~1200 pts/s** because its 780 µs
is real chip and driver work that USB removal does not touch. The large on-device win comes
specifically from **fastlock**, whose 139 µs is just SPI writes. So an on-device sequencer is
only worth building if it uses fastlock profiles *and* Y is below about 1 ms.

## 2. Recommended architecture

### Rule 1 — set one wide analog bandwidth, once, and never change it

Set `rf_bandwidth` to at least the widest requested bandwidth (up to 56 MHz) at scan setup
and leave it alone. Synthesise each frequency's requested bandwidth **digitally**, by
summing FFT bins. This is strictly better than using the analog filter:

- it costs nothing per frequency instead of 14.34 ms;
- the bandwidth is *exact and arbitrary*, not quantised to filter settings;
- several different bandwidths can come from one capture.

The only price is noise: a wider analog filter admits more out-of-band power before the
ADC, so the ADC must be kept out of overload (§3).

### Rule 2 — group frequencies that share a tune

This is the largest algorithmic win and needs no firmware work. Any requested frequencies
whose passbands fall inside one instantaneous span (~25–30 MHz at 30 MS/s) can be covered
by **one** tune and one capture, then separated by FFT. Cost goes from `N × 1.88 ms` to
`groups × (1.28 ms + Y + processing)`.

For 100 frequencies spread across 100 MHz:

| Approach | Time |
|---|---:|
| Per-frequency hop, bandwidth changed each time | ~1.62 s |
| Per-frequency hop, fixed bandwidth | 188 ms |
| Grouped (4 tunes at 25 MHz), on-device integration | **~10 ms + 4·Y** |

Only genuinely isolated frequencies should fall back to a hop-and-read.

### Rule 3 — for isolated frequencies, hop and read RSSI; do not stream

The AD9361's RSSI is **input-referred**, so power comes back without moving a single IQ
sample and the ~2.9 MS/s transport wall (E-LNK1) never applies. 1.88 ms per frequency,
~505/s. Free dwell up to ~1.9 ms.

### Rule 4 — put the sequencer on the device (the real speedup)

Everything above is still paying ~1.9 ms of USB round-trip per frequency. This repository
already builds custom Pluto firmware with its own USB gadget and a versioned protocol
(direct-USB protocol v3), so a **scan command** is a natural extension:

- host posts a descriptor: list of `{center_hz, bandwidth_hz, dwell_us}` plus gain policy;
- firmware runs the hop → dwell → measure loop locally, where an LO change is a handful of
  SPI writes and a fastlock recall is one, i.e. **tens of µs instead of 1.9 ms**;
- firmware returns a compact table of N powers — kilobytes, one transfer.

Add **on-device power integration** (accumulate `|x|²`, or an FFT with per-bin
accumulation) so a wide-BW group returns only the requested numbers. That is what makes
Rule 2 cheap: without it, host-side FFT must transfer `Y × fs` samples per group, which at
30 MS/s and Y=1 ms is 240 kB ≈ 10 ms of transfer — worse than the RSSI path it replaced.

Use the 8 fastlock profiles for the most-visited frequencies. From the host `fastlock_load`
(1.18 ms) is no cheaper than a retune (1.28 ms) so it is pointless there; on-device it is
16 SPI writes and becomes worthwhile for arbitrary N.

**First thing the firmware work must measure:** the true hardware settle after a hop. It is
below this document's 1.9 ms floor and is the actual limit on scan rate once USB is out of
the loop. Everything else here is already characterised.

## 3. Getting the power right, not just fast

- **RSSI is only a valid input-power estimate while the ADC is out of overload.** Measured
  in E-HCP1: RSSI is constant below overload and rises 1:1 with gain above it, so an
  overloaded reading silently understates the input. A scanner must either keep headroom or
  flag the bin.
- **Use the CTRL_OUT detector bits to flag validity.** E-AGC1 characterised all 8 of them:
  with `0x035 = 0x03` and `0x036 = 0xFF`, the per-channel low-power / small-ADC /
  large-ADC / large-LMT flags are readable, the low-power threshold `0x114` is programmable
  at **0.5 dB/LSB** over ~30 dB, and the hold band between low-power de-assert and
  small-ADC assert is **21–22 dB**. That is exactly the machinery for marking each result
  `valid` / `clipped` / `below floor` without a second measurement. Note the ADC overload
  bits **latch until the gain changes** (E-AGC1), so they must be cleared or interpreted
  with that in mind.
- **Fixed manual gain is fastest; AGC costs a settle.** For wide dynamic range prefer a
  two-pass scan (coarse low-gain pass to find strong signals, second pass at high gain for
  quiet bins) or a cached per-frequency gain from the previous sweep, rather than letting
  the AGC converge at every hop.
- **Keep ENSM in `fdd`/RX.** E-AGC1 showed the RX path is not clocked in `alert` or
  `sleep`, so RSSI will not update there.
- RSSI is quantised to **0.25 dB**.

## 4. Constraints worth knowing before designing the API

- **Both RX channels share one LO.** There is no frequency parallelism to be had from the
  second channel — but you do get two independent power readings per hop, from two
  antennas, for one hop's cost.
- **Resolve radios by serial, never by IP or USB address.** Both change across a firmware
  load; on 2026-08-11 a RAM reload rotated the DHCP leases and one radio inherited the
  other's address (see `learnings.md`).
- **Never spawn a process per operation.** 67 ms versus 0.54 ms.
- Sample rate ≥ bandwidth is required for the FFT approach, and 61.44 MS/s was rejected —
  so the usable instantaneous span is ~25–30 MHz until the achievable maximum is
  established.

## 5. Suggested API shape

```
scan(request) -> results

request:  { bandwidth_hz_max, gain_policy, dwell_us,
            points: [ {center_hz, bandwidth_hz}, ... ] }

results:  { points: [ {center_hz, bandwidth_hz, power_dbm|power_dbfs,
                       valid, clipped, below_floor, gain_db} ],
            timing: {total_us, groups, hops} }
```

Two properties worth designing in from the start:

1. **The planner is part of the API, not the caller's problem.** Given the point list, the
   implementation should decide the grouping, the single analog bandwidth, and the tune
   order. Callers asking for 100 nearby frequencies should not have to know that this is
   4 tunes rather than 100.
2. **Return validity per point, not just power.** A number without its clipped/floor flag
   is not usable, and §3 shows the flags are nearly free.
