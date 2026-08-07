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
