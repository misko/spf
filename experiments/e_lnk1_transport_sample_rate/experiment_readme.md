# E-LNK1 — transport × sample-rate comparison on R18

**Status:** designed 2026-08-07, not yet run. Arms A–C are runnable on the current
setup; **arm D needs an Ethernet cable plugged into the Pluto+** (see
[§5](#5-feasibility-what-blocks-what)).
**Est. bench time:** ~45–60 min for arms A–C; +30 min and a reconfiguration for arm D.
**Radio:** R18 `1040007c4a94000211000b009186843ef2` (historical `.18`), USB port `1-1.2`.

---

## 1. Purpose

SPF's production capture path is **direct-USB**: a custom bulk protocol carrying
per-frame gain/RSSI endpoint snapshots, a sample-sequence counter and a header
CRC32. The entire V7 dataset contract depends on that metadata.

"Should we use Ethernet instead?" is really three questions that get conflated:

1. Does a different **medium** (Ethernet vs USB) move more samples?
2. Does a different **protocol** (libiio vs SPF's bulk protocol) cost throughput?
3. Does either change the **measured phase**, or the metadata the pipeline needs?

A two-arm "USB vs Ethernet" test cannot separate (1) from (2), because the two
candidate Ethernet paths differ from direct-USB in *both* medium and protocol at
once. This design separates them.

## 2. The four arms, and why four

| Arm | Medium | Protocol | Available now? |
|---|---|---|---|
| **A — direct-USB** | USB 2.0 | **SPF bulk** (`PlutoDirectUsbReceiver`) | ✅ production path |
| **B — IIO/USB** | USB 2.0 | libiio USB backend (`usb:1.67.5`) | ✅ |
| **C — IIO/IP over RNDIS** | **USB 2.0** (`usb0` gadget) | libiio IP | ✅ needs address disambiguation |
| **D — IIO/IP over Ethernet** | **RJ45** (`eth0`) | libiio IP | ❌ **no cable** (`carrier = 0`) |

Read across the table:

- **A vs B** isolates the **protocol** — same cable, same medium, same silicon.
- **C vs D** isolates the **medium** — same protocol, same stack, different wire.
- **B vs C** isolates the **IP/network stack** on an unchanged medium.

> **The naming trap this design exists to avoid.** On this host today, "Ethernet"
> means arm C, and **arm C is not Ethernet** — `eth1`/`eth2` are `rndis_host`
> USB-gadget interfaces, so IP traffic to 192.168.2.1 rides *the same USB cable* as
> arm A. Comparing A against C and calling it "USB vs Ethernet" would measure the
> IP stack and attribute the result to the wire. **Only arm D is Ethernet**, and it
> is the one that is not currently cabled.

## 3. Hypotheses

**H1 (medium).** USB 2.0 High Speed caps at ~35–40 MB/s in practice. Two RX
channels of 16-bit I/Q is 8 bytes per sample instant, so **~4.4–5.0 MS/s is the
continuous-streaming ceiling on any USB-borne arm (A, B, C)**. Gigabit Ethernet
(arm D) has ~8× the headroom and should reach the AD9361's full 61.44 MS/s, making
it the only arm that can stream at the 30 MS/s SPF actually configures.

**H2 (protocol).** At rates below the medium ceiling, A and B should be
indistinguishable in throughput; A's advantage is metadata, not speed.

**H3 (phase, the one that matters).** The measured `angle(RX1) − angle(RX2)` on the
loopback tone is a property of the silicon and the harness, **not** of how the
samples reach the host. All arms must agree within the equal-gain repeatability of
this fixture. **Any transport-dependent phase difference is a defect**, and would
invalidate that transport for SPF regardless of its throughput.

## 4. Metrics, per (arm × sample rate)

| # | Metric | Why |
|---|---|---|
| 1 | Achieved vs requested MS/s, sustained | the headline |
| 2 | **Sample-gap rate** | integrity, not speed |
| 3 | Per-buffer latency p50 / p95 / max | jitter matters for the rover path |
| 4 | Host CPU %, bytes/s on the link | is the Pi the bottleneck? |
| 5 | **Phase `RX1−RX2` on the loopback tone**, mean and circular std | H3 — the SPF acceptance test |
| 6 | **Metadata carried** (binary) | decides deployability |

**Metric 2 has a built-in asymmetry that is itself a result.** Arm A can detect
dropped samples *exactly*, via the direct-USB `SAMPLE_SEQUENCE` counter and
`HEADER_CRC32`. Arms B–D have **no equivalent** — libiio hands over a buffer with
no statement about what happened between buffers. For those arms, gaps must be
inferred from a continuous-tone phase-continuity check (a dropped sample shows up
as a phase step at the known tone frequency). Report the inference as an inference.

**Metric 6 is likely to decide the experiment before throughput does.** Direct-USB
carries `GAIN_ENDPOINT_SNAPSHOTS`, `GAIN_DB_ENDPOINTS`, `RSSI_ENDPOINT_SNAPSHOTS`,
`SAMPLE_SEQUENCE`, `HEADER_CRC32`, `FPGA_GAIN_EVENTS`. The libiio paths carry
**none** of them. Every V7 capture asserts `gain_metadata_valid` and
`rssi_metadata_valid` per frame; a transport that cannot supply those cannot feed
the existing pipeline, however fast it is.

### Sample-rate ladder

`0.521, 1, 2, 5, 10, 15, 20, 30, 40, 50, 61.44` MS/s — AD9361 minimum, the
production rate (**30**), and the maximum. 11 points × 4 arms × 3 repetitions.

Each cell: fixed LO (5766 MHz, high band), RX gains fixed at 26/26, `rx_rf_bandwidth`
tracking the rate, 65536-sample buffers, N buffers per repetition sized so each cell
runs ≥5 s. Randomise cell order within each repetition and interleave arms, so
thermal drift cannot alias onto a transport.

## 5. Feasibility: what blocks what

Three concrete blockers, all discovered while designing this:

**5.1 Arm D has no cable.** The Pluto+ exposes a real `eth0`, but
`/sys/class/net/eth0/carrier = 0` and `speed = -1`. Arm D needs an RJ45 run to a
switch on the capture host's network, plus a static address on the Pluto
(`fw_setenv ipaddr`) — **a persistent, reboot-surviving change to a calibrated
radio**. Record the before/after `fw_printenv` and revert it afterwards.

**5.2 Both Plutos answer at 192.168.2.1.** The host holds 192.168.2.10 on *both*
`eth1` and `eth2`, so `iio_info -s` discovers only one IP context — R17's — and
R18 is unreachable over IP by address alone. Two options, in preference order:

- **Reuse the repo's existing isolation.** `spf/scripts/pluto_multi_firmware.py`
  already moves a chosen radio's USB-network interface into a private network
  namespace for exactly this reason (it is what the gain-table audit uses). Needs
  root; leaves no persistent change.
- Change R18's `ipaddr`. Simpler, but persistent — and it is also what arm D needs,
  so if arm D runs, do both at once and revert once.

**5.3 R18 is half of today's calibrated fixture.** Its
TX2 → 30 dB → splitter → RX1/RX2 loopback is still connected and unchanged since
E-CAL1 arm 1 — which is *why* metric 5 is cheap and meaningful here. Sample-rate
changes are chip-level and harmless. But arm D's `fw_setenv` requires a reboot, and
a reboot invalidates `/run/spf/direct_usb_ready.json` (as re-enumeration did this
morning), so plan to regenerate the readiness manifest. **A Pluto reboot does not
disturb RF connectors**, so the harness comparability that E-CAL1/E-GSP7/E-CAL5
rest on survives — but confirm with a pre/post gain-table audit anyway.

## 6. Hardware setup

```text
+------------------------ PLUTO R18 (1040007c4a94...) ------------------------+
|                                                                             |
|   TX2 o---> [ 30 dB attenuator ] ---> [ two-way splitter ] --+--> RX1       |
|                                                              \--> RX2       |
|                                                                             |
|   USB  o===[ arm A: SPF direct-USB bulk        ]============> host          |
|        o===[ arm B: libiio USB backend         ]============> host          |
|        o===[ arm C: RNDIS usb0 -> IIO over IP  ]============> host  (eth1)   |
|                                                                             |
|   RJ45 o---[ arm D: real eth0 -> IIO over IP   ]------------> switch --> host|
|            ^^^ NOT CURRENTLY CABLED (carrier = 0)                           |
+-----------------------------------------------------------------------------+
```

Unchanged loopback fixture, so metric 5 needs no re-cabling and no new parts.

## 7. Software

Existing building blocks:

- `spf/sdrpluto/test_throughput.py` — already benchmarks effective sample rate from
  a URI, and already applies the RX1/RX2 phase-inversion fix. **Insufficient as-is**:
  it times N buffers with no repetitions, no drop detection, no latency
  distribution and no phase metric. Extend rather than replace.
- `spf.sdrpluto.direct_usb_receiver.PlutoDirectUsbReceiver` for arm A.
- `spf.bench.dual_rx_phase.analyze_common_tone` for metric 5 — the same analyzer the
  calibration path uses, so the phase numbers are directly comparable to E-CAL1's.
- `spf/scripts/pluto_multi_firmware.py` for the namespace isolation of §5.2.

Outputs go to `artifacts/dual_rx_gain_frequency/e_lnk1_<session>/` (gitignored) as
one row per (arm, rate, repetition); committed analysis to
`reports/e_lnk1_transport_20260807_v1/`.

## 8. Decision rule

Pre-register before running.

| Result | Conclusion |
|---|---|
| **Any arm shows a phase difference beyond fixture repeatability** | That transport is **disqualified for SPF** regardless of throughput. Investigate before anything else — it would mean transport-dependent corruption. |
| A ≈ B in throughput below the USB ceiling | Protocol costs nothing; direct-USB's advantage is metadata alone, and there is no throughput reason to leave it. |
| C ≈ A/B and D ≫ C | The medium is the limit, as H1 predicts. Ethernet is the only path to streaming above ~5 MS/s. |
| D ≫ C **and** metadata gap is closable | Ethernet becomes worth engineering: scope the work to carry gain/RSSI/sequence over IP. |
| D ≫ C **and** metadata gap is not closable | **Ethernet is a dead end for V7 capture** — record the ceiling, keep direct-USB, and revisit only if the V7 contract changes. |
| No arm sustains 30 MS/s | Confirms that SPF's finite-buffer capture model is a *requirement*, not a convenience — worth stating explicitly in the docs, since it is currently implicit. |

## 9. Risks

| Risk | Handling |
|---|---|
| Reboot for arm D invalidates the readiness manifest | Expected; regenerate as on 2026-08-07. Pre/post gain-table audit to prove the tables and harness are untouched. |
| `fw_setenv` persists across reboots on a calibrated radio | Record `fw_printenv` before and after; revert explicitly; treat as a logged configuration change. |
| Namespace isolation needs root and can leave a stale netns | The existing helper restores it; verify `ip netns list` is clean afterwards. |
| Thermal drift aliasing onto arm order | Randomise cell order and interleave arms within each repetition. |
| High rates may fail to set exactly | Assert the read-back `sample_rate` per cell and record achieved vs requested; AD9361 quantises. |
| Comparing against R17 | Out of scope — this is a single-radio transport study. Do not infer fleet behaviour from it. |
