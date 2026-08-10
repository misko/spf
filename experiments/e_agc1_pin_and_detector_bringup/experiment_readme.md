# E-AGC1 — AD9361 gain-pin and detector bring-up, from userspace

**Status:** designed 2026-08-10, **revised 2026-08-10** after a read-only bench audit
against the part itself. Not yet run.
**Closes:** open items O-1, O-2, O-3 and O-5 of `TANDEM_AGC_V1_DESIGN.md`, all of
which currently block that design's candidate freeze. ⚠ **That document is not in this
repository** — see §6.3.
**Cost:** one radio, one session. **No FPGA change, no bitstream, no RAM boot, no
QSPI write.** Stock RC17 firmware throughout.
**Duration:** ≈1–1.5 h without any RF connection (steps 1–4, 6, restore), plus
≈2–2.5 h for the detector work in step 5. H1 — the result that gates all RTL work —
lands ≈30 min in. See §5.5.

---

## 1. Purpose

A planned firmware feature ("tandem AGC") gives an FPGA block ownership of the
AD9361 `CTRL_IN` pins so it can step RX1 and RX2 gain together. Its design contract
rests on a set of device behaviours that were established from the AD9361 reference
manual, the shipped Linux driver, and the Pluto+ schematic — but **not** from the
part itself.

Three facts make it possible to check almost all of them without writing any
firmware:

1. `CTRL_IN[3:0]` and `CTRL_OUT[7:0]` are wired to Zynq EMIO GPIO — EMIO 8–11 and
   EMIO 0–7, which are device-tree `<&gpio0 62..65>` and `<&gpio0 54..61>`.
2. **Nothing in the shipped stack claims them.** No device-tree consumer, no kernel
   driver, no init script, no Buildroot package, no USB or IP gadget, nothing in
   `spf`.
3. The shipped kernel sets `CONFIG_GPIO_SYSFS=y`, `CONFIG_GPIO_ZYNQ=y` and
   `CONFIG_DEBUG_FS=y`, so those pins are drivable and readable from userspace and
   AD9361 registers are reachable through `direct_reg_access`.

So the pins can be driven by hand, the detectors can be read by hand, and the gain
index can be read back over SPI — on an ordinary radio running ordinary firmware.

What this buys, specifically: the pin mapping is currently verified only by joining
a schematic PDF to a constraints file. That is good evidence, but a transposed pair
would mean the firmware sends *decrease* when it means *increase*, or drives RX2
when it means RX1. This experiment converts that from a paper inference into a
measurement, and it takes about ten minutes.

## 2. Hypotheses

Pre-registered. H1–H5 are predictions; H6 is genuinely open and exploratory.

**H1 — pin mapping is the identity.** A rising edge on:

| GPIO (EMIO) | Predicted effect |
|---|---|
| 62 → `CTRL_IN0` | RX1 gain index **increases** |
| 63 → `CTRL_IN1` | RX1 gain index **decreases** |
| 64 → `CTRL_IN2` | RX2 gain index **increases** |
| 65 → `CTRL_IN3` | RX2 gain index **decreases** |

with no effect on the other channel in any case.

**H2 — step size.** Each accepted edge moves the index by the programmed manual
step. The shipped configuration moves **2**; writing 1 to the increment and
decrement step fields yields exactly 1 index per edge.

*Half of this is already confirmed from the registers (§5.3): as shipped
`0x0FC = 0x0FE = 0x23`, so both step fields hold `value − 1 = 1`, i.e. a programmed
step of 2. What remains to measure is whether an accepted edge actually moves the
index by that programmed amount.*

**H3 — detector page.** With `0x035 = 0x03` and `0x036 = 0xFF`, the pins carry:

| GPIO (EMIO) | `CTRL_OUT` | Predicted signal |
|---|---|---|
| 61 | 7 | CH1 ADC low power |
| 60 | 6 | CH1 large LMT overload |
| 59 | 5 | CH1 large ADC overload |
| 58 | 4 | CH1 small ADC overload |
| 57 | 3 | CH2 low power |
| 56 | 2 | CH2 large LMT overload |
| 55 | 1 | CH2 large ADC overload |
| 54 | 0 | CH2 small ADC overload |

**H4 — latch and blank.** An asserted large-overload bit stays high until the gain
changes, then returns low for at least the Peak Overload Wait Time even if the
input still exceeds the threshold.

**H5 — low-power update rate.** The low-power bit changes state no faster than one
power-measurement period. Predicted **256–410 µs** across the supported rate range.

> ⚠ **H4 and H5 are resolution-limited by this method and are pre-declared as such.**
> Measured on the target: the fastest available sysfs GPIO read is **134 µs** (shell
> builtin redirect, no process spawn); via `cat` it is **6.0 ms**. Against a predicted
> 256–410 µs period that is 2–3 samples per period at best, and blind at worst. There
> is **no compiler on the target** and `/bin/sh` is busybox, so a tight poller means
> cross-compiling and pushing a binary — outside this experiment's "stock firmware,
> userspace only" scope. Attempt H4/H5, report what is observable, and do **not**
> report a period or a blank duration as measured unless the observed interval is
> comfortably above 500 µs. Both properly belong to the FPGA stage, alongside the
> minimum-pulse-width question §8 already defers.

**H6 — ENSM dependence (open).** Whether `CTRL_IN` edges are honoured while the
ENSM is outside the RX state is unknown and undocumented. No prediction is offered.
Both outcomes change the firmware's enable sequence, in opposite directions.

## 3. Approach

Six measurements, in this order, because each depends on the previous one being
sound. Every one of them runs against a **quiescent** radio — no capture in flight.

1. **Baseline and safety.** Record starting state: gain-control mode, `0x035`,
   `0x0FB`, both gain indices, both `hardwaregain` values, ENSM state. Confirm
   `0x0FB[1:0] == 0` (pin control disarmed, as shipped).
2. **Take the pins before arming anything.** Export EMIO 62–65, set direction out,
   drive all four low, and *verify* they read back low. Only then arm `0x0FB[1:0]`
   by read-modify-write. This order is mandatory — see §8.
3. **H1, pin mapping.** For each pin in turn: read both indices, emit one pulse,
   read both indices. Repeat 5×. A pin passes only if its own channel moved in the
   predicted direction on all 5, and the other channel never moved.
4. **H2, step size.** From the H1 traces, the index delta per edge is the step.
   Then program step 1 in both directions and repeat to confirm 1.
5. **H3–H5, detectors.** Set `0x035 = 3`, confirm `0x036 = 0xFF`, and read EMIO
   54–61. Sweep the source level and record which bits assert at which level. Then
   the timing observations for H4 and H5.
6. **H6, ENSM.** Move the ENSM out of RX, pulse, read the index. Restore.

Then **restore everything** per §5.4, and confirm the restore.

Controls:

- **Direction control.** Every pulse test is run in both directions on the same
  channel, so a pin that does nothing is distinguishable from a pin wired to the
  wrong function.
- **Cross-channel control.** Every pulse records *both* indices, so a transposed
  RX1/RX2 pair is caught rather than inferred.
- **Repeats.** 5 pulses per pin, because a single missed edge and a wrong mapping
  look identical in one trial.

## 4. Hardware setup

### 4.1 Radios

**One** Pluto+ is sufficient. Two is better for H3–H5, since the detector thresholds
are per-part, but the open items this experiment closes are not per-unit.

Available as of 2026-08-10, both with the step-5 harness already attached:

| Label | Serial | USB | LAN |
|---|---|---|---|
| R17 | `104000bac4950008230026001b440a003a` | `1-1.1` | 192.168.1.165 |
| R18 | `1040007c4a94000211000b009186843ef2` | `1-1.2` | 192.168.1.175 |

Resolve by serial, never by IP: both units expose a USB-gadget interface on a
duplicate `192.168.2.10/24`, and `iio_info -s` will mis-attribute one `192.168.2.1`
context to the wrong serial.

The radio must be **idle** — no capture, no rover service, no concurrent test. Note
that RC17 hardware burn-in work is active in this repository; coordinate before
claiming a radio. Both units currently run a **volatile RC17 candidate**
(`device-fw v0.38-plutoplus-spf-gain-series-v4-rc16-7-g1f3fe`); a power cycle reverts
them to the persistent QSPI image, which is a second reason not to rely on a reboot as
the restore mechanism (§5.4).

### 4.2 Physical schematic

Steps 1–4 and 6 need **no RF input at all** — they read the gain index over SPI and
never look at signal. Step 5 (H3–H5) needs a controlled input level.

The bench in place is the standard dual-RX harness: the radio's **own TX2** through a
30 dB attenuator into a two-way splitter feeding both receive ports. There is no
external signal generator on this bench, and none is needed — the level is set
programmatically with TX `hardwaregain`, which is *better* than a step attenuator for
a threshold sweep because every level is recorded rather than written down by hand.

```
        ┌──────────── one PLUTO (per radio) ────────────┐
        │                                               │
        │   TX2 ──► 30 dB attenuator                    │
        │              │     level set in software via  │
        │              │     TX hardwaregain, −80 → 0 dB│
        │       ┌──────┴──────┐                         │
        │       │  2-way 50 Ω │  a real splitter,       │
        │       │  splitter   │  NOT a tee              │
        │       └──┬───────┬──┘                         │
        │          │       │                            │
        │       ┌──┴──┐ ┌──┴──┐                         │
        │       │ RX1 │ │ RX2 │                         │
        │       └─────┘ └─────┘                         │
        └───────────────────────────────────────────────┘
```

**Start at TX `hardwaregain = −80 dB` (muted) and step up.** The AD9361 RF pin maximum
is +2.5 dBm peak and the detector sweep deliberately drives toward overload — see §8.

For H3 the bit map is cleanest when only **one** channel is driven, so that a CH1 bit
asserting while CH2 is driven immediately falsifies the map. With this splitter both
ports are fed together, so isolate in software instead: park the undriven channel's
gain at the bottom of the table so its detectors stay quiet, and record both channels'
gains at every point. If a physical single-channel test is wanted, disconnect one
splitter leg and terminate it in 50 Ω — and record that as a harness change.

### 4.3 Passive parts and adapters

Record every part and its position. For steps 1–4 and 6, record "no RF connection".

## 5. Software setup

Stock RC17 firmware. `iio_reg` and sysfs GPIO are both already present.

### 5.1 Discover, never hardcode

The gpiochip base shifts with kernel configuration. Compute it rather than assuming
the 960/968 numbers seen on one build:

```sh
# find the zynq gpiochip and its base
for c in /sys/class/gpio/gpiochip*; do
  echo "$c $(cat $c/label) base=$(cat $c/base) ngpio=$(cat $c/ngpio)"
done
# EMIO starts at offset 54:  global = base + 54 + emio_bit
# CTRL_OUT[0..7] = EMIO 0..7   -> global base+54 .. base+61
# CTRL_IN[0..3]  = EMIO 8..11  -> global base+62 .. base+65
```

Confirm the arithmetic against a known-good reference on the same box: the device
tree uses `<&gpio0 66>` for `EN_AGC` and `<&gpio0 67>` for `RESETB`, which are
EMIO 12 and 13.

**Measured on R17, 2026-08-10 (RC17):** one gpiochip, `label=zynq_gpio`,
**`base=906`**, `ngpio=118`. The DT cell is a direct offset from the base, so
`global = 906 + dt_index`:

| Signal | EMIO | DT index | Global GPIO |
|---|---:|---:|---:|
| `CTRL_OUT[0..7]` | 0–7 | 54–61 | **960–967** |
| `CTRL_IN[0..3]` | 8–11 | 62–65 | **968–971** |
| `EN_AGC` | 12 | 66 | 972 |
| `RESETB` | 13 | 67 | 973 |

Verified against two independent anchors in `/sys/kernel/debug/gpio`:
`gpio-973 (reset)` is `RESETB` at DT 67, and the `one-bit-adc-dac` node's
`out-gpios = <&gpio0 68>` / `in-gpios = <&gpio0 71>` appear as `gpio-974 (out)` and
`gpio-977 (in)`. So the 960/968 figures are correct on this build — but still discover
and record the base, because it is a kernel-configuration artifact.

**Nothing claims EMIO 0–11**, confirmed rather than assumed: the complete set of
requested lines is `921 (led0:green)`, `952 (ulpi resetb)`, `973 (reset)`,
`974 (out)`, `977 (in)`. The `one-bit-adc-dac` driver is the only surprise consumer in
the stack and it takes EMIO 14 and 17, not the CTRL pins.

Kernel options confirmed present on the shipped image: `CONFIG_GPIO_SYSFS=y`,
`CONFIG_GPIO_ZYNQ=y`, `CONFIG_DEBUG_FS=y`. `iio_reg`, `iio_attr` and `iio_info` are
all in `/usr/bin`.

### 5.2 Sequence

```sh
# ---- 1. baseline ----
# NOTE the -i: without it, iio_attr matches the OUTPUT (TX) channel of the same
# name. `iio_attr -c ad9361-phy voltage0 hardwaregain` returns the TX1 gain
# (-80 dB on this bench), NOT RX1. Every RX channel access below uses -i.
iio_attr -i -c ad9361-phy voltage0 gain_control_mode       # record
iio_attr -i -c ad9361-phy voltage1 gain_control_mode       # record
iio_attr -d ad9361-phy ensm_mode                           # record
iio_reg ad9361-phy 0x035 ; iio_reg ad9361-phy 0x036        # record
iio_reg ad9361-phy 0x0FA                                   # record (hybrid re-arm path)
iio_reg ad9361-phy 0x0FB                                   # expect bits[1:0]=0
iio_reg ad9361-phy 0x0FC ; iio_reg ad9361-phy 0x0FE        # step fields + PWOT
iio_reg ad9361-phy 0x2B0 ; iio_reg ad9361-phy 0x2B5        # RX1/RX2 index
iio_attr -i -c ad9361-phy voltage0 hardwaregain            # dB cross-check
iio_attr -i -c ad9361-phy voltage1 hardwaregain

# put both channels in manual gain before touching pins
iio_attr -i -c ad9361-phy voltage0 gain_control_mode manual
iio_attr -i -c ad9361-phy voltage1 gain_control_mode manual

# ---- 2. TAKE THE PINS FIRST, THEN ARM ----
for g in 62 63 64 65; do
  echo $((BASE+g)) > /sys/class/gpio/export
  echo out  > /sys/class/gpio/gpio$((BASE+g))/direction
  echo 0    > /sys/class/gpio/gpio$((BASE+g))/value
done
# verify all four read 0 before proceeding -- do not skip this
# only now arm, read-modify-write, never a bare 0x03
V=$(iio_reg ad9361-phy 0x0FB); iio_reg ad9361-phy 0x0FB $((V | 0x03))

# ---- 3. one pulse ----
echo 1 > /sys/class/gpio/gpioN/value ; echo 0 > /sys/class/gpio/gpioN/value
iio_reg ad9361-phy 0x2B0 ; iio_reg ad9361-phy 0x2B5
```

Gain index readback: `0x2B0` for RX1 and `0x2B5` for RX2, masked to the full-table
gain index field — this is the register pair `ad9361_get_full_table_gain()` reads.
Cross-check against `hardwaregain` in dB, remembering that one full-table index step
is exactly 1 dB but the index→dB offset is band-dependent.

Step size fields, for H2: manual increment step is `0x0FC[7:5]`, manual decrement
step is `0x0FE[7:5]`, both stored as `value − 1`. Note that `0x0FE[4:0]` in the same
register is the Peak Overload Wait Time — read-modify-write, do not clobber it.

Detector page, for H3: `0x035 = 0x03`, and confirm `0x036 = 0xFF`.

### 5.3 Measured baseline, R17, 2026-08-10 (read-only audit)

Recorded before the run so step 1 has something to diverge from, and because three of
these values turn §8 risk rows from theoretical into concrete. RX LO was 868 MHz.

| Item | Value | What it means |
|---|---|---|
| `gain_control_mode` (both RX) | `manual` | already manual; step 1's mode write is a no-op here, but still record and restore it |
| `ensm_mode` | **`fdd`** | not `rx`. Available: `sleep wait alert fdd pinctrl pinctrl_fdd_indep`. H6's "non-RX" means `alert`/`wait`/`sleep` |
| `0x035` | `0x00` | CTRL_OUT page 0; H3 needs `0x03`, so this must be restored |
| `0x036` | `0xFF` | already the value H3 wants — confirm, don't write |
| `0x0FA` | `0xE0` | the `hybrid` re-arm path §8 warns about |
| **`0x0FB`** | **`0x08`** | bits[1:0] = 0 → pin control disarmed as shipped ✓ — **and bit 3 is set**, so a bare `0x03` write would clear it. Read-modify-write is mandatory in fact, not in principle |
| **`0x0FC`** | **`0x23`** | bits[7:5] = 1 → increment step = **2**, confirming H2's premise from the register |
| **`0x0FE`** | **`0x23`** | bits[7:5] = 1 → decrement step = 2; bits[4:0] = **3** is a live Peak Overload Wait Time that a bare write would clobber |
| `0x2B0` / `0x2B5` | `0x2C` / `0x2C` | both gain indices = **44** |
| RX1 / RX2 `hardwaregain` | 41 dB / 41 dB | low-band table row 44 *is* 41 dB, so index↔dB and its band dependence both check out |
| TX1 `hardwaregain` | −80 dB | muted; this is the value a missing `-i` returns instead of RX1 |

### 5.4 Restore — mandatory, and verified

```sh
# disarm BEFORE releasing the pins
V=$(iio_reg ad9361-phy 0x0FB); iio_reg ad9361-phy 0x0FB $((V & ~0x03))
iio_reg ad9361-phy 0x035 <original>
for g in 62 63 64 65; do echo $((BASE+g)) > /sys/class/gpio/unexport; done
iio_attr -i -c ad9361-phy voltage0 gain_control_mode <original>
iio_attr -i -c ad9361-phy voltage1 gain_control_mode <original>
```

Then re-read every register recorded in step 1 and confirm it matches. Register
pokes are not sticky — a reboot or a debugfs `initialize` reverts them — but do not
rely on that as the restore mechanism. On these units a reboot would also drop the
volatile RC17 image (§4.1), so "reboot to recover" costs a reflash.

### 5.5 Cost and duration

Per-operation costs measured on R17 over SSH, 2026-08-10:

| Operation | Measured |
|---|---:|
| One `iio_reg` / `iio_attr` invocation (process spawn + IIO context open + SPI) | **67 ms** |
| sysfs read via `cat` (busybox process spawn) | **6.0 ms** |
| sysfs read via shell builtin redirect (`read v < …`, no spawn) | **134 µs** |

Command time is therefore negligible for everything except H4/H5 — a 20-trial pulse
sweep is about 6 s of actual I/O. The schedule is dominated by careful sequencing,
recording and verification, not by the radio.

| Phase | RF | Estimate |
|---|---|---:|
| 1. Baseline and safety | no | ~5 min |
| 2. Take pins, verify low, arm `0x0FB` | no | ~10 min |
| 3. **H1 pin mapping** — 4 pins × 5 repeats | no | ~15 min |
| 4. H2 step size | no | ~15 min |
| 6. H6 ENSM | no | ~15 min |
| Restore and verify (§5.4) | no | ~10 min |
| **Subtotal — no RF connection needed** | | **~1–1.5 h** |
| 5. Harness setup and H3 detector map | yes | ~45 min |
| 5. Threshold sweep, 4 registers × ≥5 points | yes | ~60–90 min |
| 5. Hold band fine sweep | yes | ~20 min |
| 5. H4/H5 attempt, resolution-limited (§2) | yes | ~20 min |
| **Subtotal — step 5** | | **~2–2.5 h** |
| **Total** | | **~3.5–4 h** |

**Run steps 1–4 as their own session.** They need no RF at all, and H1 — the only
outcome that can stop all tandem RTL work — lands about 30 minutes in. There is no
reason to hold that result behind bench time for the detector sweep.

## 6. Outputs

### 6.1 Raw

`artifacts/agc_pin_bringup/<run>/` (gitignored): the raw command transcript, and one
JSON per measurement.

### 6.2 Committed

`experiments/e_agc1_pin_and_detector_bringup/RESULTS.md`, plus:

| Artifact | Content | Acceptance gate |
|---|---|---|
| `pin_map.json` | per pin: channel moved, direction, delta, other-channel delta | 5/5 consistent trials per pin, other channel never moved |
| `step_size.json` | index delta per edge, as-shipped and after programming 1 | both values present |
| `detector_map.json` | per GPIO: which signal asserted, at what input level, which channel driven | all 8 bits characterised, or an explicit note on which could not be provoked |
| `threshold_sweep.json` | assert level vs register value for `0x104`, `0x105`, `0x108`, `0x114` | ≥5 points per threshold |
| `hold_band.json` | input-level gap between low-power de-assert and small-ADC assert | stated in dB, with the register values used |
| `latch_trace.json` | overload bit state vs gain change, with timing | shows the latch and the post-change blank, or refutes them |
| `lp_period.json` | observed minimum interval between low-power state changes | compared against the 256–410 µs prediction |
| `ensm_result.json` | index delta per pulse, per ENSM state | RX and at least one non-RX state |
| restore proof | every register from step 1, re-read after §5.3 | all match |

### 6.3 Downstream

Each result closes a numbered open item in `TANDEM_AGC_V1_DESIGN.md`: H1 → O-5,
hold band → O-2, blank timing → O-3, ENSM → O-1. Update that document's §12 table
and its revision history with the measured values.

> ⚠ **`TANDEM_AGC_V1_DESIGN.md` is not in this repository.** It has never been
> committed on any branch (`git log --all --diff-filter=A` finds nothing), and there is
> no `CTRL_IN`/`0x0FB` tooling anywhere under `spf/`. E-GSC6 §6.3 points at the same
> missing document.
>
> This does not block the *measurement* — every hypothesis in §2 is stated in terms of
> the part, and §5 is self-contained. It does block:
>
> - the O-1 / O-2 / O-3 / O-5 traceability this section promises;
> - the §7 consequences that reference the contract's §3, §5.2, §5.4 and §11, which
>   cannot be checked or amended against a document that does not exist.
>
> **Land the design document before recording results**, or restate §7's consequences
> in self-contained terms. Until then, treat the open-item numbering as provisional and
> record the measured values in `RESULTS.md` regardless — the measurements do not
> expire.

## 7. Decision rule

Pre-registered.

| Result | Reading | Consequence |
|---|---|---|
| **H1 fails on any pin** | the schematic-derived mapping is wrong | **Stop all tandem RTL work.** §3 of the design contract is wrong, and every pulse the firmware would emit is wrong. Re-derive the mapping from this measurement and re-review. This is the highest-consequence single outcome in the experiment |
| H1 passes | mapping confirmed on the part | close O-5; the contract's §3 table stands |
| H2 shows a step other than the programmed value | the step fields are not what the driver writes, or are being overridden | re-check `0x0FC`/`0x0FE` handling before relying on a step of 1 |
| **H4 fails — no latch** | the overload outputs are plain levels | §5.2's sampling discipline is unnecessary but harmless; more importantly the policy's assumption about self-clearing is wrong and §5.4's clamp handling must be revisited |
| H5 much shorter than predicted | the power-measurement period is faster than modelled | the ≈1 ms cooldown may be reducible, which directly improves recovery responsiveness |
| **Hold band narrower than 1 dB** | one gain step crosses from "increase" to "inhibit" | the policy will oscillate as designed. Re-plan thresholds, or add hysteresis beyond the current dwell. This is the most likely single change to come out of the run |
| **H6: edges NOT honoured outside RX** | pin control is RX-state dependent | the enable sequence in §11 must guarantee RX is active before arming, and must handle an ENSM transition while armed. A real change to the contract |
| H6: edges honoured everywhere | no ENSM dependence | close O-1; §11 unchanged |

## 8. Risks

| Risk | Why it matters | Check |
|---|---|---|
| **Arming pin control while the pins float** | `CTRL_IN` is high-Z from power-on through Linux — no pull on the board, none inside the AD9361. Armed control over four undriven traces is an uncommanded gain change on both receivers, driven by whatever couples in | §5.2 order is mandatory: export, direction out, drive low, **read back and verify low**, only then arm `0x0FB`. If the verify fails, stop |
| **Running against a live radio** | arming changes gain behaviour immediately | confirm no capture, no rover service, no concurrent burn-in. RC17 hardware testing is active in this repo — claim the radio explicitly |
| **A bare `0x03` write to `0x0FB`** | `direct_reg_access` writes unmasked, so it would clear every other bit in the register | read-modify-write, always. Same for `0x0FE`, where the low five bits are the Peak Overload Wait Time |
| **RF input overdrive** | the AD9361 RF pin maximum is +2.5 dBm peak; the detector sweep deliberately drives toward overload | start low, step up, and stop at the first large-ADC assert. Never start at the top of the range |
| **Concluding anything about fast timing** | sysfs GPIO toggling is slow — tens to hundreds of microseconds per write. The 2-ClkRF rule is a *minimum*, so slow pulses are legal and this method cannot test minimum-width rejection | do not report any conclusion about minimum pulse width, edge rate, or simultaneity. Those need the FPGA and belong to a later stage |
| **Hardcoding the GPIO base** | it shifts with kernel configuration; 960/968 were observed on one build | discover it per §5.1 and record the discovered value in the results |
| **dB / index confusion** | the readback is available both ways and one index is 1 dB, but the offset is band-dependent | record both the raw register and the dB for every reading |
| **`hybrid` gain mode** | selecting it re-arms `CTRL_IN2` through `0x0FA` without touching `0x0FB`, so it can move RX2 gain even when this experiment believes pin control is disarmed | do not use hybrid mode during the run; record the mode at every step |
| **Leaving the part armed** | an armed `0x0FB` with unexported, floating pins after the run is exactly the hazard above | §5.4 restore, then re-read and confirm. Do not rely on the reboot — and on these units a reboot also drops the volatile RC17 image |
| **`iio_attr` channel ambiguity** | without `-i`, `voltage0`/`voltage1` match the **output** (TX) channel of the same name. Measured: `iio_attr -c … voltage0 hardwaregain` returns −80 dB (TX1), not RX1's 41 dB. The dB cross-check on the gain index would be read off the transmitter | use `-i -c` for every RX channel access, as §5.2 now does; cross-check that the dB value moves with `0x2B0`/`0x2B5` |
| **`pinctrl` / `pinctrl_fdd_indep` ENSM modes** | both are offered by `ensm_mode_available` on this build, and both hand ENSM state to external pins — a second, independent pin-control surface next to `CTRL_IN`. Selecting one during H6 would confound the ENSM question with a pin hazard | do not select either during the run. H6 uses `alert`/`wait`/`sleep` only, and the mode is recorded at every step |
| **Concluding a period or blank duration from sysfs timing** | the fastest sysfs read is 134 µs against a predicted 256–410 µs period, so H4/H5 have 2–3 samples per period at best; via `cat` (6.0 ms) they are blind. There is no compiler on target to do better | pre-declared in §2: report only what is observable, and no period or blank duration unless the observed interval exceeds ~500 µs. Defer the rest to the FPGA stage |

## 9. Revision log

**2026-08-10, revision 1** — read-only bench audit against R17 on stock RC17, before
any pin was exported or any register written. Nothing in the design's reasoning was
wrong; the changes add measured values, fix one command that reads the wrong channel,
and scope two hypotheses the method cannot resolve.

1. **§5.1 GPIO map measured.** One `zynq_gpio` chip, `base=906`, `ngpio=118`, so
   `CTRL_OUT = 960–967` and `CTRL_IN = 968–971` — the hedged 960/968 figures are
   correct on this build. Verified against two independent anchors, and confirmed that
   the complete set of claimed lines is 921/952/973/974/977, so **nothing claims
   EMIO 0–11**. The `one-bit-adc-dac` driver takes EMIO 14 and 17, not the CTRL pins.
   Kernel options and `iio_reg` presence confirmed on the image.
2. **§5.2 `iio_attr -i` fix.** Without `-i`, `voltage0`/`voltage1` resolve to the TX
   channel: the baseline command returned −80 dB (TX1) rather than RX1's 41 dB. Every
   RX access now passes `-i`, and the trap is recorded in §8.
3. **§5.3 added** with the measured baseline. Three risk rows become concrete:
   `0x0FB = 0x08` has bit 3 set, so read-modify-write is required in fact; `0x0FE`
   carries a live PWOT of 3 in its low bits; and `0x0FC/0x0FE = 0x23` already confirm
   H2's programmed step of 2 from the register.
4. **§2 H4/H5 scoped.** Fastest sysfs read is 134 µs (builtin) or 6.0 ms (`cat`)
   against a predicted 256–410 µs period, and there is no compiler on target. Both are
   pre-declared resolution-limited rather than left to fail silently at write-up.
5. **§4.2 schematic replaced** with the harness actually on the bench — TX2 → 30 dB
   pad → splitter → RX1/RX2, with level set in software via TX `hardwaregain`. No
   external signal generator exists on this bench, and the software level control is
   better suited to a threshold sweep than a step attenuator. Adds how to isolate one
   channel when the splitter feeds both.
6. **§4.1** records both serials, the duplicate-IP hazard, and that both units run a
   volatile RC17 image, which makes "reboot to recover" cost a reflash.
7. **§6.3 flags that `TANDEM_AGC_V1_DESIGN.md` is not in the repository** — never
   committed on any branch. The measurement is unaffected; the open-item traceability
   and four of §7's consequences are not.
8. **§5.5 added** with measured per-operation costs and a phase-by-phase duration:
   ~1–1.5 h with no RF for steps 1–4/6, ~2–2.5 h for step 5, H1 at ~30 min.
9. **§8** gains rows for the `iio_attr` ambiguity, the `pinctrl` ENSM modes, and the
   H4/H5 timing resolution limit.
