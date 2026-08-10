# E-AGC1 — AD9361 gain-pin and detector bring-up, from userspace

**Status:** designed 2026-08-10, not yet run.
**Closes:** open items O-1, O-2, O-3 and O-5 of `TANDEM_AGC_V1_DESIGN.md`, all of
which currently block that design's candidate freeze.
**Cost:** one radio, one session. **No FPGA change, no bitstream, no RAM boot, no
QSPI write.** Stock RC17 firmware throughout.

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

Then **restore everything** per §5.3, and confirm the restore.

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

The radio must be **idle** — no capture, no rover service, no concurrent test. Note
that RC17 hardware burn-in work is active in this repository; coordinate before
claiming a radio.

Record the serial and resolve by serial, never by IP.

### 4.2 Physical schematic

Steps 1–4 and 6 need **no RF input at all** — they read the gain index over SPI and
never look at signal. Steps 5 (H3–H5) needs a controlled input level:

```
        signal generator
               │          start LOW; the AD9361 RF pin maximum
               │          is +2.5 dBm peak — see §8
        ┌──────┴──────┐
        │ step atten  │   or use the generator's own level control
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │  2-way 50 Ω │   only if driving both channels; a single
        │  splitter   │   channel needs no splitter
        └──┬───────┬──┘
           │       │
        ┌──┴──┐ ┌──┴──┐
        │ RX1 │ │ RX2 │
        └─────┘ └─────┘
```

For H3 it is sufficient — and cleaner — to drive **one channel at a time**, so that
a CH1 bit asserting when CH2 is driven immediately falsifies the bit map.

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

### 5.2 Sequence

```sh
# ---- 1. baseline ----
iio_attr -c ad9361-phy voltage0 gain_control_mode          # record
iio_reg ad9361-phy 0x035 ; iio_reg ad9361-phy 0x036        # record
iio_reg ad9361-phy 0x0FB                                   # expect bits[1:0]=0
iio_reg ad9361-phy 0x2B0 ; iio_reg ad9361-phy 0x2B5        # RX1/RX2 index
iio_attr -c ad9361-phy voltage0 hardwaregain               # dB cross-check
iio_attr -c ad9361-phy voltage1 hardwaregain

# put both channels in manual gain before touching pins
iio_attr -c ad9361-phy voltage0 gain_control_mode manual
iio_attr -c ad9361-phy voltage1 gain_control_mode manual

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

### 5.3 Restore — mandatory, and verified

```sh
# disarm BEFORE releasing the pins
V=$(iio_reg ad9361-phy 0x0FB); iio_reg ad9361-phy 0x0FB $((V & ~0x03))
iio_reg ad9361-phy 0x035 <original>
for g in 62 63 64 65; do echo $((BASE+g)) > /sys/class/gpio/unexport; done
iio_attr -c ad9361-phy voltage0 gain_control_mode <original>
iio_attr -c ad9361-phy voltage1 gain_control_mode <original>
```

Then re-read every register recorded in step 1 and confirm it matches. Register
pokes are not sticky — a reboot or a debugfs `initialize` reverts them — but do not
rely on that as the restore mechanism.

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
| **Leaving the part armed** | an armed `0x0FB` with unexported, floating pins after the run is exactly the hazard above | §5.3 restore, then re-read and confirm. Do not rely on the reboot |
