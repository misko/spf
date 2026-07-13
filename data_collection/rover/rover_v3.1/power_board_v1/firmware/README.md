# Power-board supervisor firmware (ATtiny816)

Split into a **pure state machine** (`lpd_core.h` — no hardware access, unit-tested
on the host with `make test`) and a thin **AVR shell** (`supervisor.c` — ADC/GPIO/
I2C/RTC wiring only). All thresholds live in one block at the top of `lpd_core.h`.

## Behavior contract (DESIGN.md / P1_DETAIL_DESIGN.md §4)
- CUT 10.2 V / RECONNECT 11.7 V (1.5 V hysteresis), warn (LOW_BATT) at 10.55 V
- **10 s qualifier** on every voltage transition — a motor-stall sag of even 8 s
  below the cut threshold does nothing (host test #2)
- **60 s LOW_BATT → cut handshake**; Pi can shorten it via SHDN_ACK pin or I2C
- Panel switch off = graceful: LOW_BATT asserted as "please halt", 5 s budget
- DYING has no return path (a pack at the cut threshold recovers upward under
  reduced load — allowing return would flap the rails; host test #4/#5)
- PRECHARGE aborts to a **latched FAULT** if PGOODs never rise (~8 s) — no
  chatter loop; the user must toggle the panel switch to retry (host test #10)
- AUX_CTL (motor contactor) is ON only in the healthy ON state AND when the Pi
  has requested it via I2C (defaults off after power-up)

State enum (register 0x00): 0 OFF, 1 PRECHARGE, 2 ON, 3 WARN, 4 DYING, 5 CUT,
6 SWOFF, 7 FAULT. LED blinks the state number in a 3.2 s cycle.

## I2C register map (slave addr 0x36, 100/400 kHz)
| Reg | Access | Meaning |
|---|---|---|
| 0x00 | RO | state (enum above) |
| 0x01/0x02 | RO | pack voltage mV, little-endian u16 |
| 0x03/0x04 | RO | rail A voltage mV, little-endian u16 |
| 0x05 | RO | NTC raw (8-bit; linearization on the Pi side) |
| 0x06 | RO | flags: b0 PGOOD_A, b1 PGOOD_B, b2 FAULT_USB, b3 switch, b4 LOW_BATT, b5 AUX active |
| 0x10 | RW | USB port-cycle: write b0/b1 to cycle port 1/2 (~560 ms off); self-clears |
| 0x11 | RW | b0 = AUX_CTL request (ANDed with state gate) |
| 0x12 | RW | write 0xA5 = shutdown ack/request (same effect as SHDN_ACK pin) |
| 0xF0 | RO | firmware version (0x11 = v1.1) |

Pi-side quick check: `i2cget -y 1 0x36 0x00` (state), `i2cget -y 1 0x36 0x01 w`
(pack mV).

## Scaling
- VIN: 680k/100k divider (÷7.8; high-impedance so the always-on drain stays ~16 µA,
  with a 100 nF hold cap feeding the ADC), 2.5 V internal ref, 10-bit ×16
  accumulation → 19 mV/LSB pack resolution
- Rail A: 30k/10k divider (÷4) → 9.8 mV/LSB

## Hardware notes
- FE_EN and LOW_BATT are open-drain **emulated** (drive-low / release-to-input);
  FE_EN releases to the LM74800 EN/UVLO precision divider, so an unprogrammed or
  absent MCU leaves the analog LPD fallback in charge.
- USB_EN1/2 default ON at reset so radios power with the rail even if the Pi never
  speaks I2C.
- WDT 1 s, kicked only after a full sample→step→apply cycle.

## Build / test / flash
```
make test          # host unit tests (10 scenarios) — run in CI
make avr           # needs avr-gcc + ATtiny_DFP (see Makefile DFP=)
make flash PORT=/dev/ttyUSB0   # SerialUPDI (3-pin header: UPDI/3V3/GND)
```
Bench validation of thresholds/timers is bring-up checklist item 3
(PRODUCTION_REVIEW.md checklist D).
