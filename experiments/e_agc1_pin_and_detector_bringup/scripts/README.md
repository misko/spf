# E-AGC1 measurement scripts

The scripts that produced [`RESULTS.md`](../RESULTS.md) and the committed JSON artifacts.
Kept because the raw transcripts under `/mnt/qnap01/mouse9911/spf/calibration_data/raw/agc_pin_bringup/` are gitignored — with
these, the run is re-executable and the artifacts are re-derivable; without them it is
neither.

## Layout

`*.sh` run **on the Pluto** (busybox `ash`) and emit flat `key=value` lines.
`agc1_mkjson.py` runs **on the host**: it pipes a script over SSH, runs it, and assembles
the output into a JSON artifact stamped with host time — the radio has no RTC, so its own
clock reads 1970 + uptime. The `agc1_derive*.py` scripts turn the raw JSONs into the
committed artifacts.

## Order they were run in

| Script | Step | Produces |
|---|---|---|
| `agc1_baseline_kv.sh` | 1 | baseline; also re-run afterwards as the restore proof |
| `agc1_step2_takepins.sh` | 2 | takes `CTRL_IN`, verifies low, **does not arm** |
| `agc1_step3_h1.sh` | 3 | H1 pin mapping |
| `agc1_step4_stepsize_and_ab.sh` | 4 | H2 step size + the armed-write A/B |
| `agc1_step6_ensm.sh` | 6 | H6 ENSM dependence |
| `agc1_step7_restore.sh` | §5.4 | restore |
| `agc1_step5b_ramp.sh` | 5 | TX level ramp, RSSI-controlled |
| `agc1_step5c_fast.sh` | 5 | H3 detector map + hold band (two gain sweeps) |
| `agc1_step5d_thresholds.sh` | 5 | threshold sweep |
| `agc1_step5e_h4h5.sh` | 5 | H4 latch + H5 period attempts |

Host side: `agc1_derive2.py` builds the session-1 artifacts from both radios;
`agc1_derive_step5.py` builds the step-5 artifacts.

## Invocation

```sh
python3 agc1_mkjson.py <script.sh> <radio-ip> <step> <name> <dest.json>
```

## Conventions these encode — keep them if you write more

- **Every phase that arms `0x0FB` restores it from an `EXIT INT TERM HUP` trap.** `HUP`
  matters: a host-side timeout killed one run over SSH and an `EXIT/INT/TERM`-only trap did
  not fire, leaving the tone on with registers modified.
- **Read-modify-write, always.** `0x0FB` has bit 3 set on this build and `0x0FE[4:0]` holds
  a live Peak Overload Wait Time, so a bare byte write clobbers real state.
- **Take the pins before arming**, and *verify* they read low — an unarmed pin cannot move
  gain, which is what makes that order safe. `agc1_step2_takepins.sh` deliberately stops
  before arming.
- **`iio_attr` needs `-i`** for RX channels; without it the tool matches the output (TX)
  channel of the same name.
- **Reading a register costs ~67 ms** (process spawn + context open). A GPIO read via a
  shell builtin redirect is ~134 µs on a plain sysfs attribute but **~322 µs on a GPIO
  `value` file** — which is what put H4/H5 out of reach.
- **When timing matters, step gain with a `CTRL_IN` edge, not `iio_attr`** — ~100s of µs
  versus 67 ms.
- **Ramp control is RSSI, not the detector bits.** The two low-power bits are asserted with
  no signal (quiescent `10001011`), so "any bit high" is not a usable limit.
