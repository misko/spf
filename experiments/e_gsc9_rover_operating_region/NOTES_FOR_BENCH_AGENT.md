# E-GSC9 — operational brief for the bench agent

Read [`experiment_readme.md`](experiment_readme.md) first. This file is the order of
operations only.

---

## Before you start

| check | how | abort if |
|---|---|---|
| Two radios enumerated | `iio_info -s` or the harness's identity probe | fewer than 2 |
| Direct-USB manifest is **fresh** | `stat -c%y /run/spf/direct_usb_ready.json` | mtime predates now, or the file is absent |
| Firmware | readback vs `v0.38-plutoplus-spf-libiio-metadata-v5`, QSPI boot | mismatch |
| Gain table | live readback SHA-256 vs E-GSC7's `90d34d61…a1143` | differs |
| Manual gain mode | readback per arm, not just commanded | either arm reports AGC |
| Disk | ≥ 60 GB free on `/mnt/qnap01` | less |

> **Known and expected:** the repo firmware-pin test expects rover `RC16` and will report a
> mismatch against metadata-v5. That is the documented policy difference, not a fault.
>
> **Known blocker as of 2026-08-13:** no Pluto is attached over USB on the analysis host and
> `/run/spf/direct_usb_ready.json` does not exist there. This experiment runs on the bench
> rig, not on `kalman`.

---

## Step 0 — level ladder (2.7 min), then **STOP**

Run all three, both LOs:

```bash
P=~/spf-virtualenv/bin/python3
for T in 23 29 35; do
  $P -m spf.calibrations.dual_rx_gain_frequency run --transport iio-usb \
     --config experiments/e_gsc9_rover_operating_region/configs/e_gsc9_level_ladder_tx${T}.yaml \
     --output /home/pi/gsc9_staging/e_gsc9_level_ladder_tx${T}_YYYYMMDD_v1
done
```

**Then stop and read the numbers.** For each (radio, LO, arm) extract `tone_dbfs` at
g = 62 and at g = 23.

Apply the preregistered rule:

- pick the **largest** integer `T` with `max(tone_dbfs @ g=62) ≤ −12.0 dBFS`
- subject to `min(tone_dbfs @ g=23) ≥ −58.0 dBFS`
- if both cannot hold: satisfy the **upper** bound, then raise the gain floor above 23 and
  regenerate with `analysis/gen_configs.py`. Record the new floor and its coverage
  (floor 26 → 99.98% / 100.00%; floor 30 → 99.93% / 100.00%).

Write `T` into `e_gsc9_rover_region_grid.yaml` **once**. The document hash becomes the
`calibration_run_signature`; any later edit — *including the `notes:` field* — invalidates
resume and forces a new output root.

**Do not start session A on an unread ladder.**

---

## Step 1 — session A, the grid (2.64 h)

```bash
for i in $(seq 1 12); do
  ~/spf-virtualenv/bin/python3 -m spf.calibrations.dual_rx_gain_frequency run \
    --transport iio-usb \
    --config experiments/e_gsc9_rover_operating_region/configs/e_gsc9_rover_region_grid.yaml \
    --output /home/pi/gsc9_staging/e_gsc9_session_a_YYYYMMDD_v1
  sleep 30
done
```

The loop is mandatory, not defensive: `_open_preflight_radio` has a bare `finally:` with no
`except:`, so an exhausted handoff-prime sequence propagates and kills the process. The run
resumes by signature, so re-invoking continues rather than restarting.

**Check at the end of epoch 1 (~32 min in):**

| check | expected | if not |
|---|---|---|
| cells in epoch 1 | 1,369 per radio per LO after the measured fallback (1,600 in the original plan) | abort, diagnose |
| `tone_dbfs` range | within [−65, −6] | abort — the ladder was wrong |
| `clipping_fraction` | 0 everywhere | abort |
| `gain_endpoints_equal` | true on 100% | abort — AGC re-entered |
| worst anchor drift so far | < 4° | continue but record it |

**Stopping early is safe.** The schedule is epoch-outer, so every epoch contains the whole
grid. Epoch 3 (1.6 h) is the minimum for any published number; epoch 5 is design power.

---

## Step 2 — short controls (~10 min, same session, do not re-cable)

```bash
# A2 transitions bridge - brackets the RF-word steps the grid cannot reach
--config .../configs/e_gsc9_t2_transitions_bridge.yaml

# A3 AM-PM control - run these two BACK TO BACK, nothing touched between them
--config .../configs/e_gsc9_t3a_ampm_16384.yaml
--config .../configs/e_gsc9_t3b_ampm_8192.yaml
```

A3's two legs are byte-identical except `tx-digital-amplitude` (16384 vs 8192 = −6.02 dB).
Running them adjacently is what stops drift masquerading as a level effect.

---

## Step 3 — session B, ≥12 h later

Power-cycle both radios. **Touch no connector.** Re-run:

```bash
experiments/e_gsc9_rover_operating_region/run_session_b.sh
```

This is the session-transfer control. If a connector is disturbed, say so in the results —
it converts B into a variant of C and it is no longer a clean 12 h test.

---

## Step 4 — session C, pad discriminator, with reversal

```bash
GSC9_C_PHYSICAL_STATE=no_pads GSC9_C_OPERATOR_NOTE='confirmed no added pads' \
  experiments/e_gsc9_rover_operating_region/run_session_c_leg.sh a
# insert 10 dB pads on both arms
GSC9_C_PHYSICAL_STATE=pads_installed GSC9_C_OPERATOR_NOTE='10 dB pads installed on both arms' \
  experiments/e_gsc9_rover_operating_region/run_session_c_leg.sh b
# REMOVE the pads
GSC9_C_PHYSICAL_STATE=pads_removed GSC9_C_OPERATOR_NOTE='both added pads removed; original paths restored' \
  experiments/e_gsc9_rover_operating_region/run_session_c_leg.sh aprime
```

The runner uses the current date in all three output names. If the sequence can
cross midnight, set the same `GSC9_C_RUN_DATE=YYYYMMDD` on every command.

**The A′ leg is mandatory.** Connector work has previously moved a radio's high-band mean
`|A|` from 3.49° to 29.41°. Without the reversal you cannot separate "the pad did it" from
"handling the connector did it".

---

## When to abort

- any pre-flight check above fails
- `clipping_fraction > 0` on any kept frame
- either arm reports AGC after being commanded to manual
- gain-table SHA-256 differs between radios or from E-GSC7's
- anchor drift exceeds 4° within a single session
- `tone_dbfs` leaves [−65, −6] on kept frames

## Where output must land

```
/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/e_gsc9_<session>_<date>_v1/<serial>/
```

Never write to `/mnt/md2`. Never delete anything, including failed partial runs — a partial
run is evidence.

## What to report

`RESULTS.md` in this directory, stating **H1–H7 with numbers, including any falsified**, plus
the gate table G1–G10 with pass/fail. Report the same-LO control **before** any cross-carrier
claim (G7). Publish any effect that fails the resolved-margin gate (G5) as a bound, not a value.
