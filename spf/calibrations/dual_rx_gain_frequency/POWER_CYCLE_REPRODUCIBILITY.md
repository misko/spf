# Power-cycle reproducibility check

This experiment determines whether a serial-specific phase calibration can be
reused after cold power removal, needs a small session anchor, or needs a
gain-dependent recalibration.

It uses `configs/power_cycle_subsample.yaml`: all 12 dense-survey frequencies,
the ordered Cartesian product of gains 0, 16, 26, 41, and 52 dB, and three
randomized epochs. This is 900 frames per radio, versus 10,404 in the dense
survey.

## Controlled conditions

- Do not disconnect or move TX2, attenuator, splitter, RX1, or RX2 RF cables.
- Keep the same USB ports and host.
- Use the script for both captures so firmware, protocol, configuration, radio
  serials, stored IQ validation, and quality gates are checked identically.
- A cold cycle means removing Pluto power for at least 10 seconds. A software
  reboot or RAM firmware reload alone does not count.
- Start with radios near the same temperature. Temperature is a separate
  experiment.

## Capture the before state

From the repository root:

```bash
spf/calibrations/dual_rx_gain_frequency/power_cycle_subsample.sh \
  capture-before radios_17_18_20260729
```

When it completes, physically remove power from both Plutos for at least ten
seconds. Keep every RF cable fixed, restore power, and wait for USB enumeration.

## Capture one cold-power-cycle state

The explicit confirmation prevents an ordinary repeat from being mislabeled as
a power-cycle result:

```bash
spf/calibrations/dual_rx_gain_frequency/power_cycle_subsample.sh \
  capture-after radios_17_18_20260729 1 --confirmed-power-cycle
```

## Compare

```bash
spf/calibrations/dual_rx_gain_frequency/power_cycle_subsample.sh \
  compare radios_17_18_20260729 1
```

The report is written below:

```text
artifacts/dual_rx_gain_frequency/power_cycle/
  radios_17_18_20260729/
    comparison_cycle_1/
      README.md
      power_cycle_comparison.json
```

Repeat the physical power removal, `capture-after`, and `compare` commands with
cycle numbers 2 and 3. Every cycle gets a new output root and cannot overwrite a
previous cycle.

## Decision ladder

The default acceptance threshold is at most 2 degrees circular MAE and 5
degrees P95 with at least 80% common passing-cell coverage.

1. If raw before/after drift passes, no session calibration is required on the
   measured support.
2. Otherwise, if subtracting the 26/26 dB anchor at 2.412 GHz passes, one global
   session anchor is required.
3. Otherwise, if subtracting the 26/26 dB anchor independently at every
   frequency passes, one anchor per operating frequency is required.
4. Otherwise, gain-dependent calibration changed and the stored LUT must not be
   reused without broader recalibration.

The comparator fails closed when serials, firmware provenance, configuration
hashes, complete-run status, common-cell coverage, or required anchors do not
match.
