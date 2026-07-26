# Pluto+ dual-RX manual-gain phase sweep

This bench experiment measures how the relative phase of one Pluto+ changes
across every ordered pair of RX1 and RX2 manual-gain settings. Both receivers
must be fed from the same sine-wave source. Each gain pair is measured multiple
times in a randomized order, checkpointed, quality-gated, and summarized as
tables and heatmaps.

The recorded phase convention is:

```text
phase_difference = angle(RX1) - angle(RX2)
```

This is the convention used by `spf.rf.get_phase_diff`.

The experiment uses the standard pyadi/libiio USB interface. It does not
depend on Rover services, direct-USB streaming, Zarr collection, or a second
Pluto. Do not run another receive stream against the selected Pluto while the
sweep is active.

## Physical setup

Use this signal chain:

```text
sine generator
    |
50-ohm attenuator, if required
    |
two-way 50-ohm RF splitter
    |                         |
matched coax A                matched coax B
    |                         |
Pluto+ RX1                    Pluto+ RX2
```

Requirements:

- use a proper splitter, not a tee;
- use equal-type, equal-length cables where practical;
- secure every connector before the run;
- leave the Pluto TX output disconnected and terminated if appropriate;
- do not exceed the Pluto+ RF-input or generator specifications;
- start at a conservative source level and check for clipping;
- record the generator, splitter, attenuator, cables, and source power in the
  setup label or notes.

The default example uses a 2.4 GHz LO and places the source 100 kHz above it:

```text
Pluto LO          2,400,000,000 Hz
source frequency  2,400,100,000 Hz
digital IF              +100,000 Hz
```

The offset avoids measuring a tone directly on the receiver's DC-correction
notch. The same generator output must reach both receivers simultaneously.

## Software preflight

Run the focused tests:

```bash
cd /home/pi/spf
/home/pi/spf-virtualenv/bin/python3 -m pytest -q \
  tests/test_dual_rx_phase_sweep.py
```

Discover attached Plutos:

```bash
iio_info -s
```

Then select one by stable serial number:

```bash
/home/pi/spf-virtualenv/bin/python3 \
  -m spf.scripts.dual_rx_phase_sweep discover \
  --serial PLUTO_SERIAL \
  --lo-hz 2400000000
```

Pass when discovery reports:

- the intended serial and USB URI;
- two enabled AD9361 receive channels;
- manual-gain values for both receivers;
- phase-inversion mitigation enabled;
- quadrature tracking in the requested state; and
- a source frequency of 2,400,100,000 Hz.

Stop if the serial is ambiguous, RX1/RX2 gain ranges differ, either receive
channel is unavailable, or a second program owns the receive stream.

## Small signal-and-level sweep

Before the exhaustive run, exercise low, middle, and high gain:

```bash
/home/pi/spf-virtualenv/bin/python3 \
  -m spf.scripts.dual_rx_phase_sweep run \
  --serial PLUTO_SERIAL \
  --output artifacts/dual_rx_phase/SETUP_ID_smoke \
  --lo-hz 2400000000 \
  --gains=-3,34,71 \
  --repetitions 2 \
  --min-valid-per-cell 2 \
  --source-power-dbm SOURCE_POWER_DBM \
  --setup-label SETUP_ID \
  --notes "generator/splitter/attenuator/cable identifiers" \
  --yes
```

This measures all nine ordered combinations, including unequal RX1/RX2
settings. It is not sufficient to test only equal-gain pairs.

Pass when `report.json` has `status: "pass"` and all nine cells pass. Inspect
`report.md` and the tone-level/SNR fields in `report.csv`. A completed command
can still return `fail_quality`; that means it acquired the requested data but
the RF evidence was not good enough to accept.

If high-gain cells clip, reduce source power or add attenuation. If low-gain
cells have poor SNR, increase source power. The full manual-gain span is about
74 dB while the default accepted tone-level window is 67 dB, so one fixed
source level may not make every extreme cell pass. In that case:

1. retain the first run unchanged;
2. run additional source levels into separate output directories;
3. record every level explicitly with `--source-power-dbm`; and
4. never silently combine results from different physical setups.

The SNR, clipping, coherence, and phase-stability fields determine whether a
measurement is useful; source power alone does not.

## Exhaustive repeated sweep

Omitting gain selection requests every integral manual-gain state dynamically
reported by both receivers:

```bash
/home/pi/spf-virtualenv/bin/python3 \
  -m spf.scripts.dual_rx_phase_sweep run \
  --serial PLUTO_SERIAL \
  --output artifacts/dual_rx_phase/SETUP_ID_full \
  --lo-hz 2400000000 \
  --repetitions 3 \
  --captures-per-pair 1 \
  --min-valid-per-cell 2 \
  --max-across-repeat-phase-std-deg 5 \
  --source-power-dbm SOURCE_POWER_DBM \
  --setup-label SETUP_ID \
  --notes "generator/splitter/attenuator/cable identifiers" \
  --yes
```

For the currently observed `[-3 1 71]` range this produces:

```text
75 gain states
5,625 ordered gain pairs
16,875 captures at three repetitions
```

Allow roughly 45–90 minutes. The command prints an ideal RF-time floor before
starting; USB control writes and report generation add overhead.

The order is deterministic but randomized independently per repetition. After
each gain change the runner waits, discards stale buffers, verifies the dB
readback, reads both raw gain-table indices, captures IQ, and verifies both the
dB values and raw indices again. A changed or unreadable state is recorded as
an explicit error and retried.

## Interruption, resume, and reporting

The output directory is an immutable experiment identity. To resume after an
interruption, repeat the exact same command and output path. Completed
measurement keys are skipped. Changing the serial, configuration, gain set, or
source/setup provenance requires a new output directory.

Regenerate reports without touching hardware:

```bash
/home/pi/spf-virtualenv/bin/python3 \
  -m spf.scripts.dual_rx_phase_sweep report \
  artifacts/dual_rx_phase/SETUP_ID_full
```

Artifacts:

- `manifest.json` — immutable radio, firmware, software, RF, and setup
  provenance;
- `observations.jsonl` — append-only measurements and explicit error attempts;
- `report.json` — machine-readable run and per-cell summary;
- `report.csv` — one row per ordered gain pair;
- `phase_delta_deg.csv` — RX1-gain × RX2-gain phase matrix;
- `phase_circular_std_deg.csv` — repeatability matrix;
- `phase_sweep_heatmaps.png` — phase delta, repeatability, and valid fraction;
- `report.md` — short human-readable result.

Run status:

- `pass` — every scheduled capture exists and every gain cell has the required
  valid repeats with acceptable cross-repeat circular phase deviation;
- `fail_quality` — acquisition completed, but one or more cells failed signal
  quality or repeatability;
- `partial` — one or more scheduled measurements have no successful capture.

The CLI exits zero only for `pass`.

## Acceptance checklist

- [ ] The manifest contains the intended Pluto serial and firmware version.
- [ ] The source frequency is exactly LO plus the configured offset.
- [ ] RX1 and RX2 are fed through the documented splitter and cable paths.
- [ ] Every ordered gain pair was attempted for every repetition.
- [ ] Requested and read-back dB gains match before and after every capture.
- [ ] Raw gain-table indices match before and after every capture.
- [ ] No channel clips and every accepted tone meets its SNR/level thresholds.
- [ ] Accepted captures meet coherence and within-capture phase-stability gates.
- [ ] Every cell has enough accepted repeats.
- [ ] Every cell meets the cross-repeat circular phase-deviation gate.
- [ ] `report.json` reports `status: "pass"`.
- [ ] Nonzero error attempts, phase discontinuities, and gain-table boundaries
      are reviewed rather than hidden.
- [ ] The report, manifest, and physical setup notes are archived together.

Running without the sine source is useful only as a plumbing test. It should
normally finish with `fail_quality`; a no-source run must never be presented as
a phase-characterization result.
