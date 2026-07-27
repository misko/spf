# Dual-RX gain/frequency calibration

This package measures the phase relationship between PlutoPlus RX1 and RX2 for
manual RX-gain pairs and RF frequencies. It uses the SPF direct-USB protocol-v2
firmware, writes one data-version-7 Zarr per radio serial, validates every frame
from its stored IQ, and fits a circular additive gain model.

The physical path on each radio is:

```text
TX2 -> 30 dB attenuator -> two-way splitter -> RX1 and RX2
```

Only one radio transmits at a time. Every frequency/gain Cartesian scan is
repeated in three separately randomized epochs. A frame is usable for phase
only when its gain/RSSI metadata, tone level, clipping, cross-channel
coherence, and segmented phase-stability checks pass. Failed frames remain in
the dataset with an explicit quality-reason mask.

The calibration tone comes from TX2's FPGA DDS, so TX DMA never competes with
the direct-USB receive path. Only one Pluto control/direct context is open at a
time. At every radio/frequency block, the runner negotiates the two observed
Pluto+ RX-DMA handoff sequences (direct start, then post-arm IIO prime) and
accepts one only after a direct-USB reference-gain tone preflight passes. The
selected handoff is logged with the preflight. There are no host-side IIO gain
or RSSI reads in the frame loop, and TX attenuation changes in place without
restarting the DDS.

## Commands

First RAM-load and verify every attached Pluto using the Rover v3.1 preparation
script so `/run/spf/direct_usb_ready.json` exists. Then qualify each serial:

```bash
python -m spf.calibrations.dual_rx_gain_frequency probe \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --serial SERIAL
```

Run the pilot for both radios from the verified manifest:

```bash
python -m spf.calibrations.dual_rx_gain_frequency run \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --output artifacts/dual_rx_gain_frequency/pilot
```

Validate and fit each serial-specific dataset:

```bash
python -m spf.calibrations.dual_rx_gain_frequency validate \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --dataset artifacts/dual_rx_gain_frequency/pilot/SERIAL/calibration.v7.zarr \
  --serial SERIAL \
  --output artifacts/dual_rx_gain_frequency/pilot/SERIAL/validation.json

python -m spf.calibrations.dual_rx_gain_frequency fit \
  --config spf/calibrations/dual_rx_gain_frequency/configs/pilot_5ghz.yaml \
  --dataset artifacts/dual_rx_gain_frequency/pilot/SERIAL/calibration.v7.zarr \
  --output artifacts/dual_rx_gain_frequency/pilot/SERIAL/model.json

python -m spf.calibrations.dual_rx_gain_frequency report \
  --validation artifacts/dual_rx_gain_frequency/pilot/SERIAL/validation.json \
  --model artifacts/dual_rx_gain_frequency/pilot/SERIAL/model.json \
  --output-dir artifacts/dual_rx_gain_frequency/pilot/SERIAL/analysis
```

When diagnosing an AD9361 DC-correction failure, capture a read-only snapshot
of both RF-input correction banks:

```bash
python -m spf.calibrations.dual_rx_gain_frequency dc-registers \
  --serial PLUTO_SERIAL \
  --output rf_dc_registers.json
```

This command does not change gain, LO, tracking, calibration, or streaming
state. It decodes all four packed 10-bit RX1/RX2 I/Q correction words and
flags the documented stuck value `0x200`.

After the pilot establishes usable signal levels, replace `pilot_5ghz.yaml`
with `coarse_5ghz.yaml` to enumerate all 73 manual gain states on both
receivers at the four coarse frequencies. Runs are resumable: the completion
bit is committed only after IQ, V7 metadata, coordinates, and quality metrics.
Initial and resumed capture use the same asynchronous LMDB write mode, with a
durable sync after every radio/frequency block and on clean shutdown.

## Model selection

The operational candidate is fitted independently for each radio and RF
frequency:

```text
angle(RX1) - angle(RX2)
    = frequency intercept + RX1_effect(gain1) + RX2_effect(gain2) + residual
```

The fit command evaluates predictions only on an epoch excluded from training.
It also performs paired, identical-mask comparisons against:

- a model that knows only `gain1 - gain2`;
- an additive model plus one residual for every ordered gain pair;
- one set of gain curves shared by all RF frequencies; and
- independent per-frequency baselines versus a constant phase plus a linear
  frequency slope (an effective differential-delay description); and
- the additive model adjusted from one equal-gain anchor frame in each held-out
  frequency/epoch.

Differences smaller than the declared 0.1° held-out MAE margin are treated as
practically equivalent and select the predeclared simpler operational choice.
The report also repeats additive cross-validation with both channels above
-10, 0, and 10 dB tone SNR. These are confidence/coverage diagnostics, not
permission to use a frame that failed any base quality check.

`predict_phase_offset()` accepts only an exact fitted frequency and an ordered
gain pair that passed the repeatability criterion. It raises instead of
interpolating an unsupported point, even when both individual gains were
observed elsewhere in the surface. With the recorded convention, apply a valid
prediction as:

```text
corrected_phase = wrap(measured_angle_RX1_minus_RX2 - predicted_offset)
```

## Pass/fail interpretation

- Structural validation must pass: all scheduled frames, exact V7 shape,
  direct-USB protocol v2, verified firmware/serial provenance, valid endpoint
  gain and RSSI metadata, and recomputed IQ metrics matching stored values.
- Each gain/frequency cell passes when at least two of its three epochs pass
  frame quality and repeat phase circular standard deviation is at most 5°.
- A failed cell is not assigned a phase correction. This is expected where one
  channel is too weak or either ADC clips.
- Model effects are emitted only for gain states observed in that receiver
  role. Missing effects are JSON `null`; they must never be interpreted as
  zero phase correction.
- A correction is available only for the exact radio serial and calibrated RF
  frequency, with non-null RX1/RX2 effects and a live frame that passes gain
  metadata and signal-quality checks. Never extrapolate across an unsupported
  gain, frequency, clipping region, or weak-signal region.
- The ordered gain pair matters. A model based only on gain difference is
  retained as a measured baseline, not assumed correct.
- The fitted baseline follows `phase = constant - 2*pi*frequency*delay`.
  Its reported delay and free-space-equivalent path describe the measured
  electrical group delay. They do not identify literal cable length: cables,
  splitter paths, PCB traces, analogue filters, and retune/calibration state
  can all contribute. The report gives the linear-fit residual and a paired
  held-out comparison against independent frequency baselines.
