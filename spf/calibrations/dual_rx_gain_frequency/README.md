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

The hardware adapter performs one standard-IIO RX DMA priming read immediately
before arming cyclic TX at each frequency. It destroys that IIO buffer before
direct USB starts; there are no host-side IIO gain or RSSI reads in the frame
loop. TX attenuation changes in place so the cyclic TX DMA and phase continuity
are preserved.

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

After the pilot establishes usable signal levels, replace `pilot_5ghz.yaml`
with `coarse_5ghz.yaml` to enumerate all 73 manual gain states on both
receivers at the four coarse frequencies. Runs are resumable: the completion
bit is committed only after IQ, V7 metadata, coordinates, and quality metrics.

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
- The fitted frequency slope is descriptive. LO retuning can introduce phase
  state changes, so it is not claimed to be physical cable delay.
