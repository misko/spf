# 02 · RF Signal Processing & SDR — overview

The core that turns raw antenna IQ into angle information: the beamforming math (`spf/rf.py`),
the SDR hardware drivers (`spf/sdrpluto/`), and the threaded capture loop (`spf/data_collector.py`).
Read [`00_concepts/beamforming_primer.md`](../00_concepts/beamforming_primer.md) first for the physics.

## `spf/rf.py` — the beamforming core (~1556 lines)

The single most important module. Two responsibilities: **simulate** antenna arrays, and
**process** real signals into angle estimates.

### Processing API (the part the pipeline uses)

| Function (`spf/rf.py`) | Role |
|---|---|
| `beamformer(...)` (`:1352`) | Top-level: build steering vectors + apply → `(thetas, power_spectrum, steering_vectors)`. |
| `precompute_steering_vectors(...)` (`:1173`) | Build the `(nthetas, n_elements)` steering matrix once, reuse per snapshot. |
| `beamformer_given_steering_nomean(...)` (`:1273`) | Apply precomputed steering, keep the per-sample/time axis (numpy). |
| `beamformer_given_steering_nomean_fast(...)` (`:1204`) | Numba-accelerated CPU variant. |
| `beamformer_given_steering_nomean_cp(...)` (`:1240`) | CuPy GPU variant. |
| `beamformer_given_steering(...)` (`:1309`) | Apply + mean over time. |
| `beamformer_thetas(spacing)` (`:1336`) | The θ grid that the spectrum is defined over. |

> Note: there is **no runtime dispatch** between the numpy / numba / cupy variants — the caller
> picks. Keep this in mind when chasing performance or a device mismatch.

### Phase & circular statistics

φ extraction and robust averaging that respects the ±π wraparound: `get_phase_diff`,
`get_avg_phase` (+`_fast` variants), `phase_diff_to_theta` (the `arcsin` relation),
`circular_mean` / `circular_stddev` (+`_fast`), `pi_norm`, plus `torch_*` equivalents for GPU.
Windowed versions (`windowed_trimmed_circular_mean_and_stddev_*`) are what segmentation calls
per window. Peak finding: `get_top_3_peaks`, `get_peaks_for_2rx`. Distribution rotation:
`rotate_dist`, `reduce_theta_to_positive_y`.

### Simulation classes (for synthetic data / tests)

`Source` / `IQSource` / `SinSource` (signal models) and `Detector` / `ULADetector` (uniform
linear array) / `UCADetector` (uniform circular array). These generate signal matrices for a
given source geometry — used by the data generators in [`03_datasets`](../03_datasets/overview.md)
and by tests. 🧪 The UCA path and some `Source` subclasses are simulation-only / lightly used.

### Conventions enforced here

`c = 299_792_458`; angles in radians on `[-π, π]`; signal matrix `(n_receivers, n_samples)`;
steering vectors `(nthetas, n_elements)`. See [conventions.md](../00_concepts/conventions.md).

## `spf/sdrpluto/` — SDR hardware control (~3065 lines)

Driver abstraction over **PlutoSDR** (Analog Devices AD9361) and **BladeRF**.

| Class (`spf/sdrpluto/sdr_controller.py`) | Role |
|---|---|
| `ReceiverConfig` / `EmitterConfig` | RX/TX parameter bundles (LO, bandwidth, sample rate, gains, gain mode, spacing, orientation). |
| `PPlus` | Real PlutoSDR wrapper via `adi`/libiio. `setup_rx_config()` configures the AD9361 incl. RX1/RX2 **phase-inversion calibration** (register writes); `rx()` returns one IQ buffer; `rssis()`, `gains()`. |
| `BladeRFSdr` | BladeRF 2.0 wrapper via the `bladerf` library. |
| `FakePPlus` / `FakeSdr` | Mock SDRs for hardware-free testing. |

The **phase-inversion register workaround** is load-bearing: the AD9361 has a known RX1↔RX2
phase asymmetry, and beamforming requires phase coherence, so `setup_rx_config()` forces
alignment. Also here: `detrend.py` (remove DC/linear drift from IQ before processing),
`benchmark.py`, `test_throughput.py`, and the `firmware/` + `setup/` hardware bring-up assets.

## `spf/data_collector.py` — threaded capture (~655 lines)

Bridges SDR hardware and the dataset writer during a collection run.

- **Snapshot dataclasses**: `DataSnapshotRaw` → `DataSnapshotV4` (adds GPS) → `DataSnapshotV5`
  (adds tx/rx XY in mm + heading).
- **`ThreadedRX` / `ThreadedRXRaw`**: background threads that read the SDR forever, (optionally)
  beamform with precomputed steering vectors, and queue snapshots (drop-stale, single-buffer).
- **`DataCollector`** + subclasses `DroneDataCollectorRaw`, `GrblDataCollectorRaw`,
  `GrblDataCollector` — orchestrate multiple `ThreadedRX`, timing, and zarr I/O.

This is the **handoff from 01→03**: the collector is what a `*_radio_collection.py` orchestrator
([`01`](../01_data_collection/overview.md)) drives, and what writes the `.zarr`.

## Known issues / tech debt (spot-verified)

- **No dispatch across beamformer variants** (numpy / numba `_fast` / cupy `_cp`) — caller must
  choose; easy to run the slow path or the wrong device unknowingly.
- **Multiple overlapping circular-mean implementations** (`circular_mean`,
  `circular_mean_single_fast`, `torch_*`) with differing signatures — no single canonical entry.
- 🧪 **TX/Emitter path is partial** — `EmitterConfig` exists but the SDR TX path in `PPlus` is
  incomplete; emitters in the field are usually ESP32/external, not SDR-TX.
- **AD9361 phase-inversion register writes** are hardware-version-sensitive; treat as fragile when
  changing SDR firmware.

## Planned leaf pages

- `beamforming_core.md` — **DONE** → [`../reference/spf/rf.md`](../reference/spf/rf.md). Function-level
  contract reference for `rf.py`, full 3-pass + adversarially verified. Covers **cluster A**
  (angle/phase utilities — `pi_norm`, `circular_mean`, `phase_diff_to_theta`), **cluster B**
  (beamforming & steering vectors — `precompute_steering_vectors`, `beamformer`, the
  `beamformer_given_steering*` family, `beamformer_thetas`), **cluster C** (windowed circular
  statistics), and **cluster D** (phase extraction).
- `sdr_hardware.md` — *not written yet*; current source of truth is the Pass-1 surface inventory
  [`../reference/_inventory/rf_sdr.md`](../reference/_inventory/rf_sdr.md) (covers `sdrpluto/`
  config, AD9361 calibration, detrending). PlutoSDR/BladeRF config + calibration to be deep-dived.
- `data_collector.md` — *not written yet*; current source of truth is the same inventory
  [`../reference/_inventory/rf_sdr.md`](../reference/_inventory/rf_sdr.md) (covers
  `data_collector.py` threading model, snapshot dataclasses, queueing/timing).

## Known issues

RF/SDR-relevant entries from [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) (read-only audit; nothing
fixed). Severities as currently triaged there:

- **#6 🟡 P2** — `setup_rxtx_and_phase_calibration` → `AttributeError`
  (`spf/sdrpluto/sdr_controller.py:971,982`). **Not on the live collection path** — only the
  standalone `--mode rxcal` CLI; normal collection uses `setup_rxtx`/`setup_rx` →
  `setup_rx_config()`. Fix: `setup_rx()`→`setup_rx_config()`.
- **#14 🟠 P1** — `FakePPlus.get_rssi_and_gain` reads undefined `self.dev` (only `self.sdr` is set)
  (`spf/sdrpluto/sdr_controller.py:543`).
- **#15 🟠 P1** — `sdr.py` constructs detectors and **creates matplotlib figures at import time**
  (`spf/sdrpluto/sdr.py:212-230`).
- **#16 🟠 P1** — `data_collector` writes `data.heading` where `rx_heading_in_pis` is expected
  (latent field mismatch) (`spf/data_collector.py:~540`).
- **#20 🟡 P2** — `torch_mean_phase_mean` does `np.*` ops on torch input (broken) — also dead, no
  callers (`spf/rf.py:424`).
- **#21 🟡 P2** — `windowed_…_fast2` passes scalars to an array fn (crashes); only call site is
  commented out (`spf/rf.py:557`, call site `segmentation.py:661`).
- **#37 🟡 P2** — rf.py trimmed-stat **backend divergence**: live `fast_percentile_1d` (ceil-rank)
  is untested; the *tested* `fast_percentile` (floor-rank) has zero production callers; the torch
  path uses a third convention. Trimmed stats differ by backend, unverified on the real path
  (`spf/rf.py:348` live vs `82` tested).
