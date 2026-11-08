# Glossary

Cross-cutting vocabulary. These terms appear everywhere; the data layer in particular
overloads a few of them, so this page is the canonical definition.

| Term | Meaning |
|---|---|
| **Snapshot** | One capture from **one receiver** at one timestamp: a complex IQ `signal_matrix` of shape `(2, buffer_size)` plus metadata (positions, gains, RSSI, etc.). The atomic record in a zarr store. |
| **Session** | A group of `snapshots_per_session` snapshots that a dataset returns as one item. `=1` for single-point models, `16+` for temporal/transformer models. Sessions can be tiled (overlapping) or disjoint. |
| **Window** | A fixed-length slice (default 2048 samples) **within** a snapshot's signal. Segmentation computes per-window phase statistics; `n_windows ≈ buffer_size / window_size` (~256). |
| **Receiver (rx)** | One physical 2-element antenna array + SDR. SPF datasets assume exactly **2 receivers** (`r0`, `r1`). |
| **Paired** | A dataset/model mode that returns/fuses **both receivers** together. Note: "paired" ≠ paired samples in the ML sense — it means "both radios of one craft." |
| **Craft** | The vehicle carrying the two receivers (rover or wall array). "craft-relative" angles are measured against the vehicle's heading rather than an absolute frame. |
| **Routine** | The motion pattern a collection run drives: `bounce`, `circle`, `center`, `diamond`, calibration routines. |
| **Beamformer output** | Power-vs-angle spectrum, a vector of length `nthetas` (typically 65). High value at θ ⇒ signal likely arriving from θ. |
| **Segmentation** | Pre-processing that labels which windows contain signal vs noise and caches per-window/per-session phase + beamformer stats. Cached in a `.yarr` store. |
| **φ (phi)** | **Phase difference** between the two antenna elements of a receiver. The raw observable. |
| **θ (theta)** | **Angle of arrival** of the signal. The thing we want. Related to φ by the array geometry (see [beamforming_primer.md](beamforming_primer.md)). |
| **nthetas** | Number of discrete angle bins the spectrum / network output is quantized into (65 typical). |
| **d/λ** (`rx_wavelength_spacing`) | Antenna spacing in wavelengths. Sets the φ↔θ mapping. |
| **Empirical distribution** | A pre-learned lookup table `P(θ | φ)` built from data; used as a prior/feature and as the likelihood in the non-NN particle filters. |
| **Steering vector** | The complex phase-compensation vector for a candidate angle θ; applied to the signal to test "how much energy comes from θ." |

## Format / version vocabulary

| Term | Meaning |
|---|---|
| **v4** | Dataset format with GPS-based positioning (used by rovers). Stores raw signal matrix. |
| **v5** | Dataset format with pre-computed XY positions in mm + receiver heading (used by wall arrays; preferred for new data). |
| **3pX / 3.7** | Segmentation algorithm version. Current `SEGMENTATION_VERSION = 3.7` (`spf/utils.py:14`). |
| **`.zarr`** | The raw dataset store (LMDB-backed zarr). |
| **`.yarr`** | The segmentation cache store derived from a `.zarr`. |

See [`03_datasets/overview.md`](../03_datasets/overview.md) for the full schema.
