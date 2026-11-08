# Synthetic raw datasets — the math of emission, reading, and hardware artifacts

How to generate a **raw** `signal_matrix` dataset that flows through the *entire* SPF pipeline
(segmentation → windowed beamformer → model), and the signal model behind it. Grounded in
`spf/dataset/fake_dataset.py` (parametric generator) and `spf/rf.py` (physics sim). Physics
background: [`../00_concepts/beamforming_primer.md`](../00_concepts/beamforming_primer.md).

## Why "raw" matters (vs how we actually trained)

The jun26 single/paired runs trained on **`nosig`** data — the `windowed_beamformer` and
`all_windows_stats` were precomputed, and `signal_matrix` was never loaded at train time
(`signal_matrix_input: false`). So the training path *skips* segmentation.

A synthetic **raw** dataset is different and more valuable: you generate the complex IQ
`signal_matrix (records, 2, buffer_size)` and let the real segmentation stage turn it into the
windowed beamformer. That exercises the part of the pipeline the training path skips:

```
 synthetic raw IQ ──▶ segmentation (detrend → per-window delay-and-sum → circular phase stats
 (2 × buffer_size)     → signal/noise mask) ──▶ windowed_beamformer + all_windows_stats ──▶ model
```

## The signal model (parametric generator, `fake_dataset.py`)

### 1. Emission — abstracted to "a coherent tone at the array"

The generator does **not** simulate a carrier waveform; it works in complex baseband. Each antenna
element sees a unit phasor `e^{jφ_element}` (scaled ×200 for amplitude). The transmitter is a single
narrowband source; its *absolute* phase is unobservable, so it's drawn randomly per record
(`offsets = rand·4π`, `phi_to_signal_matrix:39`) and applied to **both** elements — only the
*difference* between elements carries direction information. (The higher-fidelity path in `rf.py`
*does* synthesize a tone: `IQSource.signal(t) = e^{j2πt}`, `rf.py:827+`.)

### 2. Reading — geometry turns angle θ into inter-element phase φ

A plane wave from angle θ (measured from the array normal/broadside) reaches the second element
over an extra path `Δd = d·sin(θ)`, giving a phase difference (`theta_to_phi`, `:220-221`; derivation `:55-56`):

```
λ  = c / f_carrier                      # rx_lo
Δd = d · sin(θ)                         # d = antenna spacing (rx_spacing, metres)
φ  = 2π · Δd / λ = 2π · (d/λ) · sin(θ)
```

Each of the two receivers is mounted at a different orientation, so the world angle is rotated into
each array's frame (`:238-240`): `θ_rx = θ_world − rx_theta_in_pis · π`.

The signal matrix is then (`phi_to_signal_matrix:40-48`), for a record with inter-element phase φ:

```
row 0 (reference) = 200 · exp( j·(offset + φ_drift) )
row 1 (element 2) = 200 · exp( j·(offset + φ + per-sample noise) )
```

The pipeline recovers the observable via `get_phase_diff = pi_norm(angle(row0) − angle(row1))`
(`rf.py:745`) ≈ `−φ` (the sign matches the `−sin(θ)` convention in the dataset's `ground_truth_phi`).
Beamforming (in segmentation) then scans candidate angles with steering vectors
`s(θ) = e^{−j·2π(d/λ)sin(θ)}` and reads the power peak. **Front/back ambiguity** is baked into the
physics: `sin(θ) = sin(π−θ)`, so φ can't separate θ from π−θ — exactly the symmetry the model's
target fold encodes (see [`../04_training_inference/latest_models.md`](../04_training_inference/latest_models.md)).

### 3. Hardware artifacts — what's modeled, and what isn't

| Artifact | Modeled by parametric gen | Where | Real-world meaning |
|---|---|---|---|
| Unknown absolute phase | ✅ | `offsets = rand·4π` (`:39`) | LO phase is arbitrary per capture; only Δphase carries angle |
| Thermal / phase noise | ✅ | `+ randn·noise` on φ per sample (`:36`) and on θ per record (`:233,242`) | broadens the beamformer peak, raises circular-stddev |
| **Inter-radio phase drift / miscalibration** | ✅ | `φ_drift·π·(±1)` — **opposite sign on the two receivers** (`:254`) | the AD9361 RX1/RX2 phase-inversion / per-SDR LO offset (why `sdr_controller` has a phase-cal step) |
| **Intermittent signal (noise windows)** | ✅ | every 3rd `2048·4`-sample window replaced by `torch_random_signal_matrix` (`:262-267`) | the emitter is bursty / the buffer has noise stretches — this is *what segmentation's signal/noise mask exists to detect* |
| Amplitude / gain | ⚠️ crude | `×200`; `gains=[0,0]`, `rssis=[0,0]` placeholders | per-channel RSSI/AGC |
| Multipath / reflections | ❌ (parametric) / ✅ (physics sim, multi-source) | `rf.py` `Detector.add_source` | second arrival at a different angle/delay |
| Doppler / frequency offset | ❌ | — | moving platform, LO mismatch |
| IQ imbalance, DC offset, ADC quantization | ❌ | — | analog front-end + ADC non-idealities |
| Clock/timing jitter, AGC dynamics | ❌ | — | sample-clock drift, gain settling |
| Real emitter modulation (WiFi/LoRa burst) | ❌ (CW only) | — | actual packets vs a pure tone |

The off-signal filler `torch_random_signal_matrix` (`utils.py:73`) is uniform complex noise in
~`[-1,1] + j[-0.5,0.5]` — deliberately different scale/structure from the ×200 signal so the
segmentation mask can separate them.

## Two generation paths

| Path | Function | Fidelity | Use |
|---|---|---|---|
| **Parametric** | `create_fake_dataset` (`fake_dataset.py:165`) | φ synthesized directly from θ | fast; unit tests; the `conftest` fixtures; pipeline smoke |
| **Physics sim** | `rf.py` `Detector`/`ULADetector` + `IQSource` (via `spf_generate.py`) | true array + source geometry, per-element delays, **multi-source (multipath)**, moving detector, `sigma` noise | realistic scenes; modeling artifacts the parametric path can't |

The physics path holds real `source_positions` + `receiver_positions`, computes
`distance_receiver_to_source` → per-element delay → summed complex signal at `sampling_frequency`,
with an array `orientation` and additive `sigma`. Add several sources to model multipath.

## Recipe — a synthetic raw dataset through the full pipeline

This is exactly what the test fixtures do (`tests/conftest.py`):

```python
from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.dataset.spf_dataset import v5spfdataset

# 1) a v5 config: 2 receivers, spacing, carrier (rx_lo), bandwidth, buffer_size,
#    n-records-per-receiver, data-version: 5, per-receiver theta-in-pis.
yaml_str = fake_yaml(version=5)          # template to edit

# 2) generate raw signal_matrix + v5 metadata (tx orbits at `radius`, rx at origin)
create_fake_dataset(
    yaml_config_str=yaml_str, filename="/tmp/synth", n=200,
    noise=0.01,        # per-sample/record Gaussian phase noise
    phi_drift=0.0,     # inter-radio calibration offset (±)
    radius=10000,      # tx orbit radius (mm)
    seed=0,            # seeds numpy+torch+random → reproducible
)

# 3) segmentation runs here (raw IQ → windowed_beamformer + all_windows_stats in the .yarr)
ds = v5spfdataset("/tmp/synth", nthetas=65, precompute_cache="/tmp/cache",
                  segment_if_not_exist=True)

# 4) ds[i] now flows to the model exactly like real data
```

The tx trajectory is a circle: `tx_pos = (sin(θ)·radius, cos(θ)·radius)`, `θ` sweeping `orbits·2π`
(`:216-218, 273-274`) — so ground-truth θ is known per record and the model can be scored against it.

## Extending toward realism (design levers)

To make a synthetic raw set that stresses the pipeline the way real captures do, in rough order of
payoff:

1. **Realistic trajectories** — replace the perfect circle with recorded/plausible `tx_pos`/`rx_pos`
   paths (the model saw rover missions + wall-array sweeps, not clean orbits).
2. **Multipath** — use the `rf.py` physics sim with ≥2 sources (a direct path + a reflection at a
   different angle/amplitude); the parametric generator can't do this.
3. **Per-radio calibration + drift** — set `phi_drift` (and make it slowly time-varying) to mimic
   the RX1/RX2 phase-inversion the real hardware fights.
4. **Front-end non-idealities** — add IQ imbalance (`I·(1+ε) + jQ`), a small DC offset, and ADC
   quantization to the `signal_matrix` before writing.
5. **Frequency offset / Doppler** — multiply by `e^{j2πΔf·t}` across the buffer.
6. **Emitter waveform** — replace the CW tone with a bursty packet envelope (WiFi/LoRa) so the
   noise-window / segmentation behavior matches reality.

Each of these is a **generator-side** change (write a custom producer of `signal_matrix` + metadata,
or extend `create_fake_dataset`) and requires **no change to the consuming pipeline** — segmentation
and the model take whatever raw IQ you write, as long as the v5 schema
(`v5_data.py::v5rx_keys`) is respected.
