"""Performance regression tests for the inference path (matrix rows RT5/RT12 + new P-rows).

Philosophy: deterministic proxies first (call counts, payload sizes — cannot flake),
then wall-clock ceilings set ~10x above healthy baselines so they catch order-of-
magnitude regressions (a second forward pass, an accidental O(n^2), raw IQ through a
queue) without ever failing on CI noise. Baselines are printed so trends are visible
in CI logs.
"""

import time
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# generous ceilings (healthy values are ~10x lower; see printed baselines)
MODEL_FORWARD_CEILING_S = 0.5
WINDOW_STATS_CEILING_S = 2.0
BEAMFORMER_CEILING_S = 5.0
REALTIME_QUEUE_PAYLOAD_CEILING_BYTES = 1_000_000


# ----------------------------------------------------------------- P2 (RT12 witness)
def test_p2_realtime_generator_single_forward_per_sample(monkeypatch):
    """Exactly one model forward per realtime sample, and eval() enforced.
    (RT12: the old loop ran the model twice and accumulated an unbounded list.)"""
    import spf.model_training_and_inference.models.single_point_networks_inference as spi

    monkeypatch.setattr(spi, "global_config_to_keys_used", lambda global_config: [])
    monkeypatch.setattr(spi, "v5_collate_keys_fast", lambda keys, samples: SimpleNamespace(
        to=lambda device: samples[0]
    ))

    calls = {"n": 0, "eval": False}

    class CountingModel:
        def eval(self):
            calls["eval"] = True

        def __call__(self, x):
            calls["n"] += 1
            return {"paired": torch.zeros(2, 65)}

    samples = [object() for _ in range(7)]
    out = list(
        spi.single_example_realtime_inference(
            CountingModel(), global_config={}, optim_config={"device": "cpu"},
            realtime_ds=iter(samples),
        )
    )
    assert len(out) == 7
    assert calls["n"] == 7, f"expected 1 forward/sample, got {calls['n']} for 7 samples"
    assert calls["eval"], "model.eval() must be enforced in the realtime path"


# ----------------------------------------------------------------- P3 (RT5 red)
@pytest.mark.xfail(
    strict=True,
    reason="RT5: realtime queue payload carries the full raw signal_matrix (~8MB "
    "per snapshot per radio); features should be extracted writer-side",
)
def test_p3_realtime_queue_payload_bounded():
    from spf.data_collector import DataSnapshotV4, DroneDataCollectorRaw

    kwargs = {}
    for f in fields(DataSnapshotV4):
        if f.name == "signal_matrix":
            kwargs[f.name] = np.zeros((2, 524288), dtype=np.complex64)  # realistic
        else:
            kwargs[f.name] = 0.0
    data = DataSnapshotV4(**kwargs)

    captured = {}

    class _RT:
        def write_to_idx(self, record_idx, thread_idx, data_dict):
            captured.update(data_dict)

    collector = object.__new__(DroneDataCollectorRaw)
    collector.position_controller = SimpleNamespace(
        get_position_bearing_and_time=lambda: {
            "heading": 0.0, "gps": (0.0, 0.0), "gps_time": 0.0,
        }
    )
    collector.realtime_v5inf = _RT()
    collector.data_filename = None
    collector.write_to_record_matrix(0, 0, data)

    payload = sum(
        v.nbytes for v in captured.values() if isinstance(v, np.ndarray)
    )
    assert payload < REALTIME_QUEUE_PAYLOAD_CEILING_BYTES, (
        f"realtime queue payload is {payload/1e6:.1f} MB per snapshot per radio"
    )


# ----------------------------------------------------------------- P1 model latency
def test_p1_model_forward_latency(
    perfect_circle_dataset_n7_with_empirical,
    paired_net_checkpoint_using_single_checkpoint,
):
    from spf.dataset.spf_dataset import v5_collate_keys_fast, v5spfdataset
    from spf.model_training_and_inference.models.single_point_networks_inference import (
        load_model_and_config_from_config_fn_and_checkpoint,
    )
    from spf.scripts.train_utils import global_config_to_keys_used

    root_dir, empirical_pkl_fn, zarr_fn = perfect_circle_dataset_n7_with_empirical
    ckpt_dir = paired_net_checkpoint_using_single_checkpoint
    m, config = load_model_and_config_from_config_fn_and_checkpoint(
        f"{ckpt_dir}/config.yml", f"{ckpt_dir}/best.pth", device="cpu"
    )
    m.eval()
    ds = v5spfdataset(
        f"{zarr_fn}.zarr" if not str(zarr_fn).endswith(".zarr") else str(zarr_fn),
        nthetas=config["global"]["nthetas"],
        ignore_qc=True,
        precompute_cache=root_dir,
        empirical_data_fn=empirical_pkl_fn,
        paired=True,
        skip_fields=set(["signal_matrix"]),
    )
    keys = global_config_to_keys_used(global_config=config["global"])
    batch = v5_collate_keys_fast(keys, [ds[0]]).to("cpu")

    with torch.no_grad():
        for _ in range(3):  # warmup
            m(batch)
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            m(batch)
        per_forward = (time.perf_counter() - t0) / n
    print(f"\n[perf-baseline] model forward: {per_forward*1e3:.2f} ms/sample (cpu)")
    assert per_forward < MODEL_FORWARD_CEILING_S


# ----------------------------------------------------------------- P4 feature latency
def test_p4_window_stats_latency():
    """Per-snapshot feature extraction must stay far under capture cadence."""
    from spf.dataset.segmentation import get_all_windows_stats

    rng = np.random.default_rng(0)
    n = 524288  # production buffer
    sig = (
        rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
    ).astype(np.complex64)
    get_all_windows_stats(sig, window_size=2048, stride=2048, trim=20.0)  # warmup/jit
    t0 = time.perf_counter()
    get_all_windows_stats(sig, window_size=2048, stride=2048, trim=20.0)
    dt = time.perf_counter() - t0
    print(f"\n[perf-baseline] window stats (2x524288): {dt*1e3:.1f} ms/snapshot")
    assert dt < WINDOW_STATS_CEILING_S


def test_p4_beamformer_latency():
    from spf.rf import (
        beamformer_given_steering_nomean_fast,
        precompute_steering_vectors,
    )

    rng = np.random.default_rng(0)
    n = 524288
    sig = (
        rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
    ).astype(np.complex64)
    steering = precompute_steering_vectors(
        receiver_positions=np.array([[0.0, 0.02], [0.0, -0.02]]),
        carrier_frequency=2.412e9,
        spacing=65,
    )
    beamformer_given_steering_nomean_fast(steering, sig)  # warmup numba jit
    t0 = time.perf_counter()
    beamformer_given_steering_nomean_fast(steering, sig)
    dt = time.perf_counter() - t0
    print(f"\n[perf-baseline] beamformer 65x{n}: {dt*1e3:.1f} ms/snapshot")
    assert dt < BEAMFORMER_CEILING_S
