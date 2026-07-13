"""Green-at-birth regression guards (matrix rows G1-G5).

These pin invariants that HOLD today — including this project's own recent fixes
(#53/#54) and the numerical outputs the precompute caches depend on. They are not
xfail: a failure here means a regression, not a known bug.
"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

GOLDEN = Path(__file__).parent / "golden_windows_stats_v3p7.npz"


# ---------------------------------------------------------------------------- G1
def test_g1_windows_stats_match_golden():
    """The window-stats function both training-precompute and realtime paths share
    must be numerically stable. If this fails because of an INTENTIONAL change,
    bump segmentation_version and regenerate the golden in the same diff —
    otherwise months of precompute caches silently desync from the code."""
    from spf.dataset.segmentation import get_all_windows_stats

    rng = np.random.default_rng(1234)
    n = 65536
    t = np.arange(n)
    base = np.exp(1j * (2 * np.pi * 0.01 * t))
    sig = np.stack(
        [
            base + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)),
            base * np.exp(1j * 0.7)
            + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)),
        ]
    ).astype(np.complex64)
    sig[:, 20000:30000] *= 0.01

    out = get_all_windows_stats(sig, window_size=2048, stride=2048, trim=20.0)
    stats = np.asarray(out[1] if isinstance(out, tuple) else out)
    golden = np.load(GOLDEN)["stats"]
    np.testing.assert_allclose(stats, golden, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------- G2
def test_g2_checkpoint_carries_best_val_loss(single_net_checkpoint):
    """#53: every checkpoint must persist the best-val watermark so a resume
    cannot clobber best.pth with a worse epoch."""
    ck = torch.load(
        f"{single_net_checkpoint}/best.pth", map_location="cpu", weights_only=False
    )
    assert "best_val_loss" in ck, "#53 regression: best_val_loss missing from save"


def test_g2_load_checkpoint_returns_watermark(single_net_checkpoint):
    from spf.model_training_and_inference.models.single_point_networks_inference import (
        load_model_and_config_from_config_fn_and_checkpoint,
    )
    from spf.scripts.train_single_point import load_checkpoint, load_model

    config_fn = f"{single_net_checkpoint}/config.yml"
    ckpt_fn = f"{single_net_checkpoint}/best.pth"
    m, config = load_model_and_config_from_config_fn_and_checkpoint(
        config_fn, ckpt_fn, device="cpu"
    )
    out = load_checkpoint(
        checkpoint_fn=ckpt_fn,
        config=config,
        model=m,
        optimizer=None,
        scheduler=None,
        force_load=True,
    )
    assert len(out) == 6, "#53: load_checkpoint must return 6-tuple incl. watermark"


# ---------------------------------------------------------------------------- G3
def test_g3_pre_fix_checkpoint_still_loads(single_net_checkpoint, tmp_path):
    """Old checkpoints (saved before #53, no best_val_loss key) must keep loading,
    yielding a None watermark."""
    from spf.model_training_and_inference.models.single_point_networks_inference import (
        load_model_and_config_from_config_fn_and_checkpoint,
    )
    from spf.scripts.train_single_point import load_checkpoint

    config_fn = f"{single_net_checkpoint}/config.yml"
    ckpt_fn = f"{single_net_checkpoint}/best.pth"
    ck = torch.load(ckpt_fn, map_location="cpu", weights_only=False)
    ck.pop("best_val_loss", None)
    old_fn = str(tmp_path / "old_format.pth")
    torch.save(ck, old_fn)

    m, config = load_model_and_config_from_config_fn_and_checkpoint(
        config_fn, ckpt_fn, device="cpu"
    )
    out = load_checkpoint(
        checkpoint_fn=old_fn,
        config=config,
        model=m,
        optimizer=None,
        scheduler=None,
        force_load=True,
    )
    assert out[-1] is None, "old-format checkpoint should yield None watermark"


# ---------------------------------------------------------------------------- G5
def test_g5_paired_heads_identical(
    perfect_circle_dataset_n7_with_empirical,
    paired_net_checkpoint_using_single_checkpoint,
):
    """The PF observation reads only radio 0's paired head (PF7). That is safe
    ONLY while the paired model emits identical fused output for both radios.
    This guard trips the moment that architectural assumption changes."""
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
        out = m(batch)
    assert "paired" in out
    paired = out["paired"]
    assert torch.allclose(paired[0], paired[1], atol=1e-6), (
        "paired heads differ between radios — PF observation (radio 0 only) is no "
        "longer safe; fix PF7 before relying on this model"
    )
