"""Red/green tests for the realtime inference stack.

Row IDs reference claude_docs/reference/_realtime_redgreen_matrix.md.
Policy (G10): tests for UNFIXED bugs are strict xfail — CI fails the moment a fix
makes them pass, forcing the marker's removal in the same diff. Tests whose fix has
landed run as normal green tests (they are the red→green witnesses).
"""

import inspect
import math
import re
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import spf


# --------------------------------------------------------------------------- RT2
def test_rt2_no_live_debugger_calls():
    """No uncommented breakpoint()/pdb in the shipped package (4 sites removed)."""
    pat = re.compile(r"^\s*(breakpoint\(\)|pdb\.set_trace\(\)|import\s+ipdb)", re.M)
    offenders = []
    for f in Path(spf.__path__[0]).rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        if pat.search(f.read_text(errors="ignore")):
            offenders.append(str(f))
    assert offenders == [], f"live debugger calls in: {offenders}"


# --------------------------------------------------------------------------- RT1
def test_rt1_heading_carried_into_realtime_dict():
    """asdict() only serializes declared dataclass fields; the collector must set
    the declared rx_heading_in_pis field (deg/180), not just the dynamic .heading."""
    from spf.data_collector import DataSnapshotV4, DroneDataCollectorRaw

    kwargs = {}
    for f in fields(DataSnapshotV4):
        if f.name == "signal_matrix":
            kwargs[f.name] = np.zeros((2, 16), dtype=np.complex64)
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
            "heading": 90.0,
            "gps": (1.0, 2.0),
            "gps_time": 3.0,
        }
    )
    collector.realtime_v5inf = _RT()
    collector.data_filename = None
    collector.write_to_record_matrix(0, 0, data)

    assert captured, "realtime dict was not written"
    assert captured["rx_heading_in_pis"] == pytest.approx(90.0 / 180.0), (
        "heading dropped by asdict(): rx_heading_in_pis must carry heading/180"
    )


# --------------------------------------------------------------------------- RT4
def test_rt4_v4_heading_conversion_is_deg_over_180():
    """v4->v5 on-the-fly upgrade must convert degrees with (h/360)*2 (deg/180),
    matching v4_tx_rx_to_v5.py — not the /2 typo (deg/720)."""
    from spf.dataset.spf_dataset import v5spfdataset

    ds = object.__new__(v5spfdataset)
    ds.v4 = True
    ds.n_receivers = 1
    ds.receiver_data = [
        {
            "system_timestamp": np.zeros(3),
            "heading": np.array([0.0, 90.0, 180.0]),
        }
    ]
    ds.v4_to_v5()
    got = np.asarray(ds.receiver_data[0]["rx_heading_in_pis"][:])
    np.testing.assert_allclose(got, [0.0, 0.5, 1.0], atol=1e-9)


# --------------------------------------------------------------------------- PF1
class _StubPF:
    pass


def _make_stub_pf():
    from spf.filters.filters import ParticleFilter

    class StubPF(ParticleFilter):
        def __init__(self):
            self.ds = [0, 1, 2]

        def our_state(self, idx):
            return None

        def observation(self, idx):
            return torch.ones(5) / 5

        def predict(self, our_state, dt, noise_std):
            pass

        def update(self, z):
            pass

    return StubPF()


def test_pf1_return_particles_uses_clone():
    """trajectory(return_particles=True) crashed on Tensor.copy(); must clone."""
    pf = _make_stub_pf()
    traj = pf.trajectory(
        mean=torch.tensor([[0.0, 0.0]]),
        std=torch.tensor([[1.0, 0.1]]),
        N=8,
        steps=2,
        return_particles=True,
    )
    assert len(traj) == 2 and isinstance(traj[0]["particles"], torch.Tensor)
    # snapshot must be independent of the live particle tensor
    before = traj[-1]["particles"].clone()
    pf.particles += 100.0
    assert torch.equal(traj[-1]["particles"], before)


# --------------------------------------------------------------------------- PF2
def test_pf2_estimate_circular_at_pi_seam():
    """Arithmetic mean of wrapped angles collapses to ~0 for a cloud straddling
    +-pi; angular estimate must return ~pi."""
    from spf.filters.filters import estimate_angular_dim0
    from spf.rf import torch_pi_norm_pi

    p = torch.tensor([[math.pi - 0.05, 0.0]] * 50 + [[-math.pi + 0.05, 0.0]] * 50)
    w = torch.ones(100, dtype=torch.float64) / 100
    mu, var = estimate_angular_dim0(p, w)
    assert abs(float(torch_pi_norm_pi(mu[0] - math.pi))) < 0.2
    assert float(var[0]) < 0.1  # wrapped variance, not ~pi^2


def test_pf2_theta_filters_are_marked_angular():
    from spf.filters.particle_dual_radio_nn_filter import PFSingleThetaDualRadioNN
    from spf.filters.particle_dualradio_filter import PFSingleThetaDualRadio
    from spf.filters.particle_dualradioXY_filter import PFXYDualRadio
    from spf.filters.particle_single_radio_filter import PFSingleThetaSingleRadio

    assert PFSingleThetaSingleRadio.dim0_is_angular
    assert PFSingleThetaDualRadio.dim0_is_angular
    assert PFSingleThetaDualRadioNN.dim0_is_angular
    assert not PFXYDualRadio.dim0_is_angular  # positions are NOT angles


# --------------------------------------------------------------------------- PF3
def test_pf3_absolute_gt_circular_mean_at_seam():
    """Absolute-mode ground truth must average the two radios' bearings
    circularly; arithmetic mean of (pi-e, -pi+e) is ~0 (opposite direction)."""
    from spf.filters.particle_dual_radio_nn_filter import PFSingleThetaDualRadioNN

    pf = object.__new__(PFSingleThetaDualRadioNN)
    pf.absolute = True
    pf.ds = SimpleNamespace(
        absolute_thetas=torch.tensor(
            [[math.pi - 0.01] * 3, [-math.pi + 0.01] * 3]
        )
    )
    traj = [{"craft_theta": np.array([math.pi])} for _ in range(3)]
    m = pf.metrics(traj)
    assert m["mse_craft_theta"] < 0.01, (
        f"gt averaged non-circularly: mse={m['mse_craft_theta']}"
    )


# ------------------------------------------------------------------- xfail reds
@pytest.mark.xfail(
    strict=True, reason="RT11: mutable default skip_fields=[] is mutated across instances"
)
def test_rt11_no_mutable_default_skip_fields():
    from spf.dataset.spf_dataset import v5inferencedataset

    default = inspect.signature(v5inferencedataset.__init__).parameters[
        "skip_fields"
    ].default
    assert default is None or not isinstance(default, list)


@pytest.mark.xfail(
    strict=True,
    reason="RT7: cached_model_inference_to_absolute_north hardcodes reshape(-1, 65)",
)
def test_rt7_rotation_generic_bin_count():
    from spf.filters.particle_dual_radio_nn_filter import (
        cached_model_inference_to_absolute_north,
    )

    nbins = 7
    heading_pis = 2.0 / nbins  # one bin worth of heading (in pis): 2pi/nbins rad
    ds = SimpleNamespace(
        cached_keys={
            0: {"rx_heading_in_pis": torch.tensor([heading_pis])},
            1: {"rx_heading_in_pis": torch.tensor([heading_pis])},
        }
    )
    dist = torch.zeros(1, 2, 1, nbins)
    dist[..., 2] = 1.0
    out = cached_model_inference_to_absolute_north(ds, dist)
    assert out.shape == dist.shape
    assert int(out[0, 0, 0].argmax()) == 3  # shifted by exactly one bin


def test_rt7_rotation_correct_at_65_bins():
    """Green control for RT7: the supported 65-bin path shifts by the heading."""
    from spf.filters.particle_dual_radio_nn_filter import (
        cached_model_inference_to_absolute_north,
    )

    nbins = 65
    shift_bins = 4
    heading_pis = 2.0 * shift_bins / nbins
    ds = SimpleNamespace(
        cached_keys={
            0: {"rx_heading_in_pis": torch.tensor([heading_pis])},
            1: {"rx_heading_in_pis": torch.tensor([heading_pis])},
        }
    )
    dist = torch.zeros(1, 2, 1, nbins)
    dist[..., 10] = 1.0
    out = cached_model_inference_to_absolute_north(ds, dist)
    assert int(out[0, 0, 0].argmax()) == 10 + shift_bins
    assert int(out[0, 1, 0].argmax()) == 10 + shift_bins
