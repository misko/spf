"""Particle-filter runs must be reproducible from their `seed` argument alone.

`ParticleFilter.trajectory` has always taken a `seed`, but it only seeded the
`torch.Generator` used for particle initialisation and process noise. Systematic
resampling drew its offset from numpy's process-global, unseeded RNG, so an
identical configuration on an identical dataset returned different answers --
measured at 42% (empirical dual-radio) and 106% (NN dual-radio) spread in MSE
over eight repeats of one 539-timestep capture. That is larger than the gap
between adjacent points in the sweep grid, which made "the best hyperparameter"
partly a property of the RNG.

`test_global_numpy_rng_does_not_affect_result` is the specific regression guard;
it fails against a filter whose resampling reads the global RNG.
"""

import numpy as np
import pytest
import torch

from spf.dataset.spf_dataset import v5spfdataset_manager
from spf.filters.particle_dualradio_filter import PFSingleThetaDualRadio
from spf.filters.particle_dualradioXY_filter import PFXYDualRadio
from spf.filters.particle_single_radio_filter import PFSingleThetaSingleRadio

# Small but noisy: enough timesteps that the effective-sample-size test trips and
# resampling actually runs (asserted by test_resampling_actually_runs below --
# without resampling every test here would pass vacuously).
N_PARTICLES = 1024


def open_ds(noise1_n128_obits2):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    return v5spfdataset_manager(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_fields=set(["signal_matrix"]),
        empirical_data_fn=empirical_pkl_fn,
        paired=True,
        ignore_qc=True,
        gpu=False,
        segment_if_not_exist=True,
    )


def build(kind, ds):
    """(filter, trajectory kwargs) for each particle-filter family."""
    if kind == "single_radio":
        return PFSingleThetaSingleRadio(ds=ds, rx_idx=0), dict(
            mean=torch.tensor([[0, 0]]),
            std=torch.tensor([[20, 0.1]]),
            noise_std=torch.tensor([[0.01, 0.01]]),
            N=N_PARTICLES,
        )
    if kind == "dual_radio":
        return PFSingleThetaDualRadio(ds=ds), dict(
            mean=torch.tensor([[0, 0]]),
            std=torch.tensor([[20, 0.1]]),
            noise_std=torch.tensor([[0.01, 0.01]]),
            N=N_PARTICLES,
        )
    if kind == "xy_dual_radio":
        # dim0_is_angular is False here -- covers the non-angular estimate path
        return PFXYDualRadio(ds=ds), dict(
            mean=torch.tensor([[0, 0, 0, 0, 0]]),
            std=torch.tensor([[0, 200, 200, 0.1, 0.1]]),
            noise_std=torch.tensor([[0, 15, 15, 0.5, 0.5]]),
            N=N_PARTICLES,
        )
    raise ValueError(kind)


def track(traj):
    return torch.stack([t["mu"] for t in traj]).numpy()


KINDS = ["single_radio", "dual_radio", "xy_dual_radio"]


@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_is_bit_identical(kind, noise1_n128_obits2):
    with open_ds(noise1_n128_obits2) as ds:
        pf_a, kwargs = build(kind, ds)
        a = track(pf_a.trajectory(seed=0, **kwargs))
        pf_b, kwargs = build(kind, ds)
        b = track(pf_b.trajectory(seed=0, **kwargs))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", KINDS)
def test_global_numpy_rng_does_not_affect_result(kind, noise1_n128_obits2):
    """The regression guard: only `seed` may determine the answer.

    Perturbing numpy's global RNG between two otherwise identical runs must
    change nothing. Under the old filterpy resampler it changed the resampling
    offsets and therefore the whole trajectory.
    """
    with open_ds(noise1_n128_obits2) as ds:
        np.random.seed(1)
        pf_a, kwargs = build(kind, ds)
        a = track(pf_a.trajectory(seed=0, **kwargs))

        np.random.seed(2)
        _ = np.random.random(97)  # advance it somewhere unrelated as well
        pf_b, kwargs = build(kind, ds)
        b = track(pf_b.trajectory(seed=0, **kwargs))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", KINDS)
def test_different_seeds_give_different_results(kind, noise1_n128_obits2):
    """Guards the opposite failure: freezing the RNG entirely."""
    with open_ds(noise1_n128_obits2) as ds:
        pf_a, kwargs = build(kind, ds)
        a = track(pf_a.trajectory(seed=0, **kwargs))
        pf_b, kwargs = build(kind, ds)
        b = track(pf_b.trajectory(seed=1, **kwargs))
    assert not np.array_equal(a, b)


def test_resampling_actually_runs(noise1_n128_obits2, monkeypatch):
    """Without this, every test above could pass on a filter that never resamples."""
    import spf.filters.filters as filters_module

    calls = []
    original = filters_module.systematic_resample

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(filters_module, "systematic_resample", counting)
    with open_ds(noise1_n128_obits2) as ds:
        pf, kwargs = build("dual_radio", ds)
        pf.trajectory(seed=0, **kwargs)
    assert len(calls) > 0, "resampling never ran; the determinism tests are vacuous"


# ------------------------------------------------- the sweep's seed axis
#
# ParticleFilter.trajectory takes a seed, but the run_PF_* wrappers the sweep
# calls did not pass one, so every job in a grid silently ran seed=0 -- one draw
# per configuration, against a measured 42-106% draw-to-draw spread. These check
# the seed reaches the filter and is recorded on the result, so the report can
# group by it and give mean +- std per configuration.

RUN_PF = [
    (
        "run_PF_single_theta_dual_radio",
        dict(N=512, theta_err=0.01, theta_dot_err=0.01),
        "mse_craft_theta",
    ),
    (
        "run_PF_single_theta_single_radio",
        dict(N=512, theta_err=0.01, theta_dot_err=0.01),
        "mse_single_radio_theta",
    ),
    ("run_PF_xy_dual_radio", dict(N=512, pos_err=15, vel_err=0.5), "mse_craft_theta"),
]


@pytest.mark.parametrize("fn_name,kwargs,metric", RUN_PF)
def test_run_wrapper_honours_seed(noise1_n128_obits2, fn_name, kwargs, metric):
    import spf.filters.run_filters_on_data as runner

    fn = getattr(runner, fn_name)
    with open_ds(noise1_n128_obits2) as ds:
        a = fn(ds=ds, seed=0, **kwargs)[0]["metrics"][metric]
        b = fn(ds=ds, seed=0, **kwargs)[0]["metrics"][metric]
        c = fn(ds=ds, seed=1, **kwargs)[0]["metrics"][metric]
    assert a == b, "same seed must reproduce exactly"
    assert a != c, "different seed must give a different draw"


@pytest.mark.parametrize("fn_name,kwargs,metric", RUN_PF)
def test_run_wrapper_records_seed(noise1_n128_obits2, fn_name, kwargs, metric):
    """The reporter keys on every non-metric field, so seed must be on the result."""
    import spf.filters.run_filters_on_data as runner

    fn = getattr(runner, fn_name)
    with open_ds(noise1_n128_obits2) as ds:
        results = fn(ds=ds, seed=3, **kwargs)
    for result in results:
        assert result["seed"] == 3


def test_ekf_wrappers_carry_no_seed(noise1_n128_obits2):
    """EKFs are deterministic -- a seed axis would be five identical rows implying
    a spread that does not exist."""
    import spf.filters.run_filters_on_data as runner

    with open_ds(noise1_n128_obits2) as ds:
        results = runner.run_EKF_single_theta_dual_radio(
            ds=ds, phi_std=10.0, p=5.0, noise_std=0.001, dynamic_R=0.0
        )
    assert "seed" not in results[0]
