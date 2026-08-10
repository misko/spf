"""The empirical-table override on the inference path, and its cache-key guard.

The empirical P(theta|phi) table is inherited from the checkpoint config and was
previously unreachable from the CLI. That is fine until a capture's
(antenna spacing, carrier) postdates the table: the dataset builds a per-sample
``empirical`` field, ``get_empirical_dist`` is an exact-key lookup, and the run
dies -- even for models with ``empirical_input: false`` that never read the
values. A 48-dataset cache build silently produced 31 files that way.
"""

import pickle
import tempfile

import pytest

from spf.model_training_and_inference.models.single_point_networks_inference import (
    convert_datasets_config_to_inference,
    get_inference_on_ds_noexceptions,
)
from spf.utils import SEGMENTATION_VERSION


def _datasets_config():
    return {
        "empirical_data_fn": "/original/full.pkl",
        "precompute_cache": "/original/cache",
        "batch_size": 256,
        "shuffle": True,
        "flip": True,
    }


# --------------------------------------------------------------- override


def test_override_replaces_the_table_the_checkpoint_names():
    out = convert_datasets_config_to_inference(
        _datasets_config(),
        ds_fn="/data/x.zarr",
        precompute_cache="/new/cache",
        segmentation_version=3.7,
        empirical_data_fn="/new/full_20260809_v1.pkl",
    )
    assert out["empirical_data_fn"] == "/new/full_20260809_v1.pkl"


def test_absent_override_keeps_the_checkpoints_table():
    """Default None must be a no-op, so every existing caller is unaffected."""
    out = convert_datasets_config_to_inference(
        _datasets_config(),
        ds_fn="/data/x.zarr",
        precompute_cache="/new/cache",
        segmentation_version=3.7,
    )
    assert out["empirical_data_fn"] == "/original/full.pkl"


def test_override_does_not_mutate_the_callers_config():
    """The config dict belongs to the loaded checkpoint; it must not be edited."""
    cfg = _datasets_config()
    convert_datasets_config_to_inference(
        cfg,
        ds_fn="/data/x.zarr",
        precompute_cache="/new/cache",
        segmentation_version=3.7,
        empirical_data_fn="/new/table.pkl",
    )
    assert cfg["empirical_data_fn"] == "/original/full.pkl"


def test_override_does_not_disturb_the_other_inference_settings():
    out = convert_datasets_config_to_inference(
        _datasets_config(),
        ds_fn="/data/x.zarr",
        precompute_cache="/new/cache",
        segmentation_version=3.7,
        batch_size=64,
        empirical_data_fn="/new/table.pkl",
    )
    # inference must not augment or shuffle
    assert out["flip"] is False and out["shuffle"] is False
    assert out["batch_size"] == 64
    assert out["train_paths"] == ["/data/x.zarr"]
    assert out["precompute_cache"] == "/new/cache"


# ------------------------------------------------------- cache-key guard


def _write_config(tmp, empirical_input):
    """Minimal config file; only `global.empirical_input` matters for the guard."""
    import yaml

    fn = f"{tmp}/config.yml"
    with open(fn, "w") as f:
        yaml.safe_dump(
            {
                "global": {"empirical_input": empirical_input, "nthetas": 65},
                "datasets": _datasets_config(),
                "model": {"name": "beamformer"},
                "optim": {"device": "cpu", "checkpoint": None},
            },
            f,
        )
    return fn


def test_override_refused_when_the_model_consumes_the_table():
    """The inference cache key omits the table, so two tables would collide.

    Key is {dataset}/{segver}/{checkpoint_md5}/{config_md5}.npz -- nothing about
    the empirical table. For a model that reads the table, a second run with a
    different table would silently reuse the first run's cached outputs. Fail
    loudly instead; the key format cannot change without invalidating every
    cache ever built.
    """
    from spf.model_training_and_inference.models import (
        single_point_networks_inference as inf,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg_fn = _write_config(tmp, empirical_input=True)

        # stub the model load: the guard must fire before any model is built
        def fake_load(config_fn, checkpoint_fn, device=None):
            import yaml

            with open(config_fn) as f:
                return None, yaml.safe_load(f)

        original = inf.load_model_and_config_from_config_fn_and_checkpoint
        inf.load_model_and_config_from_config_fn_and_checkpoint = fake_load
        try:
            with pytest.raises(ValueError, match="empirical_input=true"):
                inf.run_nn_inference_on_ds(
                    ds_fn="/data/x.zarr",
                    config_fn=cfg_fn,
                    checkpoint_fn="/data/best.pth",
                    device="cpu",
                    batch_size=1,
                    workers=0,
                    precompute_cache=tmp,
                    segmentation_version=3.7,
                    empirical_data_fn="/new/table.pkl",
                )
        finally:
            inf.load_model_and_config_from_config_fn_and_checkpoint = original


def test_no_override_is_allowed_even_when_the_model_consumes_the_table():
    """The guard is about overriding, not about empirical_input itself."""
    from spf.model_training_and_inference.models import (
        single_point_networks_inference as inf,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg_fn = _write_config(tmp, empirical_input=True)

        def fake_load(config_fn, checkpoint_fn, device=None):
            import yaml

            with open(config_fn) as f:
                return None, yaml.safe_load(f)

        original = inf.load_model_and_config_from_config_fn_and_checkpoint
        inf.load_model_and_config_from_config_fn_and_checkpoint = fake_load
        try:
            # gets past the guard, then fails later on the nonexistent dataset --
            # any error EXCEPT the guard's ValueError proves the guard passed
            with pytest.raises(Exception) as excinfo:
                inf.run_nn_inference_on_ds(
                    ds_fn="/data/missing.zarr",
                    config_fn=cfg_fn,
                    checkpoint_fn="/data/best.pth",
                    device="cpu",
                    batch_size=1,
                    workers=0,
                    precompute_cache=tmp,
                    segmentation_version=3.7,
                )
            assert "empirical_input=true" not in str(excinfo.value)
        finally:
            inf.load_model_and_config_from_config_fn_and_checkpoint = original


# ------------------------------------------------- failures are reported


def test_failure_is_returned_not_just_logged():
    """A bulk build reported success with 31 of 48 files because this returned None."""
    result = get_inference_on_ds_noexceptions(
        ds_fn="/nope/missing.zarr",
        config_fn="/nope/config.yml",
        checkpoint_fn="/nope/best.pth",
        inference_cache=None,
        crash_if_not_cached=False,
        segmentation_version=3.7,
    )
    ds_fn, err = result
    assert ds_fn == "/nope/missing.zarr"
    assert err is not None and isinstance(err, str)


def test_success_returns_no_error(
    noise1_n128_obits2, paired_net_checkpoint_using_single_checkpoint
):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    ckpt_dir = paired_net_checkpoint_using_single_checkpoint
    with tempfile.TemporaryDirectory() as cache:
        got, err = get_inference_on_ds_noexceptions(
            ds_fn=ds_fn + ".zarr",
            config_fn=f"{ckpt_dir}/config.yml",
            checkpoint_fn=f"{ckpt_dir}/best.pth",
            inference_cache=cache,
            device="cpu",
            batch_size=4,
            workers=0,
            precompute_cache=dirname,
            crash_if_not_cached=False,
            # track the constant: the fixture segments at the default, and a
            # mismatch here silently yields an empty dataloader rather than a
            # clear "wrong version" error
            segmentation_version=SEGMENTATION_VERSION,
            empirical_data_fn=empirical_pkl_fn,
        )
        assert err is None, err
        assert got == ds_fn + ".zarr"
