import tempfile

import numpy as np

from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.dataset.spf_dataset import v5spfdataset


def test_dataset_load():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn = f"{tmpdirname}/test_circle"
        create_fake_dataset(
            filename=ds_fn, yaml_config_str=fake_yaml, n=5, noise=0.0, phi_drift=0.0
        )
        ds = v5spfdataset(
            ds_fn,
            nthetas=11,
            ignore_qc=True,
            precompute_cache=tmpdirname,
        )
        assert np.isclose([0.0, 0.0], ds.phi_drifts, atol=0.05).all()


def test_dataset_load_drift():
    for phi_drift in [0.1, 0.5, 0.7]:
        with tempfile.TemporaryDirectory() as tmpdirname:
            ds_fn = f"{tmpdirname}/test_circle"
            create_fake_dataset(
                filename=ds_fn,
                yaml_config_str=fake_yaml,
                n=5,
                noise=0.0,
                phi_drift=phi_drift,
            )
            ds = v5spfdataset(
                ds_fn,
                nthetas=11,
                ignore_qc=True,
                precompute_cache=tmpdirname,
            )
            assert np.isclose(
                [phi_drift * np.pi, -phi_drift * np.pi], ds.phi_drifts, atol=0.05
            ).all()
