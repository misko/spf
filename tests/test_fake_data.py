import tempfile

import numpy as np

import tempfile


from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.dataset.spf_dataset import v5spfdataset

import numpy as np


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


# is antenna1 or antenna0 on more x+
def test_fake_data_array_orientation():
    noise = 0.0
    nthetas = 65
    orbits = 2
    n = 65

    tmpdir = tempfile.TemporaryDirectory()
    tmpdirname = tmpdir.name
    ds_fn = f"{tmpdirname}/sample_dataset_for_ekf_n{n}_noise{noise}"

    create_fake_dataset(
        filename=ds_fn, yaml_config_str=fake_yaml, n=n, noise=noise, orbits=orbits
    )
    ds = v5spfdataset(
        ds_fn,
        nthetas=nthetas,
        ignore_qc=True,
        precompute_cache=tmpdirname,
        paired=True,
        skip_signal_matrix=True,
    )

    assert np.isclose(
        ds.ground_truth_phis[0] - ds.mean_phase["r0"], 0, atol=0.00001
    ).all()
    assert np.isclose(
        ds.ground_truth_phis[1] - ds.mean_phase["r1"], 0, atol=0.00001
    ).all()
