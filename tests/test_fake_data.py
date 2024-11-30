import tempfile

import numpy as np
import pytest
import torch

from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.dataset.spf_dataset import v5spfdataset
from spf.utils import identical_datasets


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
            segment_if_not_exist=True,
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
                segment_if_not_exist=True,
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
        skip_fields=set(["signal_matrix"]),
        n_parallel=0,
        segment_if_not_exist=True,
    )

    assert np.isclose(ds.ground_truth_phis[0], ds.mean_phase["r0"], atol=0.001).all()
    assert np.isclose(ds.ground_truth_phis[1], ds.mean_phase["r1"], atol=0.001).all()

    # test craft ground truth theta
    tx_pos = torch.vstack(
        [
            torch.tensor([ds[idx][0]["tx_pos_x_mm"], ds[idx][0]["tx_pos_y_mm"]])
            for idx in range(len(ds))
        ]
    )
    rx_pos = torch.vstack(
        [
            torch.tensor([ds[idx][0]["rx_pos_x_mm"], ds[idx][0]["rx_pos_y_mm"]])
            for idx in range(len(ds))
        ]
    )
    craft_theta = torch.tensor(
        [torch.tensor(ds[idx][0]["craft_ground_truth_theta"]) for idx in range(len(ds))]
    )
    d = tx_pos - rx_pos
    assert torch.arctan2(d[:, 0], d[:, 1]).isclose(craft_theta, rtol=0.0001).all()


def test_identical_datasets():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dsA_fn = f"{tmpdirname}/test_circleA"
        create_fake_dataset(
            filename=dsA_fn,
            yaml_config_str=fake_yaml,
            n=5,
            noise=0.0,
            phi_drift=0.0,
            seed=0,
        )
        dsB_fn = f"{tmpdirname}/test_circleB"
        create_fake_dataset(
            filename=dsB_fn,
            yaml_config_str=fake_yaml,
            n=6,
            noise=0.0,
            phi_drift=0.0,
            seed=10,
        )
        dsC_fn = f"{tmpdirname}/test_circleC"
        create_fake_dataset(
            filename=dsC_fn,
            yaml_config_str=fake_yaml,
            n=5,
            noise=0.0,
            phi_drift=0.0,
            seed=20,
        )

        dsA = v5spfdataset(
            dsA_fn,
            nthetas=11,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            segment_if_not_exist=True,
        )
        dsB = v5spfdataset(
            dsB_fn,
            nthetas=11,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            segment_if_not_exist=True,
        )
        dsC = v5spfdataset(
            dsC_fn,
            nthetas=11,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            segment_if_not_exist=True,
        )

        for a in [dsA, dsB, dsC]:
            for b in [dsA, dsB, dsC]:
                if a == b:
                    identical_datasets(a, b)
                else:
                    with pytest.raises(AssertionError):
                        identical_datasets(a, b)
