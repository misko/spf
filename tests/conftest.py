import tempfile

import pytest

from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.dataset.spf_dataset import v5spfdataset
from spf.scripts.create_empirical_p_dist import (
    create_empirical_p_dist,
    get_empirical_p_dist_parser,
)


@pytest.fixture(scope="session")
def perfect_circle_dataset_n1025_orbits4_noise0p3():
    breakpoint()
    n = 1025
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0.3"
        create_fake_dataset(
            filename=fn, yaml_config_str=fake_yaml, n=n, noise=0.3, orbits=4
        )
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            n_parallel=4,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n33():
    n = 33
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml, n=n, noise=0.0)
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n5_noise0():
    n = 5
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml, n=5, noise=0.0)
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n5_noise0p001():
    n = 5
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p001"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml, n=5)
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n7_with_empirical():
    n = 7
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml, n=n, noise=0.0)

        datasets = [f"{fn}.zarr"]
        parser = get_empirical_p_dist_parser()

        empirical_pkl_fn = tmpdirname + "/full.pkl"

        args = parser.parse_args(
            [
                "--out",
                empirical_pkl_fn,
                "--nbins",
                "7",
                "--nthetas",
                "7",
                "--precompute-cache",
                tmpdirname,
                "--device",
                "cpu",
                "-d",
            ]
            + datasets
        )
        create_empirical_p_dist(args)

        yield tmpdirname, empirical_pkl_fn, fn
