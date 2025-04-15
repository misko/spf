import logging
import tempfile

import pytest

from spf.dataset.fake_dataset import (
    create_empirical_dist_for_datasets,
    create_fake_dataset,
    fake_yaml_v4,
    fake_yaml_v5,
)
from spf.dataset.spf_dataset import v5spfdataset


@pytest.fixture(scope="session")
def noise1_n128_obits2():
    with tempfile.TemporaryDirectory() as tmpdirname:
        n = 128
        nthetas = 65
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p3"
        create_fake_dataset(
            filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.3, orbits=2
        )

        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=nthetas,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )

        empirical_pkl_fn = create_empirical_dist_for_datasets(
            datasets=[f"{fn}.zarr"], precompute_cache=tmpdirname, nthetas=nthetas
        )
        yield tmpdirname, empirical_pkl_fn, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n1025_orbits4_noise0p3():
    n = 1025
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p3"
        create_fake_dataset(
            filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.3, orbits=4
        )
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )

        empirical_pkl_fn = create_empirical_dist_for_datasets(
            datasets=[f"{fn}.zarr"], precompute_cache=tmpdirname, nthetas=65
        )

        yield tmpdirname, empirical_pkl_fn, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n33():
    n = 33
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.0)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_n50_0p01():
    n = 50
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=50, noise=0.01)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_n50_0p01_v4():
    n = 50
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v4, n=50, noise=0.01)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
            v4=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n5_noise0():
    n = 5
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=5, noise=0.0)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n5_noise0p001():
    n = 5
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0p001"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=5)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )
        yield tmpdirname, fn


@pytest.fixture(scope="session")
def perfect_circle_dataset_n7_with_empirical():
    n = 7
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + f"/perfect_circle_n{n}_noise0"
        create_fake_dataset(filename=fn, yaml_config_str=fake_yaml_v5, n=n, noise=0.0)
        v5spfdataset(  # make sure everything gets segmented here
            fn,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            paired=True,
            skip_fields=set(["signal_matrix"]),
            segment_if_not_exist=True,
        )

        empirical_pkl_fn = create_empirical_dist_for_datasets(
            datasets=[f"{fn}.zarr"], precompute_cache=tmpdirname, nthetas=7
        )

        yield tmpdirname, empirical_pkl_fn, fn
