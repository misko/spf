import tempfile

import numpy as np

import tempfile

import pytest


from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.dataset.spf_dataset import v5spfdataset

import numpy as np

from spf.dataset.zarr_rechunk import zarr_rechunk
from spf.utils import identical_datasets


def test_zarr_rechunk():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn = f"{tmpdirname}/test_circle"
        create_fake_dataset(
            filename=ds_fn, yaml_config_str=fake_yaml, n=5, noise=0.0, phi_drift=0.0
        )

        ds_fn_rechunked = f"{tmpdirname}/test_circle_rechunked"
        zarr_rechunk(ds_fn + ".zarr", ds_fn_rechunked + ".zarr", False)

        ds = v5spfdataset(
            ds_fn,
            nthetas=11,
            ignore_qc=True,
            precompute_cache=tmpdirname,
        )
        ds_rechunked = v5spfdataset(
            ds_fn_rechunked,
            nthetas=11,
            ignore_qc=True,
            precompute_cache=tmpdirname,
        )
        identical_datasets(ds, ds_rechunked)

        ds_nosig_fn_rechunked = f"{tmpdirname}/test_circle_rechunked_nosig"
        zarr_rechunk(ds_fn + ".zarr", ds_nosig_fn_rechunked + ".zarr", True)

        ds_nosig_rechunked = v5spfdataset(
            ds_nosig_fn_rechunked,
            nthetas=11,
            ignore_qc=True,
            precompute_cache=tmpdirname,
            skip_fields="signal_matrix",
        )


        with pytest.raises(AssertionError):
            identical_datasets(ds_nosig_rechunked, ds)
        identical_datasets(ds_nosig_rechunked, ds, skip_signal_matrix=True)
