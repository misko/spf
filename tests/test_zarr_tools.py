import tempfile

import pytest

from spf.dataset.spf_dataset import v5spfdataset
from spf.dataset.zarr_rechunk import zarr_rechunk
from spf.utils import identical_datasets


def test_zarr_rechunk(perfect_circle_dataset_n5_noise0):
    with tempfile.TemporaryDirectory() as tmpdirname:
        _, ds_fn = perfect_circle_dataset_n5_noise0

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
