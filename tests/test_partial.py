import random
import tempfile

import torch

from spf.dataset.fake_dataset import partial_dataset
from spf.dataset.open_partial_ds import open_partial_dataset_and_check_some
from spf.dataset.spf_dataset import v5spfdataset


def test_partial(noise1_n128_obits2):
    dirname, empirical_pkl_fn, ds_fn = noise1_n128_obits2
    ds_og = v5spfdataset(
        ds_fn,
        precompute_cache=dirname,
        nthetas=65,
        skip_fields=set(["signal_matrix"]),
        empirical_data_fn=empirical_pkl_fn,
        paired=True,
        ignore_qc=True,
        gpu=False,
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn_out = f"{tmpdirname}/partial"
        for partial_n in [10, 100, 128]:
            partial_dataset(ds_fn, ds_fn_out, partial_n)
            ds = v5spfdataset(
                ds_fn_out,
                precompute_cache=tmpdirname,
                nthetas=65,
                skip_fields=set(["signal_matrix"]),
                paired=True,
                ignore_qc=True,
                gpu=False,
                temp_file=True,
                temp_file_suffix="",
                empirical_data_fn=empirical_pkl_fn,
            )
            assert min(ds.valid_entries) == partial_n
            random.seed(0)
            idxs = list(range(partial_n))
            random.shuffle(idxs)
            for idx in idxs[:8]:
                for r_idx in range(2):
                    for key in ds_og[0][0].keys():
                        if isinstance(ds_og[idx][r_idx][key], torch.Tensor):
                            assert (ds_og[idx][r_idx][key] == ds[idx][r_idx][key]).all()


def test_partial_script(noise1_n128_obits2):
    _, _, ds_fn = noise1_n128_obits2
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn_out = f"{tmpdirname}/partial"
        for partial_n in [10, 20]:
            partial_dataset(ds_fn, ds_fn_out, partial_n)
            # open_partial_dataset_and_check_some(ds_fn_out, suffix="", n_parallel=0)
            open_partial_dataset_and_check_some(
                ds_fn_out,
                suffix="",
                n_parallel=0,
                skip_fields=set(["windowed_beamformer"]),
            )
