import argparse
import random
import tempfile
from spf.dataset.spf_dataset import v5spfdataset


def open_partial_dataset_and_check_some(
    ds_fn, suffix=".tmp", n_parallel=1, skip_fields=[]
):

    with tempfile.TemporaryDirectory() as tmpdirname:
        nthetas = 65
        ds = v5spfdataset(
            ds_fn,
            nthetas=nthetas,
            ignore_qc=True,
            precompute_cache=str(tmpdirname),
            paired=True,
            skip_signal_matrix=True,
            snapshots_per_session=1,
            temp_file=True,
            temp_file_suffix=suffix,
            n_parallel=n_parallel,
            skip_fields=skip_fields,
        )

        print(
            f"Opened dataset of length {len(ds)} and {ds.valid_entries} valid entries"
        )

        idxs = list(range(min(ds.valid_entries)))
        random.shuffle(idxs)

        for i in idxs[:10]:
            print(
                i,
                ds[i][0]["mean_phase_segmentation"].item(),
                ds[i][1]["mean_phase_segmentation"].item(),
            )
            if "windowed_beamformer" in skip_fields:
                assert "windowed_beamformer" not in ds[i][0].keys()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset",
        required=True,
    )
    parser.add_argument(
        "--suffix", type=str, help="suffix", required=False, default=".tmp"
    )

    parser.add_argument(
        "--nparallel", type=int, help="paralell compute", required=False, default=1
    )
    parser.add_argument(
        "--skip-beamformer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    skip_fields = []
    if args.skip_beamformer:
        skip_fields.append(["windowed_beamformer"])
    open_partial_dataset_and_check_some(
        args.dataset,
        suffix=args.suffix,
        n_parallel=args.nparallel,
        skip_fields=skip_fields,
    )
