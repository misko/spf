import argparse
import tempfile

from spf.dataset.spf_dataset import mp_segment_zarr, v5spfdataset
from spf.rf import precompute_steering_vectors


def get_parser_filter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset prefixes",
        required=True,
    )
    parser.add_argument(
        "--precompute-cache",
        type=str,
        help="precompute cache",
        required=True,
    )
    parser.add_argument(
        "--nthetas",
        type=int,
        default=65,
        required=False,
    )
    return parser


# from pyinstrument.renderers import ConsoleRenderer


def segment(args):
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds = v5spfdataset(
            args.dataset,
            precompute_cache=str(tmpdirname),
            nthetas=args.nthetas,
            skip_signal_matrix=True,
            paired=True,
            ignore_qc=True,
            gpu=False,
            snapshots_per_session=1,
            readahead=False,
            skip_simple_segmentations=False,
            temp_file=True,
            temp_file_suffix="",
        )
        mp_segment_zarr(
            args.dataset,
            str(tmpdirname) + "/tmp",
            ds.steering_vectors,
            precompute_to_idx=-1,
            gpu=False,
            n_parallel=0,
            skip_beamformer=False,
        )


if __name__ == "__main__":
    parser = get_parser_filter()
    args = parser.parse_args()
    segment(args)
