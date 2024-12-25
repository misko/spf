import argparse
import sys

from spf.dataset.spf_dataset import v5spfdataset
from spf.scripts.zarr_utils import compare_and_check
from spf.utils import identical_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_filenameA", type=str, help="zarr")
    parser.add_argument("zarr_filenameB", type=str, help="zarr")
    parser.add_argument(
        "--precompute-cache", type=str, help="precompute cache", required=True
    )
    parser.add_argument("--nthetas", type=int, help="nthetas", default=65)
    parser.add_argument(
        "-s", "--skip-signal-matrix", action="store_true", help="skip signal matrix"
    )
    parser.add_argument("-r", "--rendered", action="store_true", help="check rendered")
    args = parser.parse_args()

    dsA = v5spfdataset(
        args.zarr_filenameA,
        nthetas=args.nthetas,
        ignore_qc=True,
        precompute_cache=args.precompute_cache,
        skip_fields=["signal_matrix"] if args.skip_signal_matrix else [],
    )
    dsB = v5spfdataset(
        args.zarr_filenameB,
        nthetas=args.nthetas,
        ignore_qc=True,
        precompute_cache=args.precompute_cache,
        skip_fields=["signal_matrix"] if args.skip_signal_matrix else [],
    )
    if args.rendered:
        return_code = identical_datasets(dsA, dsB, args.skip_signal_matrix)
    else:
        ret = compare_and_check(
            "", dsA.z, dsB.z, skip_signal_matrix=args.skip_signal_matrix
        )
        if ret:
            return_code = 0
        else:
            return_code = 1
    if return_code == 0:
        print("IDENTICAL!")
    else:
        print("DIFFERENT!")
    sys.exit(return_code)
