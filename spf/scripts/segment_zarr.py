import argparse
import concurrent

from spf.dataset.spf_dataset import v5spfdataset
import cupy

def process_zarr(args):
    print(args["input_zarr"])
    ds = v5spfdataset(
        args["input_zarr"],
        nthetas=65,
        precompute_cache=args["precompute_cache"],
        gpu=args["gpu"],
        skip_fields=set(["signal_matrix"]),
        ignore_qc=True,
        # readahead=True, #this is hard coded in the segmentation code
        n_parallel=args["parallel"],
        segment_if_not_exist=True,
    )
    print(args["input_zarr"], ds.phi_drifts[0], ds.phi_drifts[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-zarrs", type=str, nargs="+", help="input zarr", required=True
    )
    parser.add_argument(
        "-c", "--precompute-cache", type=str, help="precompute cache", required=True
    )
    parser.add_argument("--gpu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-p", "--parallel", type=int, default=12, help="parallel")
    parser.add_argument("-w", "--workers", type=int, default=2, help="n workers")
    args = parser.parse_args()

    jobs = [
        {
            "input_zarr": zarr_fn,
            "precompute_cache": args.precompute_cache,
            "parallel": args.parallel,
            "gpu": args.gpu,
        }
        for zarr_fn in args.input_zarrs
    ]
    if args.debug or args.workers==0:
        list(map(process_zarr, jobs))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            executor.map(process_zarr, jobs)
