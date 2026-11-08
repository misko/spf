import argparse

from spf.dataset.spf_dataset import v5spfdataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-zarr", type=str, help="input zarr", required=True
    )
    parser.add_argument(
        "-c", "--precompute-cache", type=str, help="precompute cache", required=True
    )
    parser.add_argument("--gpu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-p", "--parallel", type=int, default=24, help="precompute cache"
    )
    args = parser.parse_args()

    ds = v5spfdataset(
        args.input_zarr,
        nthetas=65,
        precompute_cache=args.precompute_cache,
        gpu=args.gpu,
        skip_fields=set(["signal_matrix"]),
        ignore_qc=True,
        # readahead=True, #this is hard coded in the segmentation code
        n_parallel=args.parallel,
        segment_if_not_exist=True,
    )
    print(args.input_zarr, ds.phi_drifts[0], ds.phi_drifts[1])
