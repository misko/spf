import argparse

from spf.dataset.spf_dataset import v5spfdataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-zarr", type=str, help="input zarr")
    parser.add_argument("-c", "--precompute-cache", type=str, help="precompute cache")
    parser.add_argument("--gpu", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ds = v5spfdataset(
        args.input_zarr,
        nthetas=65,
        precompute_cache=args.precompute_cache,
        gpu=args.gpu,
        skip_fields=set(["signal_matrix"]),
        ignore_qc=True,
    )
    print(args.input_zarr, ds.phi_drifts[0], ds.phi_drifts[1])
