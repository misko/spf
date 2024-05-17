from spf.dataset.spf_dataset import v5spfdataset
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{sys.argv[0]} input_zarr")
        sys.exit(1)
    ds = v5spfdataset(
        sys.argv[1],
        nthetas=11,
    )
