import sys

from spf.scripts.zarr_utils import zarr_shrink

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{sys.argv[0]} zarr_filename")
        sys.exit(1)

    zarr_shrink(sys.argv[1])
