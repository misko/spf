import os
import sys

from spf.scripts.zarr_utils import new_yarr_dataset, zarr_open_from_lmdb_store

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} yarr_filename_in yarr_filename_out")
        sys.exit(1)

    input_fn = sys.argv[1]
    output_fn = sys.argv[2]

    assert not os.path.exists(output_fn)

    precomputed_zarr = zarr_open_from_lmdb_store(input_fn, mode="r")

    output_yarr = new_yarr_dataset(
        output_fn,
        all_windows_stats_shape=precomputed_zarr["r0/all_windows_stats"].shape,
        windowed_beamformer_shape=precomputed_zarr["r0/windowed_beamformer"].shape,
        n_receivers=2,
    )

    for r_idx in [0, 1]:
        output_yarr[f"r{r_idx}/all_windows_stats"][:] = precomputed_zarr[
            f"r{r_idx}/all_windows_stats"
        ][:]
        output_yarr[f"r{r_idx}/windowed_beamformer"][:] = precomputed_zarr[
            f"r{r_idx}/windowed_beamformer"
        ][:]
    output_yarr.store.close()
    output_yarr = None
