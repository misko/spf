import shutil
import sys

from spf.dataset.spf_dataset import v5spfdataset
from spf.dataset.v5_data import v5rx_new_dataset
from spf.utils import zarr_open_from_lmdb_store, zarr_shrink

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"{sys.argv[0]} input_zarr output_zarr precompute_cache")
        sys.exit(1)
    zarr_fn = sys.argv[1]
    output_fn = sys.argv[2]

    # make sure its segmented
    v5spfdataset(
        sys.argv[1],
        nthetas=65,
        precompute_cache=sys.argv[3],
    )

    z = zarr_open_from_lmdb_store(zarr_fn)

    n_records_per_receiver, n_receivers, buffer_size = (
        z.receivers.r0.signal_matrix.shape
    )

    shutil.copyfile(
        zarr_fn.replace(".zarr", "_segmentation.pkl"),
        output_fn.replace(".zarr", "_segmentation.pkl"),
    )
    min_z = v5rx_new_dataset(
        filename=output_fn,
        timesteps=n_records_per_receiver,
        buffer_size=buffer_size,
        n_receivers=n_receivers,
        chunk_size=512,
        compressor=None,
        skip_signal_matrix=True,
    )

    for receiver_idx in range(n_receivers):
        receiver_key = f"receivers/r{receiver_idx}"
        for key in z[receiver_key].keys():
            if key == "signal_matrix":
                continue
            min_z[receiver_key + "/" + key] = z[receiver_key + "/" + key]

    min_z.store.close()
    min_z = None
    zarr_shrink(output_fn)
