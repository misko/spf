import os
import sys
import zarr
from spf.utils import (
    new_yarr_dataset,
    zarr_new_dataset,
    zarr_open_from_lmdb_store,
    zarr_shrink,
)

from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys


def compare_and_check(prefix, src, dst):
    if isinstance(src, zarr.hierarchy.Group):
        for key in src.keys():
            compare_and_check(prefix + "/" + key, src[key], dst[key])
    else:
        if prefix == "/config":
            if not src.shape == ():
                if dst[:] != src[:]:
                    raise ValueError(f"{prefix} does not match!")
        else:
            for x in range(src.shape[0]):
                if not (dst[x] == src[x]).all():
                    raise ValueError(f"{prefix} does not match!")
    return True


def compare_and_copy(prefix, src, dst):
    if isinstance(src, zarr.hierarchy.Group):
        for key in src.keys():
            compare_and_copy(prefix + "/" + key, src[key], dst[key])
    else:
        if prefix == "/config":
            if src.shape != ():
                dst[:] = src[:]
        else:
            for x in range(src.shape[0]):
                dst[x] = src[x]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} zarr_filename_in zarr_filename_out")
        sys.exit(1)

    input_fn = sys.argv[1]
    output_fn = sys.argv[2]

    origianl_zarr = zarr_open_from_lmdb_store(input_fn, mode="r")

    if os.path.exists(output_fn):
        new_zarr = zarr_open_from_lmdb_store(output_fn, mode="r")
        if not compare_and_check("", origianl_zarr, new_zarr):
            print(output_fn, "IS NOT CORRECT!")
        else:
            print(output_fn, "looks good!")
        sys.exit(1)

    timesteps = origianl_zarr["receivers/r0/system_timestamp"].shape[0]
    buffer_size = origianl_zarr["receivers/r0/signal_matrix"].shape[-1]
    n_receivers = 2
    keys_f64 = v5rx_f64_keys
    keys_2xf64 = v5rx_2xf64_keys
    chunk_size = 512

    new_zarr = zarr_new_dataset(
        output_fn,
        timesteps,
        buffer_size,
        n_receivers,
        keys_f64,
        keys_2xf64,
        origianl_zarr["config"],
        chunk_size=512,  # tested , blosc1 / chunk_size=512 / buffer_size (2^18~20) = seems pretty good
        compressor=None,
        skip_signal_matrix=False,
    )

    compare_and_copy("", origianl_zarr, new_zarr)
    new_zarr.store.close()
    new_zarr = None
    zarr_shrink(output_fn)
