import argparse
import concurrent
import json
import multiprocessing
import os
import time

from tqdm import tqdm

from spf.dataset.spf_dataset import v5spfdataset

LS_VERSION = 1.4


def ls_zarr(ds_fn, force=False):
    ls_fn = ds_fn + ".ls.json"
    if force or not os.path.exists(ls_fn):
        try:
            ds = v5spfdataset(
                ds_fn,
                nthetas=65,
                precompute_cache=None,
                skip_fields=set(["signal_matrix"]),
                ignore_qc=True,
                segment_if_not_exist=False,
                temp_file=True,
                temp_file_suffix="",
            )
            ls_info = {
                "ds_fn": ds_fn,
                "frequency": ds.carrier_frequencies[0],
                "rf_bandwidth": ds.rf_bandwidths[0],
                "rx_spacing": ds.rx_spacing,
                "samples": len(ds),
                "routine": ds.yaml_config["routine"],
                "version": LS_VERSION,
                "vehicle_type": ds.vehicle_type,
                "rx_theta_in_pis_cached": {
                    f"r{rx_idx}": {
                        "cached": ds.cached_keys[rx_idx]["rx_theta_in_pis"]
                        .median()
                        .item(),
                        "config": ds.yaml_config["receivers"][rx_idx]["theta-in-pis"],
                    }
                    for rx_idx in range(2)
                },
            }
            with open(ls_fn, "w") as fp:
                json.dump(ls_info, fp, indent=4)
        except Exception as e:
            print(f"Failed to write {ds_fn} with {e}")
            return None
            # raise ValueError(f"Could not not process file {ds_fn} , {str(e)}")
    with open(ls_fn, "r") as file:
        try:
            ls_info = json.load(file)
            if "version" not in ls_info or ls_info["version"] != LS_VERSION:
                assert (
                    not force
                ), f"LS_DS version not found(?) {ls_info} vs {LS_VERSION}"
                return ls_zarr(ds_fn, force=True)
            return ls_info
        except Exception as e:
            if not force:
                print("FAILED TO LOAD< TYRING TO FORCE")
                return ls_zarr(ds_fn, force=True)
    raise ValueError(f"Could not not process file {ds_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-zarrs", type=str, nargs="+", help="input zarr", required=True
    )
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-w", "--workers", type=int, default=4, help="n workers")
    args = parser.parse_args()

    if args.debug:
        results = list(map(ls_zarr, args.input_zarrs))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            results = list(
                tqdm(
                    executor.map(ls_zarr, args.input_zarrs), total=len(args.input_zarrs)
                )
            )
    # print(results[0], len(results))

    # aggregate results
    merged_stats = {}
    for result in results:
        if result:
            key = f"{result['frequency']},{result['rx_spacing']},{result['routine']},{result['rf_bandwidth']}"
            if key not in merged_stats:
                merged_stats[key] = 0
            merged_stats[key] += result["samples"]

    print("frequency,rx_spacing,routine,rf_bandwidth,samples")
    for key in sorted(merged_stats.keys()):
        print(f"{key},{merged_stats[key]}")
