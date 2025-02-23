import argparse
import bisect
import logging
import os
from dataclasses import dataclass

import numpy as np
import yaml
import zarr
from pyproj import Proj

from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys
from spf.gps.boundaries import boundaries, find_closest_boundary
from spf.scripts.zarr_utils import zarr_new_dataset  # crissy_boundary_convex
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store, zarr_shrink


def lat_lon_to_xy(lat, lon, center_lat, center_lon):
    proj_centered = Proj(proj="aeqd", lat_0=center_lat, lon_0=center_lon, datum="WGS84")
    return np.array(proj_centered(lon, lat))


def smooth_out_timestamps_and_gps(timestamps_and_gps):
    last_valid_idx = timestamps_and_gps.times.shape[0] - 1
    while timestamps_and_gps.times[last_valid_idx] < 0.1 and last_valid_idx > 0:
        last_valid_idx -= 1
    assert last_valid_idx >= 0

    for prop in ["times", "gps_lats", "gps_longs"]:
        arr = getattr(timestamps_and_gps, prop)

        for missing_idx in np.where(arr[:last_valid_idx] == 0)[0]:
            neighbors = []
            if missing_idx > 0 and arr[missing_idx - 1] != 0:
                neighbors.append(arr[missing_idx - 1])
            if missing_idx < (len(arr) - 1) and arr[missing_idx + 1] != 0:
                neighbors.append(arr[missing_idx + 1])
            arr[missing_idx] = sum(neighbors) / len(neighbors)
    return timestamps_and_gps


@dataclass
class TimestampAndGPS:
    times: np.ndarray  # Array to store timestamps
    gps_lats: np.ndarray  # Array to store GPS coordinates
    gps_longs: np.ndarray  # Array to store GPS coordinates


# the timestamps are at the end of capture, this is off by sample acq time
def get_tx_xy_at_rx(rx_time_and_gps, tx_time_and_gps, gps_center_long_lat):
    tx_gps_lookups = []
    for rx_idx in range(rx_time_and_gps.times.shape[0]):
        idx = bisect.bisect_left(tx_time_and_gps.times, rx_time_and_gps.times[rx_idx])
        if idx != 0 and idx < tx_time_and_gps.times.shape[0]:
            # linear interpolation
            # time between steps
            tx_time_delta = tx_time_and_gps.times[idx] - tx_time_and_gps.times[idx - 1]
            rx_time_diff = (
                rx_time_and_gps.times[rx_idx] - tx_time_and_gps.times[idx - 1]
            )
            coeff = rx_time_diff / tx_time_delta
            gps_lat = tx_time_and_gps.gps_lats[idx] * coeff + tx_time_and_gps.gps_lats[
                idx - 1
            ] * (1 - coeff)
            gps_long = tx_time_and_gps.gps_longs[
                idx
            ] * coeff + tx_time_and_gps.gps_longs[idx - 1] * (1 - coeff)
            rx_xy = (
                lat_lon_to_xy(
                    lat=rx_time_and_gps.gps_lats[rx_idx],
                    lon=rx_time_and_gps.gps_longs[rx_idx],
                    center_lat=gps_center_long_lat[1],
                    center_lon=gps_center_long_lat[0],
                )
                * 1000
            )  # in mm
            tx_xy = (
                lat_lon_to_xy(
                    lat=gps_lat,
                    lon=gps_long,
                    center_lat=gps_center_long_lat[1],
                    center_lon=gps_center_long_lat[0],
                )
                * 1000
            )  # in mm

            tx_gps_lookups.append(
                {
                    "idx": rx_idx,
                    "tx_pos_x_mm": tx_xy[0],
                    "tx_pos_y_mm": tx_xy[1],
                    "rx_pos_x_mm": rx_xy[0],
                    "rx_pos_y_mm": rx_xy[1],
                }
            )
            # print(rx_timestamp, tx_time_delta, rx_time_diff, coeff, idx)
    return tx_gps_lookups


def convert_list_dict_to_dict_lists(list_dict):
    keys = list(list_dict[0].keys())
    d = {key: [] for key in keys}
    for e in list_dict:
        for key in keys:
            d[key].append(e[key])
    return d


# trim them accordingly
def trim_valid_idxs_and_tx_rx_pos(valid_idxs_and_tx_rx_pos):
    assert len(valid_idxs_and_tx_rx_pos["r0"]) > 0
    assert len(valid_idxs_and_tx_rx_pos["r1"]) > 0

    if len(valid_idxs_and_tx_rx_pos) == 1:
        return {
            "r0": convert_list_dict_to_dict_lists(valid_idxs_and_tx_rx_pos["r0"]),
            "idxs": [x["idx"] for x in valid_idxs_and_tx_rx_pos["r0"]],
        }
    start_idxs = (
        valid_idxs_and_tx_rx_pos["r0"][0]["idx"],
        valid_idxs_and_tx_rx_pos["r1"][0]["idx"],
    )
    end_idxs = (
        valid_idxs_and_tx_rx_pos["r0"][-1]["idx"],
        valid_idxs_and_tx_rx_pos["r1"][-1]["idx"],
    )
    assert abs(start_idxs[0] - start_idxs[1]) <= 1
    assert abs(end_idxs[0] - end_idxs[1]) <= 1
    if start_idxs[0] < start_idxs[1]:
        valid_idxs_and_tx_rx_pos["r0"] = valid_idxs_and_tx_rx_pos["r0"][1:]
    elif start_idxs[1] < start_idxs[0]:
        valid_idxs_and_tx_rx_pos["r1"] = valid_idxs_and_tx_rx_pos["r1"][1:]
    if end_idxs[0] > end_idxs[1]:
        valid_idxs_and_tx_rx_pos["r0"] = valid_idxs_and_tx_rx_pos["r0"][:-1]
    elif end_idxs[1] > end_idxs[0]:
        valid_idxs_and_tx_rx_pos["r1"] = valid_idxs_and_tx_rx_pos["r1"][:-1]

    # TODO this might be off by a few at the start as the timing algirhtm settles in
    assert set([x["idx"] for x in valid_idxs_and_tx_rx_pos["r0"]]) == set(
        [x["idx"] for x in valid_idxs_and_tx_rx_pos["r1"]]
    )

    valid_idxs_and_tx_rx_pos["idxs"] = [
        x["idx"] for x in valid_idxs_and_tx_rx_pos["r0"]
    ]
    valid_idxs_and_tx_rx_pos["r0"] = convert_list_dict_to_dict_lists(
        valid_idxs_and_tx_rx_pos["r0"]
    )
    valid_idxs_and_tx_rx_pos["r1"] = convert_list_dict_to_dict_lists(
        valid_idxs_and_tx_rx_pos["r1"]
    )

    return valid_idxs_and_tx_rx_pos


def compare_and_copy_with_idxs_and_aux_data(
    prefix, src, dst, idxs, skip_signal_matrix=False, aux_data={}
):
    if isinstance(src, zarr.hierarchy.Group):
        for key in dst.keys():
            if not skip_signal_matrix or key != "signal_matrix":
                src_key = key
                _aux_data = (
                    aux_data[key] if aux_data is not None and key in aux_data else None
                )
                if src_key == "rx_heading_in_pis":
                    src_key = "heading"
                    assert _aux_data is None
                    _src = (src[src_key][:] / 360) * 2
                else:
                    _src = src[src_key] if src_key in src else _aux_data
                compare_and_copy_with_idxs_and_aux_data(
                    prefix + "/" + key,
                    _src,
                    dst[key],
                    idxs=idxs,
                    skip_signal_matrix=skip_signal_matrix,
                    aux_data=_aux_data,
                )
    else:
        if prefix == "/config":
            if aux_data:
                dst[:] = aux_data
            elif src.shape != ():
                dst[:] = src[:]
        else:
            failures = 0
            for idx in range(len(idxs)):
                try:
                    dst[idx] = src[
                        idx
                    ]  # TODO why cant we just copy the whole thing at once? # too big?
                except RuntimeError as e:
                    failures += 1
            if failures > 0:
                logging.error(f"There were {failures} when copying")


def merge_v4rx_v4tx_into_v5(
    tx_fn,
    rx_fn,
    zarr_out_fn,
    fix_config,
    collector_type="unknown",
    collector_version="0.0",
    receivers=2,
    dry_run=False,
):
    tx_zarr = zarr_open_from_lmdb_store(tx_fn, readahead=True, mode="r")
    rx_zarr = zarr_open_from_lmdb_store(rx_fn, readahead=True, mode="r")

    rx_time_and_gpses = {
        f"r{rx_idx}": smooth_out_timestamps_and_gps(
            TimestampAndGPS(
                times=rx_zarr["receivers"][f"r{rx_idx}"]["gps_timestamp"][:],
                gps_lats=rx_zarr["receivers"][f"r{rx_idx}"]["gps_lat"][:],
                gps_longs=rx_zarr["receivers"][f"r{rx_idx}"]["gps_long"][:],
            )
        )
        for rx_idx in range(receivers)
    }

    tx_time_and_gps = smooth_out_timestamps_and_gps(
        TimestampAndGPS(
            times=tx_zarr["receivers"]["r0"]["gps_timestamp"][:],
            gps_lats=tx_zarr["receivers"]["r0"]["gps_lat"][:],
            gps_longs=tx_zarr["receivers"]["r0"]["gps_long"][:],
        )
    )

    # could just take the mean of TX if this is not working
    # but this would be more canonical
    boundary_name = find_closest_boundary(
        (tx_time_and_gps.gps_longs.mean(), tx_time_and_gps.gps_lats.mean())
    )
    gps_center_long_lat = boundaries[boundary_name].mean(axis=0)

    valid_idxs_and_tx_rx_pos = trim_valid_idxs_and_tx_rx_pos(
        {
            rx_idx: get_tx_xy_at_rx(
                rx_time_and_gpses[rx_idx], tx_time_and_gps, gps_center_long_lat
            )
            for rx_idx in rx_time_and_gpses
        }
    )

    # timesteps = original_zarr["receivers/r0/system_timestamp"].shape[0]
    # timesteps needs to be updated since missing some RX points because of out of sync with TX times TODO
    timesteps = len(valid_idxs_and_tx_rx_pos["idxs"])
    logging.info(
        f"Found {timesteps} valid data points, out of {rx_time_and_gpses['r0'].times.shape[0]} total"
    )
    if dry_run:
        return

    buffer_size = rx_zarr["receivers/r0/signal_matrix"].shape[-1]
    keys_f64 = v5rx_f64_keys
    keys_2xf64 = v5rx_2xf64_keys
    chunk_size = 512

    prefix = rx_fn.replace(".zarr", "")
    yaml_fn = f"{prefix}.yaml"
    config = yaml.safe_load(open(yaml_fn, "r"))

    assert (
        config["receivers"][0]["theta-in-pis"] != 0.0 or fix_config
    ), "Early rovers had this set incorrectly, refusing to run in this state, its ambiguous"
    config["receivers"][0]["theta-in-pis"] = 1.0

    if "collector" not in config:  #
        logging.warning(
            f"Adding collector type {collector_type} and collection version {collector_version} to config!"
        )
        config["collector"] = {"type": "rover", "version": "3.1"}

    with open(zarr_out_fn.replace(".zarr", ".yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    new_zarr = zarr_new_dataset(
        filename=zarr_out_fn,
        timesteps=timesteps,
        buffer_size=buffer_size,
        keys_f64=keys_f64,
        keys_2xf64=keys_2xf64,
        config=rx_zarr["config"],
        chunk_size=512,  # tested , blosc1 / chunk_size=512 / buffer_size (2^18~20) = seems pretty good
        compressor=None,
        skip_signal_matrix=False,
        n_receivers=receivers,
    )

    compare_and_copy_with_idxs_and_aux_data(
        "",
        rx_zarr,
        new_zarr,
        skip_signal_matrix=False,
        idxs=valid_idxs_and_tx_rx_pos["idxs"],
        aux_data={"receivers": valid_idxs_and_tx_rx_pos, "config": yaml.dump(config)},
    )
    if fix_config:
        new_zarr["receivers/r0/rx_theta_in_pis"][:] = config["receivers"][0][
            "theta-in-pis"
        ]

    new_zarr.store.close()
    new_zarr = None
    zarr_shrink(zarr_out_fn)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", type=str, help="input tx zarr", required=True)
    parser.add_argument("--rx", type=str, help="input rx zarr", required=True)
    parser.add_argument("--output", type=str, help="output zarr", required=True)
    parser.add_argument("--collector-type", type=str, default="rover")
    parser.add_argument("--collector-version", type=str, default="3.1")
    parser.add_argument("--dry-run", action="store_true", default=False)

    parser.add_argument(  # the early rovers had antenna array 0 off by 180deg
        "--fix-config",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if os.path.isdir(args.output):
        logging.error(f"Output file already exists {args.output}")
    else:
        logging.info(f"tx: {args.tx}, rx: {args.rx}, out: {args.output}")
        merge_v4rx_v4tx_into_v5(
            tx_fn=args.tx,
            rx_fn=args.rx,
            zarr_out_fn=args.output,
            fix_config=args.fix_config,
            collector_type=args.collector_type,
            collector_version=args.collector_version,
            dry_run=args.dry_run,
        )
