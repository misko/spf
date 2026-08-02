"""Join a rover-TX (emitter) GPS track with rover-RX v7 captures into merged zarrs.

This is a v7-aware copy of ``v4_tx_rx_to_v5.py`` (see that file / project_spf.pdf for
the original method). Same core join: for every (tx, rx) pair, linearly interpolate
the TX rover's GPS onto each RX timestamp, project both to a local aeqd XY frame in
mm, align the r0/r1 valid indices, and write the ground-truth ``tx_pos_*_mm`` /
``rx_pos_*_mm`` per snapshot. Only temporally-overlapping pairs yield >= min_timesteps.

Differences from the v4->v5 version (why this file exists):
  1. Preserves the FULL v7 payload: the merged zarr carries every v7 receiver key
     (gain/RSSI metadata, gps_*, heading, sequence/flags, signal_matrix) AS WELL AS
     the v5 ground-truth keys (tx_pos/rx_pos, rx_heading_in_pis). It loads exactly
     like the historical v5 merged data (the v5 keys are all present) and no
     gain/RSSI *array* from the direct-USB campaign is dropped.

     CAVEAT -- zarr ATTRS are NOT propagated. The source stores carry ~30 receiver
     attrs (direct_usb_serial / bus / port_path, firmware identity, ...) and 4 root
     attrs (capture_status, capture_records_written_by_receiver,
     sdr_identity_version, radio_metadata_schema_version); the merged store gets
     only radio_metadata_schema_version. So a merged dataset cannot currently tell
     you which physical Pluto is r0, what firmware it ran, or whether the source
     capture finalized cleanly. Propagating these (plus the merge revision and the
     projection center) is a known gap -- see the 2026_08_01 field report appendix.
  2. Projection center is derived from the data (mean TX GPS) instead of a
     boundary-table geofence. The July 2026 site is Fort Baker (~1 m from
     fort_baker_left_boundary), so find_closest_boundary would work; the
     data-derived center is equivalent (a translation only affects the XY origin,
     not the relative tx<->rx geometry) and avoids depending on which of the three
     fort_baker polygons is nearest / on the site being in boundaries.py at all.
  3. RX arrays are copied at the VALID indices (``src[idxs[i]]``) so the RF/GPS rows
     line up with their interpolated tx_pos even when the RX capture starts before
     the TX (the v4->v5 version copied the first N rows, assuming idxs start at 0).

Output naming mirrors the old CLI: ``<rx>.<tx>.zarr`` under --output.
"""

import argparse
import bisect
import logging
import os

import lmdb
import numpy as np
import yaml
from pyproj import Proj

from spf.dataset.v5_data import v5rx_2xf64_keys, v5rx_f64_keys
from spf.dataset.v7_data import v7rx_2x_keys, v7rx_scalar_keys
from spf.scripts.zarr_utils import (
    zarr_new_dataset,
    zarr_open_from_lmdb_store,
    zarr_shrink,
)

# v7 receiver keys we carry through in addition to the v5 ground-truth keys. gps_*
# and heading come from the v4/v7 f64 set; the scalar/2x keys are the gain/RSSI
# metadata that the whole direct-USB campaign is about.
EXTRA_F64_KEYS = ["gps_timestamp", "gps_lat", "gps_long", "heading"]


def lat_lon_to_xy(lat, lon, center_lat, center_lon):
    proj_centered = Proj(proj="aeqd", lat_0=center_lat, lon_0=center_lon, datum="WGS84")
    return np.array(proj_centered(lon, lat))


class TimestampAndGPS:
    def __init__(self, times, gps_lats, gps_longs):
        self.times = times
        self.gps_lats = gps_lats
        self.gps_longs = gps_longs


def smooth_out_timestamps_and_gps(tg):
    last_valid_idx = tg.times.shape[0] - 1
    while tg.times[last_valid_idx] < 0.1 and last_valid_idx > 0:
        last_valid_idx -= 1
    assert last_valid_idx >= 0
    for prop in ["times", "gps_lats", "gps_longs"]:
        arr = getattr(tg, prop)
        for missing_idx in np.where(arr[:last_valid_idx] == 0)[0]:
            neighbors = []
            if missing_idx > 0 and arr[missing_idx - 1] != 0:
                neighbors.append(arr[missing_idx - 1])
            if missing_idx < (len(arr) - 1) and arr[missing_idx + 1] != 0:
                neighbors.append(arr[missing_idx + 1])
            if neighbors:
                arr[missing_idx] = sum(neighbors) / len(neighbors)
    return tg


def get_non_zero_mean(x):
    return x[~np.isclose(x, 0.0)].mean()


# the timestamps are at the end of capture, this is off by sample acq time
def get_tx_xy_at_rx(rx_tg, tx_tg, center_lat, center_lon):
    lookups = []
    for rx_idx in range(rx_tg.times.shape[0]):
        idx = bisect.bisect_left(tx_tg.times, rx_tg.times[rx_idx])
        if idx != 0 and idx < tx_tg.times.shape[0]:
            tx_time_delta = tx_tg.times[idx] - tx_tg.times[idx - 1]
            rx_time_diff = rx_tg.times[rx_idx] - tx_tg.times[idx - 1]
            coeff = rx_time_diff / tx_time_delta
            gps_lat = tx_tg.gps_lats[idx] * coeff + tx_tg.gps_lats[idx - 1] * (1 - coeff)
            gps_long = tx_tg.gps_longs[idx] * coeff + tx_tg.gps_longs[idx - 1] * (
                1 - coeff
            )
            rx_xy = (
                lat_lon_to_xy(
                    rx_tg.gps_lats[rx_idx], rx_tg.gps_longs[rx_idx], center_lat, center_lon
                )
                * 1000
            )
            tx_xy = lat_lon_to_xy(gps_lat, gps_long, center_lat, center_lon) * 1000
            if np.abs(tx_xy).max() > 300000 or np.abs(rx_xy).max() > 300000:
                logging.error("Something is wrong! 300000 v7 merge")
            lookups.append(
                {
                    "idx": rx_idx,
                    "tx_pos_x_mm": tx_xy[0],
                    "tx_pos_y_mm": tx_xy[1],
                    "rx_pos_x_mm": rx_xy[0],
                    "rx_pos_y_mm": rx_xy[1],
                }
            )
    return lookups


def convert_list_dict_to_dict_lists(list_dict):
    keys = list(list_dict[0].keys())
    d = {key: [] for key in keys}
    for e in list_dict:
        for key in keys:
            d[key].append(e[key])
    return d


def trim_valid_idxs_and_tx_rx_pos(valid):
    if len(valid["r0"]) == 0 or len(valid["r1"]) == 0:
        return {"idxs": []}
    start_idxs = (valid["r0"][0]["idx"], valid["r1"][0]["idx"])
    end_idxs = (valid["r0"][-1]["idx"], valid["r1"][-1]["idx"])
    assert abs(start_idxs[0] - start_idxs[1]) <= 1
    assert abs(end_idxs[0] - end_idxs[1]) <= 1
    if start_idxs[0] < start_idxs[1]:
        valid["r0"] = valid["r0"][1:]
    elif start_idxs[1] < start_idxs[0]:
        valid["r1"] = valid["r1"][1:]
    if end_idxs[0] > end_idxs[1]:
        valid["r0"] = valid["r0"][:-1]
    elif end_idxs[1] > end_idxs[0]:
        valid["r1"] = valid["r1"][:-1]
    assert set(x["idx"] for x in valid["r0"]) == set(x["idx"] for x in valid["r1"])
    valid["idxs"] = [x["idx"] for x in valid["r0"]]
    valid["r0"] = convert_list_dict_to_dict_lists(valid["r0"])
    valid["r1"] = convert_list_dict_to_dict_lists(valid["r1"])
    return valid


def _copy_receiver(rx_group, dst_group, idxs, pos):
    """Copy one RX receiver group into the merged group at the valid indices.

    ``pos`` holds the computed tx_pos/rx_pos lists (already valid-ordered). Every
    other dst key is copied straight from the RX zarr at ``idxs[i]``. rx_heading_in_pis
    is derived from the RX ``heading`` (deg -> multiples of pi), matching
    v4_tx_rx_to_v5.py / spf_dataset.v4_to_v5.
    """
    heading = None
    if "heading" in rx_group:
        heading = rx_group["heading"][:]
    failures = 0
    for key in dst_group.keys():
        dst = dst_group[key]
        if key in ("tx_pos_x_mm", "tx_pos_y_mm", "rx_pos_x_mm", "rx_pos_y_mm"):
            vals = pos[key]
            for i in range(len(idxs)):
                dst[i] = vals[i]
        elif key == "rx_heading_in_pis":
            for i in range(len(idxs)):
                dst[i] = (heading[idxs[i]] / 360.0) * 2 if heading is not None else 0.0
        elif key in rx_group:
            src = rx_group[key]
            # Per-row fault tolerance (mirrors v4_tx_rx_to_v5.py): torn .tmp inputs
            # can have individual corrupt chunks (blosc -1 / lmdb page errors). Skip
            # the bad row (left at zeros) instead of aborting the whole capture.
            n = len(idxs)
            for i in range(n):
                try:
                    dst[i] = src[idxs[i]]
                except (RuntimeError, lmdb.Error):
                    failures += 1
                if key == "signal_matrix" and (i + 1) % 500 == 0:
                    logging.info("  %s signal_matrix %d/%d", dst_group.path, i + 1, n)
        else:
            logging.warning("merged key %s not present in RX zarr; left at zeros", key)
    if failures:
        logging.error(
            "%s: skipped %d corrupt source element(s) (torn input?)",
            dst_group.path,
            failures,
        )
    return failures


def merge_v7rx_v7tx(
    tx_fn, rx_fn, zarr_out_fn, fix_config, receivers=2, dry_run=False, min_timesteps=500
):
    tx_zarr = zarr_open_from_lmdb_store(tx_fn, readahead=True, mode="r")
    rx_zarr = zarr_open_from_lmdb_store(rx_fn, readahead=True, mode="r")

    rx_tgs = {
        f"r{r}": smooth_out_timestamps_and_gps(
            TimestampAndGPS(
                rx_zarr["receivers"][f"r{r}"]["gps_timestamp"][:],
                rx_zarr["receivers"][f"r{r}"]["gps_lat"][:],
                rx_zarr["receivers"][f"r{r}"]["gps_long"][:],
            )
        )
        for r in range(receivers)
    }
    tx_tg = smooth_out_timestamps_and_gps(
        TimestampAndGPS(
            tx_zarr["receivers"]["r0"]["gps_timestamp"][:],
            tx_zarr["receivers"]["r0"]["gps_lat"][:],
            tx_zarr["receivers"]["r0"]["gps_long"][:],
        )
    )

    # Projection center from the data (mean TX GPS) -- no hardcoded geofence needed;
    # only the relative tx<->rx geometry matters.
    center_lon = get_non_zero_mean(tx_tg.gps_longs)
    center_lat = get_non_zero_mean(tx_tg.gps_lats)
    if not np.isfinite(center_lon) or not np.isfinite(center_lat):
        raise ValueError(f"No valid TX GPS for center ({center_lon},{center_lat})")

    valid = trim_valid_idxs_and_tx_rx_pos(
        {r: get_tx_xy_at_rx(rx_tgs[r], tx_tg, center_lat, center_lon) for r in rx_tgs}
    )
    timesteps = len(valid.get("idxs", []))
    if dry_run or timesteps < min_timesteps:
        return timesteps
    logging.info(
        "Found %d valid points (of %d rx), idxs[0]=%d",
        timesteps,
        rx_tgs["r0"].times.shape[0],
        valid["idxs"][0],
    )

    buffer_size = rx_zarr["receivers/r0/signal_matrix"].shape[-1]
    config = yaml.safe_load(open(rx_fn.replace(".zarr", ".yaml"), "r"))
    assert (
        config["receivers"][0]["theta-in-pis"] != 0.0 or fix_config
    ), "theta-in-pis is 0 and --no-fix-config; ambiguous, refusing"
    config["receivers"][0]["theta-in-pis"] = 1.0
    with open(zarr_out_fn.replace(".zarr", ".yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # v5 ground-truth keys + the v7 gps/heading keys, then the v7 gain/RSSI datasets.
    new_zarr = zarr_new_dataset(
        filename=zarr_out_fn,
        timesteps=timesteps,
        buffer_size=buffer_size,
        n_receivers=receivers,
        keys_f64=list(v5rx_f64_keys) + EXTRA_F64_KEYS,
        keys_2xf64=list(v5rx_2xf64_keys),
        config=yaml.dump(config),
        chunk_size=512,
        compressor=None,
        skip_signal_matrix=False,
    )
    new_zarr.attrs["radio_metadata_schema_version"] = 2
    for r in range(receivers):
        rz = new_zarr[f"receivers/r{r}"]
        for key, dtype in v7rx_scalar_keys.items():
            rz.create_dataset(key, shape=(timesteps,), dtype=dtype,
                              chunks=(timesteps,), compressor=None)
        for key, dtype in v7rx_2x_keys.items():
            rz.create_dataset(key, shape=(timesteps, 2), dtype=dtype,
                              chunks=(timesteps, 2), compressor=None)

    for r in range(receivers):
        _copy_receiver(
            rx_zarr[f"receivers/r{r}"], new_zarr[f"receivers/r{r}"], valid["idxs"],
            valid[f"r{r}"],
        )
    # Match v4_tx_rx_to_v5.py: the array-0 "off by 180deg" correction applies to
    # receiver 0 only (historical (0, 0.5) raw -> (1.0, 0.5) merged). July RO1/RO3
    # carry the identical raw (0, 0.5), so this reproduces the historical format.
    if fix_config:
        new_zarr["receivers/r0/rx_theta_in_pis"][:] = config["receivers"][0][
            "theta-in-pis"
        ]

    new_zarr.store.close()
    zarr_shrink(zarr_out_fn)
    return timesteps


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--txs", type=str, nargs="+", required=True, help="TX (emitter) zarrs")
    parser.add_argument("--rxs", type=str, nargs="+", required=True, help="RX zarrs")
    parser.add_argument("--output", type=str, required=True, help="output dir")
    parser.add_argument("--min-timesteps", type=int, default=500)
    parser.add_argument(
        "--fix-config", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    def _base(p):
        return os.path.basename(p).replace(".zarr.tmp", "").replace(".zarr", "")

    os.makedirs(args.output, exist_ok=True)
    for tx in args.txs:
        for rx in args.rxs:
            out = f"{args.output}/{_base(rx)}.{_base(tx)}.zarr"
            if os.path.isdir(out):
                logging.error("Output already exists %s", out)
                continue
            try:
                n = merge_v7rx_v7tx(
                    tx_fn=tx, rx_fn=rx, zarr_out_fn=out, fix_config=args.fix_config,
                    dry_run=args.dry_run, min_timesteps=args.min_timesteps,
                )
                if n >= args.min_timesteps or args.dry_run:
                    logging.info("%d  tx:%s rx:%s -> %s", n, tx, rx, out)
            except (lmdb.CorruptedError, lmdb.Error) as e:
                logging.error("CORRUPT tx:%s rx:%s : %s", tx, rx, e)
            except (KeyError, ValueError, AssertionError) as e:
                logging.error("DataError tx:%s rx:%s : %s", tx, rx, e)
