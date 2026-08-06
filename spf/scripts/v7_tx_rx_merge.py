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

     Since 2026-08-06 the source zarr ATTRS are propagated too. The RX store's 31
     receiver attrs (direct_usb_serial / bus / port_path, firmware identity, ...)
     are copied verbatim into receivers/r*/.zattrs -- slots the merged store
     already had and left empty -- and its root attrs, the aeqd projection center
     and a TX source record go to root under namespaced keys. So a merged dataset
     now says which physical Pluto is r0, what firmware it ran, and whether the
     source capture finalized. See spf/dataset/provenance.py; datasets merged
     before that date carry a .provenance.json sidecar instead, and
     load_provenance() reads either.

     The TX record is deliberately four fields rather than thirty-one: the TX
     contributes GPS only, and its GPS is already here as tx_pos_*_mm -- exactly
     invertible via the recorded projection, so the TX radio's identity is not
     load-bearing.
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

WHICH SIDECARS THIS TOOL NEEDS
------------------------------
A staged capture is NOT just the ``.zarr`` -- it is the ``.zarr`` AND its ``.yaml``
sidecar (and, for the field record, its ``.log``). This merge reads the **RX**
capture's ``.yaml`` only: it is the antenna geometry / receiver config that the
merged dataset is written with. The **TX** capture contributes GPS from its zarr
and nothing else, so a TX rover that never came back online -- or whose sidecar
was left behind -- does not block a merge. Only the RX sidecars do.

Every RX sidecar is checked up front, before any pairing work, and ALL missing
ones are reported at once (see ``check_inputs``). Staging one with
``stage_captures.sh`` copies the whole family and cannot leave a sidecar behind.
"""

import argparse
import bisect
import logging
import os
import sys

import lmdb
import numpy as np
import yaml
from pyproj import Proj

from spf.dataset.provenance import build_provenance, write_provenance
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


def config_path_for(zarr_fn):
    """The ``.yaml`` sidecar that belongs to ``zarr_fn``.

    ``a.zarr -> a.yaml`` and ``a.zarr.tmp -> a.yaml.tmp``: an unfinalized capture
    keeps the suffix on all three family members, and finalization renames them
    together (ROVER_RUNBOOK 12.2). Split on the LAST ``.zarr`` rather than
    ``str.replace``, so a directory named ``.../rovers.zarr_staging/x.zarr`` is not
    mangled.
    """
    head, sep, tail = zarr_fn.rpartition(".zarr")
    if not sep:
        raise ValueError(f"not a .zarr path, cannot derive its .yaml sidecar: {zarr_fn}")
    return head + ".yaml" + tail


def check_inputs(txs, rxs):
    """Return a list of human-readable problems with the run's inputs.

    Empty list means the run has everything it needs. This exists because the
    sidecar read used to happen deep inside the per-pair merge: on 2026-08-04 a
    staging copy that took only the ``.zarr`` directories got through the whole
    TX x RX scan and then died on a FileNotFoundError naming ONE file, so the fix
    was iterative -- rerun, wait, learn about the next one.

    Only RX sidecars are required; see the module docstring on why the TX one is
    not read.
    """
    problems = []
    for tx in txs:
        if not os.path.exists(tx):
            problems.append(f"TX capture is missing: {tx}")
    missing_sidecars = []
    for rx in rxs:
        if not os.path.exists(rx):
            problems.append(f"RX capture is missing: {rx}")
            continue
        config_fn = config_path_for(rx)
        if not os.path.exists(config_fn):
            missing_sidecars.append((rx, config_fn))
    if missing_sidecars:
        problems.append(
            f"{len(missing_sidecars)} RX capture(s) staged WITHOUT their .yaml "
            "sidecar. A staged capture is the .zarr AND its .yaml (and .log); the "
            "merge reads the RX .yaml for the receiver/antenna config, so it "
            "cannot run without them:"
        )
        for rx, config_fn in missing_sidecars:
            problems.append(f"    {os.path.basename(rx)}  needs  {config_fn}")
        problems.append(
            "  Copy each capture WITH its sidecars from the rover, e.g.\n"
            "    data_collection/rover/rover_v3.1/stage_captures.sh \\\n"
            "        --from pi@roverpi1:temp --to <staging dir> --match '<capture>*'\n"
            "  (TX sidecars are NOT needed -- an offline TX rover never blocks a merge.)"
        )
    return problems


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
    # RX sidecar only -- and its presence was already checked for every RX before
    # any pairing started (check_inputs), so reaching here with it missing means
    # the file disappeared mid-run.
    with open(config_path_for(rx_fn), "r") as f:
        config = yaml.safe_load(f)
    assert (
        config["receivers"][0]["theta-in-pis"] != 0.0 or fix_config
    ), "theta-in-pis is 0 and --no-fix-config; ambiguous, refusing"
    config["receivers"][0]["theta-in-pis"] = 1.0
    with open(config_path_for(zarr_out_fn), "w") as f:
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
    # Which physical Pluto is r0, and did the RX capture finalise? Both used to be
    # unanswerable from a merged store: the 31 receiver attrs carrying sdr_serial
    # were dropped, and the merged filename discards a source's .tmp suffix. The
    # RX attrs go back into receivers/r*/.zattrs, which already existed and were
    # empty. See spf/dataset/provenance.py.
    prov_root, prov_receivers = build_provenance(
        rx_zarr=rx_zarr,
        rx_fn=rx_fn,
        tx_zarr=tx_zarr,
        tx_fn=tx_fn,
        center_lat=center_lat,
        center_lon=center_lon,
        n_receivers=receivers,
    )
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

    # After _copy_receiver, so a receiver group's attrs cannot be lost to the
    # create_dataset calls above.
    write_provenance(new_zarr, prov_root, prov_receivers)

    new_zarr.store.close()
    zarr_shrink(zarr_out_fn)
    return timesteps


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txs", type=str, nargs="+", required=True, help="TX (emitter) zarrs")
    parser.add_argument("--rxs", type=str, nargs="+", required=True, help="RX zarrs")
    parser.add_argument("--output", type=str, required=True, help="output dir")
    parser.add_argument("--min-timesteps", type=int, default=500)
    parser.add_argument(
        "--fix-config", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "report the TX/RX overlap map without copying IQ. Missing inputs are "
            "still reported (and still make the run exit non-zero)."
        ),
    )
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)

    # Fail fast, and completely. Pairing every TX against every RX takes minutes
    # on a field-sized staging directory; learning only then that a sidecar was
    # never copied off the rover -- one sidecar per rerun -- is the failure this
    # guards. A dry run gets the same report, because a dry run is exactly when
    # an operator is asking "is this staging directory good?", but it still
    # produces its overlap map first so the trip is not wasted.
    problems = check_inputs(args.txs, args.rxs)
    if problems and not args.dry_run:
        print("\n".join(["Refusing to merge -- inputs are incomplete:"] + problems),
              file=sys.stderr)
        return 2
    if problems:
        print("\n".join(["Inputs are incomplete (continuing: --dry-run):"] + problems),
              file=sys.stderr)

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

    if problems:
        # Dry run only: the map is printed, but the run is not "fine".
        print(
            "\nThe overlap map above is complete, but the input problems reported "
            "at the top of this run remain; a real merge will refuse until they "
            "are fixed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sys.exit(main())
