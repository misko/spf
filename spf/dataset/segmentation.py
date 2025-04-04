import os
import pickle
from multiprocessing import Pool, cpu_count

from spf.utils import SEGMENTATION_VERSION

try:
    import cupy as cp
except:
    pass
import numpy as np
import torch
import tqdm
from scipy.stats import trim_mean

from spf.rf import (
    beamformer_given_steering_nomean,
    beamformer_given_steering_nomean_cp,
    get_phase_diff,
    get_stats_for_signal,
    mean_phase_mean,
    pi_norm,
    reduce_theta_to_positive_y,
    windowed_trimmed_circular_mean_and_stddev,
)
from spf.scripts.zarr_utils import (
    new_yarr_dataset,
    zarr_open_from_lmdb_store,
    zarr_open_from_lmdb_store_cm,
    zarr_shrink,
)
from spf.sdrpluto.detrend import detrend_np

default_segment_args = {
    "window_size": 2048,
    "stride": 2048,
    "trim": 20.0,
    "mean_diff_threshold": 0.2,
    "max_stddev_threshold": 0.5,
    "drop_less_than_size": 3000,
    "min_abs_signal": 40,
}


def segment_single_session(
    zarr_fn, steering_vectors_for_receiver, session_idx, receiver_idx, gpu=False
):
    args = {
        "zarr_fn": zarr_fn,
        "receiver": receiver_idx,
        "session_idx": session_idx,
        "steering_vectors": steering_vectors_for_receiver,
        "gpu": gpu,
        **default_segment_args,
    }
    return segment_session(**args)


# MultiProcess segmentation and pre-processing
# This contains the heart of pre-processing
# The input here is a zarr file containing the raw radio signal
# The output is another zarr file (yarr) containing the segmentation
# and other pre-processed features. This yarr file can then be loaded
# during training, avoiding touching any files containing the very large
# raw radio features
def mp_segment_zarr(
    zarr_fn,
    results_fn,
    steering_vectors_for_all_receivers,
    precompute_to_idx=-1,
    gpu=False,
    n_parallel=20,
    skip_beamformer=False,
):
    z = zarr_open_from_lmdb_store(zarr_fn)
    yarr_fn = results_fn.replace(".pkl", ".yarr")
    already_computed = 0

    previous_simple_segmentation = {
        f"r{r_idx}": [] for r_idx in range(len(z["receivers"]))
    }
    precomputed_zarr = None
    if os.path.exists(results_fn) and os.path.exists(yarr_fn):
        previous_simple_segmentation = pickle.load(open(results_fn, "rb"))[
            "segmentation_by_receiver"
        ]
        precomputed_zarr = zarr_open_from_lmdb_store(yarr_fn, mode="rw", map_size=2**32)
        already_computed = min(
            [
                (
                    np.sum(precomputed_zarr[f"r{r_idx}/all_windows_stats"], axis=(1, 2))
                    > 0
                ).sum()
                for r_idx in range(len(z["receivers"]))
            ]
        )

    if precompute_to_idx < 0:
        precompute_to_idx = min(
            [
                (z[f"receivers/r{ridx}/system_timestamp"][:] > 0).sum()
                for ridx in range(len(z["receivers"]))
            ]
        )
    else:
        precompute_to_idx += 1

    # we dont need to segment anything
    if precompute_to_idx <= already_computed:
        return already_computed

    assert len(z["receivers"]) == 2

    n_sessions = z.receivers["r0"].system_timestamp.shape[0]

    results_by_receiver = {}
    for r_idx in [0, 1]:
        r_name = f"r{r_idx}"
        inputs = [
            {
                "zarr_fn": zarr_fn,
                "receiver": r_name,
                "session_idx": idx,
                "steering_vectors": steering_vectors_for_all_receivers[r_idx],
                "gpu": gpu,
                "skip_beamformer": skip_beamformer,
                **default_segment_args,
            }
            for idx in range(already_computed, precompute_to_idx)
        ]

        # n_parallel = 0
        if n_parallel > 0:
            with Pool(
                min(cpu_count(), n_parallel)
            ) as pool:  # cpu_count())  # cpu_count() // 4)
                results_by_receiver[r_name] = list(
                    tqdm.tqdm(
                        pool.imap(segment_session_star, inputs),
                        desc="segmentation",
                        total=len(inputs),
                    )
                )
        else:
            results_by_receiver[r_name] = list(
                tqdm.tqdm(
                    map(segment_session_star, inputs),
                    desc="segmentation",
                    total=len(inputs),
                )
            )

    segmentation_zarr_fn = results_fn.replace(".pkl", ".yarr")
    all_windows_stats_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "all_windows_stats"
    ].shape
    windowed_beamformer_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "windowed_beamformer"
    ].shape
    weighted_beamformer_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "windowed_beamformer"
    ].shape[1:]
    downsampled_segmentation_mask_shape = (n_sessions,) + results_by_receiver["r0"][0][
        "downsampled_segmentation_mask"
    ].shape

    if precomputed_zarr is None:
        os.makedirs(os.path.dirname(segmentation_zarr_fn), exist_ok=True)
        precomputed_zarr = new_yarr_dataset(
            filename=segmentation_zarr_fn,
            n_receivers=2,
            all_windows_stats_shape=all_windows_stats_shape,
            windowed_beamformer_shape=windowed_beamformer_shape,
            weighted_beamformer_shape=weighted_beamformer_shape,
            downsampled_segmentation_mask_shape=downsampled_segmentation_mask_shape,
            mean_phase_shape=(n_sessions,),
        )

    for r_idx in [0, 1]:
        # collect all windows stats
        precomputed_zarr[f"r{r_idx}/all_windows_stats"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["all_windows_stats"][None] for x in results_by_receiver[f"r{r_idx}"]]
        )
        # collect windowed beamformer
        precomputed_zarr[f"r{r_idx}/windowed_beamformer"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["windowed_beamformer"][None] for x in results_by_receiver[f"r{r_idx}"]]
        )
        # collect weighted beamformer
        precomputed_zarr[f"r{r_idx}/weighted_beamformer"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["weighted_beamformer"][None] for x in results_by_receiver[f"r{r_idx}"]]
        )
        # collect weighted_windows_stats
        precomputed_zarr[f"r{r_idx}/weighted_windows_stats"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [x["weighted_stats"][None] for x in results_by_receiver[f"r{r_idx}"]]
        )

        # collect downsampled segmentation mask
        precomputed_zarr[f"r{r_idx}/downsampled_segmentation_mask"][
            already_computed:precompute_to_idx
        ] = np.vstack(
            [
                x["downsampled_segmentation_mask"][None]
                for x in results_by_receiver[f"r{r_idx}"]
            ]
        )

        mean_phases = []
        for result in results_by_receiver[f"r{r_idx}"]:
            means = []
            weights = []
            for x in result["simple_segmentation"]:
                if x["type"] == "signal":
                    means.append(x["mean"])
                    weights.append(
                        (x["end_idx"] - x["start_idx"])
                        * x["abs_signal_median"]
                        / (x["stddev"] + 1e-6)  # weight by signal strength and region
                    )
            if len(means) == 0:
                mean_phases.append(torch.nan)
            else:
                means = np.array(means)
                weights = np.array(weights)
                # weights /= weights.sum()
                mean_phases.append(mean_phase_mean(angles=means, weights=weights))
        mean_phase = np.hstack(mean_phases)

        # mean_phase = np.hstack(
        #     [
        #         (
        #             torch.tensor(
        #                 [x["mean"] for x in result["simple_segmentation"]]
        #             ).mean()
        #             if len(result) > 0
        #             else torch.tensor(float("nan"))
        #         )
        #         for result in results_by_receiver[f"r{r_idx}"]
        #     ]
        # )
        # TODO THIS SHOULD BE FIXED!!!
        # mean_phase[~np.isfinite(mean_phase)] = 0

        # assert np.isfinite(mean_phase).all()
        precomputed_zarr[f"r{r_idx}/mean_phase"][
            already_computed:precompute_to_idx
        ] = mean_phase

    # print(previous_simple_segmentation)
    # print("WINDOWED BEAMFORMER 7", precomputed_zarr["r0/windowed_beamformer"][7])
    simple_segmentation = {}
    for r_idx in [0, 1]:
        simple_segmentation[f"r{r_idx}"] = (
            previous_simple_segmentation[f"r{r_idx}"][:already_computed]
            + [
                {"simple_segmentation": x["simple_segmentation"]}
                for x in results_by_receiver[f"r{r_idx}"]
            ]
            + [
                {"simple_segmentation": []}
                for _ in range(n_sessions - precompute_to_idx)
            ]
        )

    precomputed_zarr.store.close()
    precomputed_zarr = None
    zarr_shrink(segmentation_zarr_fn)
    pickle.dump(
        {
            "version": SEGMENTATION_VERSION,
            "segmentation_by_receiver": simple_segmentation,
        },
        open(results_fn, "wb"),
    )
    return precompute_to_idx


# helper function to expand a dictionary in keywords args
def segment_session_star(arg_dict):
    return segment_session(**arg_dict)


# take a single zarr, receiver and session_idx and segment it
def segment_session(
    zarr_fn, receiver, session_idx, gpu=True, skip_beamformer=False, **kwrgs
):
    with zarr_open_from_lmdb_store_cm(zarr_fn, mode="r", readahead=True) as z:
        # z[f"receivers/r{receiver}/system_timestamp"][session_idx] > 0

        v = z.receivers[receiver].signal_matrix[session_idx][:].astype(np.complex64)

        ##
        # detrend here
        v = detrend_np(v)
        ##

        segmentation_results = simple_segment(v, **kwrgs)

        segmentation_results["all_windows_stats"] = (
            segmentation_results["all_windows_stats"].astype(np.float16).T
        )

        _, windows = segmentation_results["all_windows_stats"].shape
        nthetas = kwrgs["steering_vectors"].shape[0]

        if not skip_beamformer:
            if gpu:
                segmentation_results["windowed_beamformer"] = cp.asnumpy(
                    beamformer_given_steering_nomean_cp(
                        steering_vectors=cp.asarray(kwrgs["steering_vectors"]),
                        signal_matrix=cp.asarray(v.astype(np.complex64)),
                    )
                    .reshape(nthetas, -1, kwrgs["window_size"])
                    .mean(axis=2)
                    .T
                )
            else:
                segmentation_results["windowed_beamformer"] = (
                    beamformer_given_steering_nomean(
                        steering_vectors=kwrgs["steering_vectors"],
                        signal_matrix=v.astype(np.complex64),
                    )
                    .reshape(nthetas, -1, kwrgs["window_size"])
                    .mean(axis=2)
                    .T
                )
            weighted_beamformer = (
                segmentation_results["windowed_beamformer"].astype(np.float32)
                * segmentation_results["downsampled_segmentation_mask"][:, None]
            ).sum(axis=0) / (
                segmentation_results["downsampled_segmentation_mask"].sum() + 0.001
            )
            segmentation_results["windowed_beamformer"] = segmentation_results[
                "windowed_beamformer"
            ].astype(np.float16)

            segmentation_results["weighted_beamformer"] = weighted_beamformer
        else:
            # do this on GPU?
            windowed_beamformer = np.zeros((windows, nthetas), dtype=np.float16)
            windowed_beamformer.fill(np.nan)
            segmentation_results["windowed_beamformer"] = windowed_beamformer

        if segmentation_results["downsampled_segmentation_mask"].sum() > 0:
            # all_window_stats = trimmed_cm, trimmed_stddev, abs_signal_median
            mean_phase = trim_mean(
                reduce_theta_to_positive_y(
                    segmentation_results["all_windows_stats"][0]
                )[segmentation_results["downsampled_segmentation_mask"]].astype(
                    np.float32
                ),
                0.1,
            )
            stddev_and_abs_signal = trim_mean(
                segmentation_results["all_windows_stats"][1:][
                    :, segmentation_results["downsampled_segmentation_mask"]
                ].astype(np.float32),
                0.1,
                axis=1,
            )
            segmentation_results["weighted_stats"] = np.array(
                [mean_phase, stddev_and_abs_signal[0], stddev_and_abs_signal[1]],
                dtype=np.float32,
            )
        else:
            segmentation_results["weighted_stats"] = np.array([-1, -1, -1])
        return segmentation_results


def simple_segment(
    v,
    window_size,
    stride,
    trim,
    mean_diff_threshold,
    max_stddev_threshold,
    drop_less_than_size,
    min_abs_signal,
    steering_vectors=None,  # not used but passed in
):
    assert v.ndim == 2 and v.shape[0] == 2

    # numba incorrectly compiles this and it returns a float64 :'(
    pd = get_phase_diff(v).astype(np.float32)
    candidate_windows = []
    window_idxs_and_stats = windowed_trimmed_circular_mean_and_stddev(
        v, pd, window_size=window_size, stride=stride, trim=trim
    )
    # window_idxs_and_stats[1]= trimmed_cm, trimmed_stddev, abs_signal_median

    original_windows = [
        {
            "start_idx": idx[0],
            "end_idx": idx[1],
            "mean": stats[0],
            "stddev": stats[1],
            "abs_signal_median": stats[2],
        }
        for idx, stats in zip(window_idxs_and_stats[0], window_idxs_and_stats[1])
    ]

    # combine windows
    candidate_windows = combine_windows(
        original_windows, max_stddev_threshold, min_abs_signal
    )

    # drop all noise windows less than 3windows in size
    candidate_windows = drop_windows_smaller_than(
        candidate_windows, drop_less_than_size
    )

    # only keep signal windows surounded by noise
    candidate_windows = keep_signal_surrounded_by_noise(candidate_windows)
    #
    # candidate_windows = drop_noise_windows(candidate_windows)

    simple_segmentation = recompute_stats_for_windows(candidate_windows, v, pd, trim)
    downsampled_segmentation_mask = compute_downsampled_segmentation_mask(
        simple_segmentation,
        window_size=window_size,
        n_windows=window_idxs_and_stats[1].shape[0],
    )

    return {
        "simple_segmentation": simple_segmentation,
        # "all_windows": original_windows,
        "downsampled_segmentation_mask": downsampled_segmentation_mask,
        "all_windows_stats": window_idxs_and_stats[
            1
        ],  # trimmed_cm, trimmed_stddev, abs_signal_median
    }


def drop_noise_windows(windows):
    valid_windows = []
    for _, window in enumerate(windows):
        if window["type"] == "signal":
            valid_windows.append(window)
    return valid_windows


# 3.11 has this
def keep_signal_surrounded_by_noise(windows):
    valid_windows = []
    for window_idx, window in enumerate(windows):
        if window["type"] == "signal":
            # check if one before was signal
            if window_idx > 0 and windows[window_idx - 1]["type"] == "signal":
                continue
            # check if one after was signal
            if (
                window_idx + 1 < len(windows)
                and windows[window_idx + 1]["type"] == "signal"
            ):
                continue
            valid_windows.append(window)
    return valid_windows


# 3.3 SEGMENTATION VERSION HAS THIS
# def keep_signal_surrounded_by_noise(windows):
#     valid_windows = []
#     for window_idx, window in enumerate(windows):
#         if window["type"] == "signal":
#             if window["stddev"] > 0.03:
#                 #     # check if one before was signal
#                 before_is_signal = False
#                 if window_idx > 0 and windows[window_idx - 1]["type"] == "signal":
#                     before_is_signal = True
#                 # check if one after was signal
#                 after_is_signal = False
#                 if (
#                     window_idx + 1 < len(windows)
#                     and windows[window_idx + 1]["type"] == "signal"
#                 ):
#                     after_is_signal = True
#                 if before_is_signal and after_is_signal:
#                     continue
#             valid_windows.append(window)
#     return valid_windows


# 3.2 SEGMENTATION VERSION HAD THIS
# def keep_signal_surrounded_by_noise(windows):
#     valid_windows = []
#     for window_idx, window in enumerate(windows):
#         if window["type"] == "signal":
#             if window["stddev"] > 1.0:
#                 continue
#             if window["stddev"] > 0.03:
#                 #     # check if one before was signal
#                 before_is_different_signal = False
#                 if (
#                     window_idx > 0
#                     and windows[window_idx - 1]["type"] == "signal"
#                     and pi_norm(windows[window_idx - 1]["mean"] - window["mean"]) > 0.5
#                 ):
#                     before_is_different_signal = True
#                 # check if one after was signal
#                 after_is_different_signal = False
#                 if (
#                     window_idx + 1 < len(windows)
#                     and windows[window_idx + 1]["type"] == "signal"
#                 ) and pi_norm(windows[window_idx + 1]["mean"] - window["mean"]) < 0.5:
#                     after_is_different_signal = True
#                 # if the means are different enough then dont
#                 if before_is_different_signal and after_is_different_signal:
#                     continue
#             valid_windows.append(window)
#     return valid_windows


def drop_windows_smaller_than(windows, drop_less_than_size):
    return [w for w in windows if (w["end_idx"] - w["start_idx"]) > drop_less_than_size]


def combine_windows(windows, max_stddev_threshold, min_abs_signal):
    # combine windows
    new_windows = []
    for window in windows:
        if (
            window["stddev"] < max_stddev_threshold
            or window["abs_signal_median"] >= min_abs_signal
        ):
            if (
                len(new_windows) > 0
                and new_windows[-1]["type"] == "signal"
                and pi_norm(abs(new_windows[-1]["mean"] - window["mean"])) < 0.2
                and abs(new_windows[-1]["stddev"] - window["stddev"]) < 0.1
            ):
                new_windows[-1]["end_idx"] = window["end_idx"]
            else:
                window["type"] = "signal"
                new_windows.append(window)
        else:
            # previous window was also noise
            if len(new_windows) > 0 and new_windows[-1]["type"] == "noise":
                new_windows[-1]["end_idx"] = window["end_idx"]
            else:
                window["type"] = "noise"
                new_windows.append(window)
    return new_windows


def recompute_stats_for_windows(windows, v, pd, trim):
    for window in windows:
        _pd = pd[window["start_idx"] : window["end_idx"]]
        _v = v[:, window["start_idx"] : window["end_idx"]]
        # TODO THIS ISNT RIGHT???
        # _v = v[window["start_idx"] : window["end_idx"]]
        r = get_stats_for_signal(_v, _pd, trim)
        window["mean"] = r[0]
        window["stddev"] = r[1]
        window["abs_signal_median"] = r[2]
    return windows


def compute_downsampled_segmentation_mask(simple_segmentation, n_windows, window_size):
    seg_mask = np.zeros(n_windows).astype(bool)
    for window in simple_segmentation:
        seg_mask[
            window["start_idx"] // window_size : window["end_idx"] // window_size,
        ] = True
    return seg_mask
