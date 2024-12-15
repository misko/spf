import argparse
import glob
import logging
import os
import pickle

import torch
import tqdm


def read_pkl(pkl_fn):
    return pickle.load(open(pkl_fn, "rb"))


def cannonical(_ty, params):
    name = _ty
    keys = sorted(params.keys())
    for key in keys:
        if key == "runtime":
            continue
        name += f",{key},{str(params[key])}"
    name = (
        name.replace("rx_idx,0", "")
        .replace("rx_idx,1", "")
        .replace("dyanmic_R", "dynamic_R")
    )
    return name


def result_to_tuple(result, header):
    l = []
    for key in header:
        v = result[key]
        if isinstance(v, torch.Tensor):
            v = f"{v.item():0.4e}"
        l.append(v)
    return tuple(l)


def merge_results(results, header):
    merged = {}

    for _results in results:
        for result in _results:
            result = result.copy()
            # _ty = result.pop("type")
            _fn = result.pop("ds_fn").lower()
            movement = ""
            if "circle" in _fn:
                movement = "circle"
            elif "bounce" in _fn:
                movement = "bounce"
            elif "calibrate" in _fn:
                movement = "calibrate"
            else:
                raise ValueError(f"Cannot figure out movement {_fn}")
            result["movement"] = movement
            key = result_to_tuple(result, header)
            if key not in merged:
                merged[key] = []
            merged[key].append(result["metrics"])
    return merged


def get_type(key):
    key = key.lower()
    if "xy" in key:
        return "XY"
    if "single_radio" in key:
        return "single_radio"
    if "dual_radio" in key:
        return "dual_radio"
    raise ValueError(key)


def write_csv(merged, output_csv_fn, header):
    a_result_key = list(merged.keys())[0]
    a_result = merged[a_result_key][0]
    for metric_key in a_result:
        header.append(metric_key)
    # find out header for metrics

    output = open(output_csv_fn, "w")
    # write header
    output.write(",".join(header) + "\n")

    # write lines
    for run_key in merged:
        summed_metrics = {}
        n = len(merged[run_key])
        for metrics in merged[run_key]:
            for key, value in metrics.items():
                if key not in summed_metrics:
                    summed_metrics[key] = 0
                summed_metrics[key] += value / n

        line = [str(x) for x in run_key]
        for header_idx in range(len(run_key), len(header)):
            line.append(f"{summed_metrics[header[header_idx]]:.4f}")

        output.write(",".join(line) + "\n")


def report_workdir_to_csv(workdir, output_csv_fn):
    results = {}  # glob.glob(f"{workdir}/*.pkl")
    algorithm_paths = glob.glob(f"{workdir}/*")
    for algorithm_path in algorithm_paths:
        logging.info(f"Processing: {algorithm_path}")
        fns = []
        for dirpath, dirnames, filenames in os.walk(algorithm_path):
            for file in filenames:
                if file.endswith(".pkl"):
                    fns.append(os.path.join(dirpath, file))
        results = list(
            tqdm.tqdm(
                map(read_pkl, fns),
                total=len(fns),
            )
        )
        header = ["type", "movement", "segmentation_version"]
        for field in results[0][0].keys():
            if field == "ds_fn":
                pass
            elif "metric" not in field and field not in header:
                header.append(field)

        merged_results = merge_results(results, header)

        write_csv(
            merged_results,
            output_csv_fn + os.path.basename(algorithm_path) + ".csv",
            header,
        )


if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output-prefix",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--work-dir",
            type=str,
            required=True,
        )

        return parser

    parser = get_parser()
    args = parser.parse_args()
    report_workdir_to_csv(args.work_dir, args.output_prefix)
