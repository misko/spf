import argparse
import glob
import os
import pickle

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


def merge_results(results):
    merged = {}

    for _results in results:
        for result in _results:
            result = result.copy()
            _ty = result.pop("type")
            _fn = result.pop("ds_fn").lower()
            movement = ""
            if "circle" in _fn:
                movement = "circle"
            elif "bounce" in _fn:
                movement = "bounce"
            else:
                raise ValueError(f"Cannot figure out movement {_fn}")
            metrics = result.pop("metrics")
            name = f"movement:{movement}," + cannonical(_ty, result)
            if name not in merged:
                merged[name] = []
            merged[name].append(metrics)
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


def write_csv(merged, output_csv_fn):
    output = open(output_csv_fn, "w")
    for name in merged:
        summed_metrics = {}
        n = len(merged[name])
        for metrics in merged[name]:
            for key, value in metrics.items():
                if key not in summed_metrics:
                    summed_metrics[key] = 0
                summed_metrics[key] += value / n
        keys = sorted(summed_metrics.keys())
        output.write(
            get_type(name)
            + ","
            + ",".join([str(key + "," + str(summed_metrics[key])) for key in keys])
            + f",{name},"
            + "\n"
        )


def report_workdir_to_csv(workdir, output_csv_fn):
    results_fns = glob.glob(f"{workdir}/*.pkl")

    results = list(
        tqdm.tqdm(
            map(read_pkl, results_fns),
            total=len(results_fns),
        )
    )

    merged_results = merge_results(results)
    write_csv(merged_results, output_csv_fn)


if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output",
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
    report_workdir_to_csv(args.work_dir, args.output)
