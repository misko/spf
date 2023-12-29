import argparse
import io
import pickle
import random

import numpy as np
import torch

from spf.baseline_algorithm import baseline_algorithm
from spf.dataset.spf_dataset import (
    SessionsDatasetTask2,
    collate_fn,
    output_cols,
    rel_to_pos,
)
from spf.plot.plot import plot_predictions_and_baseline


# from online https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--session-idx", type=int, required=True)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--test-fraction", type=float, required=False, default=0.2)
    parser.add_argument("--model-name", type=str, required=True)

    parser.add_argument(
        "--output_prefix", type=str, required=False, default="session_output"
    )
    args = parser.parse_args()

    d = CPU_Unpickler(open(args.load, "rb")).load()
    str_to_model = {model["name"]: model for model in d["models"]}
    model = str_to_model[args.model_name]

    args.snapshots_per_sample = model["snapshots_per_sample"]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = SessionsDatasetTask2(
        args.dataset, snapshots_in_sample=args.snapshots_per_sample
    )
    train_size = int(len(ds) * args.test_fraction)
    test_size = len(ds) - train_size

    ds_train = torch.utils.data.Subset(ds, np.arange(train_size))
    ds_test = torch.utils.data.Subset(ds, np.arange(train_size, train_size + test_size))

    session = ds_test[args.session_idx]
    # _in=collate_fn([session])
    _in = collate_fn([session])
    width = session["width_at_t"][0][0]

    # get the baseline
    fp, imgs, baseline_predictions = baseline_algorithm(
        session, width, steps=model["snapshots_per_sample"]
    )

    # get the predictions from the model
    with torch.no_grad():
        model_pred = model["model"].cpu()(_in["inputs"])["transformer_pred"]
        model_predictions = rel_to_pos(
            model_pred[0, :, output_cols["src_pos"]], width
        ).clamp(0, width)
    plot_predictions_and_baseline(
        session,
        args,
        model["snapshots_per_sample"] - 1,
        {"name": "baseline algorithm", "predictions": baseline_predictions},
        {"name": "NN " + args.model_name, "predictions": model_predictions},
    )
    # filenames=plot_lines(session,args.steps,args.output_prefix)

    # filenames_to_gif(filenames,"%s_lines.gif" % args.output_prefix,size=(1200,400))
