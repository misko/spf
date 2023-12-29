import argparse

from spf.dataset.spf_dataset import SessionsDatasetTask2
from spf.plot.plot import filenames_to_gif, plot_full_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--session-idx", type=int, required=True)
    parser.add_argument("--steps", type=int, required=False, default=3)
    parser.add_argument(
        "--output_prefix", type=str, required=False, default="session_output"
    )
    args = parser.parse_args()

    ds = SessionsDatasetTask2(args.dataset, snapshots_in_sample=args.steps)
    session = ds[args.session_idx]
    filenames = plot_full_session(session, args.steps, args.output_prefix)

    filenames_to_gif(filenames, "%s.gif" % args.output_prefix)
