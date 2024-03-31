import argparse

from spf.dataset.spf_dataset import SessionsDatasetSimulated
from spf.plot.plot import filenames_to_gif, plot_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--session-idx", type=int, required=True)
    parser.add_argument("--steps", type=int, required=False, default=3)
    parser.add_argument(
        "--output_prefix", type=str, required=False, default="session_output"
    )
    args = parser.parse_args()

    ds = SessionsDatasetSimulated(args.dataset, snapshots_per_session=args.steps)
    session = ds[args.session_idx]
    filenames = plot_lines(session, args.steps, args.output_prefix)

    filenames_to_gif(filenames, "%s_lines.gif" % args.output_prefix, size=(1200, 400))
