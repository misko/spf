import argparse
from pathlib import Path

from spf.dataset.spf_dataset import SessionsDatasetRealV2
from spf.plot.plot import filenames_to_gif, plot_full_session_v2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--snapshots-in-session", type=int, required=False, default=128)
    parser.add_argument("--width", type=int, required=False, default=3000)
    parser.add_argument("--session-idx", type=int, required=True)
    parser.add_argument("--steps", type=int, required=False, default=3)
    parser.add_argument("--step-size", type=int, required=False, default=16)
    parser.add_argument("--duration", type=int, required=False, default=50)
    parser.add_argument(
        "--output-folder", type=Path, required=False, default="tmp_v2plotter"
    )
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{args.output_folder}/s{args.session_idx}_steps{args.steps}_ssize{args.step_size}"

    assert args.snapshots_in_session >= args.steps
    ds = SessionsDatasetRealV2(
        root_dir=args.root_dir,
        snapshots_in_session=args.snapshots_in_session,
        nsources=1,
        step_size=args.step_size,
    )

    if ds.n_receivers == 1:
        filenames = plot_full_session_v2(
            ds[args.session_idx],
            args.steps,
            output_prefix,
            invert=True,
        )
        filenames_to_gif(filenames, f"{output_prefix}.gif", duration=args.duration)
    else:
        sessions = [
            ds[0, args.session_idx],
            ds[1, args.session_idx],
        ]
        filenames = plot_full_session_v2(
            sessions,
            args.steps,
            output_prefix,
            invert=True,
        )
        filenames_to_gif(
            filenames, f"{output_prefix}.gif", duration=args.duration, size=(900, 600)
        )
