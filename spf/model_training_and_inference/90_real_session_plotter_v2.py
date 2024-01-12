import argparse

from spf.dataset.spf_dataset import SessionsDatasetRealV2
from spf.plot.plot import filenames_to_gif, plot_full_session

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
        "--output_prefix", type=str, required=False, default="session_output"
    )
    args = parser.parse_args()
    assert args.snapshots_in_session >= args.steps
    ds = SessionsDatasetRealV2(
        root_dir=args.root_dir,
        snapshots_in_session=args.snapshots_in_session,
        nsources=1,
        step_size=args.step_size,
    )
    print(ds)
    print("DSLEN", ds.len)

    session = ds[args.session_idx]
    filenames = plot_full_session(session, args.steps, args.output_prefix, invert=True)

    filenames_to_gif(filenames, "%s.gif" % args.output_prefix, duration=args.duration)
