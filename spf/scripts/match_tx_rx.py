import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--arg1",
    dest="argument_1",
    nargs="+",
    type=str,
    help="Multiple integers for argument 1",
)
parser.add_argument(
    "--arg2",
    dest="argument_2",
    nargs="*",
    type=str,
    help="Multiple strings for argument 2",
)

args = parser.parse_args()

if args.argument_1:
    print(f"Argument 1 values: {args.argument_1}")

if args.argument_2:
    print(f"Argument 2 values: {args.argument_2}")
