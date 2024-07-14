import pickle
import sys

if len(sys.argv) != 2:
    print(sys.argv[0], "pickle_input")
    sys.exit(1)

pkl_fn = sys.argv[1]

pickle.load(open(pkl_fn, "rb"))
