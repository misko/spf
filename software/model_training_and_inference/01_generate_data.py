import argparse
import os
import pickle
import sys
import bz2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from compress_pickle import dump, load
from utils.rf import NoiseWrapper, QAMSource, UCADetector, ULADetector, beamformer

from utils.spf_generate import generate_session_and_dump

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--carrier-frequency', type=float, required=False, default=2.4e9)
	parser.add_argument('--signal-frequency', type=float, required=False, default=100e3)
	parser.add_argument('--sampling-frequency', type=float, required=False, default=10e6)
	parser.add_argument('--array-type', type=str, required=False, default='circular',choices=['linear','circular'])
	parser.add_argument('--elements', type=int, required=False, default=11)
	parser.add_argument('--sources', type=int, required=False, default=2)
	parser.add_argument('--seed', type=int, required=False, default=0)
	parser.add_argument('--width', type=int, required=False, default=128)
	parser.add_argument('--detector-trajectory', type=str, required=False, default='orbit',choices=['orbit','bounce'])
	parser.add_argument('--detector-speed', type=float, required=False, default=10.0)
	parser.add_argument('--sigma', type=float, required=False, default=1.0)
	parser.add_argument('--time-steps', type=int, required=False, default=100)
	parser.add_argument('--time-interval', type=float, required=False, default=5.0)
	parser.add_argument('--samples-per-snapshot', type=int, required=False, default=3)
	parser.add_argument('--sessions', type=int, required=False, default=1024)
	parser.add_argument('--output', type=str, required=False, default="sessions-default")
	parser.add_argument('--reference', type=bool, required=False, default=False)
	parser.add_argument('--cpus', type=int, required=False, default=8)
	parser.add_argument('--live', type=bool, required=False, default=False)
	args = parser.parse_args()
	os.mkdir(args.output)
	dump(args,"/".join([args.output,'args.pkl']),compression="lzma")
	if not args.live:
		result = Parallel(n_jobs=args.cpus)(delayed(generate_session_and_dump)((args,session_idx)) for session_idx in tqdm(range(args.sessions)))
