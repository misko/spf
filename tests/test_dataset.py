import argparse
import os
import tempfile

import numpy as np
import pytest
from compress_pickle import dump
from joblib import Parallel, delayed

from spf.dataset.spf_generate import generate_session, generate_session_and_dump


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@pytest.fixture
def default_args():
    return dotdict(
        {
            "carrier_frequency": 2.4e9,
            "signal_frequency": 100e3,
            "sampling_frequency": 10e6,
            "array_type": "linear",  # "circular"],
            "elements": 11,
            "random_silence": False,
            "detector_noise": 1e-4,
            "random_emitter_timing": False,
            "sources": 2,
            "seed": 0,
            "beam_former_spacing": 256 + 1,
            "width": 128,
            "detector_trajectory": "bounce",
            "detector_speed": 10.0,
            "source_speed": 0.0,
            "sigma": 1.0,
            "time_steps": 100,
            "time_interval": 0.3,
            "samples_per_snapshot": 3,
            "sessions": 16,
            "reference": False,
            "cpus": 8,
            "live": False,
            "profile": False,
            "fixed_detector": None,  #
        }
    )


def test_data_generation(default_args):
    with tempfile.TemporaryDirectory() as tmp:
        args = default_args
        args.output = tmp
        print(args)

        dump(args, "/".join([args.output, "args.pkl"]), compression="lzma")
        result = [
            generate_session_and_dump((args, session_idx))
            for session_idx in range(args.sessions)
        ]