import tempfile

import numpy as np
import pytest
from compress_pickle import dump

from spf.dataset.spf_dataset import SessionsDataset
from spf.dataset.spf_generate import generate_session_and_dump
from spf.rf import get_peaks_for_2rx
from spf.utils import dotdict


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
            "sigma_noise": 1.0,
            "time_steps": 1024,
            "time_interval": 0.3,
            "readings_per_snapshot": 3,
            "sessions": 2,
            "reference": False,
            "cpus": 2,
            "live": False,
            "profile": False,
            "fixed_detector": None,  #
        }
    )


def test_data_generation(default_args):
    with tempfile.TemporaryDirectory() as tmp:
        args = default_args
        args.output = tmp
        dump(
            args,
            "/".join([args.output, "args.pkl"]),
            compression="lzma",
        )
        result = [  # noqa
            generate_session_and_dump((args, session_idx))
            for session_idx in range(args.sessions)
        ]
        ds = SessionsDataset(root_dir=tmp, snapshots_in_sample=1024)
        session = ds[1]
        dump(
            session,
            "/".join([args.output, "onesession.pkl"]),
            compression="lzma",
        )


def test_closeness_to_ground_truth(default_args):
    with tempfile.TemporaryDirectory() as tmp:
        args = default_args
        args.output = tmp
        args.live = True
        args.sources = 1
        args.elements = 2
        args.detector_speed = 0.0
        args.source_speed = 10.0
        args.sigma_noise = 0.0
        args.detector_noise = 0.0
        args.beam_former_spacing = 4096 + 1
        dump(
            args,
            "/".join([args.output, "args.pkl"]),
            compression="lzma",
        )
        ds = SessionsDataset(root_dir=tmp, snapshots_in_sample=1024)
        session = ds[1]
        peaks_at_t = np.array(
            [
                get_peaks_for_2rx(bf_out)
                for bf_out in session["beam_former_outputs_at_t"]
            ]
        )
        peaks_at_t_in_radians = (
            2 * (peaks_at_t / args.beam_former_spacing - 0.5) * np.pi
        )
        peaks_at_t_in_radians_adjusted = (
            peaks_at_t_in_radians + session["detector_orientation_at_t"]
        )
        ground_truth = (
            session["detector_orientation_at_t"] + session["source_theta_at_t"]
        )
        deviation = (
            np.abs(peaks_at_t_in_radians_adjusted - ground_truth).min(axis=1).mean()
        )
        assert deviation < 0.01


def test_live_data_generation(default_args):
    with tempfile.TemporaryDirectory() as tmp:
        args = default_args
        args.output = tmp
        args.live = True
        dump(
            args,
            "/".join([args.output, "args.pkl"]),
            compression="lzma",
        )
        result = [  # noqa
            generate_session_and_dump((args, session_idx))
            for session_idx in range(args.sessions)
        ]
        ds = SessionsDataset(root_dir=tmp, snapshots_in_sample=1024)
        session = ds[1]
        dump(
            session,
            "/".join([args.output, "onesession.pkl"]),
            compression="lzma",
        )
