import tempfile
import numpy as np

from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.notebooks.simple_train import get_parser, simple_train


def test_beamnet_downsampled():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds_fn = f"{tmpdirname}/test_circle"
        create_fake_dataset(filename=ds_fn, yaml_config_str=fake_yaml, n=5)
        args_list = [
            "--device",
            "cpu",
            "--seed",
            "0",
            "--nthetas",
            "11",
            "--datasets",
            ds_fn,
            "--batch",
            "128",
            "--workers",
            "0",
            # "--batch-norm",
            "--act",
            "leaky",
            "--shuffle",
            "--segmentation-level",
            "downsampled",
            "--type",
            "direct",
            "--seg-net",
            "conv",
            "--epochs",
            "1200",
            # "--skip-segmentation",
            "--no-shuffle",
            "--symmetry",
            # "--sigmoid",
            "--no-sigmoid",
            "--block",
            # "--wandb-project",
            # "test123",
            "--plot-every",
            "50",
            "--lr",
            "0.0005",
            "--precompute-cache",
            tmpdirname,
        ]
        args = get_parser().parse_args(args_list)

        train_results = simple_train(args)
        assert np.array(train_results["losses"])[-10:].mean() < 0.2
