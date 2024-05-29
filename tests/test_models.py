import numpy as np

from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml
from spf.notebooks.simple_train import get_parser, simple_train


def test_beamnet_downsampled():
    create_fake_dataset(filename="test_circle", yaml_config_str=fake_yaml, n=5)
    args_list = [
        "--device",
        "cpu",
        "--seed",
        "0",
        "--nthetas",
        "11",
        "--datasets",
        "test_circle.zarr",
        "--batch",
        "128",
        "--workers",
        "0",
        "--act",
        "selu",
        "--segmentation-level",
        "downsampled",
        "--type",
        "direct",
        "--seg-net",
        "conv",
        "--epochs",
        "200",
        "--skip-segmentation",
        "--no-shuffle",
        "--symmetry",
        # "--no-sigmoid",
        "--plot-every",
        "50",
        "--lr",
        "0.0002",
    ]
    args = get_parser().parse_args(args_list)

    train_results = simple_train(args)
    assert np.array(train_results["losses"])[-10:].mean() < 0.2
