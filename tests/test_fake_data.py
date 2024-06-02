from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml


def test_fake_datasetv5():
    create_fake_dataset(filename="test_circle", yaml_config_str=fake_yaml, n=5)
