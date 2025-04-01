import numpy as np
import torch

from spf.model_training_and_inference.models.single_point_networks import (
    normal_correction_for_bounded_range,
)


def test_normal_correction():
    # mean is on the boundary of the bounded range, half of the dist is out of bounds
    # factor should be 2
    assert np.isclose(
        normal_correction_for_bounded_range(
            mean=torch.tensor(torch.pi), sigma=torch.tensor(1), max_y=np.pi
        ),
        2,
    )
    # mean is in center of range with tight sigma, almost all of dist is in range
    # factor should be 1
    assert np.isclose(
        normal_correction_for_bounded_range(
            mean=torch.tensor(0), sigma=torch.tensor(0.001), max_y=np.pi
        ),
        1,
    )
    # mean is in center of range with box at 1std
    # 68% of density is in the box
    # correction factor should be 1/0.68
    # factor should be ~1.47
    assert np.isclose(
        normal_correction_for_bounded_range(
            mean=torch.tensor(0), sigma=torch.tensor(1), max_y=1
        ),
        1.47,
        atol=0.01,
    )
