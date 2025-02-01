import torch

from spf.rf import rotate_dist


def test_rotate_dist():
    lines = torch.zeros(100, 65)
    lines_2x = torch.zeros(100, 65)
    angle_diffs = torch.zeros(100, 1)
    for i in range(100):
        lines[i][i % 65] = 1.0
        lines_2x[i][(2 * i) % 65] = 1.0
        angle_diffs[i] = i * 2 * torch.pi / 65

    # rotate, then rotate back, should be identity
    assert (
        (lines - rotate_dist(rotate_dist(lines, angle_diffs), -angle_diffs))
        .isclose(torch.tensor([0.0]))
        .all()
    )
    # rotate back, should be equal to unrotated
    assert (
        (lines - rotate_dist(lines_2x, -angle_diffs)).isclose(torch.tensor([0.0])).all()
    )
    # rotate forward, should be equal to rotated
    assert (
        (lines_2x - rotate_dist(lines, angle_diffs)).isclose(torch.tensor([0.0])).all()
    )
