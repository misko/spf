import numpy as np

from spf.rf import pi_norm


def test_angles():
    rx_theta_in_pis = np.array([-0.25, 1.25])
    for (x, y), expected in [
        ((0, -900), [-np.pi * 0.75, -0.25 * np.pi]),
        ((900, 0), [np.pi * 0.75, -0.75 * np.pi]),
        ((0, 900), [np.pi * 0.25, 0.75 * np.pi]),
        ((-900, 0), [-0.25 * np.pi, 0.25 * np.pi]),
    ]:
        rx_to_tx_theta = np.arctan2(x, y)
        estimated = np.array(
            [
                pi_norm((rx_to_tx_theta - rx_theta_in_pis * np.pi)),
                pi_norm((rx_to_tx_theta - rx_theta_in_pis * np.pi)),
            ]
        )
        assert np.isclose(estimated, expected).all()
