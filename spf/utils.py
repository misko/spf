import numpy as np


class dotdict(dict):
    __getattr__ = dict.get

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


def is_pi():
    try:
        import RPi.GPIO as GPIO  # noqa

        return True
    except (RuntimeError, ImportError):
        return False


def random_signal_matrix(n):
    return np.random.uniform(-1, 1, (n,)) + 1.0j * np.random.uniform(-1, 1, (n,))
