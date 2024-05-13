from numba import njit

repo_root = "/Users/miskodzamba/Dropbox/research/gits/spf/"
import sys
import numpy as np
sys.path.append(repo_root)  # go to parent dir
from spf.utils import random_signal_matrix


@njit
def pi_norm(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


@njit
def get_phase_diff(signal_matrix):
    return pi_norm(np.angle(signal_matrix[0]) - np.angle(signal_matrix[1]))


z=random_signal_matrix(n=200)

get_phase_diff(z)
