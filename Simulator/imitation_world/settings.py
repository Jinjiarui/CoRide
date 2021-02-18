"""
Contains some meta-data definition, include rendering
"""
import os.path as osp
import os
import numpy as np

from collections import namedtuple


DIST = namedtuple('Dist', 'mu, sigma, max_range')

TIME_LIMIT = 144
WIDTH, HEIGHT = 5, 5
RED_MAX = WIDTH * HEIGHT * 5
BLUE_MAX = WIDTH * HEIGHT * 3
N_PEAK = 3


# default distribution
mu_blue, mu_red = [3, 3, 5], [6, 6, 15]
sigma_blue, sigma_red = [2, 2, 3], [2, 2, 3]
