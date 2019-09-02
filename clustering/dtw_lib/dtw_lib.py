from __future__ import absolute_import, division
import numbers
import numpy as np
from collections import defaultdict
import math
import time
from pprint import pprint
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt


def __difference(a, b):
    return abs(a - b)


def __norm(p):
    return lambda a, b: np.linalg.norm(a - b, p)


def __prep_inputs(x, y, dist):
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')

    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')

    if dist is None:
        if x.ndim == 1:
            dist = __difference
        else:
            dist = __norm(1)
    elif isinstance(dist, numbers.Number):
        dist = __norm(dist)

    return x, y, dist


def dtw(x, y, relax=0, dist=None):
    x, y, dist = __prep_inputs(x, y, dist)
    return __dtw(x, y, relax, None, dist)


def fastdtw(x, y, relax=0, radius=1, dist=None):
    x, y, dist = __prep_inputs(x, y, dist)
    return __fastdtw(x, y, relax, radius, dist)


def __fastdtw(x, y, relax, radius, dist):

    min_time_size = radius + 2

    assert relax <= min_time_size, ("Make sure if relax <= radius + 2")

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, dist=dist)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)

    distance, path, _ = __fastdtw(
        x_shrinked, y_shrinked, relax=0, radius=radius, dist=dist)

    window = __expand_window(path, len(x), len(y), radius)

    return __dtw(x, y, relax, window, dist)


def __dtw(x, y, relax, window, dist):

    win = window

    len_x, len_y = len(x), len(y)

    if window is None:
        window = ((i + 1, j + 1)
                  for i, j in [(i, j) for i in range(len_x) for j in range(len_y)])
    else:
        window = ((i + 1, j + 1) for i, j in win)

    D = defaultdict(lambda: (float('inf'),))

    if(relax == 0):
        D[0, 0] = (0, 0, 0)
    else:
        for i in range(relax + 1):
            D[i, 0] = (0, 0, 0)
            D[0, i] = (0, 0, 0)

    for i, j in window:
        dt = dist(x[i - 1], y[j - 1])
        D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j), (D[i, j - 1][0] + dt, i, j - 1),
                      (D[i - 1, j - 1][0] + dt, i - 1, j - 1), key=lambda a: a[0])
    path = []

    if (relax == 0):
        i, j = len_x, len_y
        final_dist = D[len_x, len_y][0]

    else:

        row_r = [D[len(x), len(y) - i][0] for i in range(relax + 1)]
        col_r = [D[len(x) - i, len(y)][0] for i in range(relax + 1)]
        if min(row_r) <= min(col_r):
            i, j = len(x), len(y) - row_r.index(min(row_r))
        else:
            i, j = len(x) - col_r.index(min(col_r)), len(y)

        final_dist = min(min(row_r), min(col_r))

    while not (i == j == 0):
        path.append((i - 1, j - 1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()

    return (final_dist, path, D)


def __reduce_by_half(x):
    return [(x[i] + x[1 + i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius + 1)
                     for b in range(-radius, radius + 1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window
