from __future__ import annotations

import numpy as np


def _weighted_sum(out, w, V):
    np.dot(w, V, out=out)


def weighted_sum_numpy(T: int, D: int):
    return _weighted_sum
