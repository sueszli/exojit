from __future__ import annotations

import numpy as np


def _dot(result, q, k):
    result[0] = np.dot(q, k)


def dot_numpy(n: int):
    return _dot
