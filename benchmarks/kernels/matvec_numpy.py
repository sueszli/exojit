from __future__ import annotations

import numpy as np


def _matvec(y, W, x):
    np.dot(W, x, out=y)


def matvec_numpy(M: int, N: int):
    return _matvec
