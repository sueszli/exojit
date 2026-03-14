from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _softmax(out, x):
    n = x.shape[0]
    # find max
    m = x[0]
    for i in range(1, n):
        if x[i] > m:
            m = x[i]
    # exp and sum
    s = np.float32(0.0)
    for i in range(n):
        v = math.exp(x[i] - m)
        out[i] = v
        s += v
    # normalize
    inv_s = np.float32(1.0) / s
    for i in range(n):
        out[i] *= inv_s


def softmax_numba(n: int):
    """Return a compiled softmax callable, warming up the JIT on first call."""
    dummy_x = np.zeros(n, dtype=np.float32)
    dummy_out = np.zeros(n, dtype=np.float32)
    _softmax(dummy_out, dummy_x)
    return _softmax
