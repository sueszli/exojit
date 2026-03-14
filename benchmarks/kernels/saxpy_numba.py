from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _saxpy(y, x, a):
    alpha = a[0]
    for i in range(y.shape[0]):
        y[i] += alpha * x[i]


def saxpy_numba(n: int):
    """Return a compiled saxpy callable, warming up the JIT on first call."""
    dummy_y = np.zeros(n, dtype=np.float32)
    dummy_x = np.zeros(n, dtype=np.float32)
    dummy_a = np.array([1.0], dtype=np.float32)
    _saxpy(dummy_y, dummy_x, dummy_a)
    return _saxpy
