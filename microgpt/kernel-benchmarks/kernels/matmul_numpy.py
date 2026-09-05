from __future__ import annotations

import numpy as np


def _matmul(C, A, B):
    np.matmul(A, B, out=C)


def matmul_numpy(M: int, K: int, N: int):
    return _matmul
