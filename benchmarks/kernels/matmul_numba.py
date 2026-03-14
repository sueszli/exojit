from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True, parallel=True)
def _matmul(C, A, B):
    M, K = A.shape
    _, N = B.shape
    for i in nb.prange(M):
        for k in range(K):
            a_ik = A[i, k]
            for j in range(N):
                C[i, j] += a_ik * B[k, j]


def matmul_numba(M: int, K: int, N: int):
    dummy_A = np.zeros((M, K), dtype=np.float32)
    dummy_B = np.zeros((K, N), dtype=np.float32)
    dummy_C = np.zeros((M, N), dtype=np.float32)
    _matmul(dummy_C, dummy_A, dummy_B)  # warm up
    return _matmul
