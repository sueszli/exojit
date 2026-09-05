from __future__ import annotations

import torch


def matmul_torch(M: int, K: int, N: int):
    @torch.compile
    def _matmul(C, A, B):
        torch.matmul(A, B, out=C)

    def wrapper(C, A, B):
        _matmul(torch.from_numpy(C), torch.from_numpy(A), torch.from_numpy(B))

    return wrapper
