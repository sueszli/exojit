from __future__ import annotations

import torch


def matvec_torch(M: int, N: int):
    @torch.compile
    def _matvec(y, W, x):
        torch.mv(W, x, out=y)

    def wrapper(y, W, x):
        _matvec(torch.from_numpy(y), torch.from_numpy(W), torch.from_numpy(x))

    return wrapper
