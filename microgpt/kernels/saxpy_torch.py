from __future__ import annotations

import torch


def saxpy_torch(n: int):
    @torch.compile
    def _saxpy(y, x, a):
        y.add_(x, alpha=a)

    def wrapper(y, x, a):
        _saxpy(torch.from_numpy(y), torch.from_numpy(x), float(a[0]))

    return wrapper
