from __future__ import annotations

import torch


def add_torch(n: int):
    @torch.compile
    def _add(z, x, y):
        torch.add(x, y, out=z)

    def wrapper(z, x, y):
        _add(torch.from_numpy(z), torch.from_numpy(x), torch.from_numpy(y))

    return wrapper
