from __future__ import annotations

import torch


def relu_torch(n: int):
    @torch.compile
    def _relu(out, x):
        torch.clamp(x, min=0, out=out)

    def wrapper(out, x):
        _relu(torch.from_numpy(out), torch.from_numpy(x))

    return wrapper
