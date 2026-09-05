from __future__ import annotations

import torch


def dot_torch(n: int):
    @torch.compile
    def _dot(q, k):
        return torch.dot(q, k)

    def wrapper(result, q, k):
        result[0] = _dot(torch.from_numpy(q), torch.from_numpy(k)).item()

    return wrapper
