from __future__ import annotations

import torch


def weighted_sum_torch(T: int, D: int):
    @torch.compile
    def _weighted_sum(out, w, V):
        torch.matmul(w, V, out=out)

    def wrapper(out, w, V):
        _weighted_sum(torch.from_numpy(out), torch.from_numpy(w), torch.from_numpy(V))

    return wrapper
