from __future__ import annotations

import torch


def cross_entropy_torch(n: int):
    @torch.compile
    def _ce_max(x):
        return x.max()

    @torch.compile
    def _ce_sum_exp(x, mx):
        return torch.sum(torch.exp(x - mx))

    def wrapper_max(mx, x):
        mx[0] = _ce_max(torch.from_numpy(x)).item()

    def wrapper_sum_exp(sum_exp, x, mx):
        sum_exp[0] = _ce_sum_exp(torch.from_numpy(x), float(mx[0])).item()

    return wrapper_max, wrapper_sum_exp
