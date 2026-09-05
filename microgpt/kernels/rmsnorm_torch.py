from __future__ import annotations

import torch


def rmsnorm_torch(n: int):
    @torch.compile
    def _sumsq(x):
        return torch.dot(x, x)

    @torch.compile
    def _scale(out, x, s):
        torch.mul(x, s, out=out)

    def wrapper_sumsq(sumsq, x):
        sumsq[0] = _sumsq(torch.from_numpy(x)).item()

    def wrapper_scale(out, x, scale):
        _scale(torch.from_numpy(out), torch.from_numpy(x), float(scale[0]))

    return wrapper_sumsq, wrapper_scale
