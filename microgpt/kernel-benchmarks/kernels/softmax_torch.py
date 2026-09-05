from __future__ import annotations

import torch


def softmax_torch(n: int):
    @torch.compile
    def _softmax(out, x):
        m = x.max()
        torch.sub(x, m, out=out)
        torch.exp(out, out=out)
        s = out.sum()
        out.div_(s)

    def wrapper(out, x):
        _softmax(torch.from_numpy(out), torch.from_numpy(x))

    return wrapper
