from __future__ import annotations

import torch


def embedding_torch(d: int):
    @torch.compile
    def _embedding(out, row):
        out.copy_(row)

    def wrapper(out, row):
        _embedding(torch.from_numpy(out), torch.from_numpy(row))

    return wrapper
