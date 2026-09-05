from __future__ import annotations

import numpy as np


def _embedding(out, row):
    np.copyto(out, row)


def embedding_numpy(d: int):
    return _embedding
