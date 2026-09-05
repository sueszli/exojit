from __future__ import annotations

import numpy as np


def softmax_numpy(n: int):
    tmp = np.empty(n, dtype=np.float32)

    def _softmax(out, x):
        m = x.max()
        np.subtract(x, m, out=tmp)
        np.exp(tmp, out=out)
        s = out.sum()
        out *= 1.0 / s

    return _softmax
