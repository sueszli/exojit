from __future__ import annotations

import numpy as np


def _relu(out, x):
    np.maximum(0, x, out=out)


def relu_numpy(n: int):
    return _relu
