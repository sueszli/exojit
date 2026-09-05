from __future__ import annotations

import numpy as np


def _saxpy(y, x, a):
    np.add(a[0] * x, y, out=y)


def saxpy_numpy(n: int):
    return _saxpy
