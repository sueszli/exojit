from __future__ import annotations

import numpy as np


def _add(z, x, y):
    np.add(x, y, out=z)


def add_numpy(n: int):
    return _add
