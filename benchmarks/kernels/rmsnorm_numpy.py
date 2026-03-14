from __future__ import annotations

import numpy as np


def _rmsnorm_sumsq(sumsq, x):
    sumsq[0] = np.dot(x, x)


def _rmsnorm_scale(out, x, scale):
    np.multiply(x, scale[0], out=out)


def rmsnorm_numpy(n: int):
    return _rmsnorm_sumsq, _rmsnorm_scale
