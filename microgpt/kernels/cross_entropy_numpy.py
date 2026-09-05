from __future__ import annotations

import numpy as np


def _ce_max(mx, x):
    mx[0] = x.max()


def _ce_sum_exp(sum_exp, x, mx):
    sum_exp[0] = np.sum(np.exp(x - mx[0]))


def cross_entropy_numpy(n: int):
    return _ce_max, _ce_sum_exp
