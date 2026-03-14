from __future__ import annotations

import numpy as np


def _adam(param, grad, m, v, b1, b2, eps, lr, beta1_t, beta2_t):
    inv_b1 = 1.0 - b1[0]
    inv_b2 = 1.0 - b2[0]
    m[:] = b1[0] * m + inv_b1 * grad
    v[:] = b2[0] * v + inv_b2 * grad * grad
    m_hat = m / beta1_t[0]
    v_hat = v / beta2_t[0]
    param[:] = param - lr[0] * m_hat / (np.sqrt(v_hat) + eps[0])


def adam_numpy(n: int):
    return _adam
