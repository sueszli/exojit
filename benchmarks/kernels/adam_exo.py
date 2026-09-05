from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.libs.externs import sqrt
from exo.stdlib.scheduling import simplify

from exojit.main import jit


@proc
def _adam(N: size, param: f32[N] @ DRAM, grad: f32[N] @ DRAM, m: f32[N] @ DRAM, v: f32[N] @ DRAM, b1: f32[1] @ DRAM, b2: f32[1] @ DRAM, eps: f32[1] @ DRAM, lr: f32[1] @ DRAM, beta1_t: f32[1] @ DRAM, beta2_t: f32[1] @ DRAM):
    inv_b1: f32 @ DRAM
    inv_b2: f32 @ DRAM
    inv_beta1_t: f32 @ DRAM
    inv_beta2_t: f32 @ DRAM
    inv_b1 = 1.0 - b1[0]
    inv_b2 = 1.0 - b2[0]
    inv_beta1_t = 1.0 / beta1_t[0]
    inv_beta2_t = 1.0 / beta2_t[0]

    for i in seq(0, N):
        g: f32 @ DRAM
        g = grad[i]

        m_val: f32 @ DRAM
        m_val = b1[0] * m[i] + inv_b1 * g
        m[i] = m_val

        v_val: f32 @ DRAM
        v_val = b2[0] * v[i] + inv_b2 * g * g
        v[i] = v_val

        m_hat: f32 @ DRAM
        m_hat = m_val * inv_beta1_t

        v_hat: f32 @ DRAM
        v_hat = v_val * inv_beta2_t

        param[i] = param[i] - lr[0] * m_hat / (sqrt(v_hat) + eps[0])


@cache
def adam_exo(n: int) -> Callable[..., None]:
    p = _adam.partial_eval(N=n)
    p = simplify(p)
    return jit(p, raw=True)
