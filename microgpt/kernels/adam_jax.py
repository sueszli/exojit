from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def adam_jax(n: int):
    @jax.jit
    def _adam(param, grad, m, v, b1, b2, eps, lr, beta1_t, beta2_t):
        inv_b1 = 1.0 - b1
        inv_b2 = 1.0 - b2
        m_new = b1 * m + inv_b1 * grad
        v_new = b2 * v + inv_b2 * grad * grad
        m_hat = m_new / beta1_t
        v_hat = v_new / beta2_t
        param_new = param - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        return param_new, m_new, v_new

    def wrapper(param, grad, m, v, b1, b2, eps, lr, beta1_t, beta2_t):
        p_new, m_new, v_new = _adam(
            jnp.asarray(param),
            jnp.asarray(grad),
            jnp.asarray(m),
            jnp.asarray(v),
            float(b1[0]),
            float(b2[0]),
            float(eps[0]),
            float(lr[0]),
            float(beta1_t[0]),
            float(beta2_t[0]),
        )
        np.copyto(param, np.asarray(p_new))
        np.copyto(m, np.asarray(m_new))
        np.copyto(v, np.asarray(v_new))

    return wrapper
