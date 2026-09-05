from __future__ import annotations

import jax
import jax.numpy as jnp


def cross_entropy_jax(n: int):
    @jax.jit
    def _ce_max(x):
        return jnp.max(x)

    @jax.jit
    def _ce_sum_exp(x, mx):
        return jnp.sum(jnp.exp(x - mx))

    def wrapper_max(mx, x):
        mx[0] = float(_ce_max(jnp.asarray(x)))

    def wrapper_sum_exp(sum_exp, x, mx):
        sum_exp[0] = float(_ce_sum_exp(jnp.asarray(x), float(mx[0])))

    return wrapper_max, wrapper_sum_exp
