from __future__ import annotations

import jax
import jax.numpy as jnp


def dot_jax(n: int):
    @jax.jit
    def _dot(q, k):
        return jnp.dot(q, k)

    def wrapper(result, q, k):
        result[0] = float(_dot(jnp.asarray(q), jnp.asarray(k)))

    return wrapper
