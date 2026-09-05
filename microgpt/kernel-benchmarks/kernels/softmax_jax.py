from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def softmax_jax(n: int):
    @jax.jit
    def _softmax(x):
        m = jnp.max(x)
        e = jnp.exp(x - m)
        return e / jnp.sum(e)

    def wrapper(out, x):
        np.copyto(out, np.asarray(_softmax(jnp.asarray(x))))

    return wrapper
