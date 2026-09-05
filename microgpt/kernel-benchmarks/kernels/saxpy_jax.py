from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def saxpy_jax(n: int):
    @jax.jit
    def _saxpy(y, x, a):
        return y + a * x

    def wrapper(y, x, a):
        np.copyto(y, np.asarray(_saxpy(jnp.asarray(y), jnp.asarray(x), float(a[0]))))

    return wrapper
