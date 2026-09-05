from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def add_jax(n: int):
    @jax.jit
    def _add(x, y):
        return x + y

    def wrapper(z, x, y):
        np.copyto(z, np.asarray(_add(jnp.asarray(x), jnp.asarray(y))))

    return wrapper
