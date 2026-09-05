from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def relu_jax(n: int):
    @jax.jit
    def _relu(x):
        return jnp.maximum(0, x)

    def wrapper(out, x):
        np.copyto(out, np.asarray(_relu(jnp.asarray(x))))

    return wrapper
