from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def matvec_jax(M: int, N: int):
    @jax.jit
    def _matvec(W, x):
        return W @ x

    def wrapper(y, W, x):
        np.copyto(y, np.asarray(_matvec(jnp.asarray(W), jnp.asarray(x))))

    return wrapper
