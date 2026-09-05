from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def matmul_jax(M: int, K: int, N: int):
    @jax.jit
    def _matmul(A, B):
        return A @ B

    def wrapper(C, A, B):
        np.copyto(C, np.asarray(_matmul(jnp.asarray(A), jnp.asarray(B))))

    return wrapper
