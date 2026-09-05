from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def weighted_sum_jax(T: int, D: int):
    @jax.jit
    def _weighted_sum(w, V):
        return w @ V

    def wrapper(out, w, V):
        np.copyto(out, np.asarray(_weighted_sum(jnp.asarray(w), jnp.asarray(V))))

    return wrapper
