from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def embedding_jax(d: int):
    @jax.jit
    def _embedding(row):
        return row.copy()

    def wrapper(out, row):
        np.copyto(out, np.asarray(_embedding(jnp.asarray(row))))

    return wrapper
