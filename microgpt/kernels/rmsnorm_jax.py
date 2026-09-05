from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def rmsnorm_jax(n: int):
    @jax.jit
    def _sumsq(x):
        return jnp.dot(x, x)

    @jax.jit
    def _scale(x, s):
        return x * s

    def wrapper_sumsq(sumsq, x):
        sumsq[0] = float(_sumsq(jnp.asarray(x)))

    def wrapper_scale(out, x, scale):
        np.copyto(out, np.asarray(_scale(jnp.asarray(x), float(scale[0]))))

    return wrapper_sumsq, wrapper_scale
