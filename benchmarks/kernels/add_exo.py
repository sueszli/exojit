from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename, simplify

from xnumpy.main import compile_jit


@proc
def _add(N: size, z: f32[N], x: f32[N], y: f32[N]):
    for i in seq(0, N):
        z[i] = x[i] + y[i]


@cache
def add_exo(n: int) -> Callable[..., None]:
    p = _add.partial_eval(N=n)
    p = simplify(p)
    name = f"_add_{n}"
    return compile_jit(rename(p, name))[name]
