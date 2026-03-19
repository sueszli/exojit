from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import simplify

from exojit.main import jit
from exojit.patches_exo import Stack


@proc
def _dot(N: size, result: f32[1], q: f32[N], k: f32[N]):
    acc: f32 @ Stack
    acc = 0.0
    for i in seq(0, N):
        acc += q[i] * k[i]
    result[0] = acc


@cache
def dot_exo(n: int) -> Callable[..., None]:
    p = _dot.partial_eval(N=n)
    p = simplify(p)
    return jit(p, raw=True)
