from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.libs.externs import select
from exo.stdlib.scheduling import simplify

from exojit.main import jit


@proc
def _relu(N: size, out: f32[N], inp: f32[N]):
    for i in seq(0, N):
        out[i] = select(0.0, inp[i], inp[i], 0.0)


@cache
def relu_exo(n: int) -> Callable[..., None]:
    p = _relu.partial_eval(N=n)
    p = simplify(p)
    return jit(p, raw=True)
