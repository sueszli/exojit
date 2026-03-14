from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename, simplify

from xnumpy.main import compile_jit
from xnumpy.patches_exo import Stack


@proc
def _dot(N: size, result: f32[1], q: f32[N], k: f32[N]):
    acc0: f32 @ Stack
    acc1: f32 @ Stack
    acc2: f32 @ Stack
    acc3: f32 @ Stack
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    for i in seq(0, N / 4):
        acc0 += q[i * 4] * k[i * 4]
        acc1 += q[i * 4 + 1] * k[i * 4 + 1]
        acc2 += q[i * 4 + 2] * k[i * 4 + 2]
        acc3 += q[i * 4 + 3] * k[i * 4 + 3]
    result[0] = acc0 + acc1 + acc2 + acc3


@cache
def dot_exo(n: int) -> Callable[..., None]:
    assert n % 4 == 0
    p = _dot.partial_eval(N=n)
    p = simplify(p)
    name = f"_dot_{n}"
    return compile_jit(rename(p, name))[name]
