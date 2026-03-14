from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename, simplify

from xnumpy.main import compile_jit
from xnumpy.patches_exo import Stack


@proc
def _rmsnorm_sumsq(N: size, result: f32[1], inp: f32[N]):
    acc: f32 @ Stack
    acc = 0.0
    for i in seq(0, N):
        acc += inp[i] * inp[i]
    result[0] = acc


@proc
def _rmsnorm_scale(N: size, out: f32[N], inp: f32[N], scale: f32[1]):
    for i in seq(0, N):
        out[i] = inp[i] * scale[0]


@cache
def _jit_sumsq(n: int) -> Callable[..., None]:
    p = _rmsnorm_sumsq.partial_eval(N=n)
    p = simplify(p)
    name = f"_rmsnorm_sumsq_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_scale(n: int) -> Callable[..., None]:
    p = _rmsnorm_scale.partial_eval(N=n)
    p = simplify(p)
    name = f"_rmsnorm_scale_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def rmsnorm_exo(n: int) -> tuple[Callable[..., None], Callable[..., None]]:
    return _jit_sumsq(n), _jit_scale(n)
