from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.libs.externs import select
from exo.stdlib.scheduling import simplify

from exojit import jit


@proc
def _find_max(N: size, result: f32[1], inp: f32[N]):
    acc: f32 @ DRAM
    acc = inp[0]
    for i in seq(0, N):
        acc = select(acc, inp[i], inp[i], acc)
    result[0] = acc


@cache
def _jit_max(n: int) -> Callable[..., None]:
    p = _find_max.partial_eval(N=n)
    p = simplify(p)
    return jit(p, raw=True)


@proc
def _softmax_core(N: size, out: f32[N], inp: f32[N], mx: f32[1]):
    sum_val: f32 @ DRAM
    t: f32 @ DRAM
    y: f32 @ DRAM
    e5: f32 @ DRAM
    e4: f32 @ DRAM
    e3: f32 @ DRAM
    e2: f32 @ DRAM
    e1: f32 @ DRAM
    s1: f32 @ DRAM
    s2: f32 @ DRAM
    s3: f32 @ DRAM
    s4: f32 @ DRAM
    s5: f32 @ DRAM

    sum_val = 0.0
    for j in seq(0, N):
        t = inp[j] - mx[0]
        y = t * 0.03125
        e5 = y * 0.008333333 + 0.041666667
        e4 = e5 * y + 0.166666667
        e3 = e4 * y + 0.5
        e2 = e3 * y + 1.0
        e1 = e2 * y + 1.0
        s1 = e1 * e1
        s2 = s1 * s1
        s3 = s2 * s2
        s4 = s3 * s3
        s5 = s4 * s4
        out[j] = s5
        sum_val += s5

    for k in seq(0, N):
        out[k] = out[k] / sum_val


@cache
def _jit_core(n: int) -> Callable[..., None]:
    p = _softmax_core.partial_eval(N=n)
    p = simplify(p)
    return jit(p, raw=True)


@cache
def softmax_exo(n: int) -> tuple[Callable[..., None], Callable[..., None]]:
    return _jit_max(n), _jit_core(n)
