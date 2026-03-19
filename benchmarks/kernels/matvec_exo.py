from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import fission, simplify

from exojit.main import jit

_PAR_MIN_ELEMENTS = 1024


@proc
def _matvec(M: size, K: size, y: f32[M] @ DRAM, W: f32[M, K] @ DRAM, x: f32[K] @ DRAM):
    for j in seq(0, M):
        y[j] = 0.0
        for i in seq(0, K):
            y[j] += W[j, i] * x[i]


@proc
def _matvec_par(M: size, K: size, y: f32[M] @ DRAM, W: f32[M, K] @ DRAM, x: f32[K] @ DRAM):
    for j in par(0, M):
        y[j] = 0.0
        for i in seq(0, K):
            y[j] += W[j, i] * x[i]


@cache
def matvec_exo(m: int, k: int) -> Callable[..., None]:
    p = (_matvec_par if m >= _PAR_MIN_ELEMENTS else _matvec).partial_eval(M=m, K=k)
    p = fission(p, p.find("for i in _: _").before(), n_lifts=1)
    p = simplify(p)
    return jit(p)
