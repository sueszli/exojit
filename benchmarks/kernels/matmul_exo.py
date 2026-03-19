from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import divide_loop, fission, reorder_loops, simplify

from exojit.main import jit

PAR_MIN_ELEMENTS = 256


@proc
def _matmul(M: size, K: size, N: size, C: f32[M, N] @ DRAM, A: f32[M, K] @ DRAM, B: f32[K, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            C[i, j] = 0.0
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


@proc
def _matmul_par(M: size, K: size, N: size, C: f32[M, N] @ DRAM, A: f32[M, K] @ DRAM, B: f32[K, N] @ DRAM):
    for i in par(0, M):
        for j in seq(0, N):
            C[i, j] = 0.0
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


@cache
def matmul_exo(m: int, k: int, n: int) -> Callable[..., None]:
    p = (_matmul_par if m >= PAR_MIN_ELEMENTS else _matmul).partial_eval(M=m, K=k, N=n)
    p = fission(p, p.find("for k in _: _").before(), n_lifts=2)
    p = reorder_loops(p, "j k")
    do_k = k > 64
    do_j = n > 64
    if do_k:
        p = divide_loop(p, "k", 64, ["ko", "ki"], perfect=True)
    if do_j:
        p = divide_loop(p, "j #1", 64, ["jo", "ji"], perfect=True)
        if do_k:
            p = reorder_loops(p, "ki jo")
    p = simplify(p)
    return jit(p, raw=True)
