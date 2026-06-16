from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *

from exojit.main import jit
from exo.platforms.neon import Neon

PAR_MIN_ELEMENTS = 1024


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ Neon, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ Neon, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@instr("neon_add_acc_f32x4({acc_data}, {src_data});")
def neon_add_acc_f32x4(acc: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] += src[i]


# fmt: off
@proc
def _mv_neon(M: size, K: size, y: f32[M] @ DRAM, WT: f32[K, M] @ DRAM, x: f32[K] @ DRAM):
    for j in seq(0, M):
        y[j] = 0.0

    for jo in seq(0, M / 4):
        acc0: f32[4] @ Neon
        acc1: f32[4] @ Neon
        acc2: f32[4] @ Neon
        acc3: f32[4] @ Neon
        neon_loadu_f32x4(acc0, y[4 * jo : 4 * jo + 4])
        neon_loadu_f32x4(acc1, y[4 * jo : 4 * jo + 4])
        neon_loadu_f32x4(acc2, y[4 * jo : 4 * jo + 4])
        neon_loadu_f32x4(acc3, y[4 * jo : 4 * jo + 4])
        for io in seq(0, K / 4):
            w0: f32[4] @ Neon
            w1: f32[4] @ Neon
            w2: f32[4] @ Neon
            w3: f32[4] @ Neon
            x0: f32[4] @ Neon
            x1: f32[4] @ Neon
            x2: f32[4] @ Neon
            x3: f32[4] @ Neon
            neon_loadu_f32x4(w0, WT[4 * io + 0, 4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(w1, WT[4 * io + 1, 4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(w2, WT[4 * io + 2, 4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(w3, WT[4 * io + 3, 4 * jo : 4 * jo + 4])
            neon_broadcast_f32x4(x0, x[4 * io + 0 : 4 * io + 1])
            neon_broadcast_f32x4(x1, x[4 * io + 1 : 4 * io + 2])
            neon_broadcast_f32x4(x2, x[4 * io + 2 : 4 * io + 3])
            neon_broadcast_f32x4(x3, x[4 * io + 3 : 4 * io + 4])
            neon_fmadd_f32x4(acc0, w0, x0)
            neon_fmadd_f32x4(acc1, w1, x1)
            neon_fmadd_f32x4(acc2, w2, x2)
            neon_fmadd_f32x4(acc3, w3, x3)
        neon_add_acc_f32x4(acc0, acc1)
        neon_add_acc_f32x4(acc0, acc2)
        neon_add_acc_f32x4(acc0, acc3)
        neon_storeu_f32x4(y[4 * jo : 4 * jo + 4], acc0)


@proc
def _mv_neon_par(M: size, K: size, y: f32[M] @ DRAM, WT: f32[K, M] @ DRAM, x: f32[K] @ DRAM):
    for j in seq(0, M):
        y[j] = 0.0

    for jo in par(0, M / 4):
        acc0: f32[4] @ Neon
        acc1: f32[4] @ Neon
        acc2: f32[4] @ Neon
        acc3: f32[4] @ Neon
        neon_loadu_f32x4(acc0, y[4 * jo : 4 * jo + 4])
        neon_loadu_f32x4(acc1, y[4 * jo : 4 * jo + 4])
        neon_loadu_f32x4(acc2, y[4 * jo : 4 * jo + 4])
        neon_loadu_f32x4(acc3, y[4 * jo : 4 * jo + 4])
        for io in seq(0, K / 4):
            w0: f32[4] @ Neon
            w1: f32[4] @ Neon
            w2: f32[4] @ Neon
            w3: f32[4] @ Neon
            x0: f32[4] @ Neon
            x1: f32[4] @ Neon
            x2: f32[4] @ Neon
            x3: f32[4] @ Neon
            neon_loadu_f32x4(w0, WT[4 * io + 0, 4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(w1, WT[4 * io + 1, 4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(w2, WT[4 * io + 2, 4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(w3, WT[4 * io + 3, 4 * jo : 4 * jo + 4])
            neon_broadcast_f32x4(x0, x[4 * io + 0 : 4 * io + 1])
            neon_broadcast_f32x4(x1, x[4 * io + 1 : 4 * io + 2])
            neon_broadcast_f32x4(x2, x[4 * io + 2 : 4 * io + 3])
            neon_broadcast_f32x4(x3, x[4 * io + 3 : 4 * io + 4])
            neon_fmadd_f32x4(acc0, w0, x0)
            neon_fmadd_f32x4(acc1, w1, x1)
            neon_fmadd_f32x4(acc2, w2, x2)
            neon_fmadd_f32x4(acc3, w3, x3)
        neon_add_acc_f32x4(acc0, acc1)
        neon_add_acc_f32x4(acc0, acc2)
        neon_add_acc_f32x4(acc0, acc3)
        neon_storeu_f32x4(y[4 * jo : 4 * jo + 4], acc0)


@cache
def matvec_neon(m: int, k: int) -> Callable[..., None]:
    assert m % 4 == 0
    assert k % 4 == 0
    return jit((_mv_neon_par if m >= PAR_MIN_ELEMENTS else _mv_neon).partial_eval(M=m, K=k), raw=True)
