from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from kernels.softmax_neon import neon_add_acc_f32x4, neon_broadcast_f32x4, neon_loadu_f32x4, neon_mul_f32x4, neon_square_f32x4, neon_storeu_f32x4

from exojit.main import jit
from exojit.patches_exo import NEON

_PAR_MIN_ELEMENTS = 524288


@proc
def _rmsnorm_sumsq_neon(N: size, result: f32[1] @ DRAM, inp: f32[N] @ DRAM):
    zero_buf: f32[4] @ DRAM
    zero_buf[0] = 0.0
    zero_buf[1] = 0.0
    zero_buf[2] = 0.0
    zero_buf[3] = 0.0

    acc0: f32[4] @ NEON
    acc1: f32[4] @ NEON
    acc2: f32[4] @ NEON
    acc3: f32[4] @ NEON
    neon_loadu_f32x4(acc0, zero_buf[0:4])
    neon_loadu_f32x4(acc1, zero_buf[0:4])
    neon_loadu_f32x4(acc2, zero_buf[0:4])
    neon_loadu_f32x4(acc3, zero_buf[0:4])

    for i in seq(0, N / 16):
        v0: f32[4] @ NEON
        v1: f32[4] @ NEON
        v2: f32[4] @ NEON
        v3: f32[4] @ NEON
        neon_loadu_f32x4(v0, inp[16 * i : 16 * i + 4])
        neon_loadu_f32x4(v1, inp[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(v2, inp[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(v3, inp[16 * i + 12 : 16 * i + 16])
        sq0: f32[4] @ NEON
        sq1: f32[4] @ NEON
        sq2: f32[4] @ NEON
        sq3: f32[4] @ NEON
        neon_square_f32x4(sq0, v0)
        neon_square_f32x4(sq1, v1)
        neon_square_f32x4(sq2, v2)
        neon_square_f32x4(sq3, v3)
        neon_add_acc_f32x4(acc0, sq0)
        neon_add_acc_f32x4(acc1, sq1)
        neon_add_acc_f32x4(acc2, sq2)
        neon_add_acc_f32x4(acc3, sq3)

    neon_add_acc_f32x4(acc0, acc1)
    neon_add_acc_f32x4(acc2, acc3)
    neon_add_acc_f32x4(acc0, acc2)

    sum_buf: f32[4] @ DRAM
    neon_storeu_f32x4(sum_buf[0:4], acc0)
    total: f32 @ DRAM
    total = sum_buf[0]
    total += sum_buf[1]
    total += sum_buf[2]
    total += sum_buf[3]
    result[0] = total


@proc
def _rmsnorm_scale_neon(N: size, out: f32[N] @ DRAM, inp: f32[N] @ DRAM, scale: f32[1] @ DRAM):
    scale_v: f32[4] @ NEON
    neon_broadcast_f32x4(scale_v, scale[0:1])

    for i in seq(0, N / 16):
        v0: f32[4] @ NEON
        v1: f32[4] @ NEON
        v2: f32[4] @ NEON
        v3: f32[4] @ NEON
        neon_loadu_f32x4(v0, inp[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(v1, inp[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(v2, inp[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(v3, inp[16 * i + 12 : 16 * i + 16])
        r0: f32[4] @ NEON
        r1: f32[4] @ NEON
        r2: f32[4] @ NEON
        r3: f32[4] @ NEON
        neon_mul_f32x4(r0, v0, scale_v)
        neon_mul_f32x4(r1, v1, scale_v)
        neon_mul_f32x4(r2, v2, scale_v)
        neon_mul_f32x4(r3, v3, scale_v)
        neon_storeu_f32x4(out[16 * i + 0 : 16 * i + 4], r0)
        neon_storeu_f32x4(out[16 * i + 4 : 16 * i + 8], r1)
        neon_storeu_f32x4(out[16 * i + 8 : 16 * i + 12], r2)
        neon_storeu_f32x4(out[16 * i + 12 : 16 * i + 16], r3)


@proc
def _rmsnorm_scale_neon_par(N: size, out: f32[N] @ DRAM, inp: f32[N] @ DRAM, scale: f32[1] @ DRAM):
    scale_v: f32[4] @ NEON
    neon_broadcast_f32x4(scale_v, scale[0:1])

    for i in par(0, N / 16):
        v0: f32[4] @ NEON
        v1: f32[4] @ NEON
        v2: f32[4] @ NEON
        v3: f32[4] @ NEON
        neon_loadu_f32x4(v0, inp[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(v1, inp[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(v2, inp[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(v3, inp[16 * i + 12 : 16 * i + 16])
        r0: f32[4] @ NEON
        r1: f32[4] @ NEON
        r2: f32[4] @ NEON
        r3: f32[4] @ NEON
        neon_mul_f32x4(r0, v0, scale_v)
        neon_mul_f32x4(r1, v1, scale_v)
        neon_mul_f32x4(r2, v2, scale_v)
        neon_mul_f32x4(r3, v3, scale_v)
        neon_storeu_f32x4(out[16 * i + 0 : 16 * i + 4], r0)
        neon_storeu_f32x4(out[16 * i + 4 : 16 * i + 8], r1)
        neon_storeu_f32x4(out[16 * i + 8 : 16 * i + 12], r2)
        neon_storeu_f32x4(out[16 * i + 12 : 16 * i + 16], r3)


@cache
def _jit_sumsq_neon(n: int) -> Callable[..., None]:
    assert n % 16 == 0
    return jit(_rmsnorm_sumsq_neon.partial_eval(N=n))


@cache
def _jit_scale_neon(n: int) -> Callable[..., None]:
    assert n % 16 == 0
    return jit((_rmsnorm_scale_neon_par if n >= _PAR_MIN_ELEMENTS else _rmsnorm_scale_neon).partial_eval(N=n))


@cache
def rmsnorm_neon(n: int) -> tuple[Callable[..., None], Callable[..., None]]:
    return _jit_sumsq_neon(n), _jit_scale_neon(n)
