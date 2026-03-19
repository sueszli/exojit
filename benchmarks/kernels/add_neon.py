from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *

from exojit.main import jit
from exojit.patches_exo import NEON

PAR_MIN_ELEMENTS = 524288


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_add_f32x4({dst_data}, {a_data}, {b_data});")
def neon_add_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@proc
def _add_neon(N: size, z: f32[N] @ DRAM, x: f32[N] @ DRAM, y: f32[N] @ DRAM):
    for i in seq(0, N / 16):
        x0: f32[4] @ NEON
        x1: f32[4] @ NEON
        x2: f32[4] @ NEON
        x3: f32[4] @ NEON
        y0: f32[4] @ NEON
        y1: f32[4] @ NEON
        y2: f32[4] @ NEON
        y3: f32[4] @ NEON
        neon_loadu_f32x4(x0, x[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(x1, x[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(x2, x[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(x3, x[16 * i + 12 : 16 * i + 16])
        neon_loadu_f32x4(y0, y[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(y1, y[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(y2, y[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(y3, y[16 * i + 12 : 16 * i + 16])
        z0: f32[4] @ NEON
        z1: f32[4] @ NEON
        z2: f32[4] @ NEON
        z3: f32[4] @ NEON
        neon_add_f32x4(z0, x0, y0)
        neon_add_f32x4(z1, x1, y1)
        neon_add_f32x4(z2, x2, y2)
        neon_add_f32x4(z3, x3, y3)
        neon_storeu_f32x4(z[16 * i + 0 : 16 * i + 4], z0)
        neon_storeu_f32x4(z[16 * i + 4 : 16 * i + 8], z1)
        neon_storeu_f32x4(z[16 * i + 8 : 16 * i + 12], z2)
        neon_storeu_f32x4(z[16 * i + 12 : 16 * i + 16], z3)


@proc
def _add_neon_par(N: size, z: f32[N] @ DRAM, x: f32[N] @ DRAM, y: f32[N] @ DRAM):
    for i in par(0, N / 16):
        x0: f32[4] @ NEON
        x1: f32[4] @ NEON
        x2: f32[4] @ NEON
        x3: f32[4] @ NEON
        y0: f32[4] @ NEON
        y1: f32[4] @ NEON
        y2: f32[4] @ NEON
        y3: f32[4] @ NEON
        neon_loadu_f32x4(x0, x[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(x1, x[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(x2, x[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(x3, x[16 * i + 12 : 16 * i + 16])
        neon_loadu_f32x4(y0, y[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(y1, y[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(y2, y[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(y3, y[16 * i + 12 : 16 * i + 16])
        z0: f32[4] @ NEON
        z1: f32[4] @ NEON
        z2: f32[4] @ NEON
        z3: f32[4] @ NEON
        neon_add_f32x4(z0, x0, y0)
        neon_add_f32x4(z1, x1, y1)
        neon_add_f32x4(z2, x2, y2)
        neon_add_f32x4(z3, x3, y3)
        neon_storeu_f32x4(z[16 * i + 0 : 16 * i + 4], z0)
        neon_storeu_f32x4(z[16 * i + 4 : 16 * i + 8], z1)
        neon_storeu_f32x4(z[16 * i + 8 : 16 * i + 12], z2)
        neon_storeu_f32x4(z[16 * i + 12 : 16 * i + 16], z3)


@cache
def add_neon(n: int) -> Callable[..., None]:
    assert n % 16 == 0
    return jit((_add_neon_par if n >= PAR_MIN_ELEMENTS else _add_neon).partial_eval(N=n), raw=True)
