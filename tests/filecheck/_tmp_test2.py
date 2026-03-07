from __future__ import annotations

from exo import *
from exo.libs.memories import AVX2


@instr("vec_add_f32x8({dst_data}, {a_data}, {b_data});")
def vec_add_f32x8(dst: [f32][8] @ AVX2, a: [f32][8] @ AVX2, b: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 8):
        dst[i] = a[i] + b[i]


@proc
def test_add(out: f32[8] @ DRAM, x: f32[8] @ DRAM, y: f32[8] @ DRAM):
    avx_out: f32[8] @ AVX2
    avx_x: f32[8] @ AVX2
    avx_y: f32[8] @ AVX2
    for i in seq(0, 8):
        avx_x[i] = x[i]
    for i in seq(0, 8):
        avx_y[i] = y[i]
    vec_add_f32x8(avx_out, avx_x, avx_y)
    for i in seq(0, 8):
        out[i] = avx_out[i]
