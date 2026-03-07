# RUN: uv run xdsl-exo -o - %s | filecheck %s
# ConvertVecIntrinsic: vec_*/mm256_* calls -> LLVM/vector ops through full pipeline

from __future__ import annotations

from exo import *
from exo.libs.memories import AVX2
from exo.platforms.x86 import (
    mm256_broadcast_ss,
    mm256_fmadd_ps,
    mm256_loadu_ps,
    mm256_storeu_ps,
)


# custom vec_* intrinsics (not provided by exo.platforms.x86)

@instr("vec_add_f32x8({dst_data}, {a_data}, {b_data});")
def vec_add_f32x8(dst: [f32][8] @ AVX2, a: [f32][8] @ AVX2, b: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 8):
        dst[i] = a[i] + b[i]


@instr("vec_neg_f32x8({dst_data}, {src_data});")
def vec_neg_f32x8(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = -src[i]


@instr("vec_zero_f32x8({dst_data});")
def vec_zero_f32x8(dst: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        dst[i] = 0.0


# vec_add_f32x8 -> llvm.fadd on vector<8xf32>
# CHECK-LABEL: @test_add
# CHECK:       llvm.fadd {{.*}} : vector<8xf32>
@proc
def test_add(out: f32[8] @ DRAM, x: f32[8] @ DRAM, y: f32[8] @ DRAM):
    avx_out: f32[8] @ AVX2
    avx_x: f32[8] @ AVX2
    avx_y: f32[8] @ AVX2
    mm256_loadu_ps(avx_x, x)
    mm256_loadu_ps(avx_y, y)
    vec_add_f32x8(avx_out, avx_x, avx_y)
    mm256_storeu_ps(out, avx_out)


# mm256_broadcast_ss -> load f32, vector.broadcast to vector<8xf32>
# CHECK-LABEL: @test_broadcast
# CHECK:       "llvm.load"{{.*}} -> f32
# CHECK:       vector.broadcast {{.*}} : f32 to vector<8xf32>
@proc
def test_broadcast(out: f32[8] @ DRAM, val: f32[1] @ DRAM):
    avx_out: f32[8] @ AVX2
    mm256_broadcast_ss(avx_out, val)
    mm256_storeu_ps(out, avx_out)


# mm256_fmadd_ps -> 3x load + vector.fma on vector<8xf32>
# CHECK-LABEL: @test_fmadd
# CHECK:       vector.fma {{.*}} : vector<8xf32>
@proc
def test_fmadd(C: f32[8] @ DRAM, A: f32[8] @ DRAM, B: f32[8] @ DRAM):
    avx_c: f32[8] @ AVX2
    avx_a: f32[8] @ AVX2
    avx_b: f32[8] @ AVX2
    mm256_loadu_ps(avx_a, A)
    mm256_loadu_ps(avx_b, B)
    mm256_loadu_ps(avx_c, C)
    mm256_fmadd_ps(avx_c, avx_a, avx_b)
    mm256_storeu_ps(C, avx_c)


# vec_neg_f32x8 -> llvm.fneg on vector<8xf32>
# CHECK-LABEL: @test_neg
# CHECK:       llvm.fneg {{.*}} : vector<8xf32>
@proc
def test_neg(out: f32[8] @ DRAM, x: f32[8] @ DRAM):
    avx_out: f32[8] @ AVX2
    avx_x: f32[8] @ AVX2
    mm256_loadu_ps(avx_x, x)
    vec_neg_f32x8(avx_out, avx_x)
    mm256_storeu_ps(out, avx_out)


# vec_zero_f32x8 -> dense<0.0> constant
# CHECK-LABEL: @test_zero
# CHECK:       llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>)
@proc
def test_zero(out: f32[8] @ DRAM):
    avx_out: f32[8] @ AVX2
    vec_zero_f32x8(avx_out)
    mm256_storeu_ps(out, avx_out)
