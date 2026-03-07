from __future__ import annotations

from exo import *
from exo.libs.memories import AVX2
from exo.platforms.x86 import (
    mm256_broadcast_ss,
    mm256_fmadd_ps,
    mm256_loadu_ps,
    mm256_storeu_ps,
)


# test mm256_fmadd_ps: load acc from dst, load a, load b, fma, store to dst
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


# test mm256_broadcast_ss: load scalar f32, broadcast to vector<8xf32>, store
@proc
def test_broadcast(out: f32[8] @ DRAM, val: f32[1] @ DRAM):
    avx_out: f32[8] @ AVX2
    mm256_broadcast_ss(avx_out, val)
    mm256_storeu_ps(out, avx_out)
