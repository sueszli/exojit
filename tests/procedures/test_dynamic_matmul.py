from __future__ import annotations

from exo import DRAM, proc

from xdsl_exo.compiler import compile_one


def test_dynamic_matmul():
    @proc
    def dynamic_matmul(
        M: size,
        N: size,
        K: size,
        C: f32[M, N] @ DRAM,
        A: f32[M, K] @ DRAM,
        B: f32[K, N] @ DRAM,
    ):
        for i in seq(0, M):
            for j in seq(0, N):
                for k in seq(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    compile_one(dynamic_matmul)
