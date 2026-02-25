from __future__ import annotations

from exo import DRAM, proc

from xdsl_exo.compiler import compile_one


def test_fixed_matmul():
    @proc
    def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
        for i in seq(0, 16):
            for j in seq(0, 16):
                C[i, j] = 0.0
                for k in seq(0, 16):
                    C[i, j] += A[i, k] * B[k, j]

    compile_one(fixed_matmul)
