from __future__ import annotations

import pytest
from exo import DRAM, proc

from xdsl_exo.main import compile_procs


@pytest.mark.xfail(reason="upstream ConvertMemRefToPtr does not support dynamic strides")
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

    compile_procs(dynamic_matmul)
