import numpy as np
from exo import *
from exo.stdlib.scheduling import divide_loop, fission, rename, reorder_loops, simplify, unroll_loop

from exojit.main import compile_jit


@proc
def matmul(C: f32[128, 128] @ DRAM, A: f32[128, 128] @ DRAM, B: f32[128, 128] @ DRAM):
    for i in seq(0, 128):
        for j in seq(0, 128):
            C[i, j] = 0.0
            for k in seq(0, 128):
                C[i, j] += A[i, k] * B[k, j]


opt = rename(matmul, "opt")
opt = fission(opt, opt.find("for k in _: _").before(), n_lifts=2)
opt = reorder_loops(opt, "j k")
opt = divide_loop(opt, "j #1", 4, ["jo", "ji"], perfect=True)
opt = unroll_loop(opt, "ji")
opt = simplify(opt)

N = 128
A, B = np.random.randn(N, N).astype(np.float32), np.random.randn(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)
compile_jit(opt)[opt.name()](C, A, B)
assert np.allclose(C, A @ B, atol=0.5)
