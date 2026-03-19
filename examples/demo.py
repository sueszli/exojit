import math
import random

from exo import *
from exo.stdlib.scheduling import divide_loop, fission, reorder_loops, simplify, unroll_loop

from exojit.main import jit


@proc
def matmul(C: f32[32, 32] @ DRAM, A: f32[32, 32] @ DRAM, B: f32[32, 32] @ DRAM):
    for i in seq(0, 32):
        for j in seq(0, 32):
            C[i, j] = 0.0
            for k in seq(0, 32):
                C[i, j] += A[i, k] * B[k, j]


opt = fission(matmul, matmul.find("for k in _: _").before(), n_lifts=2)
opt = reorder_loops(opt, "j k")
opt = divide_loop(opt, "j #1", 4, ["jo", "ji"], perfect=True)
opt = unroll_loop(opt, "ji")
opt = simplify(opt)

A = [[random.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(32)]
B = [[random.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(32)]
expected = [[sum(A[i][k] * B[k][j] for k in range(32)) for j in range(32)] for i in range(32)]
C = [[0.0] * 32 for _ in range(32)]
jit(opt)(C, A, B)
assert all(math.isclose(C[i][j], expected[i][j], rel_tol=1e-5, abs_tol=1e-5) for i in range(32) for j in range(32))
