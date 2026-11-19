# REQUIRES: metal
# RUN: %python %s

import math
import random

from exo import *

from exojit import jit


@jit(backend="metal")
def matmul(out: f32[32, 48] @ DRAM, A: f32[32, 16] @ DRAM, B: f32[16, 48] @ DRAM):
    for i in par(0, 32):
        for j in par(0, 48):
            out[i, j] = 0.0
            for k in seq(0, 16):
                out[i, j] += A[i, k] * B[k, j]


rng = random.Random(0)
A = [[rng.uniform(-1.0, 1.0) for _ in range(16)] for _ in range(32)]
B = [[rng.uniform(-1.0, 1.0) for _ in range(48)] for _ in range(16)]
out = [[0.0] * 48 for _ in range(32)]
expected = [[sum(A[i][k] * B[k][j] for k in range(16)) for j in range(48)] for i in range(32)]

matmul(out, A, B)
assert all(math.isclose(out[i][j], expected[i][j], rel_tol=1e-5, abs_tol=1e-5) for i in range(32) for j in range(48))
