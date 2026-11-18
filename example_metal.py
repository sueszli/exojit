# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit[metal] @ git+https://github.com/sueszli/exojit@feat/metal-backend",
# ]
# ///

import math
import random

from exo import *
from exo.API import Procedure
from exo.stdlib.scheduling import parallelize_loop

from exojit import jit


def optimize_matmul(p: Procedure) -> Procedure:
    p = parallelize_loop(p, "i")
    return parallelize_loop(p, "j")


@jit(optimize=optimize_matmul, backend="metal")
def matmul(out: f32[32, 48] @ DRAM, A: f32[32, 16] @ DRAM, B: f32[16, 48] @ DRAM):
    for i in seq(0, 32):
        for j in seq(0, 48):
            out[i, j] = 0.0
            for k in seq(0, 16):
                out[i, j] += A[i, k] * B[k, j]


if __name__ == "__main__":
    A = [[random.uniform(-1.0, 1.0) for _ in range(16)] for _ in range(32)]
    B = [[random.uniform(-1.0, 1.0) for _ in range(48)] for _ in range(16)]
    out = [[0.0] * 48 for _ in range(32)]
    expected = [[sum(A[i][k] * B[k][j] for k in range(16)) for j in range(48)] for i in range(32)]

    matmul(out, A, B)
    assert all(math.isclose(out[i][j], expected[i][j], rel_tol=1e-5, abs_tol=1e-5) for i in range(32) for j in range(48))
