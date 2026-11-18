# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit[metal] @ git+https://github.com/sueszli/exojit@feat/metal-backend",
# ]
# ///

import math
import random

from exo import *

from exojit import jit


@proc
def matmul(out: f32[32, 48] @ DRAM, A: f32[32, 16] @ DRAM, B: f32[16, 48] @ DRAM):
    for i in par(0, 32):
        for j in par(0, 48):
            out[i, j] = 0.0
            for k in seq(0, 16):
                out[i, j] += A[i, k] * B[k, j]


if __name__ == "__main__":
    A = [[random.uniform(-1.0, 1.0) for _ in range(16)] for _ in range(32)]
    B = [[random.uniform(-1.0, 1.0) for _ in range(48)] for _ in range(16)]
    cpu, gpu = [[0.0] * 48 for _ in range(32)], [[0.0] * 48 for _ in range(32)]

    jit(matmul)(cpu, A, B)
    jit(matmul, backend="metal")(gpu, A, B)  # one gpu thread per output cell
    assert all(math.isclose(gpu[i][j], cpu[i][j], rel_tol=1e-5, abs_tol=1e-5) for i in range(32) for j in range(48))
    print(f"metal matmul matches cpu: out[0][:4] = {[round(x, 4) for x in gpu[0][:4]]}")
