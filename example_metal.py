# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit@feat/metal-backend",
# ]
# ///

import math
import random
from unittest.mock import patch

from exo import *

from backend_metal import execute_kernel
from exojit import jit


@jit(backend="metal")
def matmul(out: f32[32, 48] @ DRAM, A: f32[32, 16] @ DRAM, B: f32[16, 48] @ DRAM):
    for i in par(0, 32):  # parallel loop
        for j in par(0, 48):
            out[i, j] = 0.0
            for k in seq(0, 16):
                out[i, j] += A[i, k] * B[k, j]


if __name__ == "__main__":
    A = [[random.uniform(-1.0, 1.0) for _ in range(16)] for _ in range(32)]
    B = [[random.uniform(-1.0, 1.0) for _ in range(48)] for _ in range(16)]
    out = [[0.0] * 48 for _ in range(32)]
    expected = [[sum(A[i][k] * B[k][j] for k in range(16)) for j in range(48)] for i in range(32)]

    with patch("backend_metal.execute_kernel", wraps=execute_kernel) as dispatch:
        matmul(out, A, B)
    assert dispatch.call_count == 1
    device, pipeline, grid, group, _ = dispatch.call_args.args
    assert device.name().startswith("Apple") and pipeline.threadExecutionWidth() > 1
    assert (grid.width, grid.height, grid.depth) == (32 * 48, 1, 1) and group.width > 1
    assert all(math.isclose(out[i][j], expected[i][j], rel_tol=1e-5, abs_tol=1e-5) for i in range(32) for j in range(48))
