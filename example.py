# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit",
# ]
# ///

import math
import random

from exo import *
from exo.API import Procedure
from exo.stdlib.scheduling import divide_loop, fission, reorder_loops, simplify, unroll_loop

from exojit.main import jit


def optimize_matmul(p: Procedure) -> Procedure:
    p = fission(p, p.find("for k in _: _").before(), n_lifts=2)
    p = reorder_loops(p, "j k")
    p = divide_loop(p, "j #1", 4, ["jo", "ji"], perfect=True)
    p = unroll_loop(p, "ji")
    return simplify(p)


@jit(optimize=optimize_matmul)
def matmul(out: f32[32, 32] @ DRAM, A: f32[32, 32] @ DRAM, B: f32[32, 32] @ DRAM):
    for i in seq(0, 32):
        for j in seq(0, 32):
            out[i, j] = 0.0
            for k in seq(0, 32):
                out[i, j] += A[i, k] * B[k, j]


if __name__ == "__main__":
    A = [[random.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(32)]
    B = [[random.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(32)]
    out = [[0.0] * 32 for _ in range(32)]
    expected = [[sum(A[i][k] * B[k][j] for k in range(32)) for j in range(32)] for i in range(32)]

    matmul(out, A, B)
    assert all(math.isclose(out[i][j], expected[i][j], rel_tol=1e-5, abs_tol=1e-5) for i in range(32) for j in range(32))
