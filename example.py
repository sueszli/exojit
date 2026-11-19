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
from exo.libs.memories import DRAM_STACK
from exo.stdlib.scheduling import divide_loop, fission, lift_alloc, reorder_loops, set_memory, simplify, stage_mem

from exojit import jit


def optimize_matmul(p: Procedure) -> Procedure:
    # Found through LLM autoresearch.
    p = divide_loop(p, "j", 32, ["jo", "ji"], perfect=True)
    p = reorder_loops(p, "i jo")
    p = divide_loop(p, "i", 2, ["io", "ii"], perfect=True)
    p = stage_mem(p, p.find_loop("ii").as_block(), "out[2*io:2*io+2, 32*jo:32*jo+32]", "acc")
    p = fission(p, p.find_loop("k").before(), n_lifts=2)
    p = reorder_loops(p, "ji k")
    p = reorder_loops(p, "ii k")
    p = lift_alloc(simplify(p), "acc", n_lifts=2)
    return set_memory(p, "acc", DRAM_STACK)


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
