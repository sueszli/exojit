import timeit

import numpy as np
from exo import *
from exo.API import Procedure
from exo.stdlib.scheduling import divide_loop, fission, rename, reorder_loops, simplify, unroll_loop
from numpy.typing import NDArray

from exojit.main import jit


@proc
def matmul(C: f32[128, 128] @ DRAM, A: f32[128, 128] @ DRAM, B: f32[128, 128] @ DRAM):
    for i in seq(0, 128):
        for j in seq(0, 128):
            C[i, j] = 0.0
            for k in seq(0, 128):
                C[i, j] += A[i, k] * B[k, j]


def bench(s: str, p: Procedure, A: NDArray[np.float32], B: NDArray[np.float32]) -> None:
    f = jit(p)
    C = np.zeros_like(A)
    f(C, A, B)
    assert np.allclose(C, A @ B, atol=0.5)
    ms = min(timeit.repeat(lambda: f(C, A, B), number=200, repeat=5)) / 200 * 1e3
    print(f"{s:<12s} {ms:.2f} ms/call")


if __name__ == "__main__":
    N = 128
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    bench("naive", matmul, A, B)

    opt = rename(matmul, "opt")
    opt = fission(opt, opt.find("for k in _: _").before(), n_lifts=2)  # separate init from compute
    opt = reorder_loops(opt, "j k")  # j inside k -> row-major A streaming
    opt = divide_loop(opt, "j #1", 4, ["jo", "ji"], perfect=True)  # tile j by 4
    opt = unroll_loop(opt, "ji")  # unroll inner j -> 4 explicit accumulators
    opt = simplify(opt)
    bench("optimized", opt, A, B)
