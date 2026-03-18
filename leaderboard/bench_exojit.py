# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
#   "numpy",
# ]
# ///

import timeit

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
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
expected = A @ B

naive_ms = None
for label, p in [("naive", matmul), ("optimized", opt)]:
    fn = compile_jit(p)[p.name()]

    C = np.zeros((N, N), dtype=np.float32)
    fn(C, A, B)
    assert np.allclose(C, expected, atol=0.5), f"{label} wrong"

    ms = min(timeit.repeat(lambda: fn(C, A, B), number=200, repeat=5)) / 200 * 1e3
    print(f"{label:<12s} {ms:.2f} ms/call" + (f" ({naive_ms / ms:.1f}x)" if naive_ms else ""))
    naive_ms = naive_ms or ms
