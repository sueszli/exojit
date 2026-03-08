from __future__ import annotations

from collections.abc import Callable

from xnumpy.library.kernels.elementwise import _jit


def matmul(m: int, k: int, n: int) -> Callable[..., None]:
    name = f"_mm_{m}_{k}_{n}"
    return _jit(
        f"""@proc
def {name}(C: f32[{m},{n}] @ DRAM, A: f32[{m},{k}] @ DRAM, B: f32[{k},{n}] @ DRAM):
    for i in seq(0, {m}):
        for j in seq(0, {n}):
            C[i,j] = 0.0
            for ki in seq(0, {k}):
                C[i,j] += A[i,ki] * B[ki,j]
""",
        name,
    )
