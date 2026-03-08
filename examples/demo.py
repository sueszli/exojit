from __future__ import annotations

import numpy as np
from exo import *

from xdsl_exo.main import compile_procs
from xdsl_exo.patches_llvmlite import jit_compile


@proc
def matmul(C: f32[4, 4] @ DRAM, A: f32[4, 4] @ DRAM, B: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


compiled_fns = jit_compile(compile_procs(matmul))
input_matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
identity = np.eye(4, dtype=np.float32)
result = np.zeros((4, 4), dtype=np.float32)

compiled_fns["matmul"](result.ctypes.data, input_matrix.ctypes.data, identity.ctypes.data)
assert np.allclose(result, input_matrix @ identity)
print(result)
