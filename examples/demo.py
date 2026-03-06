from __future__ import annotations

import ctypes

import numpy as np
from exo import *

from xdsl_exo.main import compile_procs
from xdsl_exo.patches_llvmlite import jit_compile, to_llvmlite


@proc
def matmul(C: f32[4, 4] @ DRAM, A: f32[4, 4] @ DRAM, B: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


engine = jit_compile(to_llvmlite(compile_procs([matmul])))
fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(engine.get_function_address("matmul"))

A = np.arange(16, dtype=np.float32).reshape(4, 4)
B = np.eye(4, dtype=np.float32)
C = np.zeros((4, 4), dtype=np.float32)

fn(C.ctypes.data, A.ctypes.data, B.ctypes.data)
assert np.allclose(C, A @ B)
print(C)
