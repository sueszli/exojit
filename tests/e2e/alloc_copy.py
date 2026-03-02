# RUN: uv run xdsl-exo -o %t.mlir %s
# RUN: mlir-opt --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %t.mlir | mlir-translate --mlir-to-llvmir | llc -filetype=obj -o %t.o
# RUN: clang %t.o %S/alloc_copy.c -o %t.bin -lm
# RUN: %t.bin | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: OK
@proc
def alloc_copy(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp
