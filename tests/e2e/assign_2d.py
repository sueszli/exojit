# RUN: uv run xdsl-exo -o %t.mlir %s
# RUN: mlir-opt --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %t.mlir | mlir-translate --mlir-to-llvmir | llc -filetype=obj -o %t.o
# RUN: clang %t.o %S/assign_2d.c -o %t.bin -lm
# RUN: %t.bin | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: OK
@proc
def assign_2d(dst: f32[4, 4] @ DRAM, src: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[i, j]
