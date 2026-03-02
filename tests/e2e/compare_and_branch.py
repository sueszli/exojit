# RUN: uv run xdsl-exo -o %t.mlir %s
# RUN: mlir-opt --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %t.mlir | mlir-translate --mlir-to-llvmir | llc -filetype=obj -o %t.o
# RUN: clang %t.o %S/compare_and_branch.c -o %t.bin -lm
# RUN: %t.bin | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: 10.0
# CHECK: 20.0
@proc
def compare_and_branch(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 10.0
    else:
        out[0] = 20.0
