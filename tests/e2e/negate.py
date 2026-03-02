# RUN: uv run xdsl-exo -o %t.mlir %s
# RUN: mlir-opt --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %t.mlir | mlir-translate --mlir-to-llvmir | llc -filetype=obj -o %t.o
# RUN: clang %t.o %S/negate.c -o %t.bin -lm
# RUN: %t.bin | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: -42.0
# CHECK: -7
@proc
def negate_float(out: f32[1] @ DRAM, a: f32[1] @ DRAM):
    out[0] = -a[0]


@proc
def negate_int(out: i32[1] @ DRAM, a: i32[1] @ DRAM):
    out[0] = -a[0]
