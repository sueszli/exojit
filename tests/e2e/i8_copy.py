# RUN: uv run xdsl-exo -o %t.mlir %s
# RUN: mlir-opt --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %t.mlir | mlir-translate --mlir-to-llvmir | llc -filetype=obj -o %t.o
# RUN: clang %t.o %S/i8_copy.c -o %t.bin -lm
# RUN: %t.bin | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: OK
@proc
def i8_copy(dst: i8[8] @ DRAM, src: i8[8] @ DRAM):
    for i in seq(0, 8):
        dst[i] = src[i]
