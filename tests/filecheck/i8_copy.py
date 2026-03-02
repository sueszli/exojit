# RUN: uv run xdsl-exo -o - %s | filecheck %s

# exercises: i8 type lowering (1-byte element width)
# lowering: i8 tensors → !llvm.ptr, loads/stores as i8

from __future__ import annotations

from exo import *


# CHECK:      func.func @i8_copy({{.*}}) {
# CHECK:        "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i8
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (i8, !llvm.ptr) -> ()
@proc
def i8_copy(out: i8[8] @ DRAM, x: i8[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = x[i]
