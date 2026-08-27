# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @preserves_args(%0: !llvm.ptr {llvm.noalias}, %1: i64) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %3 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %4 = llvm.mul %1, %3 : i64
# CHECK-NEXT:     %5 = llvm.getelementptr inbounds %0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %2, %5 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def preserves_args(x: f32[16], idx: index):
    assert idx >= 0 and idx < 16
    x[idx] = 0.0
