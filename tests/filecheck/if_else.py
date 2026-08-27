# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @if_else(%0: !llvm.ptr {llvm.noalias}, %1: i64, %2: i64) {
# CHECK-NEXT:     %3 = llvm.icmp "slt" %1, %2 : i64
# CHECK-NEXT:     llvm.cond_br %3, ^bb1, ^bb2
# CHECK-NEXT:   ^bb1:
# CHECK-NEXT:     %4 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %5 = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:     %6 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %7 = llvm.mul %4, %6 : i64
# CHECK-NEXT:     %8 = llvm.getelementptr inbounds %0[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %5, %8 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %9 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %10 = llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK-NEXT:     %11 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %12 = llvm.mul %9, %11 : i64
# CHECK-NEXT:     %13 = llvm.getelementptr inbounds %0[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %10, %13 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def if_else(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 2.0
