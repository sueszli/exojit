# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @int_comparisons(%0: !llvm.ptr {llvm.noalias}, %1: i64, %2: i64) {
# CHECK-NEXT:     %3 = llvm.icmp "eq" %1, %2 : i64
# CHECK-NEXT:     llvm.cond_br %3, ^bb1, ^bb2
# CHECK-NEXT:   ^bb1:
# CHECK-NEXT:     %4 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %5 = llvm.mlir.constant(1 : i32) : i32
# CHECK-NEXT:     %6 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %7 = llvm.mul %4, %6 : i64
# CHECK-NEXT:     %8 = llvm.getelementptr inbounds %0[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %5, %8 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     %9 = llvm.icmp "slt" %1, %2 : i64
# CHECK-NEXT:     llvm.cond_br %9, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %10 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %11 = llvm.mlir.constant(2 : i32) : i32
# CHECK-NEXT:     %12 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %13 = llvm.mul %10, %12 : i64
# CHECK-NEXT:     %14 = llvm.getelementptr inbounds %0[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %11, %14 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb5:
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     %15 = llvm.icmp "sgt" %1, %2 : i64
# CHECK-NEXT:     llvm.cond_br %15, ^bb7, ^bb8
# CHECK-NEXT:   ^bb7:
# CHECK-NEXT:     %16 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %17 = llvm.mlir.constant(3 : i32) : i32
# CHECK-NEXT:     %18 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %19 = llvm.mul %16, %18 : i64
# CHECK-NEXT:     %20 = llvm.getelementptr inbounds %0[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %17, %20 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb9
# CHECK-NEXT:   ^bb8:
# CHECK-NEXT:     llvm.br ^bb9
# CHECK-NEXT:   ^bb9:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
