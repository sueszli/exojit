# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @bool_ops(%0: !llvm.ptr {llvm.noalias}, %1: i64, %2: i64, %3: i64) {
# CHECK-NEXT:     %4 = llvm.icmp "slt" %1, %2 : i64
# CHECK-NEXT:     %5 = llvm.icmp "slt" %2, %3 : i64
# CHECK-NEXT:     %6 = llvm.and %4, %5 : i1
# CHECK-NEXT:     llvm.cond_br %6, ^bb1, ^bb2
# CHECK-NEXT:   ^bb1:
# CHECK-NEXT:     %7 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:     %9 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %10 = llvm.mul %7, %9 : i64
# CHECK-NEXT:     %11 = llvm.getelementptr inbounds %0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %8, %11 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     %12 = llvm.icmp "slt" %1, %2 : i64
# CHECK-NEXT:     %13 = llvm.icmp "slt" %2, %3 : i64
# CHECK-NEXT:     %14 = llvm.or %12, %13 : i1
# CHECK-NEXT:     llvm.cond_br %14, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %15 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %16 = llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK-NEXT:     %17 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %18 = llvm.mul %15, %17 : i64
# CHECK-NEXT:     %19 = llvm.getelementptr inbounds %0[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %16, %19 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb5:
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def bool_ops(out: f32[1] @ DRAM, a: index, b: index, c: index):
    if a < b < c:
        out[0] = 1.0
    if a < b or b < c:
        out[0] = 2.0
