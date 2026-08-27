# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @assign_2d(%0: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%1 : i64)
# CHECK-NEXT:   ^bb1(%4: i64):
# CHECK-NEXT:     %5 = llvm.icmp "slt" %4, %2 : i64
# CHECK-NEXT:     llvm.cond_br %5, ^bb2, ^bb6
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %6 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %7 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %8 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb3(%6 : i64)
# CHECK-NEXT:   ^bb3(%9: i64):
# CHECK-NEXT:     %10 = llvm.icmp "slt" %9, %7 : i64
# CHECK-NEXT:     llvm.cond_br %10, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %11 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %12 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %13 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %14 = llvm.mul %12, %13 : i64
# CHECK-NEXT:     %15 = llvm.mul %4, %14 : i64
# CHECK-NEXT:     %16 = llvm.mul %9, %12 : i64
# CHECK-NEXT:     %17 = llvm.add %15, %16 : i64
# CHECK-NEXT:     %18 = llvm.getelementptr inbounds %0[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %11, %18 : f32, !llvm.ptr
# CHECK-NEXT:     %19 = llvm.add %9, %8 : i64
# CHECK-NEXT:     llvm.br ^bb3(%19 : i64)
# CHECK-NEXT:   ^bb5:
# CHECK-NEXT:     %20 = llvm.add %4, %3 : i64
# CHECK-NEXT:     llvm.br ^bb1(%20 : i64)
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def assign_2d(x: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            x[i, j] = 0.0
