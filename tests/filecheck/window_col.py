# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @set_col(%0: !llvm.ptr) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb0(%1 : i64)
# CHECK-NEXT:   ^bb0(%4: i64):
# CHECK-NEXT:     %5 = llvm.icmp "slt" %4, %2 : i64
# CHECK-NEXT:     llvm.cond_br %5, ^bb1, ^bb2
# CHECK-NEXT:   ^bb1:
# CHECK-NEXT:     %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %7 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %8 = llvm.mul %4, %7 : i64
# CHECK-NEXT:     %9 = llvm.getelementptr inbounds %0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %6, %9 : f32, !llvm.ptr
# CHECK-NEXT:     %10 = llvm.add %4, %3 : i64
# CHECK-NEXT:     llvm.br ^bb0(%10 : i64)
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @window_col(%11: !llvm.ptr) {
# CHECK-NEXT:     %12 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %13 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %14 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb3(%12 : i64)
# CHECK-NEXT:   ^bb3(%15: i64):
# CHECK-NEXT:     %16 = llvm.icmp "slt" %15, %13 : i64
# CHECK-NEXT:     llvm.cond_br %16, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %17 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %18 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %19 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %20 = llvm.mul %18, %19 : i64
# CHECK-NEXT:     %21 = llvm.mul %17, %20 : i64
# CHECK-NEXT:     %22 = llvm.mul %15, %18 : i64
# CHECK-NEXT:     %23 = llvm.add %21, %22 : i64
# CHECK-NEXT:     %24 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %25 = llvm.mul %23, %24 : i64
# CHECK-NEXT:     %26 = llvm.ptrtoint %11 : !llvm.ptr to i64
# CHECK-NEXT:     %27 = llvm.add %26, %25 : i64
# CHECK-NEXT:     %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @set_col(%28) : (!llvm.ptr) -> ()
# CHECK-NEXT:     %29 = llvm.add %15, %14 : i64
# CHECK-NEXT:     llvm.br ^bb3(%29 : i64)
# CHECK-NEXT:   ^bb5:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def set_col(col: [f32][4] @ DRAM):
    for i in seq(0, 4):
        col[i] = 0.0


@proc
def window_col(A: f32[4, 4] @ DRAM):
    for j in seq(0, 4):
        set_col(A[:, j])
