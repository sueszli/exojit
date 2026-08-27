# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @assign_from_scalar_memref(%0: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %2 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %3 = llvm.mlir.constant(4.200000e+01 : f32) : f32
# CHECK-NEXT:     %4 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %5 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %6 = llvm.mul %4, %5 : i64
# CHECK-NEXT:     %7 = llvm.getelementptr inbounds %2[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %3, %7 : f32, !llvm.ptr
# CHECK-NEXT:     %8 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %9 = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     %10 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%8 : i64)
# CHECK-NEXT:   ^bb1(%11: i64):
# CHECK-NEXT:     %12 = llvm.icmp "slt" %11, %9 : i64
# CHECK-NEXT:     llvm.cond_br %12, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %13 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %14 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %15 = llvm.mul %13, %14 : i64
# CHECK-NEXT:     %16 = llvm.getelementptr inbounds %2[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %17 = llvm.load %16 : !llvm.ptr -> f32
# CHECK-NEXT:     %18 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %19 = llvm.mul %11, %18 : i64
# CHECK-NEXT:     %20 = llvm.getelementptr inbounds %0[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %17, %20 : f32, !llvm.ptr
# CHECK-NEXT:     %21 = llvm.add %11, %10 : i64
# CHECK-NEXT:     llvm.br ^bb1(%21 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.call @free(%2) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def assign_from_scalar_memref(x: f32[8] @ DRAM):
    tmp: f32
    tmp = 42.0
    for i in seq(0, 8):
        x[i] = tmp
