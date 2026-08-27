# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @reduce_int(%0: !llvm.ptr, %1: !llvm.ptr) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%2 : i64)
# CHECK-NEXT:   ^bb1(%5: i64):
# CHECK-NEXT:     %6 = llvm.icmp "slt" %5, %3 : i64
# CHECK-NEXT:     llvm.cond_br %6, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %7 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %8 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %9 = llvm.mul %5, %8 : i64
# CHECK-NEXT:     %10 = llvm.getelementptr inbounds %0[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     %11 = llvm.load %10 : !llvm.ptr -> i32
# CHECK-NEXT:     %12 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %13 = llvm.mul %7, %12 : i64
# CHECK-NEXT:     %14 = llvm.getelementptr inbounds %1[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     %15 = llvm.load %14 : !llvm.ptr -> i32
# CHECK-NEXT:     %16 = llvm.add %15, %11 : i32
# CHECK-NEXT:     %17 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %18 = llvm.mul %7, %17 : i64
# CHECK-NEXT:     %19 = llvm.getelementptr inbounds %1[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %16, %19 : i32, !llvm.ptr
# CHECK-NEXT:     %20 = llvm.add %5, %4 : i64
# CHECK-NEXT:     llvm.br ^bb1(%20 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def reduce_int(x: i32[8] @ DRAM, out: i32[1] @ DRAM):
    for i in seq(0, 8):
        out[0] += x[i]
