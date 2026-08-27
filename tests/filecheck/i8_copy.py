# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @i8_copy(%0: !llvm.ptr {llvm.noalias}, %1: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%2 : i64)
# CHECK-NEXT:   ^bb1(%5: i64):
# CHECK-NEXT:     %6 = llvm.icmp "slt" %5, %3 : i64
# CHECK-NEXT:     llvm.cond_br %6, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %7 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %8 = llvm.mul %5, %7 : i64
# CHECK-NEXT:     %9 = llvm.getelementptr inbounds %1[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
# CHECK-NEXT:     %10 = llvm.load %9 : !llvm.ptr -> i8
# CHECK-NEXT:     %11 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %12 = llvm.mul %5, %11 : i64
# CHECK-NEXT:     %13 = llvm.getelementptr inbounds %0[%12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
# CHECK-NEXT:     llvm.store %10, %13 : i8, !llvm.ptr
# CHECK-NEXT:     %14 = llvm.add %5, %4 : i64
# CHECK-NEXT:     llvm.br ^bb1(%14 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def i8_copy(out: i8[8] @ DRAM, x: i8[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = x[i]
