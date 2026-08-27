# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @usub_float(%0: !llvm.ptr, %1: !llvm.ptr) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %4 = llvm.mul %2, %3 : i64
# CHECK-NEXT:     %5 = llvm.getelementptr inbounds %1[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %6 = llvm.load %5 : !llvm.ptr -> f32
# CHECK-NEXT:     %7 = llvm.fneg %6 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %8 = llvm.getelementptr inbounds %0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %7, %8 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @usub_int(%9: !llvm.ptr, %10: !llvm.ptr) {
# CHECK-NEXT:     %11 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %12 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %13 = llvm.mul %11, %12 : i64
# CHECK-NEXT:     %14 = llvm.getelementptr inbounds %10[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     %15 = llvm.load %14 : !llvm.ptr -> i32
# CHECK-NEXT:     %16 = llvm.mlir.constant(0 : i32) : i32
# CHECK-NEXT:     %17 = llvm.sub %16, %15 : i32
# CHECK-NEXT:     %18 = llvm.getelementptr inbounds %9[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %17, %18 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def usub_float(out: f32[1] @ DRAM, x: f32[1] @ DRAM):
    out[0] = -x[0]


@proc
def usub_int(out: i32[1] @ DRAM, x: i32[1] @ DRAM):
    out[0] = -x[0]
