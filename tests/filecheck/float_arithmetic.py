# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @float_arithmetic(%0: !llvm.ptr {llvm.noalias}, %1: !llvm.ptr {llvm.noalias}, %2: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %5 = llvm.mul %3, %4 : i64
# CHECK-NEXT:     %6 = llvm.getelementptr inbounds %1[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr -> f32
# CHECK-NEXT:     %8 = llvm.getelementptr inbounds %2[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %9 = llvm.load %8 : !llvm.ptr -> f32
# CHECK-NEXT:     %10 = llvm.fadd %7, %9 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %11 = llvm.getelementptr inbounds %0[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %10, %11 : f32, !llvm.ptr
# CHECK-NEXT:     %12 = llvm.load %6 : !llvm.ptr -> f32
# CHECK-NEXT:     %13 = llvm.load %8 : !llvm.ptr -> f32
# CHECK-NEXT:     %14 = llvm.fsub %12, %13 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     llvm.store %14, %11 : f32, !llvm.ptr
# CHECK-NEXT:     %15 = llvm.load %6 : !llvm.ptr -> f32
# CHECK-NEXT:     %16 = llvm.load %8 : !llvm.ptr -> f32
# CHECK-NEXT:     %17 = llvm.fmul %15, %16 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     llvm.store %17, %11 : f32, !llvm.ptr
# CHECK-NEXT:     %18 = llvm.load %6 : !llvm.ptr -> f32
# CHECK-NEXT:     %19 = llvm.load %8 : !llvm.ptr -> f32
# CHECK-NEXT:     %20 = llvm.fdiv %18, %19 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     llvm.store %20, %11 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def float_arithmetic(out: f32[1] @ DRAM, x: f32[1] @ DRAM, y: f32[1] @ DRAM):
    out[0] = x[0] + y[0]
    out[0] = x[0] - y[0]
    out[0] = x[0] * y[0]
    out[0] = x[0] / y[0]
