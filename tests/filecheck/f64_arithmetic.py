# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @f64_arithmetic(%0: !llvm.ptr {llvm.noalias}, %1: !llvm.ptr {llvm.noalias}, %2: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %5 = llvm.mul %3, %4 : i64
# CHECK-NEXT:     %6 = llvm.getelementptr inbounds %1[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
# CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr -> f64
# CHECK-NEXT:     %8 = llvm.getelementptr inbounds %2[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
# CHECK-NEXT:     %9 = llvm.load %8 : !llvm.ptr -> f64
# CHECK-NEXT:     %10 = llvm.fadd %7, %9 {fastmathFlags = #llvm.fastmath<fast>} : f64
# CHECK-NEXT:     %11 = llvm.getelementptr inbounds %0[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
# CHECK-NEXT:     llvm.store %10, %11 : f64, !llvm.ptr
# CHECK-NEXT:     %12 = llvm.load %6 : !llvm.ptr -> f64
# CHECK-NEXT:     %13 = llvm.load %8 : !llvm.ptr -> f64
# CHECK-NEXT:     %14 = llvm.fmul %12, %13 {fastmathFlags = #llvm.fastmath<fast>} : f64
# CHECK-NEXT:     llvm.store %14, %11 : f64, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def f64_arithmetic(out: f64[1] @ DRAM, a: f64[1] @ DRAM, b: f64[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] * b[0]
