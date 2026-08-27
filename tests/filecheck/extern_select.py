# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @uses_select(%0: !llvm.ptr {llvm.noalias}, %1: !llvm.ptr {llvm.noalias}, %2: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %5 = llvm.mul %3, %4 : i64
# CHECK-NEXT:     %6 = llvm.getelementptr inbounds %1[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr -> f32
# CHECK-NEXT:     %8 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %9 = llvm.getelementptr inbounds %2[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %10 = llvm.load %9 : !llvm.ptr -> f32
# CHECK-NEXT:     %11 = llvm.fcmp "olt" %8, %7 : f32
# CHECK-NEXT:     %12 = llvm.select %11, %7, %10 : i1, f32
# CHECK-NEXT:     %13 = llvm.getelementptr inbounds %0[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %12, %13 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *
from exo.platforms.x86 import *


@proc
def uses_select(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = select(0.0, a[0], a[0], b[0])
