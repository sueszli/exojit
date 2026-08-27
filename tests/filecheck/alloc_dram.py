# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @alloc_dram(%0: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %2 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %5 = llvm.mul %3, %4 : i64
# CHECK-NEXT:     %6 = llvm.getelementptr inbounds %0[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr -> f32
# CHECK-NEXT:     %8 = llvm.getelementptr inbounds %2[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %7, %8 : f32, !llvm.ptr
# CHECK-NEXT:     %9 = llvm.load %8 : !llvm.ptr -> f32
# CHECK-NEXT:     llvm.store %9, %6 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%2) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def alloc_dram(x: f32[8] @ DRAM):
    tmp: f32[4]
    tmp[0] = x[0]
    x[0] = tmp[0]
