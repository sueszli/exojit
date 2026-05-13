# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @index_modulo(%0: !llvm.ptr, %1: i64) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(10) : i64
# CHECK-NEXT:     %3 = llvm.srem %1, %2 : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(42 : i32) : i32
# CHECK-NEXT:     %5 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %6 = llvm.mul %3, %5 : i64
# CHECK-NEXT:     %7 = llvm.getelementptr inbounds %0[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %4, %7 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def index_modulo(out: i32[10] @ DRAM, i: index):
    assert i >= 0
    out[i % 10] = 42
