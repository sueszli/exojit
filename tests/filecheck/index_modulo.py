# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @index_modulo(%offset_pointer: !llvm.ptr, %0: i64) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(10) : i64
# CHECK-NEXT:     %2 = llvm.srem %0, %1 : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(42 : i32) : i32
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %2, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.add %offset_pointer_1, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.inttoptr %offset_pointer_2 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %3, %offset_pointer_3 : i32, !llvm.ptr
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
