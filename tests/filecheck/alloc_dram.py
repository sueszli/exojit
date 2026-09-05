# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @alloc_dram(%offset_pointer: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %1 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %1, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.add %offset_pointer_2, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.inttoptr %offset_pointer_3 : i64 to !llvm.ptr
# CHECK-NEXT:     %2 = llvm.load %offset_pointer_4 : !llvm.ptr -> f32
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %2, %offset_pointer_7 : f32, !llvm.ptr
# CHECK-NEXT:     %3 = llvm.load %offset_pointer_7 : !llvm.ptr -> f32
# CHECK-NEXT:     llvm.store %3, %offset_pointer_4 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%offset_pointer_1) : (!llvm.ptr) -> ()
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
