# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @multi_type_alloc(%offset_pointer: !llvm.ptr, %offset_pointer_1: !llvm.ptr) {
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.call @malloc(%bytes_per_element) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %offset_pointer_3 = llvm.call @malloc(%bytes_per_element) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %0 = llvm.mlir.constant(3.140000e+00 : f32) : f32
# CHECK-NEXT:     %1 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %1, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.ptrtoint %offset_pointer_2 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.add %offset_pointer_4, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.inttoptr %offset_pointer_5 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %0, %offset_pointer_6 : f32, !llvm.ptr
# CHECK-NEXT:     %2 = llvm.mlir.constant(42 : i32) : i32
# CHECK-NEXT:     %offset_pointer_7 = llvm.ptrtoint %offset_pointer_3 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.add %offset_pointer_7, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_9 = llvm.inttoptr %offset_pointer_8 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %2, %offset_pointer_9 : i32, !llvm.ptr
# CHECK-NEXT:     %3 = llvm.load %offset_pointer_6 : !llvm.ptr -> f32
# CHECK-NEXT:     %offset_pointer_10 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_11 = llvm.add %offset_pointer_10, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_12 = llvm.inttoptr %offset_pointer_11 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %3, %offset_pointer_12 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%offset_pointer_2) : (!llvm.ptr) -> ()
# CHECK-NEXT:     %4 = llvm.load %offset_pointer_9 : !llvm.ptr -> i32
# CHECK-NEXT:     %offset_pointer_13 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_14 = llvm.add %offset_pointer_13, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_15 = llvm.inttoptr %offset_pointer_14 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %4, %offset_pointer_15 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%offset_pointer_3) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def multi_type_alloc(out_f: f32[1] @ DRAM, out_i: i32[1] @ DRAM):
    tmp_f: f32
    tmp_i: i32
    tmp_f = 3.14
    tmp_i = 42
    out_f[0] = tmp_f
    out_i[0] = tmp_i
