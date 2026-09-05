# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @usub_float(%offset_pointer: !llvm.ptr, %offset_pointer_1: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %0, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.add %offset_pointer_2, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.inttoptr %offset_pointer_3 : i64 to !llvm.ptr
# CHECK-NEXT:     %1 = llvm.load %offset_pointer_4 : !llvm.ptr -> f32
# CHECK-NEXT:     %2 = llvm.fneg %1 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %2, %offset_pointer_7 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @usub_int(%offset_pointer_8: !llvm.ptr, %offset_pointer_9: !llvm.ptr) {
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %3, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.ptrtoint %offset_pointer_9 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_11 = llvm.add %offset_pointer_10, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_12 = llvm.inttoptr %offset_pointer_11 : i64 to !llvm.ptr
# CHECK-NEXT:     %4 = llvm.load %offset_pointer_12 : !llvm.ptr -> i32
# CHECK-NEXT:     %5 = llvm.mlir.constant(0 : i32) : i32
# CHECK-NEXT:     %6 = llvm.sub %5, %4 : i32
# CHECK-NEXT:     %offset_pointer_13 = llvm.ptrtoint %offset_pointer_8 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_14 = llvm.add %offset_pointer_13, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_15 = llvm.inttoptr %offset_pointer_14 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %6, %offset_pointer_15 : i32, !llvm.ptr
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
