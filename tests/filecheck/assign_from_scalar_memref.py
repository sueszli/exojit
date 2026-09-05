# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @assign_from_scalar_memref(%offset_pointer: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %1 = llvm.mlir.constant(4.200000e+01 : f32) : f32
# CHECK-NEXT:     %2 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %2, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.add %offset_pointer_2, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.inttoptr %offset_pointer_3 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %1, %offset_pointer_4 : f32, !llvm.ptr
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     %5 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%3 : i64)
# CHECK-NEXT:   ^bb1(%6: i64):
# CHECK-NEXT:     %7 = llvm.icmp "slt" %6, %4 : i64
# CHECK-NEXT:     llvm.cond_br %7, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %8 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %8, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     %9 = llvm.load %offset_pointer_7 : !llvm.ptr -> f32
# CHECK-NEXT:     %bytes_per_element_2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_2 = llvm.mul %6, %bytes_per_element_2 : i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_9 = llvm.add %offset_pointer_8, %scaled_pointer_offset_2 : i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.inttoptr %offset_pointer_9 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %9, %offset_pointer_10 : f32, !llvm.ptr
# CHECK-NEXT:     %10 = llvm.add %6, %5 : i64
# CHECK-NEXT:     llvm.br ^bb1(%10 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.call @free(%offset_pointer_1) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def assign_from_scalar_memref(x: f32[8] @ DRAM):
    tmp: f32
    tmp = 42.0
    for i in seq(0, 8):
        x[i] = tmp
