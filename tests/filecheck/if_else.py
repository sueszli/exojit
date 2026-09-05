# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @if_else(%offset_pointer: !llvm.ptr, %0: i64, %1: i64) {
# CHECK-NEXT:     %2 = llvm.icmp "slt" %0, %1 : i64
# CHECK-NEXT:     llvm.cond_br %2, ^bb1, ^bb2
# CHECK-NEXT:   ^bb1:
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %3, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.add %offset_pointer_1, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.inttoptr %offset_pointer_2 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %4, %offset_pointer_3 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %5 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %6 = llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %5, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.add %offset_pointer_4, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.inttoptr %offset_pointer_5 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %6, %offset_pointer_6 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def if_else(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 2.0
