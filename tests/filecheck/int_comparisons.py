# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @int_comparisons(%offset_pointer: !llvm.ptr, %0: i64, %1: i64) {
# CHECK-NEXT:     %2 = llvm.icmp "eq" %0, %1 : i64
# CHECK-NEXT:     llvm.cond_br %2, ^bb1, ^bb2
# CHECK-NEXT:   ^bb1:
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(1 : i32) : i32
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %3, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.add %offset_pointer_1, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.inttoptr %offset_pointer_2 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %4, %offset_pointer_3 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     %5 = llvm.icmp "slt" %0, %1 : i64
# CHECK-NEXT:     llvm.cond_br %5, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %6 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %7 = llvm.mlir.constant(2 : i32) : i32
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %6, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.add %offset_pointer_4, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.inttoptr %offset_pointer_5 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %7, %offset_pointer_6 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb5:
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     %8 = llvm.icmp "sgt" %0, %1 : i64
# CHECK-NEXT:     llvm.cond_br %8, ^bb7, ^bb8
# CHECK-NEXT:   ^bb7:
# CHECK-NEXT:     %9 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %10 = llvm.mlir.constant(3 : i32) : i32
# CHECK-NEXT:     %bytes_per_element_2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_2 = llvm.mul %9, %bytes_per_element_2 : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.add %offset_pointer_7, %scaled_pointer_offset_2 : i64
# CHECK-NEXT:     %offset_pointer_9 = llvm.inttoptr %offset_pointer_8 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %10, %offset_pointer_9 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb9
# CHECK-NEXT:   ^bb8:
# CHECK-NEXT:     llvm.br ^bb9
# CHECK-NEXT:   ^bb9:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
