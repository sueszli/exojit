# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @bool_ops(%offset_pointer: !llvm.ptr, %0: i64, %1: i64, %2: i64) {
# CHECK-NEXT:     %3 = llvm.icmp "slt" %0, %1 : i64
# CHECK-NEXT:     %4 = llvm.icmp "slt" %1, %2 : i64
# CHECK-NEXT:     %5 = llvm.and %3, %4 : i1
# CHECK-NEXT:     llvm.cond_br %5, ^bb1, ^bb2
# CHECK-NEXT:   ^bb1:
# CHECK-NEXT:     %6 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %7 = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %6, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.add %offset_pointer_1, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.inttoptr %offset_pointer_2 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %7, %offset_pointer_3 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     llvm.br ^bb3
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     %8 = llvm.icmp "slt" %0, %1 : i64
# CHECK-NEXT:     %9 = llvm.icmp "slt" %1, %2 : i64
# CHECK-NEXT:     %10 = llvm.or %8, %9 : i1
# CHECK-NEXT:     llvm.cond_br %10, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %11 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %12 = llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %11, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.add %offset_pointer_4, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.inttoptr %offset_pointer_5 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %12, %offset_pointer_6 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb5:
# CHECK-NEXT:     llvm.br ^bb6
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def bool_ops(out: f32[1] @ DRAM, a: index, b: index, c: index):
    if a < b < c:
        out[0] = 1.0
    if a < b or b < c:
        out[0] = 2.0
