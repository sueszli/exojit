# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @set_row(%offset_pointer: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%0 : i64)
# CHECK-NEXT:   ^bb1(%3: i64):
# CHECK-NEXT:     %4 = llvm.icmp "slt" %3, %1 : i64
# CHECK-NEXT:     llvm.cond_br %4, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %3, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.add %offset_pointer_1, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.inttoptr %offset_pointer_2 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %5, %offset_pointer_3 : f32, !llvm.ptr
# CHECK-NEXT:     %6 = llvm.add %3, %2 : i64
# CHECK-NEXT:     llvm.br ^bb1(%6 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @window_row(%offset_pointer_4: !llvm.ptr) {
# CHECK-NEXT:     %7 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %8 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %9 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%7 : i64)
# CHECK-NEXT:   ^bb1(%10: i64):
# CHECK-NEXT:     %11 = llvm.icmp "slt" %10, %8 : i64
# CHECK-NEXT:     llvm.cond_br %11, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %12 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %c4 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %increment = llvm.mul %c4, %10 : i64
# CHECK-NEXT:     %subview = llvm.add %increment, %12 : i64
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %subview, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer_4 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @set_row(%offset_pointer_7) : (!llvm.ptr) -> ()
# CHECK-NEXT:     %13 = llvm.add %10, %9 : i64
# CHECK-NEXT:     llvm.br ^bb1(%13 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def set_row(row: [f32][4] @ DRAM):
    for i in seq(0, 4):
        row[i] = 0.0


@proc
def window_row(A: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        set_row(A[i, :])
