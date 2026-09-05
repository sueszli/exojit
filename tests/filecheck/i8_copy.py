# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @i8_copy(%offset_pointer: !llvm.ptr, %offset_pointer_1: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%0 : i64)
# CHECK-NEXT:   ^bb1(%3: i64):
# CHECK-NEXT:     %4 = llvm.icmp "slt" %3, %1 : i64
# CHECK-NEXT:     llvm.cond_br %4, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %3, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.add %offset_pointer_2, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.inttoptr %offset_pointer_3 : i64 to !llvm.ptr
# CHECK-NEXT:     %5 = llvm.load %offset_pointer_4 : !llvm.ptr -> i8
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %3, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %5, %offset_pointer_7 : i8, !llvm.ptr
# CHECK-NEXT:     %6 = llvm.add %3, %2 : i64
# CHECK-NEXT:     llvm.br ^bb1(%6 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def i8_copy(out: i8[8] @ DRAM, x: i8[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = x[i]
