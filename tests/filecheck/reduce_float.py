# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @reduce_float(%offset_pointer: !llvm.ptr, %offset_pointer_1: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%0 : i64)
# CHECK-NEXT:   ^bb1(%3: i64):
# CHECK-NEXT:     %4 = llvm.icmp "slt" %3, %1 : i64
# CHECK-NEXT:     llvm.cond_br %4, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %5 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %3, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.add %offset_pointer_2, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.inttoptr %offset_pointer_3 : i64 to !llvm.ptr
# CHECK-NEXT:     %6 = llvm.load %offset_pointer_4 : !llvm.ptr -> f32
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %5, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     %7 = llvm.load %offset_pointer_7 : !llvm.ptr -> f32
# CHECK-NEXT:     %8 = llvm.fadd %7, %6 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %bytes_per_element_2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_2 = llvm.mul %5, %bytes_per_element_2 : i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_9 = llvm.add %offset_pointer_8, %scaled_pointer_offset_2 : i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.inttoptr %offset_pointer_9 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %8, %offset_pointer_10 : f32, !llvm.ptr
# CHECK-NEXT:     %9 = llvm.add %3, %2 : i64
# CHECK-NEXT:     llvm.br ^bb1(%9 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def reduce_float(x: f32[8] @ DRAM, out: f32[1] @ DRAM):
    for i in seq(0, 8):
        out[0] += x[i]
