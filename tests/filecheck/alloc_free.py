# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @alloc_free(%0: i64, %offset_pointer: !llvm.ptr {exojit.dim.0 = 0 : i64}) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%1 : i64)
# CHECK-NEXT:   ^bb1(%3: i64):
# CHECK-NEXT:     %4 = llvm.icmp "slt" %3, %0 : i64
# CHECK-NEXT:     llvm.cond_br %4, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %5 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.call @malloc(%5) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %3, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.add %offset_pointer_2, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.inttoptr %offset_pointer_3 : i64 to !llvm.ptr
# CHECK-NEXT:     %6 = llvm.load %offset_pointer_4 : !llvm.ptr -> f32
# CHECK-NEXT:     %7 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %7, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %6, %offset_pointer_7 : f32, !llvm.ptr
# CHECK-NEXT:     %8 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element_2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_2 = llvm.mul %8, %bytes_per_element_2 : i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_9 = llvm.add %offset_pointer_8, %scaled_pointer_offset_2 : i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.inttoptr %offset_pointer_9 : i64 to !llvm.ptr
# CHECK-NEXT:     %9 = llvm.load %offset_pointer_10 : !llvm.ptr -> f32
# CHECK-NEXT:     %bytes_per_element_3 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_3 = llvm.mul %3, %bytes_per_element_3 : i64
# CHECK-NEXT:     %offset_pointer_11 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_12 = llvm.add %offset_pointer_11, %scaled_pointer_offset_3 : i64
# CHECK-NEXT:     %offset_pointer_13 = llvm.inttoptr %offset_pointer_12 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %9, %offset_pointer_13 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%offset_pointer_1) : (!llvm.ptr) -> ()
# CHECK-NEXT:     %10 = llvm.add %3, %2 : i64
# CHECK-NEXT:     llvm.br ^bb1(%10 : i64)
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def alloc_free(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp
