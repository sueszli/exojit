# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @set_first(%offset_pointer: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %0, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.add %offset_pointer_1, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.inttoptr %offset_pointer_2 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %1, %offset_pointer_3 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @inner(%offset_pointer_4: !llvm.ptr) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %c4 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %increment = llvm.mul %c4, %2 : i64
# CHECK-NEXT:     %subview = llvm.add %increment, %3 : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %subview, %c4 : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.ptrtoint %offset_pointer_4 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.add %offset_pointer_5, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.inttoptr %offset_pointer_6 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @set_first(%offset_pointer_7) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @outer(%offset_pointer_8: !llvm.ptr) {
# CHECK-NEXT:     %4 = llvm.mlir.constant(2) : i64
# CHECK-NEXT:     %5 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %c16 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %increment_1 = llvm.mul %c16, %4 : i64
# CHECK-NEXT:     %c4_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %increment_2 = llvm.mul %c4_1, %5 : i64
# CHECK-NEXT:     %subview_1 = llvm.add %increment_1, %increment_2 : i64
# CHECK-NEXT:     %subview_2 = llvm.add %subview_1, %5 : i64
# CHECK-NEXT:     %scaled_pointer_offset_2 = llvm.mul %subview_2, %c4_1 : i64
# CHECK-NEXT:     %offset_pointer_9 = llvm.ptrtoint %offset_pointer_8 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.add %offset_pointer_9, %scaled_pointer_offset_2 : i64
# CHECK-NEXT:     %offset_pointer_11 = llvm.inttoptr %offset_pointer_10 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @inner(%offset_pointer_11) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def set_first(x: [f32][4] @ DRAM):
    x[0] = 1.0


@proc
def inner(A: [f32][4, 4] @ DRAM):
    set_first(A[1, :])


@proc
def outer(A: f32[4, 4, 4] @ DRAM):
    inner(A[2, :, :])
