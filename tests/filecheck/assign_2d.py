# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @assign_2d(%offset_pointer: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%0 : i64)
# CHECK-NEXT:   ^bb1(%3: i64):
# CHECK-NEXT:     %4 = llvm.icmp "slt" %3, %1 : i64
# CHECK-NEXT:     llvm.cond_br %4, ^bb2, ^bb6
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %5 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %6 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %7 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb3(%5 : i64)
# CHECK-NEXT:   ^bb3(%8: i64):
# CHECK-NEXT:     %9 = llvm.icmp "slt" %8, %6 : i64
# CHECK-NEXT:     llvm.cond_br %9, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %10 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %pointer_dim_stride = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %pointer_dim_offset = llvm.mul %3, %pointer_dim_stride : i64
# CHECK-NEXT:     %pointer_dim_stride_1 = llvm.add %pointer_dim_offset, %8 : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %pointer_dim_stride_1, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_1 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_2 = llvm.add %offset_pointer_1, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.inttoptr %offset_pointer_2 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %10, %offset_pointer_3 : f32, !llvm.ptr
# CHECK-NEXT:     %11 = llvm.add %8, %7 : i64
# CHECK-NEXT:     llvm.br ^bb3(%11 : i64)
# CHECK-NEXT:   ^bb5:
# CHECK-NEXT:     %12 = llvm.add %3, %2 : i64
# CHECK-NEXT:     llvm.br ^bb1(%12 : i64)
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def assign_2d(x: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            x[i, j] = 0.0
