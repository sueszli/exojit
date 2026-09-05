# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @fixed_matmul(%offset_pointer: !llvm.ptr, %offset_pointer_1: !llvm.ptr, %offset_pointer_2: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%0 : i64)
# CHECK-NEXT:   ^bb1(%3: i64):
# CHECK-NEXT:     %4 = llvm.icmp "slt" %3, %1 : i64
# CHECK-NEXT:     llvm.cond_br %4, ^bb2, ^bb9
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %5 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %6 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %7 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb3(%5 : i64)
# CHECK-NEXT:   ^bb3(%8: i64):
# CHECK-NEXT:     %9 = llvm.icmp "slt" %8, %6 : i64
# CHECK-NEXT:     llvm.cond_br %9, ^bb4, ^bb8
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %10 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %pointer_dim_stride = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %pointer_dim_offset = llvm.mul %3, %pointer_dim_stride : i64
# CHECK-NEXT:     %pointer_dim_stride_1 = llvm.add %pointer_dim_offset, %8 : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %pointer_dim_stride_1, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.add %offset_pointer_3, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.inttoptr %offset_pointer_4 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %10, %offset_pointer_5 : f32, !llvm.ptr
# CHECK-NEXT:     %11 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %12 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %13 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb5(%11 : i64)
# CHECK-NEXT:   ^bb5(%14: i64):
# CHECK-NEXT:     %15 = llvm.icmp "slt" %14, %12 : i64
# CHECK-NEXT:     llvm.cond_br %15, ^bb6, ^bb7
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     %pointer_dim_stride_2 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %pointer_dim_offset_1 = llvm.mul %3, %pointer_dim_stride_2 : i64
# CHECK-NEXT:     %pointer_dim_stride_3 = llvm.add %pointer_dim_offset_1, %14 : i64
# CHECK-NEXT:     %bytes_per_element_1 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_1 = llvm.mul %pointer_dim_stride_3, %bytes_per_element_1 : i64
# CHECK-NEXT:     %offset_pointer_6 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.add %offset_pointer_6, %scaled_pointer_offset_1 : i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.inttoptr %offset_pointer_7 : i64 to !llvm.ptr
# CHECK-NEXT:     %16 = llvm.load %offset_pointer_8 : !llvm.ptr -> f32
# CHECK-NEXT:     %pointer_dim_stride_4 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %pointer_dim_offset_2 = llvm.mul %14, %pointer_dim_stride_4 : i64
# CHECK-NEXT:     %pointer_dim_stride_5 = llvm.add %pointer_dim_offset_2, %8 : i64
# CHECK-NEXT:     %bytes_per_element_2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_2 = llvm.mul %pointer_dim_stride_5, %bytes_per_element_2 : i64
# CHECK-NEXT:     %offset_pointer_9 = llvm.ptrtoint %offset_pointer_2 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.add %offset_pointer_9, %scaled_pointer_offset_2 : i64
# CHECK-NEXT:     %offset_pointer_11 = llvm.inttoptr %offset_pointer_10 : i64 to !llvm.ptr
# CHECK-NEXT:     %17 = llvm.load %offset_pointer_11 : !llvm.ptr -> f32
# CHECK-NEXT:     %18 = llvm.fmul %16, %17 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %pointer_dim_stride_6 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %pointer_dim_offset_3 = llvm.mul %3, %pointer_dim_stride_6 : i64
# CHECK-NEXT:     %pointer_dim_stride_7 = llvm.add %pointer_dim_offset_3, %8 : i64
# CHECK-NEXT:     %bytes_per_element_3 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_3 = llvm.mul %pointer_dim_stride_7, %bytes_per_element_3 : i64
# CHECK-NEXT:     %offset_pointer_12 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_13 = llvm.add %offset_pointer_12, %scaled_pointer_offset_3 : i64
# CHECK-NEXT:     %offset_pointer_14 = llvm.inttoptr %offset_pointer_13 : i64 to !llvm.ptr
# CHECK-NEXT:     %19 = llvm.load %offset_pointer_14 : !llvm.ptr -> f32
# CHECK-NEXT:     %20 = llvm.fadd %19, %18 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %pointer_dim_stride_8 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %pointer_dim_offset_4 = llvm.mul %3, %pointer_dim_stride_8 : i64
# CHECK-NEXT:     %pointer_dim_stride_9 = llvm.add %pointer_dim_offset_4, %8 : i64
# CHECK-NEXT:     %bytes_per_element_4 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset_4 = llvm.mul %pointer_dim_stride_9, %bytes_per_element_4 : i64
# CHECK-NEXT:     %offset_pointer_15 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_16 = llvm.add %offset_pointer_15, %scaled_pointer_offset_4 : i64
# CHECK-NEXT:     %offset_pointer_17 = llvm.inttoptr %offset_pointer_16 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %20, %offset_pointer_17 : f32, !llvm.ptr
# CHECK-NEXT:     %21 = llvm.add %14, %13 : i64
# CHECK-NEXT:     llvm.br ^bb5(%21 : i64)
# CHECK-NEXT:   ^bb7:
# CHECK-NEXT:     %22 = llvm.add %8, %7 : i64
# CHECK-NEXT:     llvm.br ^bb3(%22 : i64)
# CHECK-NEXT:   ^bb8:
# CHECK-NEXT:     %23 = llvm.add %3, %2 : i64
# CHECK-NEXT:     llvm.br ^bb1(%23 : i64)
# CHECK-NEXT:   ^bb9:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]
