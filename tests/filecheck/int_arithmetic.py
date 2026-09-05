# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @int_arithmetic(%offset_pointer: !llvm.ptr, %offset_pointer_1: !llvm.ptr, %offset_pointer_2: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %0, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.add %offset_pointer_3, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.inttoptr %offset_pointer_4 : i64 to !llvm.ptr
# CHECK-NEXT:     %1 = llvm.load %offset_pointer_5 : !llvm.ptr -> i32
# CHECK-NEXT:     %offset_pointer_6 = llvm.ptrtoint %offset_pointer_2 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.add %offset_pointer_6, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.inttoptr %offset_pointer_7 : i64 to !llvm.ptr
# CHECK-NEXT:     %2 = llvm.load %offset_pointer_8 : !llvm.ptr -> i32
# CHECK-NEXT:     %3 = llvm.add %1, %2 : i32
# CHECK-NEXT:     %offset_pointer_9 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.add %offset_pointer_9, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_11 = llvm.inttoptr %offset_pointer_10 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %3, %offset_pointer_11 : i32, !llvm.ptr
# CHECK-NEXT:     %4 = llvm.load %offset_pointer_5 : !llvm.ptr -> i32
# CHECK-NEXT:     %5 = llvm.load %offset_pointer_8 : !llvm.ptr -> i32
# CHECK-NEXT:     %6 = llvm.sub %4, %5 : i32
# CHECK-NEXT:     llvm.store %6, %offset_pointer_11 : i32, !llvm.ptr
# CHECK-NEXT:     %7 = llvm.load %offset_pointer_5 : !llvm.ptr -> i32
# CHECK-NEXT:     %8 = llvm.load %offset_pointer_8 : !llvm.ptr -> i32
# CHECK-NEXT:     %9 = llvm.mul %7, %8 : i32
# CHECK-NEXT:     llvm.store %9, %offset_pointer_11 : i32, !llvm.ptr
# CHECK-NEXT:     %10 = llvm.load %offset_pointer_5 : !llvm.ptr -> i32
# CHECK-NEXT:     %11 = llvm.load %offset_pointer_8 : !llvm.ptr -> i32
# CHECK-NEXT:     %12 = llvm.sdiv %10, %11 : i32
# CHECK-NEXT:     llvm.store %12, %offset_pointer_11 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def int_arithmetic(out: i32[1] @ DRAM, x: i32[1] @ DRAM, y: i32[1] @ DRAM):
    out[0] = x[0] + y[0]
    out[0] = x[0] - y[0]
    out[0] = x[0] * y[0]
    out[0] = x[0] / y[0]
