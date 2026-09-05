# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @f64_arithmetic(%offset_pointer: !llvm.ptr, %offset_pointer_1: !llvm.ptr, %offset_pointer_2: !llvm.ptr) {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %bytes_per_element = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     %scaled_pointer_offset = llvm.mul %0, %bytes_per_element : i64
# CHECK-NEXT:     %offset_pointer_3 = llvm.ptrtoint %offset_pointer_1 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_4 = llvm.add %offset_pointer_3, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_5 = llvm.inttoptr %offset_pointer_4 : i64 to !llvm.ptr
# CHECK-NEXT:     %1 = llvm.load %offset_pointer_5 : !llvm.ptr -> f64
# CHECK-NEXT:     %offset_pointer_6 = llvm.ptrtoint %offset_pointer_2 : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_7 = llvm.add %offset_pointer_6, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_8 = llvm.inttoptr %offset_pointer_7 : i64 to !llvm.ptr
# CHECK-NEXT:     %2 = llvm.load %offset_pointer_8 : !llvm.ptr -> f64
# CHECK-NEXT:     %3 = llvm.fadd %1, %2 {fastmathFlags = #llvm.fastmath<fast>} : f64
# CHECK-NEXT:     %offset_pointer_9 = llvm.ptrtoint %offset_pointer : !llvm.ptr to i64
# CHECK-NEXT:     %offset_pointer_10 = llvm.add %offset_pointer_9, %scaled_pointer_offset : i64
# CHECK-NEXT:     %offset_pointer_11 = llvm.inttoptr %offset_pointer_10 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.store %3, %offset_pointer_11 : f64, !llvm.ptr
# CHECK-NEXT:     %4 = llvm.load %offset_pointer_5 : !llvm.ptr -> f64
# CHECK-NEXT:     %5 = llvm.load %offset_pointer_8 : !llvm.ptr -> f64
# CHECK-NEXT:     %6 = llvm.fmul %4, %5 {fastmathFlags = #llvm.fastmath<fast>} : f64
# CHECK-NEXT:     llvm.store %6, %offset_pointer_11 : f64, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def f64_arithmetic(out: f64[1] @ DRAM, a: f64[1] @ DRAM, b: f64[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] * b[0]
