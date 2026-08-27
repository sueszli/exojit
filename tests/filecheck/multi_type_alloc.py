# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @multi_type_alloc(%0: !llvm.ptr {llvm.noalias}, %1: !llvm.ptr {llvm.noalias}) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %3 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %4 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %5 = llvm.mlir.constant(3.140000e+00 : f32) : f32
# CHECK-NEXT:     %6 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %7 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %8 = llvm.mul %6, %7 : i64
# CHECK-NEXT:     %9 = llvm.getelementptr inbounds %3[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %5, %9 : f32, !llvm.ptr
# CHECK-NEXT:     %10 = llvm.mlir.constant(42 : i32) : i32
# CHECK-NEXT:     %11 = llvm.getelementptr inbounds %4[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %10, %11 : i32, !llvm.ptr
# CHECK-NEXT:     %12 = llvm.load %9 : !llvm.ptr -> f32
# CHECK-NEXT:     %13 = llvm.getelementptr inbounds %0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %12, %13 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%3) : (!llvm.ptr) -> ()
# CHECK-NEXT:     %14 = llvm.load %11 : !llvm.ptr -> i32
# CHECK-NEXT:     %15 = llvm.getelementptr inbounds %1[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK-NEXT:     llvm.store %14, %15 : i32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%4) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def multi_type_alloc(out_f: f32[1] @ DRAM, out_i: i32[1] @ DRAM):
    tmp_f: f32
    tmp_i: i32
    tmp_f = 3.14
    tmp_i = 42
    out_f[0] = tmp_f
    out_i[0] = tmp_i
