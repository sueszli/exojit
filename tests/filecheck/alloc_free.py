# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @alloc_free(%0: i64, %1: !llvm.ptr) {
# CHECK-NEXT:     %2 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%2 : i64)
# CHECK-NEXT:   ^bb1(%4: i64):
# CHECK-NEXT:     %5 = llvm.icmp "slt" %4, %0 : i64
# CHECK-NEXT:     llvm.cond_br %5, ^bb2, ^bb3
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %6 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %7 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %8 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %9 = llvm.mul %4, %8 : i64
# CHECK-NEXT:     %10 = llvm.getelementptr inbounds %1[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %11 = llvm.load %10 : !llvm.ptr -> f32
# CHECK-NEXT:     %12 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %13 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %14 = llvm.mul %12, %13 : i64
# CHECK-NEXT:     %15 = llvm.getelementptr inbounds %7[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %11, %15 : f32, !llvm.ptr
# CHECK-NEXT:     %16 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %17 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %18 = llvm.mul %16, %17 : i64
# CHECK-NEXT:     %19 = llvm.getelementptr inbounds %7[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %20 = llvm.load %19 : !llvm.ptr -> f32
# CHECK-NEXT:     %21 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %22 = llvm.mul %4, %21 : i64
# CHECK-NEXT:     %23 = llvm.getelementptr inbounds %1[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %20, %23 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.call @free(%7) : (!llvm.ptr) -> ()
# CHECK-NEXT:     %24 = llvm.add %4, %3 : i64
# CHECK-NEXT:     llvm.br ^bb1(%24 : i64)
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
