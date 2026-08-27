# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @set_first(%0: !llvm.ptr) {
# CHECK-NEXT:     %1 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:     %3 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %4 = llvm.mul %1, %3 : i64
# CHECK-NEXT:     %5 = llvm.getelementptr inbounds %0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %2, %5 : f32, !llvm.ptr
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @inner(%6: !llvm.ptr) {
# CHECK-NEXT:     %7 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %8 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %9 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %10 = llvm.mul %7, %9 : i64
# CHECK-NEXT:     %11 = llvm.mul %7, %10 : i64
# CHECK-NEXT:     %12 = llvm.mul %8, %7 : i64
# CHECK-NEXT:     %13 = llvm.add %11, %12 : i64
# CHECK-NEXT:     %14 = llvm.mul %13, %9 : i64
# CHECK-NEXT:     %15 = llvm.ptrtoint %6 : !llvm.ptr to i64
# CHECK-NEXT:     %16 = llvm.add %15, %14 : i64
# CHECK-NEXT:     %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @set_first(%17) : (!llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @outer(%18: !llvm.ptr) {
# CHECK-NEXT:     %19 = llvm.mlir.constant(2) : i64
# CHECK-NEXT:     %20 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %21 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %22 = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     %23 = llvm.mul %21, %22 : i64
# CHECK-NEXT:     %24 = llvm.mul %23, %22 : i64
# CHECK-NEXT:     %25 = llvm.mul %19, %24 : i64
# CHECK-NEXT:     %26 = llvm.mul %20, %23 : i64
# CHECK-NEXT:     %27 = llvm.add %25, %26 : i64
# CHECK-NEXT:     %28 = llvm.mul %20, %21 : i64
# CHECK-NEXT:     %29 = llvm.add %27, %28 : i64
# CHECK-NEXT:     %30 = llvm.mul %29, %22 : i64
# CHECK-NEXT:     %31 = llvm.ptrtoint %18 : !llvm.ptr to i64
# CHECK-NEXT:     %32 = llvm.add %31, %30 : i64
# CHECK-NEXT:     %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
# CHECK-NEXT:     llvm.call @inner(%33) : (!llvm.ptr) -> ()
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
