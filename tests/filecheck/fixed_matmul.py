# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @fixed_matmul(%0: !llvm.ptr, %1: !llvm.ptr, %2: !llvm.ptr) {
# CHECK-NEXT:     %3 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %5 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb1(%3 : i64)
# CHECK-NEXT:   ^bb1(%6: i64):
# CHECK-NEXT:     %7 = llvm.icmp "slt" %6, %4 : i64
# CHECK-NEXT:     llvm.cond_br %7, ^bb2, ^bb9
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     %8 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %9 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %10 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb3(%8 : i64)
# CHECK-NEXT:   ^bb3(%11: i64):
# CHECK-NEXT:     %12 = llvm.icmp "slt" %11, %9 : i64
# CHECK-NEXT:     llvm.cond_br %12, ^bb4, ^bb8
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %13 = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     %14 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %15 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %16 = llvm.mul %14, %15 : i64
# CHECK-NEXT:     %17 = llvm.mul %6, %16 : i64
# CHECK-NEXT:     %18 = llvm.mul %11, %14 : i64
# CHECK-NEXT:     %19 = llvm.add %17, %18 : i64
# CHECK-NEXT:     %20 = llvm.getelementptr inbounds %0[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %13, %20 : f32, !llvm.ptr
# CHECK-NEXT:     %21 = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     %22 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %23 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br ^bb5(%21 : i64)
# CHECK-NEXT:   ^bb5(%24: i64):
# CHECK-NEXT:     %25 = llvm.icmp "slt" %24, %22 : i64
# CHECK-NEXT:     llvm.cond_br %25, ^bb6, ^bb7
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     %26 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %27 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %28 = llvm.mul %26, %27 : i64
# CHECK-NEXT:     %29 = llvm.mul %6, %28 : i64
# CHECK-NEXT:     %30 = llvm.mul %24, %26 : i64
# CHECK-NEXT:     %31 = llvm.add %29, %30 : i64
# CHECK-NEXT:     %32 = llvm.getelementptr inbounds %1[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %33 = llvm.load %32 : !llvm.ptr -> f32
# CHECK-NEXT:     %34 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %35 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %36 = llvm.mul %34, %35 : i64
# CHECK-NEXT:     %37 = llvm.mul %24, %36 : i64
# CHECK-NEXT:     %38 = llvm.mul %11, %34 : i64
# CHECK-NEXT:     %39 = llvm.add %37, %38 : i64
# CHECK-NEXT:     %40 = llvm.getelementptr inbounds %2[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %41 = llvm.load %40 : !llvm.ptr -> f32
# CHECK-NEXT:     %42 = llvm.fmul %33, %41 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %43 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %44 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %45 = llvm.mul %43, %44 : i64
# CHECK-NEXT:     %46 = llvm.mul %6, %45 : i64
# CHECK-NEXT:     %47 = llvm.mul %11, %43 : i64
# CHECK-NEXT:     %48 = llvm.add %46, %47 : i64
# CHECK-NEXT:     %49 = llvm.getelementptr inbounds %0[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     %50 = llvm.load %49 : !llvm.ptr -> f32
# CHECK-NEXT:     %51 = llvm.fadd %50, %42 {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     %52 = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     %53 = llvm.mlir.constant(16) : i64
# CHECK-NEXT:     %54 = llvm.mul %52, %53 : i64
# CHECK-NEXT:     %55 = llvm.mul %6, %54 : i64
# CHECK-NEXT:     %56 = llvm.mul %11, %52 : i64
# CHECK-NEXT:     %57 = llvm.add %55, %56 : i64
# CHECK-NEXT:     %58 = llvm.getelementptr inbounds %0[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
# CHECK-NEXT:     llvm.store %51, %58 : f32, !llvm.ptr
# CHECK-NEXT:     %59 = llvm.add %24, %23 : i64
# CHECK-NEXT:     llvm.br ^bb5(%59 : i64)
# CHECK-NEXT:   ^bb7:
# CHECK-NEXT:     %60 = llvm.add %11, %10 : i64
# CHECK-NEXT:     llvm.br ^bb3(%60 : i64)
# CHECK-NEXT:   ^bb8:
# CHECK-NEXT:     %61 = llvm.add %6, %5 : i64
# CHECK-NEXT:     llvm.br ^bb1(%61 : i64)
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
