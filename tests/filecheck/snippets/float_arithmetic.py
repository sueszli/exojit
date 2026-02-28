# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import *


# CHECK:      func.func @float_arithmetic(%0 : memref<1xf32, "DRAM">, %1 : memref<1xf32, "DRAM">, %2 : memref<1xf32, "DRAM">) {
# CHECK-NEXT:   %3 = arith.constant 0 : i64
# CHECK-NEXT:   %4 = exo.read %1[%3] -> f32
# CHECK-NEXT:   %5 = exo.read %2[%3] -> f32
# CHECK-NEXT:   %6 = arith.addf %4, %5 : f32
# CHECK:        %9 = arith.subf %7, %8 : f32
# CHECK:        %12 = arith.mulf %10, %11 : f32
# CHECK:        %15 = arith.divf %13, %14 : f32
@proc
def float_arithmetic(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]
