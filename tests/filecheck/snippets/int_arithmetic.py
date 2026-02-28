# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import *


# CHECK:      func.func @int_arithmetic(%0 : memref<1xi32, "DRAM">, %1 : memref<1xi32, "DRAM">, %2 : memref<1xi32, "DRAM">) {
# CHECK-NEXT:   %3 = arith.constant 0 : i64
# CHECK-NEXT:   %4 = exo.read %1[%3] -> i32
# CHECK-NEXT:   %5 = exo.read %2[%3] -> i32
# CHECK-NEXT:   %6 = arith.addi %4, %5 : i32
# CHECK:        %9 = arith.subi %7, %8 : i32
# CHECK:        %12 = arith.muli %10, %11 : i32
# CHECK:        %15 = arith.divsi %13, %14 : i32
@proc
def int_arithmetic(out: i32[1] @ DRAM, a: i32[1] @ DRAM, b: i32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]
