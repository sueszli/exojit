# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import *


# CHECK:      func.func @index_modulo(%0 : memref<10xi32, "DRAM">, %1 : i64) {
# CHECK-NEXT:   %2 = arith.constant 10 : i64
# CHECK-NEXT:   %3 = arith.remsi %1, %2 : i64
# CHECK-NEXT:   %4 = arith.constant 42 : i32
# CHECK-NEXT:   exo.assign %4, %0[%3]
@proc
def index_modulo(out: i32[10] @ DRAM, n: index):
    assert n >= 0
    assert n < 10
    out[n % 10] = 42
