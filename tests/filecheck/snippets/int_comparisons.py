# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import *


# CHECK:      func.func @int_comparisons(%0 : memref<1xi32, "DRAM">, %1 : i64, %2 : i64) {
# CHECK-NEXT:   %3 = arith.cmpi eq, %1, %2 : i64
# CHECK-NEXT:   scf.if %3 {
# CHECK:        %6 = arith.cmpi slt, %1, %2 : i64
# CHECK-NEXT:   scf.if %6 {
# CHECK:        %9 = arith.cmpi sgt, %1, %2 : i64
# CHECK-NEXT:   scf.if %9 {
@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
