# RUN: uv run xdsl-exo -o - %s | filecheck %s

# exercises: unary negation for float (arith.negf) and int (0 - x via arith.subi)
# lowering: USub → arith.negf (float), arith.subi with zero (int)

from __future__ import annotations

from exo import *


# CHECK:      func.func @usub_float({{.*}}) {
# CHECK:        %2 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %3 = arith.negf %2 : f32
# CHECK:        "llvm.store"(%3, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
@proc
def usub_float(out: f32[1] @ DRAM, x: f32[1] @ DRAM):
    out[0] = -x[0]


# CHECK:      func.func @usub_int({{.*}}) {
# CHECK:        %2 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   %3 = arith.constant 0 : i32
# CHECK-NEXT:   %4 = arith.subi %3, %2 : i32
# CHECK:        "llvm.store"(%4, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
@proc
def usub_int(out: i32[1] @ DRAM, x: i32[1] @ DRAM):
    out[0] = -x[0]
