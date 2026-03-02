# RUN: uv run xdsl-exo -o - %s | filecheck %s

# exercises: f64 type lowering (8-byte element width, arith ops on f64)
# lowering: f64 tensors → !llvm.ptr, loads/stores as f64, arith.addf/mulf on f64

from __future__ import annotations

from exo import *


# CHECK:      func.func @f64_arithmetic({{.*}}) {
# CHECK:        "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f64
# CHECK:        "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f64
# CHECK:        arith.addf {{.*}} : f64
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
# CHECK:        arith.mulf {{.*}} : f64
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
@proc
def f64_arithmetic(out: f64[1] @ DRAM, a: f64[1] @ DRAM, b: f64[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] * b[0]
