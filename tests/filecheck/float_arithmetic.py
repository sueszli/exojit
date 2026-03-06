# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @float_arithmetic(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr, %offset_pointer_2 : !llvm.ptr) {
# CHECK:        {{.*}} = "llvm.load"(%offset_pointer_6) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        {{.*}} = "llvm.load"(%offset_pointer_9) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   {{.*}} = arith.addf {{.*}}, {{.*}} : f32
# CHECK:        {{.*}} = arith.subf {{.*}}, {{.*}} : f32
# CHECK:        {{.*}} = arith.mulf {{.*}}, {{.*}} : f32
# CHECK:        {{.*}} = arith.divf {{.*}}, {{.*}} : f32
@proc
def float_arithmetic(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]
