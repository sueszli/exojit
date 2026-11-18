# REQUIRES: metal
# RUN: mkdir -p "%t dir;x"
# RUN: env TMPDIR="%t dir;x" %python %s | filecheck %s

# CHECK: target triple = "air64_v27-apple-macosx
# CHECK: define void @metal_matmul({{.*}}addrspace(1)*{{.*}}"air-buffer-no-alias"{{.*}}, i32 %{{[^,]*}}) #0 {
# CHECK: udiv i64
# CHECK: urem i64
# CHECK-NOT: malloc
# CHECK: !air.kernel = !{
# CHECK: !0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read",
# CHECK-NEXT: !1 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write",
# CHECK-NEXT: !2 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read",
# CHECK-NEXT: !3 = !{i32 3, !"air.thread_position_in_grid",
# CHECK: gpu 19 -5 4 -10 -4

import os
import tempfile

from exo import *

from backend_metal import kernel_to_air
from exojit import jit


@proc
def metal_matmul(A: f32[37, 7] @ DRAM, C: f32[37, 41] @ DRAM, B: f32[7, 41] @ DRAM):
    for i in par(0, 37):
        for j in par(0, 41):
            C[i, j] = 0.0
            for k in seq(0, 7):
                C[i, j] += A[i, k] * B[k, j]


A = [[float((i + 2 * k) % 5 - 2) for k in range(7)] for i in range(37)]
B = [[float((3 * k + j) % 7 - 3) for j in range(41)] for k in range(7)]
C_cpu, C_gpu = [[0.5] * 41 for _ in range(37)], [[0.5] * 41 for _ in range(37)]
assert os.path.samefile(tempfile.gettempdir(), os.environ.get("TMPDIR", tempfile.gettempdir()))
print(kernel_to_air(metal_matmul))
jit(metal_matmul)(A, C_cpu, B)
jit(metal_matmul, backend="metal")(A, C_gpu, B)
reference = [[sum(A[i][k] * B[k][j] for k in range(7)) for j in range(41)] for i in range(37)]
assert C_gpu == C_cpu == reference
print("gpu", *(f"{v:g}" for v in (C_gpu[0][0], C_gpu[0][40], C_gpu[36][0], C_gpu[36][40], sum(map(sum, C_gpu)))))
