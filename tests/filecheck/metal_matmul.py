# REQUIRES: metal
# RUN: mkdir -p "%t dir;x"
# RUN: env TMPDIR="%t dir;x" %python %s | filecheck %s

# CHECK: target datalayout = "e-p:64:64:64-
# CHECK-NEXT: target triple = "air64_v27-apple-macosx15.0.0"
# CHECK: define void @"metal_matmul"(float addrspace(1)* nocapture noundef "air-buffer-no-alias" %{{[^,]+}}, float addrspace(1)* nocapture noundef "air-buffer-no-alias" %{{[^,]+}}, float addrspace(1)* nocapture noundef "air-buffer-no-alias" %{{[^,]+}}, i32 %{{[^,)]+}}) {
# CHECK-NEXT: {{.*}}:
# CHECK-NEXT: zext i32 %{{.*}} to i64
# CHECK-NEXT: udiv i64 %{{.*}}, 41
# CHECK-NEXT: urem i64 %{{.*}}, 41
# CHECK-NOT: ptr %
# CHECK-NOT: call
# CHECK: icmp slt i64 %{{.*}}, 7
# CHECK-NOT: ptr %
# CHECK-NOT: call
# CHECK: ret void
# CHECK: !air.kernel = !{!5}
# CHECK: !0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, {{.*}}, !"air.arg_name", !"out"}
# CHECK-NEXT: !1 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, {{.*}}, !"air.arg_name", !"A"}
# CHECK-NEXT: !2 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, {{.*}}, !"air.arg_name", !"B"}
# CHECK-NEXT: !3 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
# CHECK-NEXT: !4 = !{}
# CHECK-NEXT: !5 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"metal_matmul", !4, !{!0, !1, !2, !3}}
# CHECK-NEXT: !6 = !{i32 2, i32 7, i32 0}
# CHECK-NEXT: !7 = !{!"Metal", i32 3, i32 2, i32 0}
# CHECK-NOT: air.compile
# CHECK-NOT: air.source_file_name
# CHECK: gpu 19 -5 4 -10 -4

import os
import tempfile
from unittest.mock import patch

from exo import *

from backend_metal import execute_kernel, kernel_to_air
from exojit import jit


@proc
def metal_matmul(C: f32[37, 41] @ DRAM, A: f32[37, 7] @ DRAM, B: f32[7, 41] @ DRAM):
    for i in par(0, 37):
        for j in par(0, 41):
            C[i, j] = 0.0
            for k in seq(0, 7):
                C[i, j] += A[i, k] * B[k, j]


A = [[float((i + 2 * k) % 5 - 2) for k in range(7)] for i in range(37)]
B = [[float((3 * k + j) % 7 - 3) for j in range(41)] for k in range(7)]
# nonzero sentinel catches missing zeroing
C_gpu = [[0.5] * 41 for _ in range(37)]

# temp files must honor TMPDIR
assert os.path.samefile(tempfile.gettempdir(), os.environ.get("TMPDIR", tempfile.gettempdir()))
print(kernel_to_air(metal_matmul))

# wraps keeps the real GPU submission
with patch("backend_metal.execute_kernel", wraps=execute_kernel) as dispatch:
    jit(metal_matmul, backend="metal")(C_gpu, A, B)

assert dispatch.call_count == 1
queue, pipeline, grid, group, _ = dispatch.call_args.args
assert queue.device().name().startswith("Apple") and pipeline.threadExecutionWidth() > 1
assert (grid.width, grid.height, grid.depth) == (37 * 41, 1, 1) and group.width > 1

# independent reference, not the backend
reference = [[sum(A[i][k] * B[k][j] for k in range(7)) for j in range(41)] for i in range(37)]
assert C_gpu == reference
print("gpu", *(f"{v:g}" for v in (C_gpu[0][0], C_gpu[0][40], C_gpu[36][0], C_gpu[36][40], sum(map(sum, C_gpu)))))
