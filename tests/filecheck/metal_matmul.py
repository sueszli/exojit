# REQUIRES: metal
# RUN: %python %s | filecheck %s

# rectangular 2x3 grid, one gpu thread per output cell: id -> (id / 3, id % 3)
# CHECK: target triple = "air64_v27-apple-macosx
# CHECK: define void @metal_matmul({{.*}}addrspace(1)*{{.*}}"air-buffer-no-alias"{{.*}}, i32 %{{[^,]*}}) #0 {
# CHECK: zext i32
# CHECK: udiv i64
# CHECK: urem i64
# CHECK: !"air.read_write"
# CHECK: !"air.thread_position_in_grid"
# CHECK: !{i32 2, i32 7, i32 0}
# CHECK: gpu 12 1 10; 28 5 30
# CHECK-NEXT: gpu == cpu == reference

from exo import *

from backend_metal import kernel_to_air
from exojit import jit


@proc
def metal_matmul(C: f32[2, 3] @ DRAM, A: f32[2, 4] @ DRAM, B: f32[4, 3] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 3):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


A = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
B = [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0], [2.0, -1.0, 1.0]]
C_cpu, C_gpu = [[-7.0] * 3 for _ in range(2)], [[-7.0] * 3 for _ in range(2)]  # stale output, every cell must be overwritten
print(kernel_to_air(metal_matmul))
jit(metal_matmul)(C_cpu, A, B)
jit(metal_matmul, backend="metal")(C_gpu, A, B)
reference = [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(3)] for i in range(2)]
print("gpu", "; ".join(" ".join(f"{v:g}" for v in row) for row in C_gpu))
assert C_gpu == C_cpu == reference, (C_gpu, C_cpu, reference)
print("gpu == cpu == reference")
