# REQUIRES: metal
# RUN: %python %s | filecheck %s

# accumulates into own cell, id -> (id / 2, id % 2)
# CHECK: define void @metal_accumulate({{.*}}addrspace(1)*{{.*}}"air-buffer-no-alias"{{.*}}, i32 %{{[^,]*}}) #0 {
# CHECK: udiv i64
# CHECK: urem i64
# CHECK: !"air.read_write"
# CHECK: !"air.thread_position_in_grid"
# CHECK: call 1 gpu 6 2; 14 3; 22 4
# CHECK-NEXT: call 2 gpu 11 2; 25 2; 39 2

from exo import *

from backend_metal import kernel_to_air
from exojit import jit


@proc
def metal_accumulate(C: f32[3, 2] @ DRAM, A: f32[3, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 3):
        for j in par(0, 2):
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
B = [[1.0, -1.0], [2.0, 0.5]]
C_cpu, C_gpu, reference = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
print(kernel_to_air(metal_accumulate))
cpu, gpu = jit(metal_accumulate), jit(metal_accumulate, backend="metal")
for call in (1, 2):  # reused kernel sees prior results
    cpu(C_cpu, A, B)
    gpu(C_gpu, A, B)
    reference = [[reference[i][j] + sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)] for i in range(3)]
    print("call", call, "gpu", "; ".join(" ".join(f"{v:g}" for v in row) for row in C_gpu))
    assert C_gpu == C_cpu == reference, (call, C_gpu, C_cpu, reference)
