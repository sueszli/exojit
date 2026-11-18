# REQUIRES: metal
# RUN: %python %s | filecheck %s

# CHECK: call 1 gpu 1 2; 2 2; 1 0.25
# CHECK-NEXT: call 2 gpu 1 2; 2 1.5; 0 1.25

from exo import *

from exojit import jit


@proc
def metal_accumulate(C: f32[3, 2] @ DRAM, A: f32[3, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 3):
        for j in par(0, 2):
            for k in seq(0, i):
                C[i, j] += -(A[i, k] - B[k, j]) / 2.0


C_cpu, C_gpu, reference = ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] for _ in range(3))
cpu, gpu = jit(metal_accumulate), jit(metal_accumulate, backend="metal")
for call, (A, B) in enumerate((([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, -1.0], [2.0, 0.5]]), ([[-1.0, 0.0], [2.0, 1.0], [0.0, 3.0]], [[2.0, 1.0], [-1.0, 4.0]])), 1):
    cpu(C_cpu, A, B)
    gpu(C_gpu, A, B)
    reference = [[reference[i][j] + sum(-(A[i][k] - B[k][j]) / 2.0 for k in range(i)) for j in range(2)] for i in range(3)]
    assert C_gpu == C_cpu == reference, (call, C_gpu, C_cpu, reference)
    print("call", call, "gpu", "; ".join(" ".join(f"{v:g}" for v in row) for row in C_gpu))
