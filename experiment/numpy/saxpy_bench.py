# /// script
# requires-python = ">=3.11"
# dependencies = ["xdsl-exo"]
# [tool.uv.sources]
# xdsl-exo = { path = "../.." }
# ///

"""
SAXPY benchmark: Exo NEON JIT vs NumPy.

y += a * x  (BLAS Level-1, N=1024 f32 elements)

Four JIT kernel variants with progressive optimization:
  v0_naive        — scalar loop
  v1_vectorized   — 4-wide NEON FMA (256 iterations)
  v2_unrolled_4x  — 4x unrolled NEON FMA (64 iterations, 16 elements/iter)
  v3_unrolled_8x  — 8x unrolled NEON FMA (32 iterations, 32 elements/iter)

JIT kernels use {name}_repeat wrappers (native repeat loop) to eliminate
ctypes call overhead from measurements. LLVM O2 inlines the kernel into
the repeat loop for zero per-iteration overhead.
"""

from __future__ import annotations

import platform
import time

import numpy as np
from exo import *

from xdsl_exo.main import compile_procs
from xdsl_exo.patches_exo import NEON
from xdsl_exo.patches_llvmlite import emit_assembly, jit_compile

N = 1024
WARMUP = 5
REPEATS = 50
BATCH = 1000  # iterations per timing sample (high count → stable measurements)


# ── NEON intrinsic declarations ──────────────────────────────────────────


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ NEON, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


# ── SAXPY kernel variants ───────────────────────────────────────────────


@proc
def v0_naive(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    for i in seq(0, 1024):
        y[i] += a[0] * x[i]


@proc
def v1_vectorized(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    a_vec: f32[4] @ NEON
    neon_broadcast_f32x4(a_vec, a[0:1])
    for i in seq(0, 256):
        x_vec: f32[4] @ NEON
        y_vec: f32[4] @ NEON
        neon_loadu_f32x4(x_vec, x[4 * i : 4 * i + 4])
        neon_loadu_f32x4(y_vec, y[4 * i : 4 * i + 4])
        neon_fmadd_f32x4(y_vec, a_vec, x_vec)
        neon_storeu_f32x4(y[4 * i : 4 * i + 4], y_vec)


@proc
def v2_unrolled_4x(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    a_vec: f32[4] @ NEON
    neon_broadcast_f32x4(a_vec, a[0:1])
    for i in seq(0, 64):
        x0: f32[4] @ NEON
        x1: f32[4] @ NEON
        x2: f32[4] @ NEON
        x3: f32[4] @ NEON
        y0: f32[4] @ NEON
        y1: f32[4] @ NEON
        y2: f32[4] @ NEON
        y3: f32[4] @ NEON
        neon_loadu_f32x4(x0, x[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(x1, x[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(x2, x[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(x3, x[16 * i + 12 : 16 * i + 16])
        neon_loadu_f32x4(y0, y[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(y1, y[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(y2, y[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(y3, y[16 * i + 12 : 16 * i + 16])
        neon_fmadd_f32x4(y0, a_vec, x0)
        neon_fmadd_f32x4(y1, a_vec, x1)
        neon_fmadd_f32x4(y2, a_vec, x2)
        neon_fmadd_f32x4(y3, a_vec, x3)
        neon_storeu_f32x4(y[16 * i + 0 : 16 * i + 4], y0)
        neon_storeu_f32x4(y[16 * i + 4 : 16 * i + 8], y1)
        neon_storeu_f32x4(y[16 * i + 8 : 16 * i + 12], y2)
        neon_storeu_f32x4(y[16 * i + 12 : 16 * i + 16], y3)


@proc
def v3_unrolled_8x(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    a_vec: f32[4] @ NEON
    neon_broadcast_f32x4(a_vec, a[0:1])
    for i in seq(0, 32):
        x0: f32[4] @ NEON
        x1: f32[4] @ NEON
        x2: f32[4] @ NEON
        x3: f32[4] @ NEON
        x4: f32[4] @ NEON
        x5: f32[4] @ NEON
        x6: f32[4] @ NEON
        x7: f32[4] @ NEON
        y0: f32[4] @ NEON
        y1: f32[4] @ NEON
        y2: f32[4] @ NEON
        y3: f32[4] @ NEON
        y4: f32[4] @ NEON
        y5: f32[4] @ NEON
        y6: f32[4] @ NEON
        y7: f32[4] @ NEON
        neon_loadu_f32x4(x0, x[32 * i + 0 : 32 * i + 4])
        neon_loadu_f32x4(x1, x[32 * i + 4 : 32 * i + 8])
        neon_loadu_f32x4(x2, x[32 * i + 8 : 32 * i + 12])
        neon_loadu_f32x4(x3, x[32 * i + 12 : 32 * i + 16])
        neon_loadu_f32x4(x4, x[32 * i + 16 : 32 * i + 20])
        neon_loadu_f32x4(x5, x[32 * i + 20 : 32 * i + 24])
        neon_loadu_f32x4(x6, x[32 * i + 24 : 32 * i + 28])
        neon_loadu_f32x4(x7, x[32 * i + 28 : 32 * i + 32])
        neon_loadu_f32x4(y0, y[32 * i + 0 : 32 * i + 4])
        neon_loadu_f32x4(y1, y[32 * i + 4 : 32 * i + 8])
        neon_loadu_f32x4(y2, y[32 * i + 8 : 32 * i + 12])
        neon_loadu_f32x4(y3, y[32 * i + 12 : 32 * i + 16])
        neon_loadu_f32x4(y4, y[32 * i + 16 : 32 * i + 20])
        neon_loadu_f32x4(y5, y[32 * i + 20 : 32 * i + 24])
        neon_loadu_f32x4(y6, y[32 * i + 24 : 32 * i + 28])
        neon_loadu_f32x4(y7, y[32 * i + 28 : 32 * i + 32])
        neon_fmadd_f32x4(y0, a_vec, x0)
        neon_fmadd_f32x4(y1, a_vec, x1)
        neon_fmadd_f32x4(y2, a_vec, x2)
        neon_fmadd_f32x4(y3, a_vec, x3)
        neon_fmadd_f32x4(y4, a_vec, x4)
        neon_fmadd_f32x4(y5, a_vec, x5)
        neon_fmadd_f32x4(y6, a_vec, x6)
        neon_fmadd_f32x4(y7, a_vec, x7)
        neon_storeu_f32x4(y[32 * i + 0 : 32 * i + 4], y0)
        neon_storeu_f32x4(y[32 * i + 4 : 32 * i + 8], y1)
        neon_storeu_f32x4(y[32 * i + 8 : 32 * i + 12], y2)
        neon_storeu_f32x4(y[32 * i + 12 : 32 * i + 16], y3)
        neon_storeu_f32x4(y[32 * i + 16 : 32 * i + 20], y4)
        neon_storeu_f32x4(y[32 * i + 20 : 32 * i + 24], y5)
        neon_storeu_f32x4(y[32 * i + 24 : 32 * i + 28], y6)
        neon_storeu_f32x4(y[32 * i + 28 : 32 * i + 32], y7)


# ── JIT compilation helpers ──────────────────────────────────────────────

KERNELS = [v0_naive, v1_vectorized, v2_unrolled_4x, v3_unrolled_8x]

_cache: dict[str, tuple] = {}


def _get_fns(p):
    """Return (single_call_fn, repeat_fn) for the given proc."""
    name = p.name()
    if name not in _cache:
        fns = jit_compile(compile_procs(p))
        _cache[name] = (fns[name], fns[f"{name}_repeat"])
    return _cache[name]


def bench(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ── Setup ────────────────────────────────────────────────────────────────

np.random.seed(42)
x = np.random.randn(N).astype(np.float32)
y_orig = np.random.randn(N).astype(np.float32)
a_val = np.float32(2.5)
a_arr = np.array([a_val], dtype=np.float32)
expected = y_orig + a_val * x

# ── Correctness verification ────────────────────────────────────────────

print("=" * 60)
print("CORRECTNESS VERIFICATION")
print("=" * 60)

for k in KERNELS:
    y_test = y_orig.copy()
    fn, _ = _get_fns(k)
    fn(y_test.ctypes.data, x.ctypes.data, a_arr.ctypes.data)
    assert np.allclose(y_test, expected, atol=1e-5), f"{k.name()} FAILED"
    print(f"  {k.name():20s} OK")

# ── Assembly verification ────────────────────────────────────────────────

arch = platform.machine()
if arch in ("aarch64", "arm64"):
    print()
    print("=" * 60)
    print("ASSEMBLY VERIFICATION (v1_vectorized)")
    print("=" * 60)
    module = compile_procs(v1_vectorized)
    asm = emit_assembly(module)
    for line in asm.split("\n"):
        s = line.strip()
        if s and not s.startswith(".") and not s.startswith(";"):
            print(f"  {line}")
    neon_patterns = ["fmla.4s", "fadd.4s", ".4s", "ldr\tq", "str\tq", "ldp\tq", "stp\tq"]
    found = [p for p in neon_patterns if p in asm]
    print(f"\n  NEON patterns found: {found}")
    assert len(found) > 0, "No NEON instructions in assembly!"
    print("  VERIFIED: JIT code uses NEON vector instructions")

# ── Benchmark ────────────────────────────────────────────────────────────

print()
print("=" * 60)
print(f"SAXPY BENCHMARK  (N={N}, {WARMUP} warmup, {REPEATS} repeats, median)")
print("=" * 60)

results: list[tuple[str, float]] = []

# JIT variants: use _repeat wrappers (1 ctypes call runs BATCH iterations natively)
for k in KERNELS:
    _, fn_repeat = _get_fns(k)
    y_bench = y_orig.copy()
    y_ptr, x_ptr, a_ptr = y_bench.ctypes.data, x.ctypes.data, a_arr.ctypes.data
    t = bench(lambda fn=fn_repeat, y=y_ptr, xp=x_ptr, ap=a_ptr: fn(y, xp, ap, BATCH))
    results.append((k.name(), t / BATCH))

# NumPy: y += a * x (Python loop — each iteration pays numpy dispatch overhead)
y_np = y_orig.copy()


def _np_saxpy_batch(y=y_np, a=a_val, xv=x, n=BATCH):
    for _ in range(n):
        y += a * xv


t_np = bench(_np_saxpy_batch)
results.append(("numpy (y+=a*x)", t_np / BATCH))

# Print results table
np_time = results[-1][1]

# SAXPY: 2 FLOPs per element (multiply + add), read x + read y + write y = 3 * 4 bytes per element
flops_per_call = 2 * N
bytes_per_call = 3 * N * 4  # read x, read y, write y (f32)

print()
print(f"  {'Variant':<20s}  {'Time (ns)':>10s}  {'vs numpy':>10s}  {'GFLOP/s':>8s}  {'BW (GB/s)':>10s}")
print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 10}")
for name, t in results:
    speedup = np_time / t if t > 0 else float("inf")
    gflops = flops_per_call / t / 1e9 if t > 0 else float("inf")
    bw_gbs = bytes_per_call / t / 1e9 if t > 0 else float("inf")
    print(f"  {name:<20s}  {t * 1e9:10.1f}  {speedup:9.1f}x  {gflops:7.1f}  {bw_gbs:9.1f}")

# Theoretical peak analysis
best_name, best_time = min(results[:-1], key=lambda x: x[1])  # best JIT variant
best_ns = best_time * 1e9
best_gflops = flops_per_call / best_time / 1e9
best_bw = bytes_per_call / best_time / 1e9

print()
print("=" * 60)
print("THEORETICAL PEAK ANALYSIS")
print("=" * 60)
print(f"  Best JIT variant:     {best_name} ({best_ns:.1f} ns)")
print(f"  Throughput:           {best_gflops:.1f} GFLOP/s, {best_bw:.1f} GB/s")
print()
print("  Apple Silicon NEON constraints (per cycle @ ~3.5 GHz):")
print("    - 4 NEON execution units (2 FMLA throughput)")
print("    - 3 load ports, 2 store ports")
print("    - SAXPY needs: 2 loads + 1 store + 1 FMA per 4 elements")
print("    - Bottleneck: load ports (3/cycle) → 1.5 FMA/cycle → 6 elements/cycle")
print()
# Theoretical minimum: N elements / (6 elements/cycle) / freq
freq_ghz = 3.5  # conservative Apple Silicon P-core estimate
elems_per_cycle = 6.0  # load-port limited: 3 loads/cycle, each FMA needs 2 loads
cycles = N / elems_per_cycle
theoretical_ns = cycles / freq_ghz
print(f"  Theoretical minimum:  {theoretical_ns:.1f} ns ({N} elems / {elems_per_cycle:.0f} elems/cycle / {freq_ghz} GHz)")
efficiency = theoretical_ns / best_ns * 100
print(f"  Achieved efficiency:  {efficiency:.0f}% of theoretical peak")
print(f"  Speedup vs NumPy:     {np_time / best_time:.1f}x")
