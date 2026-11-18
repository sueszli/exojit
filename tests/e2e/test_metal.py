import importlib.util
import os
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from exo import *
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i32

from backend_metal import MetalIRGenerator, thread_grid
from exojit import JITRuntime, jit, to_mlir

ROOT = Path(__file__).resolve().parents[2]
HAS_METAL = platform.system() == "Darwin" and platform.machine() == "arm64" and all(importlib.util.find_spec(name) is not None for name in ("Metal", "llvm_to_air"))
requires_metal = pytest.mark.skipif(not HAS_METAL, reason="needs apple silicon, pyobjc Metal and llvm_to_air on PYTHONPATH")


@proc
def matmul_par(C: f32[3, 5] @ DRAM, A: f32[3, 4] @ DRAM, B: f32[4, 5] @ DRAM):
    for i in par(0, 3):
        for j in par(0, 5):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


@proc
def outer_accumulate_par(C: f32[4, 3] @ DRAM, A: f32[4, 1] @ DRAM, B: f32[1, 3] @ DRAM):
    # reduction length 1 that accumulates into the incoming output values
    for i in par(0, 4):
        for j in par(0, 3):
            for k in seq(0, 1):
                C[i, j] += A[i, k] * B[k, j]


def rand_matrix(rng: np.random.Generator, rows: int, cols: int) -> list[list[float]]:
    return rng.uniform(-2.0, 2.0, (rows, cols)).astype(np.float32).tolist()


@requires_metal
@pytest.mark.parametrize(("p", "m", "k", "n"), [(matmul_par, 3, 4, 5), (outer_accumulate_par, 4, 1, 3)])
def test_metal_matches_cpu(p, m, k, n):
    rng = np.random.default_rng(0)
    A, B, C = rand_matrix(rng, m, k), rand_matrix(rng, k, n), rand_matrix(rng, m, n)
    A_before, B_before, C_before = deepcopy(A), deepcopy(B), deepcopy(C)
    C_cpu, C_gpu = deepcopy(C), deepcopy(C)
    jit(p)(C_cpu, A, B)
    jit(p, backend="metal")(C_gpu, A, B)
    assert A == A_before and B == B_before
    assert all(C_gpu[i][j] != C_before[i][j] for i in range(m) for j in range(n))
    expected = (0.0 if p is matmul_par else np.asarray(C_before)) + np.asarray(A) @ np.asarray(B)
    np.testing.assert_allclose(C_cpu, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(C_gpu, C_cpu, rtol=1e-5, atol=1e-5)


@requires_metal
def test_metal_kernel_is_reusable():
    fn = jit(matmul_par, backend="metal")
    rng = np.random.default_rng(1)
    for _ in range(2):
        A, B = rand_matrix(rng, 3, 4), rand_matrix(rng, 4, 5)
        C = [[0.0] * 5 for _ in range(3)]
        fn(C, A, B)
        np.testing.assert_allclose(C, np.asarray(A) @ np.asarray(B), rtol=1e-5, atol=1e-5)


@proc
def matmul_seq(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in seq(0, 2):
        for j in seq(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def matmul_dynamic(N: size, C: f32[N, N] @ DRAM, A: f32[N, N] @ DRAM, B: f32[N, N] @ DRAM):
    for i in par(0, N):
        for j in par(0, N):
            C[i, j] = 0.0
            for k in seq(0, N):
                C[i, j] += A[i, k] * B[k, j]


@proc
def matmul_f64(C: f64[2, 2] @ DRAM, A: f64[2, 2] @ DRAM, B: f64[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def matmul_alloc(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            acc: f32 @ DRAM
            acc = 0.0
            for k in seq(0, 2):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


@proc
def scale_matrix(x: f32[2, 2] @ DRAM):
    for a in seq(0, 2):
        for b in seq(0, 2):
            x[a, b] = x[a, b] * 0.5


@proc
def matmul_call(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            scale_matrix(C)
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def matmul_window(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            row = A[i, :]
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += row[k] * B[k, j]


@proc
def matmul_racy(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            for k in seq(0, 2):
                C[i, 0] += A[i, k] * B[k, j]


@proc
def matmul_transposed(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[j, i] = 0.0
            for k in seq(0, 2):
                C[j, i] += A[i, k] * B[k, j]


@proc
def matmul_neighbor_read(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = C[1 - i, j] + A[i, j] * B[i, j]


@proc
def matmul_nested_par(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in par(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def matmul_partial_grid(C: f32[4, 4] @ DRAM, A: f32[4, 4] @ DRAM, B: f32[4, 4] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 4):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


@proc
def two_outputs(C: f32[2, 2] @ DRAM, D: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = A[i, j]
            D[i, j] = A[i, j]


@proc
def fill_too_many_threads(C: f32[65536, 65536] @ DRAM):
    # 2**32 threads, one past the largest uint32 thread id
    for i in par(0, 65536):
        for j in par(0, 65536):
            C[i, j] = 0.0


@pytest.mark.parametrize(("p", "match"), [(matmul_seq, r"par\(0, M\)"), (matmul_dynamic, "static f32 tensor"), (matmul_f64, "static f32 tensor"), (matmul_alloc, "unsupported statement Alloc"), (matmul_call, "unsupported statement Call"), (matmul_window, "unsupported statement WindowStmt"), (matmul_racy, "may only write"), (matmul_transposed, "may only write"), (matmul_neighbor_read, "may only be read"), (matmul_nested_par, "nested inside the thread grid"), (matmul_partial_grid, "must span"), (two_outputs, "exactly one written argument"), (fill_too_many_threads, "uint32 thread id")])
def test_metal_rejects_unsupported_procs(p, match):
    # validation runs before compiling or importing the optional metal dependencies
    with pytest.raises(AssertionError, match=match):
        jit(p, backend="metal")


def test_metal_rejects_raw_and_unknown_backends():
    with pytest.raises(AssertionError, match="raw=True is not supported"):
        jit(matmul_par, raw=True, backend="metal")
    with pytest.raises(AssertionError, match="unsupported jit backend"):
        jit(matmul_par, backend="cuda")


def test_metal_generator_reuses_cpu_lowering():
    # the to_mlir generator seam: a single grid kernel with a trailing i32 thread id, no noalias attrs and no malloc/free
    _, row, col, _, cols = thread_grid(matmul_par._loopir_proc)
    funcs = [op for op in to_mlir(matmul_par, MetalIRGenerator(row, col, cols)).body.block.ops if isinstance(op, llvm.FuncOp)]
    assert [func.sym_name.data for func in funcs] == ["matmul_par"]
    assert funcs[0].function_type.inputs.data[-1] == i32 and funcs[0].properties.get("arg_attrs") is None
    cpu_funcs = [op for op in to_mlir(matmul_par).body.block.ops if isinstance(op, llvm.FuncOp)]
    assert sorted(func.sym_name.data for func in cpu_funcs) == ["free", "malloc", "matmul_par"]


def test_jit_runtime_compiler_seam():
    # a custom low-level compiler receives the converters' buffer addresses and the same ffi, and writes sync back
    def compiler(_proc, ffi):
        def run(*ptrs):
            assert all(isinstance(ptr, int) for ptr in ptrs)
            ffi.cast("float *", ptrs[0])[0] = 42.0

        return run

    C = [[0.0] * 5 for _ in range(3)]
    JITRuntime.compile(matmul_par, compiler=compiler)(C, [[1.0] * 4] * 3, [[1.0] * 5] * 4)
    assert C[0][0] == 42.0 and C[2][4] == 0.0


@pytest.mark.parametrize("block_optional_deps", [False, True])
def test_cpu_jit_never_imports_metal_deps(tmp_path, block_optional_deps):
    script = tmp_path / "cpu_only.py"
    script.write_text(
        dedent(f"""
            import sys
            if {block_optional_deps}:
                sys.modules.update(dict.fromkeys(("Metal", "objc", "llvm_to_air")))  # None entries make imports fail like absent packages
            from exo import *
            from exojit import jit

            @proc
            def scale(dst: f32[2] @ DRAM, src: f32[2] @ DRAM):
                for i in par(0, 2):
                    dst[i] = src[i] * 2.0

            dst = [0.0, 0.0]
            jit(scale)(dst, [1.0, 2.5])
            assert dst == [2.0, 5.0], dst
            assert all(sys.modules.get(name) is None for name in ("Metal", "objc", "llvm_to_air")), "cpu jit imported a metal dependency"
            assert "backend_metal" not in sys.modules, "cpu jit imported the metal backend"
            """)
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))}
    result = subprocess.run([sys.executable, str(script)], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
