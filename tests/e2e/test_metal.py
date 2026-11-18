import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from exo import *

from backend_metal import XCODE, check_matmul, kernel_to_air, metal_toolchain_env
from exojit import jit

ROOT = Path(__file__).resolve().parents[2]
CUSTOM = "/opt/Xcode 16;beta.app/Contents/Developer"


@proc
def matmul(C: f32[2, 3] @ DRAM, A: f32[2, 4] @ DRAM, B: f32[4, 3] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 3):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


@proc
def racy(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, 0] += A[i, k] * B[k, j]


@proc
def wrong_init_cell(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, 0] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def wrong_init_value(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 1.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def nonzero_start(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(1, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def transposed_read(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[k, i] * B[k, j]


@proc
def neighbor_read(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * C[1 - i, j]


@proc
def nested_par(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in par(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def extra_stmt(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]
            C[i, j] = A[i, j]


@proc
def accumulate(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def partial_grid(C: f32[4, 2] @ DRAM, A: f32[4, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def seq_grid(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, B: f32[2, 2] @ DRAM):
    for i in seq(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def bad_shapes(C: f32[2, 3] @ DRAM, A: f32[2, 4] @ DRAM, B: f32[5, 3] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 3):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


@proc
def f64_matmul(C: f64[2, 2] @ DRAM, A: f64[2, 2] @ DRAM, B: f64[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = 0.0
            for k in seq(0, 2):
                C[i, j] += A[i, k] * B[k, j]


@proc
def scalar_arg(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM, s: f32):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = A[i, j] * s


@proc
def two_args(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = A[i, j]


@proc
def too_many_threads(C: f32[65536, 65536] @ DRAM, A: f32[65536, 1] @ DRAM, B: f32[1, 65536] @ DRAM):
    for i in par(0, 65536):
        for j in par(0, 65536):
            C[i, j] = 0.0
            for k in seq(0, 1):
                C[i, j] += A[i, k] * B[k, j]


@pytest.mark.parametrize(
    "p",
    [
        racy,
        transposed_read,
        neighbor_read,
        nonzero_start,
        wrong_init_cell,
        wrong_init_value,
        nested_par,
        extra_stmt,
        accumulate,
        partial_grid,
        seq_grid,
        bad_shapes,
        f64_matmul,
        scalar_arg,
        two_args,
        too_many_threads,
    ],
)
def test_metal_rejects_unsupported_procs(p, monkeypatch):
    monkeypatch.setitem(sys.modules, "Metal", None)  # validation must precede optional imports
    with pytest.raises(AssertionError):
        jit(p, backend="metal")


def test_metal_accepts_matmul_and_emits_air_without_toolchain():
    assert check_matmul(matmul._loopir_proc)[:3] == (2, 3, 4)
    with patch("backend_metal.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
        air = kernel_to_air(matmul)
    assert 'target triple = "air64_v27-apple-macosx15.0.0"' in air and " ptr " not in air
    assert air.count('!"air.read_write"') == 1 and '!"air.arg_name", !"out"' in air and '!"air.thread_position_in_grid"' in air


def test_metal_rejects_raw_and_unknown_backends():
    with pytest.raises(AssertionError):
        jit(matmul, raw=True, backend="metal")
    with pytest.raises(AssertionError, match="unsupported jit backend"):
        jit(matmul, backend="cuda")


def test_metal_missing_pyobjc_fails_without_cpu_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "Metal", None)  # None fails import
    with pytest.raises(ModuleNotFoundError, match="Metal"):
        jit(matmul, backend="metal")


def test_cpu_jit_never_imports_metal_deps(tmp_path):
    script = tmp_path / "cpu_only.py"
    script.write_text(
        dedent("""
            import sys
            sys.modules.update(dict.fromkeys(("Metal", "Foundation", "objc")))  # None blocks import
            from exo import *
            from exojit import jit

            @proc
            def scale(dst: f32[2] @ DRAM, src: f32[2] @ DRAM):
                for i in par(0, 2):
                    dst[i] = src[i] * 2.0

            dst = [0.0, 0.0]
            jit(scale)(dst, [1.0, 2.5])
            assert dst == [2.0, 5.0], dst
            assert "backend_metal" not in sys.modules
            """)
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))}
    result = subprocess.run([sys.executable, str(script)], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("environ", "usable", "expected"),
    [
        ({}, {None, XCODE}, {}),
        ({}, {XCODE}, {"DEVELOPER_DIR": XCODE}),
        ({"DEVELOPER_DIR": CUSTOM}, {CUSTOM}, {"DEVELOPER_DIR": CUSTOM}),
        ({"DEVELOPER_DIR": CUSTOM}, {XCODE}, None),
        ({}, set(), None),
    ],
)
def test_metal_toolchain_env(environ, usable, expected):
    def run(cmd, env, **_):
        assert isinstance(cmd, list) and "shell" not in _
        return SimpleNamespace(returncode=int(env.get("DEVELOPER_DIR") not in usable))

    with patch.dict(os.environ, environ, clear=True), patch("backend_metal.subprocess.run", run):
        if expected is None:
            with pytest.raises(AssertionError):
                metal_toolchain_env()
        else:
            assert metal_toolchain_env() == expected | {"MACOSX_DEPLOYMENT_TARGET": "15.0"} and dict(os.environ) == environ
