import os
import platform
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from exo import *

from backend_metal import get_target_datalayout, kernel_to_air, metal_toolchain_env
from exojit import jit

ROOT = Path(__file__).resolve().parents[2]
XCODE, CUSTOM = "/Applications/Xcode.app/Contents/Developer", "/opt/Xcode.app/Contents/Developer"


@proc
def matmul(C: f32[2, 3] @ DRAM, A: f32[2, 4] @ DRAM, B: f32[4, 3] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 3):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


@proc
def racy(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, 0] += A[i, j]


@proc
def neighbor_read(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = C[1 - i, j] + A[i, j]


@proc
def nested_par(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            for k in par(0, 2):
                C[i, j] += A[i, k]


@proc
def partial_grid(C: f32[4, 2] @ DRAM, A: f32[4, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = A[i, j]


@proc
def two_outputs(C: f32[2, 2] @ DRAM, D: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = A[i, j]
            D[i, j] = A[i, j]


@proc
def copy_f64(C: f64[2, 2] @ DRAM, A: f64[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            C[i, j] = A[i, j]


@proc
def window(C: f32[2, 2] @ DRAM, A: f32[2, 2] @ DRAM):
    for i in par(0, 2):
        for j in par(0, 2):
            row = A[i, :]
            C[i, j] = row[j]


@proc
def too_many_threads(C: f32[65536, 65536] @ DRAM):
    for i in par(0, 65536):
        for j in par(0, 65536):
            C[i, j] = 0.0


@pytest.mark.parametrize(("p", "match"), [(racy, "may only write"), (neighbor_read, "may only be read"), (nested_par, "nested inside the thread grid"), (partial_grid, "must span"), (two_outputs, "exactly one written argument"), (copy_f64, "static f32 tensor"), (window, "unsupported statement WindowStmt"), (too_many_threads, "uint32 thread id")])
def test_metal_rejects_unsafe_procs(p, match):
    with pytest.raises(AssertionError, match=match):
        jit(p, backend="metal")


def test_metal_rejects_raw_and_unknown_backends():
    with pytest.raises(AssertionError, match="raw=True is not supported"):
        jit(matmul, raw=True, backend="metal")
    with pytest.raises(AssertionError, match="unsupported jit backend"):
        jit(matmul, backend="cuda")


def test_metal_missing_pyobjc_fails_without_cpu_fallback(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setitem(sys.modules, "Metal", None)  # None fails import
    with pytest.raises(ModuleNotFoundError, match="Metal"):
        jit(matmul, backend="metal")


def test_off_mac_translation_ignores_metal_on_path():
    get_target_datalayout.cache_clear()  # cached probe could hide subprocess
    try:
        with patch("backend_metal.platform.system", return_value="Linux"), patch("backend_metal.shutil.which", return_value="/usr/local/bin/metal"), patch("backend_metal.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
            air = kernel_to_air(matmul)
    finally:
        get_target_datalayout.cache_clear()
    assert 'target triple = "air64_v27-apple-macosx' in air and air.count('!"air.read_write"') == 1 and '!"air.thread_position_in_grid"' in air


def test_cpu_jit_never_imports_metal_deps(tmp_path):
    script = tmp_path / "cpu_only.py"
    script.write_text(
        dedent("""
            import sys
            sys.modules.update(dict.fromkeys(("Metal", "objc")))  # None blocks import
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
    ("environ", "usable", "apps", "expected"),
    [
        ({}, {XCODE}, ["/Applications/Xcode.app"], {"DEVELOPER_DIR": XCODE, "MACOSX_DEPLOYMENT_TARGET": "15.0"}),
        ({"DEVELOPER_DIR": CUSTOM, "MACOSX_DEPLOYMENT_TARGET": "26.0"}, {CUSTOM, XCODE}, ["/Applications/Xcode.app"], {"DEVELOPER_DIR": CUSTOM, "MACOSX_DEPLOYMENT_TARGET": "26.0"}),
        ({"DEVELOPER_DIR": CUSTOM}, {XCODE}, ["/Applications/Xcode.app"], "no usable metal compiler in DEVELOPER_DIR"),
        ({}, set(), [], "no full Xcode"),
    ],
)
def test_metal_toolchain_env(environ, usable, apps, expected):
    def run(cmd, env, **_):
        return SimpleNamespace(returncode=int(env.get("DEVELOPER_DIR") not in usable))

    with patch.dict(os.environ, environ, clear=True), patch("backend_metal.subprocess.run", run), patch("backend_metal.glob", return_value=apps):
        if isinstance(expected, str):
            with pytest.raises(AssertionError, match=expected):
                metal_toolchain_env.__wrapped__()
        else:
            assert metal_toolchain_env.__wrapped__() == expected and dict(os.environ) == environ
