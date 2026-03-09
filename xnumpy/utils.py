from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from xdsl.dialects.builtin import ModuleOp


@cache
def llvm_bin_path() -> Path:
    if env := os.environ.get("LLVM_BIN"):
        return Path(env)
    if mlir_opt := shutil.which("mlir-opt"):
        return Path(mlir_opt).parent
    return Path(subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True, check=True).stdout.strip()) / "bin"


def exo_bin_path(proc: Procedure) -> Path:
    d = Path(tempfile.mkdtemp())
    exo_compile_procs([proc], d, "o.c", "o.h")
    subprocess.run(
        ["clang", "-shared", "-fPIC", "-O2", "-I", str(d), "-o", str(d / "lib.so"), str(d / "o.c")],
        check=True,
    )
    return d / "lib.so"


def mlir_bin_path(module: ModuleOp) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "o.mlir").write_text(str(module))
    subprocess.run(
        f"{llvm_bin_path()}/mlir-translate --mlir-to-llvmir {d / 'o.mlir'} | clang -shared -x ir -o {d / 'lib.so'} -",
        shell=True,
        check=True,
    )
    return d / "lib.so"
