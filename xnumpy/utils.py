from __future__ import annotations

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path


@cache
def llvm_bin_path() -> Path:
    if env := os.environ.get("LLVM_BIN"):
        return Path(env)
    if mlir_opt := shutil.which("mlir-opt"):
        return Path(mlir_opt).parent
    return Path(subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True, check=True).stdout.strip()) / "bin"
