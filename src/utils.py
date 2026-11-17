import functools
import platform
import re
import shutil
import subprocess
import tempfile


def get_mac_version() -> str:
    mac_version = platform.mac_ver()[0]
    if not mac_version:
        return "14.0.0"
    return mac_version


def get_metal_version() -> str:
    default_version = "Apple metal version 32023.830 (metalfe-32023.830.2)"
    metal_path = shutil.which("metal")
    if not metal_path:
        return default_version
    result = subprocess.run([metal_path, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return default_version
    first_line = result.stdout.splitlines()[0]
    if "Apple metal version" in first_line:
        return first_line.strip()
    return default_version


@functools.lru_cache(maxsize=1)
def get_target_datalayout() -> str:
    default = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
    if not (metal := shutil.which("metal")):
        return default
    with tempfile.TemporaryDirectory() as d:
        with open(src := f"{d}/t.metal", "w") as f:
            f.write("kernel void e(){}")
        cmd = ["xcrun", "-sdk", "macosx", "metal"] if shutil.which("xcrun") else [metal]
        subprocess.run(cmd + ["-S", "-emit-llvm", src, "-o", out := f"{d}/t.ll"], check=True, capture_output=True)
        if m := re.search(r'target datalayout = "([^"]+)"', open(out).read()):
            return m.group(1)
    return default
