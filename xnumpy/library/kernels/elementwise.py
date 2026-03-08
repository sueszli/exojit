from __future__ import annotations

import linecache
from collections.abc import Callable

from xnumpy.main import compile_procs
from xnumpy.patches_llvmlite import jit_compile

_cache: dict[str, Callable[..., None]] = {}


def _jit(code: str, name: str) -> Callable[..., None]:
    if name not in _cache:
        code = code.lstrip("\n")
        ns: dict[str, object] = {}
        exec("from exo import *", ns)
        filename = f"<xnumpy:{name}>"
        compiled = compile(code, filename, "exec")
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        exec(compiled, ns)
        _cache[name] = jit_compile(compile_procs(ns[name]))[name]
    return _cache[name]


def add(n: int) -> Callable[..., None]:
    name = f"_add_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        out[i] = a[i] + b[i]
""",
        name,
    )


def sub(n: int) -> Callable[..., None]:
    name = f"_sub_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        out[i] = a[i] - b[i]
""",
        name,
    )


def mul(n: int) -> Callable[..., None]:
    name = f"_mul_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        out[i] = a[i] * b[i]
""",
        name,
    )


def neg(n: int) -> Callable[..., None]:
    name = f"_neg_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        out[i] = -a[i]
""",
        name,
    )


def scalar_add(n: int) -> Callable[..., None]:
    name = f"_sadd_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        out[i] = a[i] + s[0]
""",
        name,
    )


def scalar_sub(n: int) -> Callable[..., None]:
    name = f"_ssub_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        out[i] = a[i] - s[0]
""",
        name,
    )


def scalar_mul(n: int) -> Callable[..., None]:
    name = f"_smul_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        out[i] = a[i] * s[0]
""",
        name,
    )


def scalar_rsub(n: int) -> Callable[..., None]:
    name = f"_srsub_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[{n}] @ DRAM, a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        out[i] = s[0] - a[i]
""",
        name,
    )


def iadd(n: int) -> Callable[..., None]:
    name = f"_iadd_{n}"
    return _jit(
        f"""@proc
def {name}(a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        a[i] += b[i]
""",
        name,
    )


def isub(n: int) -> Callable[..., None]:
    name = f"_isub_{n}"
    return _jit(
        f"""@proc
def {name}(a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        a[i] = a[i] - b[i]
""",
        name,
    )


def imul(n: int) -> Callable[..., None]:
    name = f"_imul_{n}"
    return _jit(
        f"""@proc
def {name}(a: f32[{n}] @ DRAM, b: f32[{n}] @ DRAM):
    for i in seq(0, {n}):
        a[i] = a[i] * b[i]
""",
        name,
    )


def iscalar_add(n: int) -> Callable[..., None]:
    name = f"_isadd_{n}"
    return _jit(
        f"""@proc
def {name}(a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        a[i] += s[0]
""",
        name,
    )


def iscalar_sub(n: int) -> Callable[..., None]:
    name = f"_issub_{n}"
    return _jit(
        f"""@proc
def {name}(a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        a[i] = a[i] - s[0]
""",
        name,
    )


def iscalar_mul(n: int) -> Callable[..., None]:
    name = f"_ismul_{n}"
    return _jit(
        f"""@proc
def {name}(a: f32[{n}] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, {n}):
        a[i] = a[i] * s[0]
""",
        name,
    )


def sum_reduce(n: int) -> Callable[..., None]:
    name = f"_sum_{n}"
    return _jit(
        f"""@proc
def {name}(out: f32[1] @ DRAM, a: f32[{n}] @ DRAM):
    out[0] = 0.0
    for i in seq(0, {n}):
        out[0] += a[i]
""",
        name,
    )
