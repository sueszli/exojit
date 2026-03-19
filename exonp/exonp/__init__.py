from __future__ import annotations

import ctypes
import random
from math import prod

from exo import *
from exo.core.memory import Memory
from exo.libs.externs import select, sqrt

__all__ = [
    "Ptr",
    "Tensor",
    "Stack",
    "ctype",
    "array",
    "empty",
    "empty_like",
    "full",
    "zeros",
    "zeros_like",
    "reshape",
    "normal",
    "fill",
    "fill3",
    "add",
    "matmul",
    "matmul_left_t",
    "matmul_right_t",
    "relu",
    "rmsnorm",
    "rmsnorm_bwd",
    "zero2d",
    "zero3",
    "matmul_nn",
    "matmul_nt",
    "matmul_tn",
    "rmsnorm_fwd",
    "rmsnorm_residual_bwd",
]


class Ptr:
    __slots__ = ("data",)

    def __init__(self, data: int):
        self.data = data


def ctype(dtype):
    if dtype is float:
        return ctypes.c_double
    if dtype is int:
        return ctypes.c_int32
    raise TypeError(dtype)


class Tensor:
    __slots__ = ("shape", "dtype", "_ctype", "_buf", "_offset", "_size")

    def __init__(self, shape: tuple[int, ...], dtype=float, *, buffer=None, offset: int = 0, fill=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._ctype = ctype(dtype)
        self._offset = offset
        self._size = prod(self.shape)
        self._buf = (self._ctype * self._size)() if buffer is None else buffer
        if buffer is None and fill not in (None, 0, 0.0):
            for i in range(self._size):
                self._buf[i] = fill

    @property
    def ctypes(self):
        return Ptr(ctypes.addressof(self._buf) + self._offset * ctypes.sizeof(self._ctype))

    def flat_index(self, key) -> int:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != len(self.shape):
            raise IndexError(key)
        if len(key) == 1:
            return self._offset + key[0]
        if len(key) == 2:
            return self._offset + key[0] * self.shape[1] + key[1]
        if len(key) == 3:
            return self._offset + (key[0] * self.shape[1] + key[1]) * self.shape[2] + key[2]
        raise ValueError(self.shape)

    def __getitem__(self, key):
        return self._buf[self.flat_index(key)]

    def __setitem__(self, key, value):
        self._buf[self.flat_index(key)] = value


def array(data, dtype=float) -> Tensor:
    if not isinstance(data, list) or len(data) == 0:
        raise TypeError("array expects a non-empty Python list")
    if isinstance(data[0], list):
        out = empty((len(data), len(data[0])), dtype=dtype)
        for i, row in enumerate(data):
            if len(row) != out.shape[1]:
                raise ValueError("ragged nested lists are unsupported")
            for j, value in enumerate(row):
                out[i, j] = value
        return out
    out = empty((len(data),), dtype=dtype)
    for i, value in enumerate(data):
        out[i] = value
    return out


def empty(shape: tuple[int, ...], dtype=float) -> Tensor:
    return Tensor(shape, dtype=dtype)


def empty_like(x: Tensor) -> Tensor:
    return empty(x.shape, dtype=x.dtype)


def full(shape: tuple[int, ...], fill_value, dtype=float) -> Tensor:
    return Tensor(shape, dtype=dtype, fill=fill_value)


def zeros(shape: tuple[int, ...], dtype=float) -> Tensor:
    return full(shape, 0.0 if dtype is float else 0, dtype=dtype)


def zeros_like(x: Tensor) -> Tensor:
    return zeros(x.shape, dtype=x.dtype)


def reshape(x: Tensor, shape: tuple[int, ...], *, offset: int = 0) -> Tensor:
    return Tensor(shape, dtype=x.dtype, buffer=x._buf, offset=x._offset + offset)


def normal(shape: tuple[int, ...], loc: float = 0.0, scale: float = 1.0) -> Tensor:
    out = empty(shape, dtype=float)
    for i in range(out._size):
        out._buf[i] = random.gauss(loc, scale)
    return out


class Stack(Memory):
    @classmethod
    def alloc(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        c_types = {"float": "float", "double": "double", "int8_t": "int8_t", "int32_t": "int32_t"}
        c_type = c_types.get(prim_type, prim_type)
        if not shape:
            return f"{c_type} {new_name};"
        return f'{c_type} {new_name}[{"][".join(shape)}];'

    @classmethod
    def can_read(cls) -> bool:
        return True

    @classmethod
    def write(cls, s, lhs: str, rhs: str) -> str:
        return f"{lhs} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs: str, rhs: str) -> str:
        return f"{lhs} += {rhs};"

    @classmethod
    def free(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        return ""


# Array update and compute kernels.
# The public names stay close to NumPy where possible, and the layout-specific
# Exo kernels remain available as low-level aliases for hot paths that need them.
@proc
def fill(M: size, N: size, x: f64[M, N] @ DRAM, value: f64[1] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            x[i, j] = value[0]


@proc
def add(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] += x[i, j]


@proc
def matmul_right_t(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[N, K] @ DRAM, zero: f64[1] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = zero[0]
            for k in seq(0, K):
                acc += x[i, k] * w[j, k]
            out[i, j] = acc


@proc
def matmul(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[K, N] @ DRAM, zero: f64[1] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = zero[0]
            for k in seq(0, K):
                acc += x[i, k] * w[k, j]
            out[i, j] = acc


@proc
def matmul_left_t(M: size, N: size, K: size, out: f64[N, K] @ DRAM, x: f64[M, N] @ DRAM, w: f64[M, K] @ DRAM, zero: f64[1] @ DRAM):
    for j in seq(0, N):
        for k in seq(0, K):
            acc: f64 @ Stack
            acc = zero[0]
            for i in seq(0, M):
                acc += x[i, j] * w[i, k]
            out[j, k] = acc


@proc
def rmsnorm(M: size, N: size, out: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, x: f64[M, N] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    for i in seq(0, M):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = zero[0]
        for j in seq(0, N):
            sumsq += x[i, j] * x[i, j]
        scale = one[0] / sqrt(sumsq * inv_n[0] + eps[0])
        rms[i, 0] = scale
        for j in seq(0, N):
            out[i, j] = x[i, j] * scale


@proc
def rmsnorm_bwd(M: size, N: size, out: f64[M, N] @ DRAM, dx: f64[M, N] @ DRAM, x_pre: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    for i in seq(0, M):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = zero[0]
        scale = rms[i, 0]
        for j in seq(0, N):
            dot += out[i, j] * x_pre[i, j]
        corr = scale * scale * scale * inv_n[0] * dot
        for j in seq(0, N):
            out[i, j] = out[i, j] * scale - x_pre[i, j] * corr + dx[i, j]


@proc
def relu(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM, zero: f64[1] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] = select(zero[0], x[i, j], x[i, j], zero[0])


@proc
def fill3(M: size, N: size, a: f64[M, N] @ DRAM, b: f64[M, N] @ DRAM, c: f64[M, N] @ DRAM, value: f64[1] @ DRAM):
    fill(M, N, a, value)
    fill(M, N, b, value)
    fill(M, N, c, value)


zero2d = fill
zero3 = fill3
matmul_nn = matmul
matmul_nt = matmul_right_t
matmul_tn = matmul_left_t
rmsnorm_fwd = rmsnorm
rmsnorm_residual_bwd = rmsnorm_bwd
