from __future__ import annotations

import ctypes
import random
from math import prod
from typing import Any


class Ptr:
    __slots__ = ("data",)

    def __init__(self, data: int):
        self.data = data


def ctype(dtype: type[float] | type[int]) -> type[ctypes.c_double] | type[ctypes.c_int64]:
    # python dtype -> ctypes scalar
    if dtype is float:
        return ctypes.c_double
    if dtype is int:
        return ctypes.c_int64
    raise TypeError(dtype)


class Tensor:
    __slots__ = ("shape", "dtype", "_ctype", "_buf", "_offset", "_size")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: type[float] | type[int] = float,
        *,
        buffer: Any = None,
        offset: int = 0,
        fill: float | int | None = None,
    ) -> None:
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
    def ctypes(self) -> Ptr:
        return Ptr(ctypes.addressof(self._buf) + self._offset * ctypes.sizeof(self._ctype))

    @property
    def ptr(self) -> int:
        return self.ctypes.data

    @property
    def numel(self) -> int:
        return self._size

    @property
    def itemsize(self) -> int:
        return ctypes.sizeof(self._ctype)

    def view(self, shape: tuple[int, ...], *, offset: int = 0) -> Tensor:
        # shared-buffer view starting at element offset
        return Tensor(shape, dtype=self.dtype, buffer=self._buf, offset=self._offset + offset)

    def flat_index(self, key: int | tuple[int, ...]) -> int:
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

    def __getitem__(self, key: int | tuple[int, ...]) -> float | int:
        return self._buf[self.flat_index(key)]

    def __setitem__(self, key: int | tuple[int, ...], value: float | int) -> None:
        self._buf[self.flat_index(key)] = value


def empty(shape: tuple[int, ...], dtype: type[float] | type[int] = float) -> Tensor:
    # allocate uninitialized storage
    return Tensor(shape, dtype=dtype)


def full(shape: tuple[int, ...], fill_value: float | int, dtype: type[float] | type[int] = float) -> Tensor:
    # allocate and fill with a scalar
    return Tensor(shape, dtype=dtype, fill=fill_value)


def zeros(shape: tuple[int, ...], dtype: type[float] | type[int] = float) -> Tensor:
    # allocate and fill with 0
    return full(shape, 0.0 if dtype is float else 0, dtype=dtype)


def zeros_like(x: Tensor) -> Tensor:
    # allocate zeros with x.shape and x.dtype
    return zeros(x.shape, dtype=x.dtype)


def reshape(x: Tensor, shape: tuple[int, ...], *, offset: int = 0) -> Tensor:
    # shared-buffer reshape view
    return x.view(shape, offset=offset)


def normal(shape: tuple[int, ...], loc: float = 0.0, scale: float = 1.0) -> Tensor:
    # iid gaussian fill: n(loc, scale^2)
    out = empty(shape, dtype=float)
    for i in range(out._size):
        out._buf[i] = random.gauss(loc, scale)
    return out


def pack_tensors(tensors: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor], int]:
    # flatten tensors into one shared storage block and return typed views
    total = sum(t.numel for t in tensors.values())
    flat = empty((total,), dtype=float)
    flat_ptr = flat.ptr
    elt_bytes = flat.itemsize
    offset = 0
    views = {}
    for name, tensor in tensors.items():
        ctypes.memmove(flat_ptr + offset * elt_bytes, tensor.ptr, tensor.numel * elt_bytes)
        views[name] = reshape(flat, tensor.shape, offset=offset)
        offset += tensor.numel
    return flat, views, elt_bytes


def view_tensors(flat: Tensor, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    # rebuild tensor views over existing flat storage
    offset = 0
    views = {}
    for name, tensor in tensors.items():
        views[name] = reshape(flat, tensor.shape, offset=offset)
        offset += tensor.numel
    return views


def tensor_ptrs(tensors: dict[str, Tensor]) -> dict[str, int]:
    # expose raw data pointers for raw-jit entry points
    return {name: tensor.ptr for name, tensor in tensors.items()}


def alloc_layout(spec: dict[str, tuple[int, ...]], dtype: type[float] | type[int] = float) -> tuple[Tensor, dict[str, Tensor]]:
    # allocate one flat buffer and typed views over it
    total = sum(prod(shape) for shape in spec.values())
    flat = empty((total,), dtype=dtype)
    return flat, view_layout(flat, spec)


def view_layout(flat: Tensor, spec: dict[str, tuple[int, ...]]) -> dict[str, Tensor]:
    # create named views over an existing flat buffer
    offset = 0
    views = {}
    for name, shape in spec.items():
        views[name] = flat.view(shape, offset=offset)
        offset += prod(shape)
    return views
