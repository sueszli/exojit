from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from xnumpy.library.kernels.elementwise import add, iadd, imul, iscalar_add, iscalar_mul, iscalar_sub, isub, mul, neg, scalar_add, scalar_mul, scalar_rsub, scalar_sub, sub, sum_reduce
from xnumpy.library.kernels.matmul import matmul

float32 = np.float32

_scalar = lambda v: np.array([np.float32(v)])


class ndarray:
    _data: np.ndarray[Any, np.dtype[np.float32]]

    def __init__(self, data: npt.ArrayLike) -> None:
        self._data = np.ascontiguousarray(data, dtype=np.float32)

    shape = property(lambda self: self._data.shape)
    dtype = property(lambda self: self._data.dtype)
    ndim = property(lambda self: self._data.ndim)
    size = property(lambda self: self._data.size)
    T = property(lambda self: ndarray(self._data.T))

    def __repr__(self) -> str:
        return f"xnumpy.{repr(self._data)}"

    def __str__(self) -> str:
        return str(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def tolist(self) -> list[Any]:
        return self._data.tolist()

    def copy(self) -> ndarray:
        return ndarray(self._data.copy())

    def __add__(self, other: ndarray | float | int) -> ndarray:
        n = self.size
        out = empty(self.shape)
        if isinstance(other, ndarray):
            add(n)(out._data.ctypes.data, self._data.ctypes.data, other._data.ctypes.data)
        else:
            s = _scalar(other)
            scalar_add(n)(out._data.ctypes.data, self._data.ctypes.data, s.ctypes.data)
        return out

    def __sub__(self, other: ndarray | float | int) -> ndarray:
        n = self.size
        out = empty(self.shape)
        if isinstance(other, ndarray):
            sub(n)(out._data.ctypes.data, self._data.ctypes.data, other._data.ctypes.data)
        else:
            s = _scalar(other)
            scalar_sub(n)(out._data.ctypes.data, self._data.ctypes.data, s.ctypes.data)
        return out

    def __mul__(self, other: ndarray | float | int) -> ndarray:
        n = self.size
        out = empty(self.shape)
        if isinstance(other, ndarray):
            mul(n)(out._data.ctypes.data, self._data.ctypes.data, other._data.ctypes.data)
        else:
            s = _scalar(other)
            scalar_mul(n)(out._data.ctypes.data, self._data.ctypes.data, s.ctypes.data)
        return out

    def __radd__(self, other: float | int) -> ndarray:
        return self.__add__(other)

    def __rmul__(self, other: float | int) -> ndarray:
        return self.__mul__(other)

    def __rsub__(self, other: float | int) -> ndarray:
        n = self.size
        out = empty(self.shape)
        s = _scalar(other)
        scalar_rsub(n)(out._data.ctypes.data, self._data.ctypes.data, s.ctypes.data)
        return out

    def __iadd__(self, other: ndarray | float | int) -> ndarray:
        n = self.size
        if isinstance(other, ndarray):
            iadd(n)(self._data.ctypes.data, other._data.ctypes.data)
        else:
            s = _scalar(other)
            iscalar_add(n)(self._data.ctypes.data, s.ctypes.data)
        return self

    def __isub__(self, other: ndarray | float | int) -> ndarray:
        n = self.size
        if isinstance(other, ndarray):
            isub(n)(self._data.ctypes.data, other._data.ctypes.data)
        else:
            s = _scalar(other)
            iscalar_sub(n)(self._data.ctypes.data, s.ctypes.data)
        return self

    def __imul__(self, other: ndarray | float | int) -> ndarray:
        n = self.size
        if isinstance(other, ndarray):
            imul(n)(self._data.ctypes.data, other._data.ctypes.data)
        else:
            s = _scalar(other)
            iscalar_mul(n)(self._data.ctypes.data, s.ctypes.data)
        return self

    def __neg__(self) -> ndarray:
        out = empty(self.shape)
        neg(self.size)(out._data.ctypes.data, self._data.ctypes.data)
        return out

    def __matmul__(self, other: ndarray) -> ndarray:
        assert isinstance(other, ndarray) and self.ndim == 2 and other.ndim == 2
        m, k = self.shape
        k2, n = other.shape
        assert k == k2
        out = zeros((m, n))
        matmul(m, k, n)(out._data.ctypes.data, self._data.ctypes.data, other._data.ctypes.data)
        return out

    def sum(self) -> float:
        out = np.zeros(1, dtype=np.float32)
        sum_reduce(self.size)(out.ctypes.data, self._data.ctypes.data)
        return float(out[0])

    def __getitem__(self, key: Any) -> Any:
        raise NotImplementedError("xnumpy.ndarray.__getitem__")

    def __setitem__(self, key: Any, value: Any) -> None:
        raise NotImplementedError("xnumpy.ndarray.__setitem__")

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(f"xnumpy.ndarray.{name}")


def array(data: npt.ArrayLike) -> ndarray:
    return ndarray(data)


def zeros(shape: int | tuple[int, ...]) -> ndarray:
    return ndarray(np.zeros(shape, dtype=np.float32))


def ones(shape: int | tuple[int, ...]) -> ndarray:
    return ndarray(np.ones(shape, dtype=np.float32))


def empty(shape: int | tuple[int, ...]) -> ndarray:
    return ndarray(np.empty(shape, dtype=np.float32))


def allclose(a: ndarray | npt.ArrayLike, b: ndarray | npt.ArrayLike, **kw: Any) -> bool:
    a = a._data if isinstance(a, ndarray) else a
    b = b._data if isinstance(b, ndarray) else b
    return np.allclose(a, b, **kw)
