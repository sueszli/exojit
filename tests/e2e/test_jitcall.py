from __future__ import annotations

import numpy as np
import pytest
from exo import *

from exojit.main import jit


@proc
def copy4(dst: f32[4] @ DRAM, src: f32[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = src[i]


@proc
def scale4(dst: f32[4] @ DRAM, src: f32[4] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, 4):
        dst[i] = src[i] * s[0]


@proc
def copy_n(N: size, dst: f32[N] @ DRAM, src: f32[N] @ DRAM):
    for i in seq(0, N):
        dst[i] = src[i]


@proc
def copy_2x2(dst: f32[2, 2] @ DRAM, src: f32[2, 2] @ DRAM):
    for i in seq(0, 2):
        for j in seq(0, 2):
            dst[i, j] = src[i, j]


def test_jit_rejects_direct_numpy_buffers():
    fn = jit(copy4)
    src = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
    with pytest.raises(AssertionError, match="direct buffer inputs are not supported"):
        fn([0.0, 0.0, 0.0, 0.0], src)


def test_jit_accepts_python_lists_and_scalar_tensor_inputs():
    fn = jit(scale4)
    dst = [0.0, 0.0, 0.0, 0.0]
    fn(dst, [1.0, -2.0, 3.5, 4.25], 2.0)
    np.testing.assert_allclose(dst, [2.0, -4.0, 7.0, 8.5])


def test_jit_validates_python_list_length_against_dynamic_shape():
    fn = jit(copy_n)
    with pytest.raises(AssertionError, match="expected 4 values, got 3"):
        fn(4, [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0])


def test_jit_syncs_nested_writable_sequences():
    fn = jit(copy_2x2)
    dst = [[0.0, 0.0], [0.0, 0.0]]
    src = [[1.0, 2.0], [3.0, 4.0]]
    fn(dst, src)
    np.testing.assert_allclose(dst, src)


def test_jit_rejects_immutable_nested_writable_sequences():
    fn = jit(copy_2x2)
    dst = [[0.0, 0.0], (0.0, 0.0)]
    src = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(AssertionError, match="mutable at every level"):
        fn(dst, src)


def test_jit_rejects_keyword_args():
    fn = jit(copy4)
    dst = [0.0, 0.0, 0.0, 0.0]
    src = [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(TypeError, match="keyword"):
        fn._raw(dst=dst, src=src)


def test_jit_raw_rejects_python_lists():
    raw = jit(copy4, raw=True)
    with pytest.raises(TypeError):
        raw([0.0] * 4, [1.0, 2.0, 3.0, 4.0])


def test_jit_exposes_raw_entrypoint():
    wrapped = jit(copy4)
    raw = wrapped._raw
    src = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
    dst = np.zeros_like(src)
    raw(dst, src)
    np.testing.assert_allclose(dst, src)
    raw(dst.ctypes.data, src.ctypes.data)
    np.testing.assert_allclose(dst, src)


def test_jit_raw_rejects_non_contiguous_buffers():
    raw = jit(copy4, raw=True)
    src = np.arange(8, dtype=np.float32)[::2]
    dst = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError):
        raw(dst, src)


def test_jit_raw_bind_marshals_once():
    src = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
    dst = np.zeros_like(src)
    call = jit(copy4, raw=True).bind(dst, src)
    call()
    np.testing.assert_allclose(dst, src)
    src[:] = [9.0, 8.0, 7.0, 6.0]
    call()
    np.testing.assert_allclose(dst, src)


def test_jit_raw_bind_checks_arity():
    with pytest.raises(ValueError):
        jit(copy4, raw=True).bind(np.zeros(4, dtype=np.float32))


def test_jit_raw_rejects_a_float_for_a_size_arg():
    raw = jit(copy_n, raw=True)
    with pytest.raises(TypeError):
        raw(4.0, np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_jit_raw_accepts_a_read_only_buffer_where_nothing_writes_it():
    raw = jit(copy4, raw=True)
    src = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
    src.setflags(write=False)
    dst = np.zeros(4, dtype=np.float32)
    raw(dst, src)
    np.testing.assert_allclose(dst, src)


def test_jit_raw_rejects_a_read_only_buffer_in_a_written_slot():
    raw = jit(copy4, raw=True)
    dst = np.zeros(4, dtype=np.float32)
    dst.setflags(write=False)
    with pytest.raises(ValueError):
        raw(dst, np.zeros(4, dtype=np.float32))
