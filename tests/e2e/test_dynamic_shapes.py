from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def partial_copy(M: size, N: size, dst: f32[M, N] @ DRAM, src: f32[M, N] @ DRAM):
    # the inner loop stops short of N, so the row stride is NOT the trip count
    assert N > 1
    for i in seq(0, M):
        for j in seq(0, N - 1):
            dst[i, j] = src[i, j]


def test_row_stride_is_the_dimension_not_the_trip_count():
    assert_match(partial_copy, M=2, N=3, dst=[[0.0] * 3 for _ in range(2)], src=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])


@proc
def offset_copy(M: size, N: size, dst: f32[M, N] @ DRAM, src: f32[M, N] @ DRAM):
    assert N > 2
    for i in seq(0, M):
        for j in seq(0, N - 2):
            dst[i, j + 1] = src[i, j]


def test_row_stride_with_offset_index():
    assert_match(offset_copy, M=3, N=4, dst=[[0.0] * 4 for _ in range(3)], src=[[float(i * 4 + j) for j in range(4)] for i in range(3)])


@proc
def scale_rows(M: size, N: size, out: f32[M, N] @ DRAM, x: f32[M, N] @ DRAM, s: f32[M] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] = x[i, j] * s[i]


def test_full_extent_dynamic_2d():
    assert_match(scale_rows, M=3, N=5, out=[[0.0] * 5 for _ in range(3)], x=[[float(i * 5 + j) for j in range(5)] for i in range(3)], s=[2.0, 3.0, 4.0])
