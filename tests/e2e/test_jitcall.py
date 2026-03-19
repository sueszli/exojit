from __future__ import annotations

import numpy as np
import pytest
from exo import *

from exojit.main import jit


@proc
def copy4(dst: f32[4] @ DRAM, src: f32[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = src[i]


def test_jit_accepts_direct_numpy_buffers():
    fn = jit(copy4)
    src = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
    dst = np.zeros_like(src)
    fn(dst, src)
    np.testing.assert_allclose(dst, src)


def test_jit_rejects_keyword_args():
    fn = jit(copy4)
    dst = np.zeros(4, dtype=np.float32)
    src = np.arange(4, dtype=np.float32)
    with pytest.raises(TypeError, match="keyword"):
        fn(dst=dst, src=src)


def test_jit_respects_buffer_writability():
    fn = jit(copy4)

    src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    src.flags.writeable = False
    dst = np.zeros(4, dtype=np.float32)
    fn(dst, src)
    np.testing.assert_allclose(dst, src)

    dst.flags.writeable = False
    with pytest.raises(TypeError, match="writable buffer"):
        fn(dst, np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
