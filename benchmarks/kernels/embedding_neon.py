from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from kernels.add_neon import neon_loadu_f32x4, neon_storeu_f32x4

from exojit.main import jit
from exojit.patches_exo import NEON

_PAR_MIN_ELEMENTS = 524288


@proc
def _embedding_neon(D: size, out: f32[D] @ DRAM, row: f32[D] @ DRAM):
    for i in seq(0, D / 16):
        v0: f32[4] @ NEON
        v1: f32[4] @ NEON
        v2: f32[4] @ NEON
        v3: f32[4] @ NEON
        neon_loadu_f32x4(v0, row[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(v1, row[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(v2, row[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(v3, row[16 * i + 12 : 16 * i + 16])
        neon_storeu_f32x4(out[16 * i + 0 : 16 * i + 4], v0)
        neon_storeu_f32x4(out[16 * i + 4 : 16 * i + 8], v1)
        neon_storeu_f32x4(out[16 * i + 8 : 16 * i + 12], v2)
        neon_storeu_f32x4(out[16 * i + 12 : 16 * i + 16], v3)


@proc
def _embedding_neon_par(D: size, out: f32[D] @ DRAM, row: f32[D] @ DRAM):
    for i in par(0, D / 16):
        v0: f32[4] @ NEON
        v1: f32[4] @ NEON
        v2: f32[4] @ NEON
        v3: f32[4] @ NEON
        neon_loadu_f32x4(v0, row[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(v1, row[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(v2, row[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(v3, row[16 * i + 12 : 16 * i + 16])
        neon_storeu_f32x4(out[16 * i + 0 : 16 * i + 4], v0)
        neon_storeu_f32x4(out[16 * i + 4 : 16 * i + 8], v1)
        neon_storeu_f32x4(out[16 * i + 8 : 16 * i + 12], v2)
        neon_storeu_f32x4(out[16 * i + 12 : 16 * i + 16], v3)


@cache
def embedding_neon(d: int) -> Callable[..., None]:
    assert d % 16 == 0
    return jit((_embedding_neon_par if d >= _PAR_MIN_ELEMENTS else _embedding_neon).partial_eval(D=d))
