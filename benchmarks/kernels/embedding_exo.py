from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename, simplify

from xnumpy.main import compile_jit


@proc
def _embedding(D: size, out: f32[D], row: f32[D]):
    for i in seq(0, D):
        out[i] = row[i]


@cache
def embedding_exo(d: int) -> Callable[..., None]:
    p = _embedding.partial_eval(D=d)
    p = simplify(p)
    name = f"_embedding_{d}"
    return compile_jit(rename(p, name))[name]
