from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import simplify

from exojit.main import jit


@proc
def _embedding(D: size, out: f32[D], row: f32[D]):
    for i in seq(0, D):
        out[i] = row[i]


@cache
def embedding_exo(d: int) -> Callable[..., None]:
    p = _embedding.partial_eval(D=d)
    p = simplify(p)
    return jit(p, raw=True)
