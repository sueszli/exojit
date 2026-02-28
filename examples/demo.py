from __future__ import annotations

from exo import *


@proc
def noop():
    pass


@proc
def add(n: size, a: f32[n] @ DRAM, b: f32[n] @ DRAM, out: f32[n] @ DRAM):
    for i in seq(0, n):
        out[i] = a[i] + b[i]
