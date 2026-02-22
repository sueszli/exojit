from __future__ import annotations

from exo import *


@proc
def vadd(n: size, x: [R][n], y: [R][n], z: [R][n]):
    for i in seq(0, n):
        z[i] = x[i] + y[i]
