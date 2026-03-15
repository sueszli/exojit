# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
# ]
# ///

import numpy as np
from exo import proc
from exojit.main import compile_jit


@proc
def scale(n: size, x: f32[n], alpha: f32[1]):
    for i in seq(0, n):
        x[i] = alpha[0] * x[i]


x = np.ones(4, dtype=np.float32)
alpha = np.array([3.0], dtype=np.float32)
compile_jit(scale)["scale"](4, x, alpha)
assert list(x) == [3.0, 3.0, 3.0, 3.0]
print("ok")
