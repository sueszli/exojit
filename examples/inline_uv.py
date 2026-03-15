# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
#   "numpy",
# ]
# ///

import numpy as np
from exo import *

from exojit.main import compile_jit


@proc
def scale(out: f32[4] @ DRAM, factor: f32[1] @ DRAM):
    for i in seq(0, 4):
        out[i] = out[i] * factor[0]


out = np.ones(4, dtype=np.float32)
factor = np.array([42.0], dtype=np.float32)
scale = compile_jit(scale)["scale"](out, factor)
assert list(out) == [42.0, 42.0, 42.0, 42.0]
print("ok")
