# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
# ]
# ///

import numpy as np
from exo import *

from exojit.main import compile_jit


@proc
def scale(out: f32[4] @ DRAM, alpha: f32[1] @ DRAM):
    for i in seq(0, 4):
        out[i] = alpha[0] * out[i]


out = np.ones(4, dtype=np.float32)
func = compile_jit(scale)["scale"]
alpha = np.array([3.0], dtype=np.float32)
func(out, alpha)
print(out)
