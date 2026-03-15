# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
# ]
# ///

from exo import proc
from exojit.main import compile_jit, to_mlir
import numpy as np


@proc
def scale(n: size, x: f32[n], alpha: f32):
    for i in seq(0, n):
        x[i] = alpha * x[i]


# print the generated MLIR
module = to_mlir(scale)
print(module)

# JIT-compile and call it
fns = compile_jit(scale)
x = np.ones(8, dtype=np.float32)
fns["scale"](8, x, np.float32(2.0))
print(x)  # [2. 2. 2. 2. 2. 2. 2. 2.]
