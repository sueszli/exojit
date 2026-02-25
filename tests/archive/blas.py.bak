from __future__ import annotations

from exo import *
from exo.platforms.x86 import *
from exoblas.blaslib import *
from exoblas.codegen_helpers import *


@proc
def scalar_load(x: f32 @ DRAM, src: [f32][8] @ AVX2):
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        x += src[i]


variants_generator(optimize_level_1, opt_precisions=("f32"), targets=("avx2"))(scalar_load, "i", 8, globals=globals())
