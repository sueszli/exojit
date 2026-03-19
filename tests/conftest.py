from __future__ import annotations

import gc as _gc
import re
from copy import deepcopy
from typing import Any

import numpy as np
from _utils import compile_exo, compile_mlir
from exo.API import Procedure

from exojit.main import jit, to_mlir

# xdsl irdl holds raw ctypes pointers. gc finalizer ordering -> dangling ptr -> segfault
_gc.disable()
_gc.set_threshold(0)
_gc.enable = lambda: None
_gc.collect = lambda *a, **kw: 0


def assert_match(proc: Procedure, **kwargs: Any) -> None:
    # compile proc on all backends, verify outputs match exo_c (reference)
    ir = proc._loopir_proc
    module = to_mlir(proc)
    jit_fn = jit(proc)
    jit_kwargs = deepcopy(kwargs)
    jit_args = [jit_kwargs[re.sub(r"_\d+$", "", str(arg.name))] for arg in ir.args]
    jit_fn(*jit_args)
    exo_c = compile_exo(proc)(**kwargs)
    results = {
        "exo_c": exo_c,
        "xdsl_mlir": compile_mlir(proc, module)(**kwargs),
        "jit": {name: np.asarray(jit_kwargs[name], dtype=exo_c[name].dtype) for name in exo_c},
    }

    ref = results["exo_c"]
    for key in ("xdsl_mlir", "jit"):
        for name in ref:
            np.testing.assert_allclose(results[key][name], ref[name], atol=1e-6, err_msg=f"{key} mismatch on buffer '{name}'")
