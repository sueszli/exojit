from __future__ import annotations

import gc as _gc
import json
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from _utils import _call, compile_exo, compile_mlir
from exo.API import Procedure
from filelock import FileLock
from plotnine import aes, coord_flip, element_text, geom_col, ggplot, labs, scale_fill_manual, theme, theme_minimal

from xnumpy.main import compile_jit, to_mlir

# xDSL IRDL holds raw ctypes pointers. GC finalizer ordering -> dangling ptr -> segfault
_gc.disable()
_gc.set_threshold(0)
_gc.enable = lambda: None
_gc.collect = lambda *a, **kw: 0


#
# timing + reporting
#


JSONL = Path(tempfile.gettempdir()) / "bench" / "timings.jsonl"


def _log_timing(name: str, times: dict[str, float]) -> None:
    # append one timing row to JSONL log (process-safe via filelock)
    JSONL.parent.mkdir(exist_ok=True)
    with FileLock(str(JSONL) + ".lock"), open(JSONL, "a") as f:
        f.write(json.dumps({"kernel": name} | times) + "\n")


def _save_report(rows: list[dict]) -> None:
    # JSONL rows -> CSV + grouped bar chart PDF
    curr_dir = Path(__file__).resolve().parent / "e2e"
    backends = ["exo_c", "xdsl_mlir", "jit"]
    df = pl.DataFrame(rows)
    df.write_csv(curr_dir / "e2e_test_times.csv")
    df = df.group_by("kernel").agg(*(pl.col(b).mean() for b in backends))
    long = df.unpivot(index="kernel", on=backends, variable_name="backend", value_name="time_s").sort("kernel")
    # fmt: off
    colors = {"exo_c": "#4e79a7", "xdsl_mlir": "#f28e2b", "jit": "#e15759"}
    p = (
        ggplot(long.to_pandas(), aes(x="kernel", y="time_s", fill="backend"))
        + geom_col(position="dodge") + coord_flip()
        + scale_fill_manual(values=colors)
        + labs(x="", y="time (s)", fill="backend", title="e2e test runtimes")
        + theme_minimal()
        + theme(figure_size=(10, max(6, df.height * 0.22)))
        + theme(axis_text_y=element_text(size=7))
        + theme(legend_position="bottom")
    )
    # fmt: on
    p.save(curr_dir / "e2e_test_times.pdf", verbose=False)


#
# pytest hooks + test helper
#


def pytest_sessionstart(session):
    is_main = not hasattr(session.config, "workerinput")
    if not is_main:
        return
    JSONL.parent.mkdir(exist_ok=True)
    JSONL.unlink(missing_ok=True)


def pytest_sessionfinish(session, exitstatus):
    is_main = not hasattr(session.config, "workerinput")
    if not is_main or not JSONL.exists():
        return
    rows = sorted((json.loads(line) for line in JSONL.read_text().splitlines() if line.strip()), key=lambda r: r["kernel"])
    if not rows:
        return
    _save_report(rows)


def _timed(fn):
    # decorator that records wall-clock elapsed on wrapper.elapsed
    def wrapper(**kw):
        t0 = time.perf_counter()
        r = fn(**kw)
        wrapper.elapsed = time.perf_counter() - t0
        return r

    return wrapper


def assert_match(proc: Procedure, **kwargs: Any) -> None:
    # compile proc on all backends, verify outputs match exo_c (reference), log timings
    ir = proc._loopir_proc
    jit_fn = compile_jit(proc)[ir.name]
    backends = {
        "exo_c": _timed(compile_exo(proc)),
        "xdsl_mlir": _timed(compile_mlir(proc, to_mlir(proc))),
        "jit": _timed(lambda **kw: _call(jit_fn, ir, deepcopy(kw))),
    }
    results = {k: fn(**kwargs) for k, fn in backends.items()}
    _log_timing(ir.name, {k: fn.elapsed for k, fn in backends.items()})

    ref = results["exo_c"]
    for key in ("xdsl_mlir", "jit"):
        for name in ref:
            np.testing.assert_allclose(results[key][name], ref[name], atol=1e-6, err_msg=f"{key} mismatch on buffer '{name}'")
