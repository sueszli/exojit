from functools import partial
from pathlib import Path
from statistics import median
from timeit import Timer

import numpy as np
import polars as pl
from exo.API import Procedure
from exo.stdlib.scheduling import divide_loop, fission, reorder_loops, simplify, stage_mem, unroll_loop
from matmul_exo import _matmul

from exojit import jit

SIZE = 32
CALLS = 10_000
ROUNDS = 9
OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "matmul-scheduling.csv"


def optimize_matmul(p: Procedure) -> Procedure:
    p = divide_loop(p, "i", 2, ["io", "ii"], perfect=True)
    p = stage_mem(p, p.find_loop("ii").as_block(), "C[2*io:2*io+2, 0:32]", "acc")
    p = fission(p, p.find_loop("k").before(), n_lifts=2)
    p = reorder_loops(p, "j k")
    p = reorder_loops(p, "ii k")
    return simplify(unroll_loop(p, "ii #1"))


def main() -> None:
    rng = np.random.default_rng(41)
    a, b = (rng.uniform(-1, 1, (SIZE, SIZE)).astype(np.float32) for _ in range(2))
    expected = a @ b
    kernel = _matmul.partial_eval(M=SIZE, K=SIZE, N=SIZE)
    out = np.empty_like(a)
    timers = {}
    for name, schedule in (("unscheduled", None), ("scheduled", optimize_matmul)):
        out.fill(np.nan)
        call = partial(jit(kernel, optimize=schedule, raw=True), out.ctypes.data, a.ctypes.data, b.ctypes.data)
        timer = Timer(call)
        timer.timeit(number=CALLS)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
        timers[name] = timer

    times = {name: [] for name in timers}
    order = tuple(timers)
    # alternate order so either schedule can run first
    for i in range(ROUNDS):
        for name in order if i % 2 == 0 else order[::-1]:
            times[name].append(timers[name].timeit(number=CALLS) * 1e6 / CALLS)

    medians = {name: median(samples) for name, samples in times.items()}
    rows = [(SIZE, str(a.dtype), name, us, medians["unscheduled"] / us) for name, us in medians.items()]
    pl.DataFrame(rows, schema=["size", "dtype", "schedule", "median_us_per_call", "speedup"], orient="row").write_csv(OUTPUT, float_precision=2)


if __name__ == "__main__":
    main()
