from functools import partial
from pathlib import Path
from statistics import median
from timeit import Timer

import numpy as np
import polars as pl
from exo.API import Procedure
from exo.libs.memories import DRAM_STACK
from exo.stdlib.scheduling import divide_loop, fission, lift_alloc, reorder_loops, set_memory, simplify, stage_mem
from matmul_exo import _matmul

from exojit import jit


def optimize_matmul(p: Procedure) -> Procedure:
    p = divide_loop(p, "j", 32, ["jo", "ji"], perfect=True)
    p = reorder_loops(p, "i jo")
    p = divide_loop(p, "i", 2, ["io", "ii"], perfect=True)
    p = stage_mem(p, p.find_loop("ii").as_block(), "C[2*io:2*io+2, 32*jo:32*jo+32]", "acc")
    p = fission(p, p.find_loop("k").before(), n_lifts=2)
    p = reorder_loops(p, "ji k")
    p = reorder_loops(p, "ii k")
    p = lift_alloc(simplify(p), "acc", n_lifts=2)
    return set_memory(p, "acc", DRAM_STACK)


def main(calls: int = 10_000, rounds: int = 9, output: Path | None = None) -> None:
    assert calls > 0 and rounds > 0
    output = output or Path(__file__).resolve().parents[1] / "assets" / "matmul-scheduling.csv"
    size = 32
    rng = np.random.default_rng(41)
    a, b = (rng.uniform(-1, 1, (size, size)).astype(np.float32) for _ in range(2))
    expected = a @ b
    kernel = _matmul.partial_eval(M=size, K=size, N=size)
    out = np.empty_like(a)
    benches = []
    for name, schedule in (("unscheduled", None), ("scheduled", optimize_matmul)):
        out.fill(np.nan)
        call = partial(jit(kernel, optimize=schedule, raw=True), out.ctypes.data, a.ctypes.data, b.ctypes.data)
        timer = Timer(call)
        timer.timeit(number=calls)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
        benches.append((name, timer, []))
    # alternate order so either schedule can run first
    for i in range(rounds):
        for _, timer, samples in benches if i % 2 == 0 else reversed(benches):
            samples.append(timer.timeit(number=calls) * 1e6 / calls)

    medians = {name: median(samples) for name, _, samples in benches}
    rows = [(size, str(a.dtype), name, us, medians["unscheduled"] / us) for name, us in medians.items()]
    pl.DataFrame(rows, schema=["size", "dtype", "schedule", "median_us_per_call", "speedup"], orient="row").write_csv(output, float_precision=2)


if __name__ == "__main__":
    main()
