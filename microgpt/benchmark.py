import os
import subprocess
import sys
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns

import polars as pl
import tyro

BACKENDS = ("original", "plain", "numpy", "torch", "jax", "exojit")
RUNS = 5
STEPS = 1000
SCHEMA = {"backend": pl.String, "run": pl.Int64, "steps": pl.Int64, "elapsed_ns": pl.Int64}
CPU_ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), "1") | {"JAX_PLATFORMS": "cpu"}


def measure(backend: str) -> pl.DataFrame:
    assert backend in BACKENDS
    impl = import_module(f"{backend}_impl")
    assert impl.NUM_STEPS == STEPS
    impl.reset()
    impl.train()
    impl.check()
    print(f"{backend}: warm-up passed", flush=True)

    rows = []
    for run in range(1, RUNS + 1):
        impl.reset()
        start = perf_counter_ns()
        impl.train()
        elapsed = perf_counter_ns() - start
        impl.check()
        rows.append((backend, run, STEPS, elapsed))
        print(f"{backend}: run {run}/{RUNS}, {elapsed / 1e9:.3f} s", flush=True)
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def read_runs(source: Path | bytes) -> pl.DataFrame:
    runs = pl.read_csv(source, infer_schema=False, raise_if_empty=False, glob=False)
    assert runs.columns == list(SCHEMA), "invalid timing csv header"
    return runs.cast(SCHEMA, strict=False)


def collect_runs() -> pl.DataFrame:
    runs = []
    for backend in BACKENDS:
        cmd = [sys.executable, __file__, "--worker", backend]
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, env=os.environ | CPU_ENV)
        assert result.returncode == 0, f"{backend}: benchmark failed"
        runs.append(read_runs(result.stdout))
    return pl.concat(runs)


def medians_in_us(runs: pl.DataFrame) -> dict[str, float]:
    assert set(runs["backend"]) == set(BACKENDS), "unexpected backends"
    assert runs.height == runs.n_unique(["backend", "run"]) == len(BACKENDS) * RUNS, "expected five runs per backend"
    assert runs.filter(pl.col("run").is_between(1, RUNS), pl.col("steps") == STEPS, pl.col("elapsed_ns") > 0).height == runs.height, "invalid timing sample"
    return dict(runs.group_by("backend").agg(pl.col("elapsed_ns").median() / (STEPS * 1000)).iter_rows())


def format_table(medians: dict[str, float]) -> str:
    header = f"{'name':<8}  {'median us':<9}  speedup"
    baseline = medians["original"]
    rows = [f"{name:<8}  {medians[name]:<9.0f}  {baseline / medians[name]:.0f}x" for name in sorted(medians, key=medians.get)]
    return "\n".join((header, "─" * len(header), *rows))


def main(output: Path | None = None, table_from: Path | None = None, plot: Path | None = None, worker: str | None = None) -> None:
    """Benchmark warmed 1,000-step MicroGPT runs."""
    if worker is not None:
        # keep backend diagnostics out of the timing csv
        with redirect_stdout(sys.stderr):
            runs = measure(worker)
        runs.write_csv(sys.stdout)
        return

    runs = collect_runs() if table_from is None else read_runs(table_from)
    medians = medians_in_us(runs)
    if output is not None:
        runs.write_csv(output)
    print(format_table(medians))
    if plot is not None:
        from benchmark_plot import save_median_plot

        save_median_plot(medians, plot)


if __name__ == "__main__":
    tyro.cli(main)
