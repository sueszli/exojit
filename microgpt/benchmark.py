import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns

import polars as pl
import tyro
from benchmark_plot import save_benchmark_plot


def measure(backend: str, runs: int, steps: int) -> pl.DataFrame:
    impl = import_module(f"{backend}_impl")
    assert impl.NUM_STEPS == steps
    impl.reset()
    impl.train()
    impl.check()

    timings = []
    for _ in range(runs):
        impl.reset()
        start = perf_counter_ns()
        impl.train()
        elapsed = perf_counter_ns() - start
        impl.check()
        timings.append(elapsed)
    return pl.DataFrame({"backend": backend, "run": range(1, runs + 1), "steps": steps, "elapsed_ns": timings})


def read_runs(source: Path | bytes) -> pl.DataFrame:
    schema = {"backend": pl.String, "run": pl.Int64, "steps": pl.Int64, "elapsed_ns": pl.Int64}
    runs = pl.read_csv(source, schema_overrides=schema, ignore_errors=True, raise_if_empty=False, glob=False)
    assert runs.columns == list(schema), "invalid timing csv header"
    return runs


def collect_runs(backends: tuple[str, ...]) -> pl.DataFrame:
    env = os.environ | dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), "1") | {"JAX_PLATFORMS": "cpu"}
    samples = []
    for backend in backends:
        cmd = [sys.executable, __file__, "--worker", backend]
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, env=env)
        assert result.returncode == 0, f"{backend}: benchmark failed"
        samples.append(read_runs(result.stdout))
    return pl.concat(samples)


def medians_in_us(samples: pl.DataFrame, backends: tuple[str, ...], runs: int, steps: int) -> dict[str, float]:
    assert set(samples["backend"]) == set(backends), "unexpected backends"
    assert samples.height == samples.n_unique(["backend", "run"]) == len(backends) * runs, f"expected {runs} runs per backend"
    assert samples.filter(pl.col("run").is_between(1, runs), pl.col("steps") == steps, pl.col("elapsed_ns") > 0).height == samples.height, "invalid timing sample"
    return dict(samples.group_by("backend").agg(pl.col("elapsed_ns").median() / (steps * 1000)).iter_rows())


def main(output: Path | None = None, from_csv: Path | None = None, plot: Path | None = None, worker: str | None = None) -> None:
    backends = ("original", "plain", "numpy", "torch", "jax", "exojit")
    runs, steps = 5, 1000
    if worker is not None:
        assert worker in backends
        measure(worker, runs, steps).write_csv(sys.stdout)
        return

    assets = Path(__file__).resolve().parent / "assets"
    samples = collect_runs(backends) if from_csv is None else read_runs(from_csv)
    medians = medians_in_us(samples, backends, runs, steps)
    if from_csv is None or output is not None:
        samples.write_csv(output or assets / "benchmark.csv")
    save_benchmark_plot(medians, plot or assets / "benchmark.svg")


if __name__ == "__main__":
    tyro.cli(main)
