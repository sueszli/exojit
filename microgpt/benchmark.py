import argparse
import csv
import io
import os
import statistics
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns

BACKENDS = ("original", "plain", "numpy", "torch", "jax", "exojit")
RUNS = 5
STEPS = 1000
HEADER = "backend,run,steps,elapsed_ns\n"
CPU_ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), "1") | {"JAX_PLATFORMS": "cpu"}


def measure(backend: str, path: Path) -> None:
    impl = import_module(f"{backend}_impl")
    assert impl.NUM_STEPS == STEPS
    impl.reset()
    impl.train()
    impl.check()
    print(f"{backend}: warm-up passed", file=sys.stderr, flush=True)

    rows = []
    for run in range(1, RUNS + 1):
        impl.reset()
        start = perf_counter_ns()
        impl.train()
        elapsed = perf_counter_ns() - start
        impl.check()
        rows.append((backend, run, STEPS, elapsed))
        print(f"{backend}: run {run}/{RUNS}, {elapsed / 1e9:.3f} s", file=sys.stderr, flush=True)
    with path.open("a", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def collect_runs() -> str:
    with tempfile.TemporaryDirectory(prefix="exojit-microgpt-") as temp:
        path = Path(temp) / "benchmark.csv"
        path.write_text(HEADER)
        for backend in BACKENDS:
            cmd = [sys.executable, __file__, "--worker", backend, "--output", str(path)]
            status = subprocess.call(cmd, stdout=sys.stderr, env=os.environ | CPU_ENV)
            assert status == 0, f"{backend}: benchmark failed"
        return path.read_text()


def medians_in_us(results: str) -> dict[str, float]:
    timings = {backend: [] for backend in BACKENDS}
    rows = csv.reader(io.StringIO(results))
    assert next(rows, None) == HEADER.strip().split(","), "invalid timing csv header"
    for row in rows:
        assert len(row) == 4, "invalid timing csv row"
        name, run, steps, elapsed = row
        assert name in timings and steps == str(STEPS)
        assert run == str(len(timings[name]) + 1)
        assert elapsed.isdecimal() and int(elapsed) > 0
        timings[name].append(int(elapsed))
    assert all(len(samples) == RUNS for samples in timings.values()), "expected five runs per backend"
    return {name: statistics.median(samples) / STEPS / 1000 for name, samples in timings.items()}


def format_table(medians: dict[str, float]) -> str:
    header = f"{'name':<8}  {'median us':<9}  speedup"
    baseline = medians["original"]
    rows = [f"{name:<8}  {medians[name]:<9.0f}  {baseline / medians[name]:.0f}x" for name in sorted(medians, key=medians.get)]
    return "\n".join((header, "─" * len(header), *rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark warmed 1,000-step MicroGPT runs.")
    parser.add_argument("--output", type=Path, help="Save raw timing samples as CSV")
    parser.add_argument("--table-from", type=Path, help="Read saved CSV timings without rerunning")
    parser.add_argument("--plot", type=Path, help="Save a chart of median runtimes")
    parser.add_argument("--archived-plot", type=Path, help="Rebuild the original chart from archived CSVs")
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        assert args.output is not None
        measure(args.worker, args.output)
        return
    if args.archived_plot:
        from benchmark_plot import save_archived_plot

        save_archived_plot(args.archived_plot)
        return

    results = args.table_from.read_text() if args.table_from else collect_runs()
    medians = medians_in_us(results)
    if args.output:
        args.output.write_text(results)
    print(format_table(medians))
    if args.plot:
        from benchmark_plot import save_median_plot

        save_median_plot(medians, args.plot)


if __name__ == "__main__":
    main()
