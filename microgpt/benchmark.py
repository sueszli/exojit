import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from importlib import import_module, metadata
from pathlib import Path

BACKENDS = ("original", "plain", "numpy", "torch", "jax", "exojit")
THREAD_ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), "1")
BENCH_ENV = {**THREAD_ENV, "JAX_PLATFORMS": "cpu"}
ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return n


def measure(backend: str, warmups: int, runs: int, reference_path: Path) -> dict:
    from utils import assert_weights_match

    impl = import_module(f"{backend}_impl")
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    if reference is None and backend != "original":
        raise ValueError("benchmark Python (Original) first to generate the reference weights")
    samples_ns = []
    for i in range(warmups + runs):
        impl.reset()
        t0 = time.perf_counter_ns()
        impl.train()
        elapsed = time.perf_counter_ns() - t0
        actual = impl.weights()
        assert_weights_match(actual, reference=actual if reference is None else reference)
        if reference is None:
            reference = actual
            reference_path.write_text(json.dumps(reference, separators=(",", ":")))
        if i < warmups:
            print(f"{backend}: warm-up {i + 1}/{warmups} passed", file=sys.stderr, flush=True)
        else:
            samples_ns.append(elapsed)
            print(f"{backend}: run {i + 1 - warmups}/{runs}, {elapsed / 1e9:.3f} s ({elapsed} ns)", file=sys.stderr, flush=True)
    return {
        "steps_per_run": impl.NUM_STEPS,
        "samples_ns": samples_ns,
        "median_step_us": statistics.median(samples_ns) / impl.NUM_STEPS / 1000,
    }


def medians_in_us(report: dict) -> dict[str, float]:
    return {name: statistics.median(result["samples_ns"]) / result["steps_per_run"] / 1000 for name, result in report["backends"].items()}


def format_table(medians: dict[str, float]) -> str:
    header = f"{'name':<8}  {'median us':<9}  speedup"
    baseline = medians["original"]
    rows = [f"{name:<8}  {medians[name]:<9.0f}  {baseline / medians[name]:.0f}x" for name in sorted(medians, key=medians.get)]
    return "\n".join((header, "─" * len(header), *rows))


def archived_samples_in_us() -> dict[str, list[float]]:
    samples = {}
    for name in BACKENDS:
        with (ROOT / "times" / f"{name}.csv").open(newline="") as f:
            samples[name] = [float(row["time_ms"]) * 1000 for row in csv.DictReader(f)]
    return samples


PLOT_NAMES = {"exojit": "ExoJIT", "jax": "JAX", "numpy": "NumPy", "torch": "PyTorch", "plain": "Python (Optimized)", "original": "Python (Original)"}
PLOT_BLUE = "#2459b5"
PLOT_STYLE = {"font.family": "cmr10", "font.size": 7.5, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "svg.fonttype": "path"}


def save_figure(fig, path: Path, title: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=216, metadata={"Title": title, "Description": description, "Date": None})
    if path.suffix.lower() == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def save_archived_plot(samples: dict[str, list[float]], path: Path) -> None:
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    centers = {name: int(statistics.mean(values)) for name, values in samples.items()}
    order = sorted(centers, key=centers.get)
    whole_loop = {"exojit", "jax"}

    with rc_context(PLOT_STYLE | {"svg.hashsalt": "exojit-microgpt-training"}):
        width, height = 360, 126
        fig = Figure(figsize=(width / 72, height / 72), facecolor="white")
        ax = fig.add_axes((96 / width, 29 / height, 175 / width, 81 / height))
        step_rows = []
        for row, name in enumerate(order):
            if name in whole_loop:
                # the archived csv repeats one whole-loop mean, not individual samples
                if len(set(samples[name])) != 1:
                    raise ValueError(f"{name}: expected a repeated whole-loop mean")
                ax.scatter([centers[name]], [row], s=11, facecolors="none", edgecolors=PLOT_BLUE, linewidths=0.63, zorder=3)
            else:
                step_rows.append(row)
        ax.boxplot(
            [samples[order[row]] for row in step_rows],
            orientation="horizontal",
            manage_ticks=False,
            positions=step_rows,
            widths=0.46,
            showfliers=True,
            patch_artist=True,
            boxprops={"facecolor": "#e7eef9", "edgecolor": PLOT_BLUE, "linewidth": 0.55},
            medianprops={"color": "#173d83", "linewidth": 0.65},
            whiskerprops={"color": PLOT_BLUE, "linewidth": 0.45},
            capprops={"color": PLOT_BLUE, "linewidth": 0.45},
            flierprops={"marker": ".", "markersize": 1.6, "markerfacecolor": PLOT_BLUE, "markeredgewidth": 0, "alpha": 0.28, "zorder": 2},
        )
        ticks = [10, 100, 1000, 10_000, 100_000]
        ax.set_xscale("log")
        ax.set_xlim(ticks[0], 2 * ticks[-1])
        ax.set_xticks(ticks, [f"{tick / 1000:g} ms" if tick >= 1000 else "$" + rf"{tick:g}\,\mu\mathrm{{s}}" + "$" for tick in ticks])
        ax.set_yticks(range(len(order)), [PLOT_NAMES[name] for name in order])
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.set_xlabel("time per training step (log scale)", fontsize=7, labelpad=6)
        ax.tick_params(axis="both", which="both", length=0)
        ax.tick_params(axis="x", pad=3)
        ax.tick_params(axis="y", pad=5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        columns = [(306, "mean"), (348, "vs. original")]
        row_text = blended_transform_factory(fig.transFigure, ax.transData)
        for x, heading in columns:
            fig.text(x / width, 1 - 10.5 / height, heading, ha="right", va="center")
        for row, name in enumerate(order):
            us = centers[name]
            elapsed = "$" + (rf"{us:.0f}\,\mu\mathrm{{s}}" if us < 1000 else rf"{us / 1000:.3f}\,\mathrm{{ms}}") + "$"
            speedup = f"{centers['original'] / us:,.0f}x"
            for (x, _), label in zip(columns, (elapsed, speedup)):
                ax.text(x / width, row, label, transform=row_text, ha="right", va="center")

        description = "Four horizontal boxplots summarize individual training steps. Hollow circles mark the whole-loop means for ExoJIT and JAX. Durations are shown on a logarithmic axis. " + "; ".join(f"{PLOT_NAMES[name]}: {centers[name]} microseconds" for name in order) + f". The ExoJIT port is about {centers['original'] / centers['exojit']:,.0f} times faster than the original implementation."
        save_figure(fig, path, "MicroGPT training time", description)


def save_median_plot(medians: dict[str, float], path: Path) -> None:
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    names = PLOT_NAMES | {"plain": "Optimized Python", "original": "Original Python"}
    order = sorted(medians, key=medians.get)
    times = [medians[name] for name in order]
    lo = math.floor(math.log10(min(times)))
    if 10**lo == min(times):
        lo -= 1
    hi = max(lo + 1, math.ceil(math.log10(max(times))))
    ticks = [10**exponent for exponent in range(lo, hi + 1)]
    left = ticks[0]

    with rc_context(PLOT_STYLE | {"svg.hashsalt": "microgpt-benchmark"}):
        width, height = 360, 126
        fig = Figure(figsize=(width / 72, height / 72), facecolor="white")
        ax = fig.add_axes((96 / width, 29 / height, 201 / width, 81 / height))
        ax.set_xscale("log")
        ax.barh(range(len(order)), [time - left for time in times], left=left, height=0.48, color=PLOT_BLUE)
        ax.set_xlim(left, ticks[-1])
        ax.set_xticks(ticks, [f"{tick / 1000:g} ms" if tick >= 1000 else "$" + rf"{tick:g}\,\mu\mathrm{{s}}" + "$" for tick in ticks])
        ax.set_yticks(range(len(order)), [names[name] for name in order])
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.set_xlabel("median time per training step (log scale, lower is better)", fontsize=7, labelpad=6)
        ax.tick_params(axis="both", which="both", length=0, pad=4)
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        row_text = blended_transform_factory(fig.transFigure, ax.transData)
        fig.text(348 / width, 1 - 10.5 / height, "median", ha="right", va="center")
        for row, us in enumerate(times):
            elapsed = "$" + (rf"{us:.0f}\,\mu\mathrm{{s}}" if us < 1000 else rf"{us / 1000:.1f}\,\mathrm{{ms}}") + "$"
            ax.text(348 / width, row, elapsed, transform=row_text, ha="right", va="center")

        save_figure(fig, path, "MicroGPT training time", "Horizontal bars show median runtime per training step on a logarithmic scale. Each bar starts at the smallest labeled tick, not zero. Shorter is faster.")


def show_report(report: dict, plot: Path | None) -> None:
    medians = medians_in_us(report)
    print(format_table(medians))
    if plot is not None:
        save_median_plot(medians, plot)
        print(f"Plot: {plot}", file=sys.stderr)


def hardware() -> dict:
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            chip = result.stdout.strip()
    return {"cpu": chip, "architecture": platform.machine(), "logical_cores": os.cpu_count(), "os": platform.platform()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark warmed MicroGPT training loops, recording one sample per run.")
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, help="Write run data to a new JSON file")
    parser.add_argument("--table-from", type=Path, help="Print the table for a saved JSON report without rerunning")
    parser.add_argument("--plot", type=Path, help="Save a horizontal boxplot after printing the table")
    parser.add_argument("--archived-plot", type=Path, help="Rebuild the original chart from the archived per-step CSVs")
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.reference or not args.output:
            parser.error("--worker requires --reference and --output")
        result = measure(args.worker, args.warmups, args.runs, args.reference)
        with args.output.open("x") as f:
            json.dump(result, f)
        return

    if args.archived_plot:
        save_archived_plot(archived_samples_in_us(), args.archived_plot)
        print(f"Plot: {args.archived_plot}", file=sys.stderr)
        return

    if args.table_from:
        show_report(json.loads(args.table_from.read_text()), args.plot)
        return

    if args.output and args.output.exists():
        parser.error(f"file already exists: {args.output}")
    report = {
        "hardware": hardware(),
        "python": platform.python_version(),
        "versions": {name: metadata.version(name) for name in ("exojit", "exo-lang", "xdsl", "llvmlite", "numpy", "torch", "jax", "jaxlib", "optax")},
        "input_sha256": hashlib.sha256((ROOT / "input.txt").read_bytes()).hexdigest(),
        "environment": BENCH_ENV,
        "warmup_runs": args.warmups,
        "measured_runs": args.runs,
        "backends": {},
    }
    with tempfile.TemporaryDirectory(prefix="exojit-microgpt-") as temp:
        reference = Path(temp) / "weights.json"
        for backend in BACKENDS:
            result = reference.with_name(f"{backend}.json")
            command = [sys.executable, __file__, "--worker", backend, "--warmups", str(args.warmups), "--runs", str(args.runs), "--reference", str(reference), "--output", str(result)]
            subprocess.run(command, stdout=sys.stderr, env={**os.environ, **BENCH_ENV}, check=True)
            report["backends"][backend] = json.loads(result.read_text())
        report["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()

    if args.output:
        with args.output.open("x") as f:
            f.write(json.dumps(report, indent=2) + "\n")
    show_report(report, args.plot)


if __name__ == "__main__":
    main()
