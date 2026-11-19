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
STEP_BACKENDS = {"original", "plain", "numpy", "torch"}
THREAD_ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), "1")
BENCH_ENV = {**THREAD_ENV, "JAX_PLATFORMS": "cpu"}
ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return n


def record_steps(impl, reference: dict[str, list[list[float]]]) -> list[int]:
    from utils import assert_weights_match

    samples_ns: list[int] = []
    impl.reset()
    impl.train(samples_ns)
    assert len(samples_ns) == impl.NUM_STEPS
    assert_weights_match(impl.weights(), reference=reference)
    return samples_ns


def measure(backend: str, warmups: int, runs: int, reference_path: Path, *, profile_steps: bool = False) -> dict:
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
        if reference is None:
            assert_weights_match(actual, reference=actual)
            reference = actual
            reference_path.write_text(json.dumps(reference, separators=(",", ":")))
        else:
            assert_weights_match(actual, reference=reference)
        if i < warmups:
            print(f"{backend}: warm-up {i + 1}/{warmups} passed", file=sys.stderr, flush=True)
        else:
            samples_ns.append(elapsed)
            print(f"{backend}: run {i + 1 - warmups}/{runs}, {elapsed / 1e9:.3f} s ({elapsed} ns)", file=sys.stderr, flush=True)
    result = {
        "steps_per_run": impl.NUM_STEPS,
        "samples_ns": samples_ns,
        "median_step_us": statistics.median(samples_ns) / impl.NUM_STEPS / 1000,
    }
    if profile_steps:
        assert backend in STEP_BACKENDS and reference is not None
        result["step_samples_ns"] = record_steps(impl, reference)
        print(f"{backend}: recorded {impl.NUM_STEPS} individual steps in a separate warmed run", file=sys.stderr, flush=True)
    return result


def samples_in_us(report: dict) -> dict[str, list[float]]:
    return {name: [ns / (result["steps_per_run"] * 1000) for ns in result["samples_ns"]] for name, result in report["backends"].items()}


def format_table(report: dict) -> str:
    samples_us = samples_in_us(report)
    medians = {name: statistics.median(samples) for name, samples in samples_us.items()}
    baseline = medians["original"]
    headings = ("name", "median us", "IQR", "min", "max", "speedup")
    rows = []
    for name in sorted(medians, key=medians.get):
        samples = samples_us[name]
        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else samples * 3
        iqr = q3 - q1
        spread = "<1" if 0 < iqr < 1 else f"{iqr:.0f}"
        rows.append((name, f"{medians[name]:.0f}", spread, f"{min(samples):.0f}", f"{max(samples):.0f}", f"{baseline / medians[name]:.0f}x"))

    widths = [max(map(len, column)) for column in zip(headings, *rows)]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()

    header = render(headings)
    return "\n".join((header, "─" * len(header), *(render(row) for row in rows)))


def archived_samples_in_us() -> dict[str, list[float]]:
    samples = {}
    for name in BACKENDS:
        with (ROOT / "times" / f"{name}.csv").open(newline="") as f:
            samples[name] = [float(row["time_ms"]) * 1000 for row in csv.DictReader(f)]
    return samples


def save_plot(samples: dict[str, list[float]], path: Path, *, archived: bool = False, step_samples: dict[str, list[float]] | None = None) -> None:
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    names = {"exojit": "ExoJIT", "jax": "JAX", "numpy": "NumPy", "torch": "PyTorch", "plain": "Python (Optimized)", "original": "Python (Original)"}
    if archived:
        centers = {name: int(statistics.mean(values)) for name, values in samples.items()}
        whole_loop = {"exojit", "jax"}
    else:
        centers = {name: statistics.median(values) for name, values in samples.items()}
        whole_loop = set(samples) - set(step_samples) if step_samples is not None else set()
    per_step_plot = archived or step_samples is not None
    if step_samples is not None and step_samples.keys() != STEP_BACKENDS:
        raise ValueError("expected per-step samples for all four step-timed backends")
    order = sorted(centers, key=centers.get)
    data = [step_samples[name] if step_samples is not None and name in step_samples else samples[name] for name in order]
    blue = "#2459b5"

    salt = "exojit-microgpt-training" if archived else "microgpt-benchmark"
    with rc_context({"font.family": "cmr10", "font.size": 7.5, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "svg.fonttype": "path", "svg.hashsalt": salt}):
        width, height = 360, 126
        fig = Figure(figsize=(width / 72, height / 72), facecolor="white")
        ax = fig.add_axes((96 / width, 29 / height, 175 / width, 81 / height))
        step_rows = []
        for row, name in enumerate(order):
            if name in whole_loop:
                # The archived CSV repeats one whole-loop mean, not individual samples.
                if archived and set(samples[name]) != {centers[name]}:
                    raise ValueError(f"{name}: expected a repeated whole-loop mean")
                ax.scatter([centers[name]], [row], s=11, facecolors="none", edgecolors=blue, linewidths=0.63, zorder=3)
            else:
                step_rows.append(row)
        ax.boxplot(
            [data[row] for row in step_rows],
            orientation="horizontal",
            manage_ticks=False,
            positions=step_rows,
            widths=0.46,
            showfliers=per_step_plot,
            patch_artist=True,
            boxprops={"facecolor": "#e7eef9", "edgecolor": blue, "linewidth": 0.55},
            medianprops={"color": "#173d83", "linewidth": 0.65},
            whiskerprops={"color": blue, "linewidth": 0.45},
            capprops={"color": blue, "linewidth": 0.45},
            flierprops={"marker": ".", "markersize": 1.6, "markerfacecolor": blue, "markeredgewidth": 0, "alpha": 0.28, "zorder": 2},
        )
        if not per_step_plot:
            for row, values in enumerate(data):
                offsets = [row + 0.3 * (i - (len(values) - 1) / 2) / max(len(values) - 1, 1) for i in range(len(values))]
                ax.scatter(sorted(values), offsets, s=2.4, color=blue, linewidths=0, alpha=0.55, zorder=3)

        lo = 1 if archived else math.floor(math.log10(min(map(min, data))))
        if step_samples is not None:
            xmax = max(10 ** (lo + 1) * 1.05, 1.08 * max(map(max, data)))
            hi = math.floor(math.log10(xmax))
        else:
            hi = 5 if archived else max(lo + 1, math.ceil(math.log10(max(map(max, data)))))
            xmax = 2 * 10**hi
        ticks = [10**exponent for exponent in range(lo, hi + 1)]
        ax.set_xscale("log")
        ax.set_xlim(ticks[0], xmax)
        ax.set_xticks(ticks, [f"{tick / 1000:g} ms" if tick >= 1000 else "$" + rf"{tick:g}\,\mu\mathrm{{s}}" + "$" for tick in ticks])
        ax.set_yticks(range(len(order)), [names[name] for name in order])
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.set_xlabel("time per training step (log scale)", fontsize=7, labelpad=6)
        ax.tick_params(axis="both", which="both", length=0)
        ax.tick_params(axis="x", pad=3)
        ax.tick_params(axis="y", pad=5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        center_label = "mean" if archived else ("run median" if per_step_plot else "median")
        columns = [(306, center_label), (348, "vs. original")]
        row_text = blended_transform_factory(fig.transFigure, ax.transData)
        for x, heading in columns:
            fig.text(x / width, 1 - 10.5 / height, heading, ha="right", va="center")
        for row, name in enumerate(order):
            us = centers[name]
            elapsed = "$" + (rf"{us:.0f}\,\mu\mathrm{{s}}" if us < 1000 else rf"{us / 1000:.3f}\,\mathrm{{ms}}") + "$"
            speedup = f"{centers['original'] / us:,.0f}x"
            for (x, _), label in zip(columns, (elapsed, speedup)):
                ax.text(x / width, row, label, transform=row_text, ha="right", va="center")

        if archived:
            description = "Four horizontal boxplots summarize individual training steps. Hollow circles mark the whole-loop means for ExoJIT and JAX. Durations are shown on a logarithmic axis. " + "; ".join(f"{names[name]}: {centers[name]} microseconds" for name in order) + f". The ExoJIT port is about {centers['original'] / centers['exojit']:,.0f} times faster than the original implementation."
        elif per_step_plot:
            description = "Each box summarizes individual steps from a separate warmed profiling pass. Hollow circles show ExoJIT and JAX run medians because their loops are timed as a whole. The columns use the medians and speedups from the uninstrumented warmed runs."
        else:
            description = "Horizontal boxplots of time per training step on a log scale. Each dot is one measured run. Columns show the median and its speedup relative to the original Python implementation."

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=216,
            metadata={
                "Title": "MicroGPT training time" if archived else "MicroGPT training times",
                "Description": description,
                "Date": None,
            },
        )
        if path.suffix.lower() == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def show_report(report: dict, plot: Path | None) -> None:
    print(format_table(report))
    if plot is not None:
        steps = {name: [ns / 1000 for ns in result["step_samples_ns"]] for name, result in report["backends"].items() if "step_samples_ns" in result}
        save_plot(samples_in_us(report), plot, step_samples=steps or None)
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
    parser.add_argument("--plot", type=Path, help="Save the timing chart, with a separate per-step profiling pass for new runs")
    parser.add_argument("--archived-plot", type=Path, help="Rebuild the original chart from the archived per-step CSVs")
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--profile-steps", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.reference:
            parser.error("--worker requires --reference")
        print(json.dumps(measure(args.worker, args.warmups, args.runs, args.reference, profile_steps=args.profile_steps)))
        return

    if args.archived_plot:
        save_plot(archived_samples_in_us(), args.archived_plot, archived=True)
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
            command = [sys.executable, __file__, "--worker", backend, "--warmups", str(args.warmups), "--runs", str(args.runs), "--reference", str(reference)]
            if args.plot and backend in STEP_BACKENDS:
                command.append("--profile-steps")
            process = subprocess.run(command, stdout=subprocess.PIPE, text=True, env={**os.environ, **BENCH_ENV}, check=True)
            report["backends"][backend] = json.loads(process.stdout)
        report["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()

    if args.output:
        with args.output.open("x") as f:
            f.write(json.dumps(report, indent=2) + "\n")
    show_report(report, args.plot)


if __name__ == "__main__":
    main()
