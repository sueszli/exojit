import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from importlib import import_module, metadata
from pathlib import Path

from utils import assert_weights_match

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
    return {
        "steps_per_run": impl.NUM_STEPS,
        "samples_ns": samples_ns,
        "median_step_us": statistics.median(samples_ns) / impl.NUM_STEPS / 1000,
    }


def hardware() -> dict:
    chip = platform.processor()
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            chip = result.stdout.strip()
    elif sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                chip = line.partition(":")[2].strip()
                break
    return {"cpu": chip, "architecture": platform.machine(), "logical_cores": os.cpu_count(), "os": platform.platform()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark warmed MicroGPT training loops, recording one sample per run.")
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, help="Write run data to a new JSON file")
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.reference:
            parser.error("--worker requires --reference")
        print(json.dumps(measure(args.worker, args.warmups, args.runs, args.reference)))
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
            process = subprocess.run(command, stdout=subprocess.PIPE, text=True, env={**os.environ, **BENCH_ENV}, check=True)
            report["backends"][backend] = json.loads(process.stdout)
        report["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()

    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()
