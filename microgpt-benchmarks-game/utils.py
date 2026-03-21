# /// script
# requires-python = "==3.14.*"
# dependencies = ["polars"]
# ///

import csv
import inspect
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights.json"
TIMES_DIR = ROOT / "times"


def dump_weights(state_dict) -> None:
    WEIGHTS_PATH.write_text(json.dumps({k: [[v.data for v in row] for row in mat] for k, mat in state_dict.items()}))


def assert_weights_match(state_dict, atol: float = 1e-5) -> None:
    assert WEIGHTS_PATH.exists(), f"weights file not found: {WEIGHTS_PATH}"

    ref = json.load(open(WEIGHTS_PATH))
    cur = {k: [[v.data for v in row] for row in mat] for k, mat in state_dict.items()}
    assert set(ref) == set(cur), f"key mismatch: ref={set(ref)-set(cur)} cur={set(cur)-set(ref)}"

    for k in ref:
        assert len(ref[k]) == len(cur[k]) and len(ref[k][0]) == len(cur[k][0]), f"shape mismatch '{k}': {len(ref[k])}x{len(ref[k][0])} vs {len(cur[k])}x{len(cur[k][0])}"

    max_diff = 0.0
    max_loc = ""
    violations = 0
    total = 0
    rows = ((k, i, rr, cr) for k in ref for i, (rr, cr) in enumerate(zip(ref[k], cur[k])))
    all_cells = itertools.chain.from_iterable(((k, i, j, r, c) for j, (r, c) in enumerate(zip(rr, cr))) for k, i, rr, cr in rows)
    for k, i, j, r, c in all_cells:
        d = abs(r - c)
        total += 1
        violations += d > atol
        if d <= max_diff:
            continue
        max_diff, max_loc = d, f"{k}[{i}][{j}]"
    assert violations == 0, f"weights mismatch (atol={atol}): {violations}/{total} params exceed tolerance, max diff={max_diff:.2e} at {max_loc}"


def save_times(step_times: list[float]) -> None:
    name = Path(inspect.stack()[1].filename).stem
    path = TIMES_DIR / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_ms"])
        writer.writerows([[i + 1, f"{t * 1000:.3f}"] for i, t in enumerate(step_times)])
    print_times(path)


def print_times(path: Path) -> None:
    if not path.exists():
        return
    with open(path, "r") as f:
        times = [float(row["time_ms"]) * 1000 for row in csv.DictReader(f)]
    if not times:
        return
    n = len(times)
    mean = sum(times) / n
    variance = sum((x - mean) ** 2 for x in times) / (n - 1) if n > 1 else 0
    stddev = math.sqrt(variance)
    print(f"  {path.stem}: mean={mean:.0f}\u03bcs \u00b1{stddev:.0f} min={min(times):.0f} max={max(times):.0f} ({n} runs)")


def print_times_all() -> None:
    import polars as pl

    if not TIMES_DIR.exists():
        return

    rows = []
    for path in sorted(TIMES_DIR.glob("*.csv")):
        t: list[float] = (pl.read_csv(path)["time_ms"].cast(pl.Float64) * 1000).to_list()
        if not t:
            continue
        n, mean = len(t), sum(t) / len(t)
        sigma = math.sqrt(sum((x - mean) ** 2 for x in t) / (n - 1)) if n > 1 else 0
        rows.append({"name": path.stem, "mean": int(mean), "sigma": int(sigma), "min": int(min(t)), "max": int(max(t))})
    if not rows:
        return

    rows.sort(key=lambda r: r["mean"])

    original_mean = next((r["mean"] for r in rows if r["name"] == "original"), max(r["mean"] for r in rows))
    speedups = {r["name"]: original_mean / r["mean"] for r in rows}
    max_speedup = max(speedups.values())
    name_width = max(len(r["name"]) for r in rows)

    print("\n# speedup over original version\n")
    for r in sorted(rows, key=lambda r: -speedups[r["name"]]):
        bar = "▇" * round(speedups[r["name"]] / max_speedup * 60) or "▏"
        print(f"  {r['name']:<{name_width}}  {bar} {speedups[r['name']]:.0f}x")

    cols = ["name", "mean \u03bcs", "\u00b1\u03c3", "min", "max"]
    table_rows = [[r["name"], r["mean"], r["sigma"], r["min"], r["max"]] for r in rows]
    col_w = [max(len(str(cols[i])), max(len(str(row[i])) for row in table_rows)) for i in range(len(cols))]

    print("\n\n# leaderboard\n")
    print("  " + "  ".join(f"{cols[i]:<{col_w[i]}}" for i in range(len(cols))))
    print("  " + "  ".join("\u2500" * col_w[i] for i in range(len(cols))))
    for row in table_rows:
        print("  " + "  ".join(f"{str(row[i]):>{col_w[i]}}" if i > 0 else f"{row[i]:<{col_w[i]}}" for i in range(len(cols))))


if __name__ == "__main__":
    print_times_all()
