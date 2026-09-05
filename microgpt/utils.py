import csv
import inspect
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights.json"
TIMES_DIR = ROOT / "times"


def assert_weights_match(state_dict, atol: float = 1e-5) -> None:
    assert WEIGHTS_PATH.exists(), f"weights file not found: {WEIGHTS_PATH}"
    with WEIGHTS_PATH.open() as f:
        ref = json.load(f)
    cur = {k: [[v.data for v in row] for row in mat] for k, mat in state_dict.items()}
    assert set(ref) == set(cur), f"key mismatch: ref={set(ref) - set(cur)} cur={set(cur) - set(ref)}"
    for k in ref:
        assert len(ref[k]) == len(cur[k]) and len(ref[k][0]) == len(cur[k][0]), f"shape mismatch '{k}': {len(ref[k])}x{len(ref[k][0])} vs {len(cur[k])}x{len(cur[k][0])}"
    max_diff = 0.0
    max_loc = ""
    violations = 0
    total = 0
    rows = ((k, i, rr, cr) for k in ref for i, (rr, cr) in enumerate(zip(ref[k], cur[k])))
    all_cells = itertools.chain.from_iterable(((k, i, j, r, c) for j, (r, c) in enumerate(zip(rr, cr))) for k, i, rr, cr in rows)
    nan_loc = ""
    for k, i, j, r, c in all_cells:
        d = abs(r - c)
        total += 1
        if d <= atol:
            continue
        violations += 1
        if math.isnan(d):
            nan_loc = nan_loc or f"{k}[{i}][{j}]"
        elif d > max_diff:
            max_diff, max_loc = d, f"{k}[{i}][{j}]"
    detail = f"first nan at {nan_loc}" if nan_loc else f"max diff={max_diff:.2e} at {max_loc}"
    assert violations == 0, f"weights mismatch (atol={atol}): {violations}/{total} params exceed tolerance, {detail}"


def save_times(step_times: list[float]) -> None:
    name = Path(inspect.stack()[1].filename).stem
    path = TIMES_DIR / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_ms"])
        writer.writerows([[i + 1, f"{t * 1000:.3f}"] for i, t in enumerate(step_times)])
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
