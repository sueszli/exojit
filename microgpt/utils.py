import itertools
import json
import math
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights.json"
TIMES_DIR = ROOT / "times"


def assert_weights_match(state_dict: dict[str, list[list[float]]], atol: float = 1e-5) -> None:
    assert WEIGHTS_PATH.exists(), f"weights file not found: {WEIGHTS_PATH}"

    with WEIGHTS_PATH.open() as f:
        ref = json.load(f)
    assert set(ref) == set(state_dict), f"key mismatch: ref={set(ref) - set(state_dict)} cur={set(state_dict) - set(ref)}"

    for k in ref:
        assert len(ref[k]) == len(state_dict[k]) and len(ref[k][0]) == len(state_dict[k][0]), f"shape mismatch '{k}': {len(ref[k])}x{len(ref[k][0])} vs {len(state_dict[k])}x{len(state_dict[k][0])}"

    max_diff = 0.0
    max_loc = ""
    violations = 0
    total = 0
    rows = ((k, i, rr, cr) for k in ref for i, (rr, cr) in enumerate(zip(ref[k], state_dict[k])))
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


def print_leaderboard() -> None:
    if not TIMES_DIR.exists():
        return

    frames = [pl.read_csv(p).with_columns(pl.lit(p.stem).alias("name")) for p in sorted(TIMES_DIR.glob("*.csv"))]
    if not frames:
        return

    stats = pl.concat(frames).with_columns((pl.col("time_ms") * 1000).alias("us")).group_by("name").agg(pl.col("us").mean().cast(pl.Int64).alias("mean \u03bcs"), pl.col("us").std().cast(pl.Int64).alias("\u00b1\u03c3"), pl.col("us").min().cast(pl.Int64).alias("min"), pl.col("us").max().cast(pl.Int64).alias("max")).sort("mean \u03bcs")

    original_mean = stats.filter(pl.col("name") == "original")["mean \u03bcs"][0]
    stats = stats.with_columns((original_mean / pl.col("mean \u03bcs")).cast(pl.Int64).alias("speedup"))

    pl.Config.set_tbl_formatting("NOTHING")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    lines = str(stats).splitlines()
    lines.insert(1, "\u2500" * len(lines[0]))
    print("\n".join(lines))


if __name__ == "__main__":
    print_leaderboard()
