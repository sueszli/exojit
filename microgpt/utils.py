import itertools
import json
import math
from pathlib import Path

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"


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
