from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import pytest

MICROGPT = Path(__file__).resolve().parent.parent.parent / "microgpt"
W = namedtuple("W", ["data"])


def _load_assert_weights_match():
    """Import assert_weights_match without triggering exo/jit module-level side effects."""
    import itertools
    import math

    weights_path = MICROGPT / "weights.json"

    def assert_weights_match(state_dict, atol: float = 1e-5) -> None:
        assert weights_path.exists(), f"weights file not found: {weights_path}"
        with weights_path.open() as f:
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

    return assert_weights_match


def _reference_state_dict() -> dict:
    ref = json.loads((MICROGPT / "weights.json").read_text())
    return {k: [[W(v) for v in row] for row in mat] for k, mat in ref.items()}


def _perturbed(value: float) -> dict:
    state = _reference_state_dict()
    key = min(state)
    state[key][0][0] = W(value)
    return state


def test_reference_weights_match_themselves():
    assert_weights_match = _load_assert_weights_match()
    assert_weights_match(_reference_state_dict())


def test_weight_check_rejects_a_perturbed_parameter():
    assert_weights_match = _load_assert_weights_match()
    with pytest.raises(AssertionError, match="weights mismatch"):
        assert_weights_match(_perturbed(5.0))


def test_weight_check_rejects_nan():
    assert_weights_match = _load_assert_weights_match()
    with pytest.raises(AssertionError, match="first nan at"):
        assert_weights_match(_perturbed(float("nan")))
