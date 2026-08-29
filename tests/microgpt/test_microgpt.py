from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

MICROGPT = Path(__file__).resolve().parents[2] / "examples" / "microgpt"
W = namedtuple("W", ["data"])


def _load_utils():
    spec = importlib.util.spec_from_file_location("microgpt_utils", MICROGPT / "utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_state_dict() -> dict:
    ref = json.loads((MICROGPT / "weights.json").read_text())
    return {k: [[W(v) for v in row] for row in mat] for k, mat in ref.items()}


def test_microgpt_trains_and_matches_reference_weights(tmp_path):
    # copied out of tree so the run cannot rewrite the tracked times/ csv
    workdir = tmp_path / "microgpt"
    shutil.copytree(MICROGPT, workdir)
    tracked_times = MICROGPT / "times" / "train.csv"
    before = tracked_times.read_bytes() if tracked_times.exists() else None

    result = subprocess.run([sys.executable, "train.py"], cwd=workdir, capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"microgpt training failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "train: mean=" in result.stdout, f"training did not reach the end of the loop\nstdout:\n{result.stdout}"
    after = tracked_times.read_bytes() if tracked_times.exists() else None
    assert after == before, "the training run must not rewrite the tracked times csv"


def test_reference_weights_match_themselves():
    _load_utils().assert_weights_match(_reference_state_dict())


def test_weight_check_rejects_a_perturbed_parameter():
    utils = _load_utils()
    state = _reference_state_dict()
    key = min(state)
    state[key][0][0] = W(state[key][0][0].data + 1.0)
    with pytest.raises(AssertionError, match="weights mismatch"):
        utils.assert_weights_match(state)


def test_weight_check_rejects_nan():
    # a miscompiled kernel produces nan, and `d > atol` is False for nan, so the
    # comparison has to be written as `not (d <= atol)` to catch it
    utils = _load_utils()
    state = _reference_state_dict()
    key = min(state)
    state[key][0][0] = W(float("nan"))
    with pytest.raises(AssertionError, match="first nan at"):
        utils.assert_weights_match(state)
