from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

MICROGPT = Path(__file__).resolve().parent
TIMES = MICROGPT / "times" / "train.csv"
W = namedtuple("W", ["data"])


def _utils():
    spec = importlib.util.spec_from_file_location("microgpt_utils", MICROGPT / "utils.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_state_dict() -> dict:
    ref = json.loads((MICROGPT / "weights.json").read_text())
    return {k: [[W(v) for v in row] for row in mat] for k, mat in ref.items()}


def _perturbed(value: float) -> dict:
    state = _reference_state_dict()
    key = min(state)
    state[key][0][0] = W(value)
    return state


def test_microgpt_trains_and_matches_reference_weights(tmp_path):
    workdir = tmp_path / "microgpt"
    shutil.copytree(MICROGPT, workdir, ignore=shutil.ignore_patterns("test_*.py", "__pycache__"))
    before = TIMES.read_bytes()

    result = subprocess.run([sys.executable, "train.py"], cwd=workdir, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "train: mean=" in result.stdout, result.stdout
    assert TIMES.read_bytes() == before, "training rewrote the tracked times csv"


def test_reference_weights_match_themselves():
    _utils().assert_weights_match(_reference_state_dict())


def test_weight_check_rejects_a_perturbed_parameter():
    with pytest.raises(AssertionError, match="weights mismatch"):
        _utils().assert_weights_match(_perturbed(5.0))


def test_weight_check_rejects_nan():
    with pytest.raises(AssertionError, match="first nan at"):
        _utils().assert_weights_match(_perturbed(float("nan")))
