from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

MICROGPT = Path(__file__).resolve().parent
TIMES = MICROGPT / "times" / "train.csv"
W = namedtuple("W", ["data"])
PERF_REGRESSION_TOLERANCE = 0.20


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


def _baseline_mean_us() -> float:
    with open(TIMES) as f:
        times_us = [float(row["time_ms"]) * 1000 for row in csv.DictReader(f)]
    assert times_us
    return sum(times_us) / len(times_us)


def _parse_mean_us(stdout: str) -> float:
    match = re.search(r"train: mean=(\d+)[µμ]s", stdout)
    assert match, f"could not parse mean from: {stdout!r}"
    return float(match.group(1))


def test_microgpt_correctness_and_performance(tmp_path):
    workdir = tmp_path / "microgpt"
    shutil.copytree(MICROGPT, workdir, ignore=shutil.ignore_patterns("test_*.py", "__pycache__"))

    result = subprocess.run([sys.executable, "train.py"], cwd=workdir, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train: mean=" in result.stdout, result.stdout

    baseline_us = _baseline_mean_us()
    actual_us = _parse_mean_us(result.stdout)
    max_allowed_us = baseline_us * (1 + PERF_REGRESSION_TOLERANCE)
    assert actual_us <= max_allowed_us, f"{actual_us:.0f}µs > {max_allowed_us:.0f}µs (baseline {baseline_us:.0f}µs)"


def test_reference_weights_match_themselves():
    _utils().assert_weights_match(_reference_state_dict())


def test_weight_check_rejects_a_perturbed_parameter():
    with pytest.raises(AssertionError, match="weights mismatch"):
        _utils().assert_weights_match(_perturbed(5.0))


def test_weight_check_rejects_nan():
    with pytest.raises(AssertionError, match="first nan at"):
        _utils().assert_weights_match(_perturbed(float("nan")))
