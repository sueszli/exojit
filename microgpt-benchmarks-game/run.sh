#!/usr/bin/env bash
set -euox pipefail
cd "$(dirname "$0")"

[ -f weights.json ] || uv run original.py

uv run bench_plain.py
uv run bench_numpy.py
uv run bench_torch.py
uv run bench_jax.py
uv run bench_exojit.py
uv run bench_exojit_full.py

uv run utils.py
