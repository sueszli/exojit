#!/usr/bin/env bash
set -euox pipefail
cd "$(dirname "$0")"

[ -f weights.json ] || uv run original.py

for file in *.py; do
  [[ "$file" == "utils.py" || "$file" == "original.py" ]] || uv run "$file"
done

uv run utils.py
