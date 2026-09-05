#!/usr/bin/env bash
set -euox pipefail

cd "$(dirname "$0")"

# format
uvx isort .
uvx autoflake --remove-all-unused-imports --recursive --in-place .
uvx black --line-length 5000 .
uvx ruff check --fix --ignore F403,F405,F821,E731,E402 .

# generate weights
[ -f weights.json ] || uv run original.py

# benchmark
for file in *.py; do
  [[ "$file" == "utils.py" || "$file" == "original.py" ]] || uv run "$file"
done

# plot
rm README
toilet -t -f pagga "microgpt benchmarks game" >> README
cat >> README << 'EOF'


community inference benchmarks game for karpathy's microgpt.
a minimal gpt trained on a names dataset: https://karpathy.github.io/2026/02/12/microgpt/

can you beat me? to submit: add a .py file prefixed with your username and open a pr.
any python library is welcome, but no other language embedded within the file.

EOF
uv run utils.py | tee >(perl -pe 's/\e\[[0-9;]*m//g' >> README)
