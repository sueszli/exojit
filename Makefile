.PHONY: precommit-hook
precommit-hook:
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	hook="$$common_dir/hooks/pre-push"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ] && [ ! -e "$$hook" ] && [ ! -L "$$hook" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$hook" && chmod +x "$$hook"; \
	fi

.PHONY: fmt
fmt:
	uv run ruff check --fix --line-length 5000 --target-version py314 --extend-select I --ignore F403,F405,F821,E731,E402,PLE0643,B008,UP040,RUF016,PLC0206,SIM115 exojit.py backend_metal.py microgpt tests example.py
	uv run ruff format --line-length 5000 --target-version py314 exojit.py backend_metal.py microgpt tests example.py

.PHONY: lint
lint:
	uv run vulture --min-confidence 80 exojit.py backend_metal.py microgpt tests example.py
	uv run pyright exojit.py backend_metal.py

.PHONY: microgpt
microgpt:
	uv run python microgpt/exojit_impl.py

.PHONY: tests
tests:
	uv run pytest -W ignore tests/
	uv run lit -j $$(nproc 2>/dev/null || sysctl -n hw.logicalcpu) tests/filecheck/

.PHONY: precommit
precommit:
	uv sync
	$(MAKE) precommit-hook
	$(MAKE) fmt
	$(MAKE) lint
	uv run python example.py
	$(MAKE) tests

.PHONY: leaderboard
leaderboard:
	uv run python microgpt/utils.py

.PHONY: benchmark
benchmark:
	@uv run --locked python microgpt/benchmark.py

.PHONY: benchmark-kernels
benchmark-kernels:
	uv run --locked python microgpt/kernels/run.py

.PHONY: benchmark-matmul
benchmark-matmul:
	@uv run --locked python microgpt/kernels/bench_matmul.py
