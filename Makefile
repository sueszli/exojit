.PHONY: precommit-hook
precommit-hook:
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$common_dir/hooks/pre-push" && chmod +x "$$common_dir/hooks/pre-push"; \
	fi

.PHONY: fmt
fmt:
	uvx ruff check --fix --line-length 5000 --target-version py314 --extend-select I --ignore F403,F405,F821,E731,E402,PLE0643,B008,UP040,RUF016,PLC0206,SIM115 exojit.py microgpt tests example.py
	uvx ruff format --line-length 5000 --target-version py314 exojit.py microgpt tests example.py

.PHONY: lint
lint:
	uv run --with vulture vulture --min-confidence 80 exojit.py microgpt tests example.py
	uv run --with pyright pyright exojit.py

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
	$(MAKE) microgpt
	$(MAKE) tests

.PHONY: leaderboard
leaderboard:
	uv run python microgpt/utils.py

.PHONY: benchmark
benchmark:
	uv sync
	uv run python microgpt/original_impl.py
	uv run python microgpt/plain_impl.py
	uv run python microgpt/numpy_impl.py
	uv run python microgpt/torch_impl.py
	uv run python microgpt/jax_impl.py
	uv run python microgpt/exojit_impl.py
	uv run python microgpt/kernels/run.py
