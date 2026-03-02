.PHONY: install
install:
	brew install llvm pkg-config ninja ccache

.PHONY: venv
venv:
	uv sync

.PHONY: tests
tests:
	uv run pytest -W ignore tests/
	uv run lit tests/filecheck/
	uv run lit tests/e2e/

.PHONY: coverage
coverage: tests
	rm -f .coverage
	for f in tests/filecheck/*.py tests/e2e/*.py; do uv run --with coverage coverage run --branch --source=xdsl_exo -a -m xdsl_exo.main -o /dev/null "$$f" > /dev/null 2>&1 || true; done
	uv run --with coverage coverage report --show-missing

.PHONY: precommit
precommit: tests
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
	uvx ruff check --fix --ignore F403,F405,F821 .
	find . -name "*.c" -o -name "*.h" | xargs clang-format -i
