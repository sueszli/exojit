.PHONY: install
install:
	brew install llvm pkg-config google-benchmark ninja ccache

.PHONY: venv
venv:
	uv sync

.PHONY: precommit
precommit:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
	# uvx ruff check --fix .

.PHONY: tests
tests:
	.venv/bin/pytest tests/
	.venv/bin/lit tests/filecheck/

.PHONY: bench
bench:
	.venv/bin/snakemake --cores all
