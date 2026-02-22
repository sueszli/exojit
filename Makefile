.PHONY: venv
venv:
	uv venv .venv --python 3.11
	uv pip install -r requirements.txt
	@echo "activate venv with: \033[1;33msource .venv/bin/activate\033[0m"

.PHONY: lock
lock:
	uv pip freeze > requirements.in
	uv pip compile requirements.in -o requirements.txt

.PHONY: precommit
precommit:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
	uvx ruff check --fix .

.PHONY: tests
tests:
	.venv/bin/pytest tests/
	.venv/bin/lit tests/filecheck/

.PHONY: bench
bench:
	.venv/bin/snakemake --cores all
