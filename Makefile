.PHONY: test
test:
	uv run --with pytest --with pyobjc-framework-Metal --with pyobjc-framework-Cocoa pytest test/

.PHONY: fmt
fmt:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
