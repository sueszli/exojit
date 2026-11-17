.PHONY: test
test:
	uv run --with pytest --with pyobjc-framework-Metal --with pyobjc-framework-Cocoa pytest test/

.PHONY: test-cpu
test-cpu:
	python3 -m unittest discover -s test -p test_llvm_to_air_unit.py

.PHONY: fmt
fmt:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
