# full Xcode provides metal. macOS 15 matches the emitted AIR 2.7.
.PHONY: test example
test example:
	@if [ -n "$$DEVELOPER_DIR" ]; then \
		xcrun metal --version >/dev/null 2>&1 || { echo "error: no usable metal compiler in DEVELOPER_DIR=$$DEVELOPER_DIR" >&2; exit 1; }; \
	elif ! xcrun metal --version >/dev/null 2>&1; then \
		for app in /Applications/Xcode.app /Applications/Xcode-*.app; do \
			DEVELOPER_DIR="$$app/Contents/Developer" xcrun metal --version >/dev/null 2>&1 && { export DEVELOPER_DIR="$$app/Contents/Developer"; break; }; \
		done; \
		[ -n "$$DEVELOPER_DIR" ] || { echo "error: no full Xcode with the Metal Toolchain found in /Applications. Install one or set DEVELOPER_DIR" >&2; exit 1; }; \
	fi; \
	export MACOSX_DEPLOYMENT_TARGET="$${MACOSX_DEPLOYMENT_TARGET:-15.0}"; \
	if [ "$@" = example ]; then \
		uv run --with pyobjc-framework-Metal python example.py; \
	else \
		uv run --with pytest --with pyobjc-framework-Metal --with pyobjc-framework-Cocoa pytest test/; \
	fi

.PHONY: test-cpu
test-cpu:
	python3 -m unittest discover -s test -p test_llvm_to_air_unit.py

.PHONY: fmt
fmt:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
