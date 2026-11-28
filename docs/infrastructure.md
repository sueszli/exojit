# Setup

Run `uv sync` from the repository root. The project requires Python 3.14.

The CPU reference tests use `clang`, `mlir-opt` and `mlir-translate`. Put `clang` on `PATH`. Set `LLVM_BIN` to the directory with both MLIR tools or put that directory on `PATH`.

# Make targets

- `make fmt` applies Ruff fixes and formatting.
- `make lint` runs Vulture and Pyright.
- `make tests` runs pytest, then runs the FileCheck cases with lit.
- `make precommit` runs `uv sync`, sets up the pre-push hook when eligible, formats and lints the code, runs `example.py` and runs `make tests`. The hook setup checks for an existing hook and a custom hooks path.
- `make benchmark` runs the CPU MicroGPT benchmark with `uv run --locked`. It writes `bench_microgpt/assets/benchmark.csv` and `bench_microgpt/assets/benchmark.svg`.

# Metal and Linux

Running Metal kernels requires macOS, a Metal device, PyObjC and Xcode's Metal tools. The backend uses `xcrun` with the macOS SDK to run `metal` and `metallib`. If the selected toolchain lacks Metal and `DEVELOPER_DIR` is unset, the backend tries `/Applications/Xcode.app/Contents/Developer`.

Linux runs the CPU JIT, the CPU example and the CPU benchmark. Pytest also runs Metal validation and AIR emission checks without the Xcode toolchain. Lit skips the two Metal GPU tests on Linux.
