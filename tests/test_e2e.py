"""
End-to-end tests that compile Exo → xDSL IR → MLIR → LLVM IR → native object,
link with a C harness, execute, and verify runtime output.

Requires: LLVM toolchain (mlir-opt, mlir-translate, llc, clang) installed via brew.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest
from exo import DRAM, proc

from xdsl_exo.main import compile_procs

LLVM_PREFIX = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True).stdout.strip()
MLIR_OPT = os.path.join(LLVM_PREFIX, "bin", "mlir-opt")
MLIR_TRANSLATE = os.path.join(LLVM_PREFIX, "bin", "mlir-translate")
LLC = os.path.join(LLVM_PREFIX, "bin", "llc")
CLANG = os.path.join(LLVM_PREFIX, "bin", "clang")

HAS_MLIR_TOOLCHAIN = all(shutil.which(t) or os.path.isfile(t) for t in [MLIR_OPT, MLIR_TRANSLATE, LLC, CLANG])

MLIR_LOWER_FLAGS = [
    "--convert-arith-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]

pytestmark = pytest.mark.skipif(not HAS_MLIR_TOOLCHAIN, reason="LLVM/MLIR toolchain not found")


def _compile_and_run(exo_procs, c_harness: str) -> subprocess.CompletedProcess:
    """
    Compile Exo procs through the full pipeline and link with a C harness.
    Returns the CompletedProcess from running the resulting executable.
    """
    module = compile_procs(exo_procs)
    mlir_text = str(module)

    with tempfile.TemporaryDirectory() as tmpdir:
        mlir_file = os.path.join(tmpdir, "kernel.mlir")
        obj_file = os.path.join(tmpdir, "kernel.o")
        c_file = os.path.join(tmpdir, "main.c")
        exe_file = os.path.join(tmpdir, "test_exe")

        Path(mlir_file).write_text(mlir_text)
        Path(c_file).write_text(c_harness)

        # MLIR → lowered MLIR → LLVM IR → object
        mlir_opt = subprocess.run(
            [MLIR_OPT, *MLIR_LOWER_FLAGS, mlir_file],
            capture_output=True,
            text=True,
        )
        assert mlir_opt.returncode == 0, f"mlir-opt failed:\n{mlir_opt.stderr}"

        mlir_translate = subprocess.run(
            [MLIR_TRANSLATE, "--mlir-to-llvmir"],
            input=mlir_opt.stdout,
            capture_output=True,
            text=True,
        )
        assert mlir_translate.returncode == 0, f"mlir-translate failed:\n{mlir_translate.stderr}"

        llc = subprocess.run(
            [LLC, "-filetype=obj", "-o", obj_file],
            input=mlir_translate.stdout.encode(),
            capture_output=True,
        )
        assert llc.returncode == 0, f"llc failed:\n{llc.stderr.decode()}"

        # link with C harness
        clang = subprocess.run(
            [CLANG, obj_file, c_file, "-o", exe_file, "-lm"],
            capture_output=True,
            text=True,
        )
        assert clang.returncode == 0, f"clang link failed:\n{clang.stderr}"

        # run
        result = subprocess.run([exe_file], capture_output=True, text=True, timeout=10)
        return result


# ---------------------------------------------------------------------------
# Integer arithmetic
# ---------------------------------------------------------------------------


@proc
def int_arithmetic(out: i32[1] @ DRAM, a: i32[1] @ DRAM, b: i32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]


def test_int_arithmetic():
    result = _compile_and_run(
        int_arithmetic,
        dedent("""\
            #include <stdio.h>
            extern void int_arithmetic(int* out, int* a, int* b);
            int main() {
                int a = 10, b = 3, out = 0;
                int_arithmetic(&out, &a, &b);
                // last op is a/b = 10/3 = 3 (integer division)
                printf("%d\\n", out);
                return (out == 3) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "3"


# ---------------------------------------------------------------------------
# Float arithmetic
# ---------------------------------------------------------------------------


@proc
def float_arithmetic(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]


def test_float_arithmetic():
    result = _compile_and_run(
        float_arithmetic,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void float_arithmetic(float* out, float* a, float* b);
            int main() {
                float a = 10.0f, b = 3.0f, out = 0.0f;
                float_arithmetic(&out, &a, &b);
                // last op is a/b = 10.0/3.0 ≈ 3.333...
                printf("%.6f\\n", out);
                return (fabsf(out - 10.0f/3.0f) < 1e-5f) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


# ---------------------------------------------------------------------------
# f64 arithmetic
# ---------------------------------------------------------------------------


@proc
def f64_arithmetic(out: f64[1] @ DRAM, a: f64[1] @ DRAM, b: f64[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]


def test_f64_arithmetic():
    result = _compile_and_run(
        f64_arithmetic,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void f64_arithmetic(double* out, double* a, double* b);
            int main() {
                double a = 10.0, b = 3.0, out = 0.0;
                f64_arithmetic(&out, &a, &b);
                printf("%.15f\\n", out);
                return (fabs(out - 10.0/3.0) < 1e-12) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


# ---------------------------------------------------------------------------
# Unary negation
# ---------------------------------------------------------------------------


@proc
def negate_float(out: f32[1] @ DRAM, a: f32[1] @ DRAM):
    out[0] = -a[0]


def test_negate_float():
    result = _compile_and_run(
        negate_float,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void negate_float(float* out, float* a);
            int main() {
                float a = 42.0f, out = 0.0f;
                negate_float(&out, &a);
                printf("%.1f\\n", out);
                return (fabsf(out - (-42.0f)) < 1e-5f) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "-42.0"


@proc
def negate_int(out: i32[1] @ DRAM, a: i32[1] @ DRAM):
    out[0] = -a[0]


def test_negate_int():
    result = _compile_and_run(
        negate_int,
        dedent("""\
            #include <stdio.h>
            extern void negate_int(int* out, int* a);
            int main() {
                int a = 7, out = 0;
                negate_int(&out, &a);
                printf("%d\\n", out);
                return (out == -7) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "-7"


# ---------------------------------------------------------------------------
# If/else branching
# ---------------------------------------------------------------------------


@proc
def if_else(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 2.0


def test_if_else_true_branch():
    result = _compile_and_run(
        if_else,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void if_else(float* out, long a, long b);
            int main() {
                float out = 0.0f;
                if_else(&out, 1, 5);  // a < b → out = 1.0
                printf("%.1f\\n", out);
                return (fabsf(out - 1.0f) < 1e-5f) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "1.0"


def test_if_else_false_branch():
    result = _compile_and_run(
        if_else,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void if_else(float* out, long a, long b);
            int main() {
                float out = 0.0f;
                if_else(&out, 5, 1);  // a >= b → out = 2.0
                printf("%.1f\\n", out);
                return (fabsf(out - 2.0f) < 1e-5f) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "2.0"


# ---------------------------------------------------------------------------
# For loop with reduce (accumulation)
# ---------------------------------------------------------------------------


@proc
def reduce_float(x: f32[8] @ DRAM, y: f32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


def test_reduce_float():
    result = _compile_and_run(
        reduce_float,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void reduce_float(float* x, float* y);
            int main() {
                float x[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                float y[1] = {0.0f};
                reduce_float(x, y);
                // sum = 1+2+...+8 = 36
                printf("%.1f\\n", y[0]);
                return (fabsf(y[0] - 36.0f) < 1e-5f) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "36.0"


@proc
def reduce_int(x: i32[8] @ DRAM, y: i32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


def test_reduce_int():
    result = _compile_and_run(
        reduce_int,
        dedent("""\
            #include <stdio.h>
            extern void reduce_int(int* x, int* y);
            int main() {
                int x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                int y[1] = {0};
                reduce_int(x, y);
                printf("%d\\n", y[0]);
                return (y[0] == 36) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "36"


# ---------------------------------------------------------------------------
# Alloc / free (internal malloc/free through the pipeline)
# ---------------------------------------------------------------------------


@proc
def alloc_copy(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp


def test_alloc_copy():
    result = _compile_and_run(
        alloc_copy,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void alloc_copy(long N, float* x);
            int main() {
                float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                alloc_copy(4, x);
                int ok = (fabsf(x[0]-1.0f)<1e-5f && fabsf(x[1]-2.0f)<1e-5f
                       && fabsf(x[2]-3.0f)<1e-5f && fabsf(x[3]-4.0f)<1e-5f);
                printf("%s\\n", ok ? "OK" : "FAIL");
                return ok ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# 2D array access (assign_2d)
# ---------------------------------------------------------------------------


@proc
def assign_2d(dst: f32[4, 4] @ DRAM, src: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[i, j]


def test_assign_2d():
    result = _compile_and_run(
        assign_2d,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void assign_2d(float* dst, float* src);
            int main() {
                float src[16], dst[16] = {0};
                for (int i = 0; i < 16; i++) src[i] = (float)i;
                assign_2d(dst, src);
                int ok = 1;
                for (int i = 0; i < 16; i++) {
                    if (fabsf(dst[i] - src[i]) > 1e-5f) { ok = 0; break; }
                }
                printf("%s\\n", ok ? "OK" : "FAIL");
                return ok ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# Fixed-size matrix multiply (16x16)
# ---------------------------------------------------------------------------


@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]


def test_matmul_identity():
    """A = Identity, B = known values → C should equal B."""
    result = _compile_and_run(
        fixed_matmul,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void fixed_matmul(float* C, float* A, float* B);
            int main() {
                float A[256] = {0}, B[256], C[256] = {0};
                for (int i = 0; i < 16; i++) A[i*16+i] = 1.0f;
                for (int i = 0; i < 256; i++) B[i] = (float)i;
                fixed_matmul(C, A, B);
                int ok = 1;
                for (int i = 0; i < 256; i++) {
                    if (fabsf(C[i] - B[i]) > 1e-3f) {
                        printf("MISMATCH at %d: C=%f B=%f\\n", i, C[i], B[i]);
                        ok = 0;
                    }
                }
                printf("%s\\n", ok ? "OK" : "FAIL");
                return ok ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "OK"


def test_matmul_values():
    """Verify actual matrix multiplication with known small values."""
    result = _compile_and_run(
        fixed_matmul,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void fixed_matmul(float* C, float* A, float* B);
            int main() {
                float A[256] = {0}, B[256] = {0}, C[256] = {0};
                // A[0,0]=1 A[0,1]=2, A[1,0]=3 A[1,1]=4 (rest zero)
                A[0*16+0] = 1; A[0*16+1] = 2;
                A[1*16+0] = 3; A[1*16+1] = 4;
                // B[0,0]=5 B[0,1]=6, B[1,0]=7 B[1,1]=8
                B[0*16+0] = 5; B[0*16+1] = 6;
                B[1*16+0] = 7; B[1*16+1] = 8;
                fixed_matmul(C, A, B);
                // C[0,0] = 1*5+2*7 = 19
                // C[0,1] = 1*6+2*8 = 22
                // C[1,0] = 3*5+4*7 = 43
                // C[1,1] = 3*6+4*8 = 50
                int ok = (fabsf(C[0*16+0]-19.0f)<1e-3f &&
                          fabsf(C[0*16+1]-22.0f)<1e-3f &&
                          fabsf(C[1*16+0]-43.0f)<1e-3f &&
                          fabsf(C[1*16+1]-50.0f)<1e-3f);
                printf("C[0,0]=%.0f C[0,1]=%.0f C[1,0]=%.0f C[1,1]=%.0f\\n",
                       C[0*16+0], C[0*16+1], C[1*16+0], C[1*16+1]);
                return ok ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "19" in result.stdout and "22" in result.stdout


# ---------------------------------------------------------------------------
# Scalar memref (mutable scalar argument)
# ---------------------------------------------------------------------------


@proc
def scalar_double(x: f32[1] @ DRAM):
    x[0] = x[0] + x[0]


def test_scalar_double():
    result = _compile_and_run(
        scalar_double,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void scalar_double(float* x);
            int main() {
                float x[1] = {21.0f};
                scalar_double(x);
                printf("%.1f\\n", x[0]);
                return (fabsf(x[0] - 42.0f) < 1e-5f) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "42.0"


# ---------------------------------------------------------------------------
# Boolean / comparison operations
# ---------------------------------------------------------------------------


@proc
def compare_and_branch(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 10.0
    else:
        out[0] = 20.0


def test_compare_and_branch():
    result = _compile_and_run(
        compare_and_branch,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void compare_and_branch(float* out, long a, long b);
            int main() {
                float out = 0.0f;
                compare_and_branch(&out, 3, 7);  // a < b → 10.0
                printf("%.1f\\n", out);
                int ok1 = (fabsf(out - 10.0f) < 1e-5f);

                compare_and_branch(&out, 7, 3);  // a >= b → 20.0
                printf("%.1f\\n", out);
                int ok2 = (fabsf(out - 20.0f) < 1e-5f);

                return (ok1 && ok2) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    lines = result.stdout.strip().split("\n")
    assert lines[0] == "10.0"
    assert lines[1] == "20.0"


# ---------------------------------------------------------------------------
# Nested loops with index arithmetic
# ---------------------------------------------------------------------------


@proc
def vec_add(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] + b[i]


def test_vec_add():
    result = _compile_and_run(
        vec_add,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void vec_add(long N, float* out, float* a, float* b);
            int main() {
                float a[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
                float b[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
                float out[5] = {0};
                vec_add(5, out, a, b);
                int ok = 1;
                float expected[5] = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f};
                for (int i = 0; i < 5; i++) {
                    if (fabsf(out[i] - expected[i]) > 1e-5f) ok = 0;
                }
                printf("%.0f %.0f %.0f %.0f %.0f\\n", out[0],out[1],out[2],out[3],out[4]);
                return ok ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "11 22 33 44 55"


# ---------------------------------------------------------------------------
# Dot product (reduce with multiply)
# ---------------------------------------------------------------------------


@proc
def dot_product(N: size, out: f32[1] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[0] += a[i] * b[i]


def test_dot_product():
    result = _compile_and_run(
        dot_product,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void dot_product(long N, float* out, float* a, float* b);
            int main() {
                float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
                float out[1] = {0.0f};
                dot_product(4, out, a, b);
                // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
                printf("%.1f\\n", out[0]);
                return (fabsf(out[0] - 70.0f) < 1e-3f) ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "70.0"


# ---------------------------------------------------------------------------
# i8 copy
# ---------------------------------------------------------------------------


@proc
def i8_copy(dst: i8[8] @ DRAM, src: i8[8] @ DRAM):
    for i in seq(0, 8):
        dst[i] = src[i]


def test_i8_copy():
    result = _compile_and_run(
        i8_copy,
        dedent("""\
            #include <stdio.h>
            #include <string.h>
            extern void i8_copy(char* dst, char* src);
            int main() {
                char src[8] = {10, 20, 30, 40, 50, 60, 70, 80};
                char dst[8] = {0};
                i8_copy(dst, src);
                int ok = (memcmp(dst, src, 8) == 0);
                printf("%s\\n", ok ? "OK" : "FAIL");
                return ok ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# Window / subview (row extraction via sub-procedure)
# ---------------------------------------------------------------------------


@proc
def zero_row(row: [f32][4] @ DRAM):
    for j in seq(0, 4):
        row[j] = 0.0


@proc
def zero_matrix_rows(A: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        zero_row(A[i, :])


@pytest.mark.xfail(reason="window stride/offset lowering produces incorrect pointer arithmetic at runtime")
def test_window_row():
    result = _compile_and_run(
        zero_matrix_rows,
        dedent("""\
            #include <stdio.h>
            #include <math.h>
            extern void zero_matrix_rows(float* A);
            int main() {
                float A[16];
                for (int i = 0; i < 16; i++) A[i] = (float)(i + 1);
                zero_matrix_rows(A);
                int ok = 1;
                for (int i = 0; i < 16; i++) {
                    if (fabsf(A[i]) > 1e-5f) { ok = 0; break; }
                }
                printf("%s\\n", ok ? "OK" : "FAIL");
                return ok ? 0 : 1;
            }
        """),
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.strip() == "OK"
