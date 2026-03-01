from __future__ import annotations

from exo import DRAM, proc

from xdsl_exo.main import compile_procs


def test_window_row():
    """Window into a row: A[i, :] creates a 1D subview of a 2D matrix."""

    @proc
    def set_row(row: [f32][4] @ DRAM):
        for j in seq(0, 4):
            row[j] = 0.0

    @proc
    def zero_rows(A: f32[4, 4] @ DRAM):
        for i in seq(0, 4):
            set_row(A[i, :])

    compile_procs(zero_rows)


def test_window_col():
    """Window into a column: A[:, j] creates a 1D subview along the other axis."""

    @proc
    def set_col(col: [f32][4] @ DRAM):
        for i in seq(0, 4):
            col[i] = 0.0

    @proc
    def zero_cols(A: f32[4, 4] @ DRAM):
        for j in seq(0, 4):
            set_col(A[:, j])

    compile_procs(zero_cols)


def test_window_submatrix():
    """Window into a submatrix: T[k, :, :] extracts a 2D slice from a 3D tensor."""

    @proc
    def process_matrix(M: [f32][4, 4] @ DRAM):
        for i in seq(0, 4):
            for j in seq(0, 4):
                M[i, j] = 0.0

    @proc
    def process_3d(T: f32[3, 4, 4] @ DRAM):
        for k in seq(0, 3):
            process_matrix(T[k, :, :])

    compile_procs(process_3d)


def test_window_write_after_window():
    """Write through a window: modifications through the subview hit the original buffer."""

    @proc
    def write_to_row(row: [f32][4] @ DRAM):
        row[1] = 42.0

    @proc
    def window_then_write(A: f32[4, 4] @ DRAM):
        write_to_row(A[2, :])

    compile_procs(window_then_write)


def test_window_read_write_through_window():
    """
    The core example:
    Read from and write to a buffer through a windowed subview.
    Tests that the two-stage lowering (exo -> memref -> ptr -> llvm)
    correctly preserves offset metadata through subview.
    """

    @proc
    def copy_elem(row: [f32][4] @ DRAM):
        # read row[1] and write it to row[0]
        # verifies that offset arithmetic through the subview is correct
        row[0] = row[1]

    @proc
    def window_then_read_write(A: f32[4, 4] @ DRAM):
        copy_elem(A[2, :])

    compile_procs(window_then_read_write)


def test_window_chained():
    """Chained windows: take a window of a window (subview of a subview)."""

    @proc
    def set_first(x: [f32][4] @ DRAM):
        x[0] = 1.0

    @proc
    def inner(M: [f32][4, 4] @ DRAM):
        set_first(M[1, :])

    @proc
    def outer(T: f32[3, 4, 4] @ DRAM):
        inner(T[2, :, :])

    compile_procs(outer)
