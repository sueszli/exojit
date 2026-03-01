from __future__ import annotations

import pytest
from exo import DRAM, proc
from xdsl.context import Context
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import Builtin, FunctionType, MemRefType, ModuleOp, NoneAttr, StringAttr, f32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

from xdsl_exo.dialects.exo import Exo, InvertWindowOp, WindowOp
from xdsl_exo.main import compile_procs
from xdsl_exo.rewrites.convert_tensor_ref import ConvertTensorRefPass


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


def _make_ctx() -> Context:
    ctx = Context()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(Exo)
    return ctx


def _build_window_invert_module(
    source_type: MemRefType,
    window_type: MemRefType,
) -> ModuleOp:
    """Build: func(@arg) { window = WindowOp(arg); orig = InvertWindowOp(window); return }"""
    block = Block(arg_types=[source_type])
    arg = block.args[0]

    # WindowOp: source_type -> window_type (point-index dim 0, keep dim 1)
    window_op = WindowOp(
        input=arg,
        indices=[],
        input_sizes=[source_type.get_shape()[0], source_type.get_shape()[1]],
        output_sizes=[window_type.get_shape()[0]],
        result_type=window_type,
    )
    block.add_op(window_op)

    invert_op = InvertWindowOp(input=window_op.result, result_type=source_type)
    block.add_op(invert_op)

    ret = ReturnOp()
    block.add_op(ret)

    func_type = FunctionType.from_lists([source_type], [])
    func_op = FuncOp("test_fn", func_type, Region(block))
    return ModuleOp([func_op])


def test_invert_window_trivial():
    """WindowOp -> InvertWindowOp: both eliminated after pass, original source forwarded."""
    source_type = MemRefType(f32, [4, 4], NoneAttr(), StringAttr("DRAM"))
    window_type = MemRefType(f32, [4], NoneAttr(), StringAttr("DRAM"))

    module = _build_window_invert_module(source_type, window_type)
    module.verify()

    ctx = _make_ctx()
    ConvertTensorRefPass().apply(ctx, module)
    module.verify()

    # After the pass, InvertWindowOp should be gone (and WindowOp lowered to SubviewOp)
    ir_text = str(module)
    assert "exo.invert_window" not in ir_text
    assert "exo.window" not in ir_text


def test_invert_window_type_matches():
    """Forwarded value has the correct original type."""
    source_type = MemRefType(f32, [8, 8], NoneAttr(), StringAttr("DRAM"))
    window_type = MemRefType(f32, [8], NoneAttr(), StringAttr("DRAM"))

    module = _build_window_invert_module(source_type, window_type)

    # Before pass: InvertWindowOp result type should match source_type
    func_op = module.body.block.first_op
    ops = list(func_op.body.block.ops)
    invert_op = ops[1]
    assert isinstance(invert_op, InvertWindowOp)
    assert invert_op.result.type == source_type

    ctx = _make_ctx()
    ConvertTensorRefPass().apply(ctx, module)
    module.verify()


def test_invert_window_block_arg_panics():
    """InvertWindowOp on a block arg (not a WindowOp/SubviewOp result) should raise."""
    source_type = MemRefType(f32, [4], NoneAttr(), StringAttr("DRAM"))
    original_type = MemRefType(f32, [4, 4], NoneAttr(), StringAttr("DRAM"))

    # Build: func(@arg: memref<4xf32>) { InvertWindowOp(@arg) -> memref<4x4xf32> }
    block = Block(arg_types=[source_type])
    arg = block.args[0]

    invert_op = InvertWindowOp(input=arg, result_type=original_type)
    block.add_op(invert_op)

    ret = ReturnOp()
    block.add_op(ret)

    func_type = FunctionType.from_lists([source_type], [])
    func_op = FuncOp("test_fn", func_type, Region(block))
    module = ModuleOp([func_op])
    module.verify()

    ctx = _make_ctx()
    with pytest.raises(AssertionError, match="InvertWindowOp"):
        ConvertTensorRefPass().apply(ctx, module)
