from __future__ import annotations

import pytest
from exo.API import proc
from exo.core.LoopIR import LoopIR, T
from exo.core.memory import DRAM
from exo.core.prelude import SrcInfo, Sym

from xdsl.dialects.builtin import i32
from xdsl.utils.scoped_dict import ScopedDict
from xdsl.utils.test_value import create_ssa_value
from xdsl_exo.compiler import IRGenerator, compile_procs

SRC_INFO = SrcInfo("test_mlir.py", 0)
TENSOR_TYPE = T.Tensor(
    [
        LoopIR.Const(32, T.index, SRC_INFO),
    ],
    False,
    T.f32,
)


def test_emit_procedure():
    @proc
    def noop():
        pass

    module = compile_procs([noop])
    print(module)


def test_emit_procedure_with_args():
    @proc
    def unary_noop(x: f32[16]):
        pass

    module = compile_procs([unary_noop])
    print(module)


def test_emit_procedure_preserves_args():
    @proc
    def unary_preserves_args(x: f32[16], idx: index):
        assert idx >= 0 and idx < 16
        x[idx] = 0.0

    module = compile_procs([unary_preserves_args])
    print(module)


def test_get_sym():
    gen = IRGenerator()._with_empty_scope()
    sym = Sym("test")

    with pytest.raises(AssertionError, match="unknown symbol test"):
        gen._get_sym(sym)

    # Test symbol found
    test_value = create_ssa_value(i32)
    same_value = gen._declare_value(sym, test_value)

    assert test_value is same_value

    res_value = gen._get_sym(sym)

    assert res_value is test_value


def test_emit_assign_op():
    # x[0] = 1
    sym_x = Sym("x")
    ir = LoopIR.Assign(
        sym_x,
        TENSOR_TYPE,
        [LoopIR.Const(0, T.index, SRC_INFO)],
        LoopIR.Const(0.0, T.f32, SRC_INFO),
        SRC_INFO,
    )

    gen = IRGenerator()._with_empty_scope()._with_test_op(sym_x, TENSOR_TYPE)
    gen._assign_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_reduce_op():
    # x[0] += 1
    sym_x = Sym("x")
    ir = LoopIR.Reduce(
        sym_x,
        TENSOR_TYPE,
        [LoopIR.Const(0, T.int, SRC_INFO)],
        LoopIR.Const(0.0, T.f32, SRC_INFO),
        SRC_INFO,
    )

    gen = IRGenerator()._with_empty_scope()._with_test_op(sym_x, TENSOR_TYPE)
    gen._reduce_stmt(ir)

    print(gen.module)
    gen.module.verify()


# def test_emit_write_config_op():
#     # TODO: discover what exactly WriteConfig does
#     raise NotImplementedError


def test_emit_if_op():
    # if True:
    #     pass
    # else:
    #     pass
    ir = LoopIR.If(LoopIR.Const(True, T.bool, SRC_INFO), [], [], SRC_INFO)

    gen = IRGenerator()
    gen._if_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_for_op():
    # for i in seq(0, 10):
    #   pass
    ir = LoopIR.For(
        Sym("i"),
        LoopIR.Const(0, T.int, SRC_INFO),
        LoopIR.Const(10, T.int, SRC_INFO),
        [],
        LoopIR.Seq(),
        SRC_INFO,
    )

    gen = IRGenerator()._with_empty_scope()
    gen._for_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_alloc_op():
    ir = LoopIR.Alloc(
        Sym("x"),
        TENSOR_TYPE,
        DRAM,
        SRC_INFO,
    )

    gen = IRGenerator()._with_empty_scope()
    gen._alloc_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_free_op():
    sym_x = Sym("x")
    ir = LoopIR.Free(
        sym_x,
        TENSOR_TYPE,
        DRAM,
        SRC_INFO,
    )

    gen = IRGenerator()._with_empty_scope()._with_test_op(sym_x, TENSOR_TYPE)
    gen._free_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_read_op():
    sym_x = Sym("x")
    ir = LoopIR.Read(sym_x, [LoopIR.Const(0, T.index, SRC_INFO)], T.f32, SRC_INFO)

    gen = (
        IRGenerator()
        ._with_empty_scope()
        ._with_test_op(
            sym_x,
            TENSOR_TYPE,
        )
    )
    gen._read_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_int():
    ir = LoopIR.Const(0, T.int, SRC_INFO)

    gen = IRGenerator()
    gen._const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_float():
    ir = LoopIR.Const(0.0, T.f32, SRC_INFO)

    gen = IRGenerator()
    gen.symbol_table = ScopedDict()
    gen._const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_bool():
    ir = LoopIR.Const(True, T.bool, SRC_INFO)

    gen = IRGenerator()
    gen.symbol_table = ScopedDict()
    gen._const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_usub_op():
    ir = LoopIR.USub(LoopIR.Const(0, T.int, SRC_INFO), T.int, SRC_INFO)

    gen = IRGenerator()
    gen._usub_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_bin_op():
    ir = LoopIR.BinOp(
        "+",
        LoopIR.Const(0, T.int, SRC_INFO),
        LoopIR.Const(0, T.int, SRC_INFO),
        T.int,
        SRC_INFO,
    )

    gen = IRGenerator()
    gen._binop_expr(ir)

    print(gen.module)
    gen.module.verify()
