from __future__ import annotations

import pytest
from exo.API import proc
from exo.core.LoopIR import LoopIR, T
from exo.core.memory import DRAM
from exo.core.prelude import SrcInfo, Sym

from xdsl.dialects.builtin import i32
from xdsl.utils.scoped_dict import ScopedDict
from xdsl.utils.test_value import create_ssa_value
from xdsl_exo.compiler import compile_one
from xdsl_exo.generator import IRGenerator, IRGeneratorError

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

    module = compile_one(noop)
    print(module)


def test_emit_procedure_with_args():
    @proc
    def unary_noop(x: f32[16]):
        pass

    module = compile_one(unary_noop)
    print(module)


def test_emit_procedure_preserves_args():
    @proc
    def unary_preserves_args(x: f32[16], idx: index):
        assert idx >= 0 and idx < 16
        x[idx] = 0.0

    module = compile_one(unary_preserves_args)
    print(module)


def test_get_sym():
    gen = IRGenerator().with_empty_scope()
    sym = Sym("test")

    with pytest.raises(IRGeneratorError, match="Unknown symbol test"):
        gen.get_sym(sym)

    # Test symbol found
    test_value = create_ssa_value(i32)
    same_value = gen.declare_value(sym, test_value)

    assert test_value is same_value

    res_value = gen.get_sym(sym)

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

    gen = IRGenerator().with_empty_scope()._with_test_op(sym_x, TENSOR_TYPE)
    gen.generate_assign_stmt(ir)

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

    gen = IRGenerator().with_empty_scope()._with_test_op(sym_x, TENSOR_TYPE)
    gen.generate_reduce_stmt(ir)

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
    gen.generate_if_stmt(ir)

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

    gen = IRGenerator().with_empty_scope()
    gen.generate_for_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_alloc_op():
    ir = LoopIR.Alloc(
        Sym("x"),
        TENSOR_TYPE,
        DRAM,
        SRC_INFO,
    )

    gen = IRGenerator().with_empty_scope()
    gen.generate_alloc_stmt(ir)

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

    gen = IRGenerator().with_empty_scope()._with_test_op(sym_x, TENSOR_TYPE)
    gen.generate_free_stmt(ir)

    print(gen.module)
    gen.module.verify()


def test_read_op():
    sym_x = Sym("x")
    ir = LoopIR.Read(sym_x, [LoopIR.Const(0, T.index, SRC_INFO)], T.f32, SRC_INFO)

    gen = (
        IRGenerator()
        .with_empty_scope()
        ._with_test_op(
            sym_x,
            TENSOR_TYPE,
        )
    )
    gen.generate_read_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_int():
    ir = LoopIR.Const(0, T.int, SRC_INFO)

    gen = IRGenerator()
    gen.generate_const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_float():
    ir = LoopIR.Const(0.0, T.f32, SRC_INFO)

    gen = IRGenerator()
    gen.symbol_table = ScopedDict()
    gen.generate_const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_const_op_bool():
    ir = LoopIR.Const(True, T.bool, SRC_INFO)

    gen = IRGenerator()
    gen.symbol_table = ScopedDict()
    gen.generate_const_expr(ir)

    print(gen.module)
    gen.module.verify()


def test_emit_usub_op():
    ir = LoopIR.USub(LoopIR.Const(0, T.int, SRC_INFO), T.int, SRC_INFO)

    gen = IRGenerator()
    gen.generate_usub_expr(ir)

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
    gen.generate_binop_expr(ir)

    print(gen.module)
    gen.module.verify()
