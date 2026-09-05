from __future__ import annotations

import math
import numbers
import re
import tempfile
from collections.abc import Callable, MutableSequence, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, SupportsInt, TypeGuard, cast

import click
import exo.frontend.boundscheck as _boundscheck
import exo.frontend.pyparser as _pyparser
from cffi import FFI
from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from exo.backend.LoopIR_compiler import find_all_subprocs
from exo.backend.mem_analysis import MemoryAnalysis
from exo.backend.prec_analysis import PrecisionAnalysis
from exo.backend.win_analysis import WindowAnalysis
from exo.core.LoopIR import UAST, LoopIR, T, get_writes_of_stmts
from exo.core.prelude import Sym
from exo.frontend.pyparser import DummyScope, Parser, get_ast_from_python
from exo.main import load_user_code
from exo.rewrite.range_analysis import constant_bound
from xdsl.backend.llvm.convert_op import _CAST_OP_NAMES
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, llvm, memref
from xdsl.dialects.builtin import DYNAMIC_INDEX, AnyFloat, ArrayAttr, Builtin, DictionaryAttr, FixedBitwidthType, FloatAttr, IndexType, IntegerAttr, IntegerType, MemRefType, ModuleOp, NoneAttr, StringAttr, UnitAttr, UnrealizedConversionCastOp, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.llvm import GEP_USE_SSA_VAL, BrOp, FNegOp, GenericCastOp, LLVMPointerType
from xdsl.dialects.utils import get_dynamic_index_list, split_dynamic_index_list
from xdsl.ir import Attribute, Block, BlockArgument, Operation, OpResult, Region, SSAValue
from xdsl.irdl import irdl_op_definition
from xdsl.jit.llvm.backend import LLVMJITBackend
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.convert_memref_to_ptr import ConvertCastOp
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.hints import isa
from xdsl.utils.scoped_dict import ScopedDict

# ===----------------------------------------------------------------------=== #
# exo patches
# ===----------------------------------------------------------------------=== #

_pyparser._prim_types["size"] = UAST.Size()
_pyparser._prim_types["index"] = UAST.Index()


ORIGINAL_LIFT_EXPR = _boundscheck.lift_expr
LIFTED_INDEX_SYMS: dict[tuple[object, ...], Sym] = {}


def patched_lift_expr(e):
    def expr_key(e) -> tuple[object, ...]:
        match e:
            case LoopIR.Read(name=name, idx=idx):
                return ("read", name, tuple(expr_key(i) for i in idx))
            case LoopIR.Const(val=val, type=type_):
                return ("const", val, str(type_))
            case LoopIR.USub(arg=arg):
                return ("usub", expr_key(arg))
            case LoopIR.BinOp(op=op, lhs=lhs, rhs=rhs):
                return ("binop", op, expr_key(lhs), expr_key(rhs))
            case LoopIR.StrideExpr(name=name, dim=dim):
                return ("stride", name, dim)
            case LoopIR.ReadConfig(config=config, field=field):
                return ("config", config.name(), field)
            case _:
                assert False, f"unsupported index expression: {type(e).__name__}"

    if not (isinstance(e, LoopIR.Read) and e.idx and e.type.is_indexable()):
        return ORIGINAL_LIFT_EXPR(e)
    key = expr_key(e)
    sym = LIFTED_INDEX_SYMS.get(key)
    if sym is None:
        sym = Sym(f"lifted_index_{len(LIFTED_INDEX_SYMS)}")
        LIFTED_INDEX_SYMS[key] = sym
    return _boundscheck.E.Var(sym, e.type, e.srcinfo)


_boundscheck.lift_expr = patched_lift_expr

# ===----------------------------------------------------------------------=== #
# xdsl patches
# ===----------------------------------------------------------------------=== #


@irdl_op_definition
class FPTruncOp(GenericCastOp):
    name = "llvm.fptrunc"


# xdsl's llvm dialect has fpext but no fptrunc, so teach its llvmlite converter about ours
_CAST_OP_NAMES[FPTruncOp] = "fptrunc"


# `memref` -> `llvm.ptr` lowering: replace structured memory ops with raw pointer arithmetic
#
# standard mlir lowers memref through a "descriptor" struct (base ptr, offset, sizes, strides).
# we skip that and go straight to flat pointer math because exo only emits statically-shaped,
# row-major memrefs with no affine maps. the descriptor is unnecessary overhead.
#
# pipeline order matters:
# ----------------------
#     1. extendedconvertmemreftoptr   — rewrite load/store/subview while shape info is still on the memreftype
#     2. rewritememreftypes           — erase memreftype -> llvm.ptr everywhere
#     3. reconcile-unrealized-casts   — clean up identity casts left behind
#
# example (convertloadstorepattern):
# ----------------------------------
#     memref.store %v, %buf[%i, %j] : memref<4x4xf32>
#     =>
#     %c1     = llvm.mlir.constant(1)   ; stride[1] = 1
#     %c4     = llvm.mlir.constant(4)   ; dim[1] = 4
#     %stride = llvm.mul %c1, %c4       ; stride[0] = dim[1] = 4
#     %off0   = llvm.mul %i, %stride    ; i * 4
#     %off1   = llvm.mul %j, %c1        ; j * 1
#     %flat   = llvm.add %off0, %off1   ; i*4 + j
#     %ptr    = llvm.getelementptr inbounds %buf[%flat] : f32
#     llvm.store %v, %ptr : f32


def _loop_upper_bound_as_i64(index: SSAValue) -> SSAValue | None:
    # for dynamic dims: walk index -> block_arg#0 -> find llvm.icmp in the loop header -> extract the bound
    # e.g. `icmp slt %iv, %n` => return %n as the dim size
    if isinstance(index, OpResult) and isinstance(index.op, UnrealizedConversionCastOp):
        inputs = list(index.op.operands)
        iv = inputs[0] if len(inputs) == 1 else index
    else:
        iv = index
    if not isinstance(iv, BlockArgument) or iv.index != 0:
        return None
    for op in iv.block.ops:
        if isinstance(op, llvm.ICmpOp):
            if op.lhs == iv:
                return op.rhs if op.rhs.type == i64 else None
            if op.rhs == iv:
                return op.lhs if op.lhs.type == i64 else None
    return None


def _iconst(ins, n: int) -> SSAValue:
    return ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result


def _base_and_offset(base: SSAValue, indices: Sequence[SSAValue], shape: tuple[int, ...], ins) -> tuple[SSAValue, SSAValue | None]:
    def dim_size(i: int) -> SSAValue:
        # static constant, or the dynamic loop bound the index is derived from
        if shape[i] != DYNAMIC_INDEX:
            return _iconst(ins, shape[i])
        ub = _loop_upper_bound_as_i64(indices[i])
        assert ub is not None
        return ub

    # row-major strides: stride[last]=1, stride[i]=stride[i+1]*dim[i+1]
    strides: list[SSAValue] = [_iconst(ins, 1)] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = ins(llvm.MulOp(strides[i + 1], dim_size(i + 1))).res

    # flat element offset = sum(index_i * stride_i)
    flat: SSAValue | None = None
    for idx, stride in zip(indices, strides):
        # peek through unrealized_cast(x:i64 -> index) to recover the original i64
        if isinstance(idx, OpResult) and isinstance(idx.op, UnrealizedConversionCastOp) and len(idx.op.operands) == 1 and idx.op.operands[0].type == i64:
            idx = idx.op.operands[0]
        term = ins(llvm.MulOp(idx, stride)).res
        flat = term if flat is None else ins(llvm.AddOp(flat, term)).res

    return ins(UnrealizedConversionCastOp.get([base], [LLVMPointerType()])).results[0], flat


class ConvertLoadStorePattern(RewritePattern):
    # memref.load/store %buf[%i, %j] => ptr arithmetic + llvm.load/store
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp | memref.StoreOp, rewriter: PatternRewriter, /):
        memref_type = op.memref.type
        assert isa(memref_type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        ins = rewriter.insert_op
        ptr, flat = _base_and_offset(op.memref, list(op.indices), memref_type.get_shape(), ins)
        if flat is not None:
            # gep with element type (rather than raw byte math) enables llvm auto-vectorization
            ptr = ins(llvm.GEPOp(ptr, [GEP_USE_SSA_VAL], memref_type.element_type, ssa_indices=[flat], inbounds=True)).result
        rewriter.replace_op(op, llvm.LoadOp(ptr, op.res.type) if isinstance(op, memref.LoadOp) else llvm.StoreOp(op.value, ptr))


class ConvertSubviewPattern(RewritePattern):
    # memref.subview %buf[offsets] => ptr to the start of the slice
    #
    # subview carries both static offsets (baked into the op) and dynamic offsets (ssa values).
    # mlir encodes "this offset is dynamic" by setting the static value to dynamic_index (-1);
    # the actual ssa value then comes from the op.offsets list in order.
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        src_type = op.source.type
        assert isa(src_type, builtin.MemRefType)
        if not isa(src_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        src_shape = src_type.get_shape()
        assert all(d != DYNAMIC_INDEX for d in src_shape), "dynamic source dims in subview not supported"
        assert isinstance(src_type.element_type, builtin.FixedBitwidthType)

        ins = rewriter.insert_op

        # merge static_offsets (constants) and dynamic offsets (ssa values) into one list
        dyn_iter = iter(op.offsets)
        all_offsets = [next(dyn_iter) if soff == DYNAMIC_INDEX else _iconst(ins, soff) for soff in op.static_offsets.iter_values()]

        # ptrtoint/inttoptr rather than gep, so the result stays type-agnostic
        ptr, flat = _base_and_offset(op.source, all_offsets, src_shape, ins)
        if flat is not None:
            byte_offset = ins(llvm.MulOp(flat, _iconst(ins, src_type.element_type.size))).res
            ptr_int = ins(llvm.PtrToIntOp(ptr)).output
            ptr = ins(llvm.IntToPtrOp(ins(llvm.AddOp(ptr_int, byte_offset)).res)).output

        # wrap result as memreftype so downstream load/store patterns still see the right shape for stride computation
        rewriter.replace_op(op, UnrealizedConversionCastOp.get([ptr], [op.result.type]))


@dataclass(frozen=True)
class ExtendedConvertMemRefToPtr(ModulePass):
    name = "extended-convert-memref-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([ConvertCastOp(), ConvertLoadStorePattern(), ConvertSubviewPattern()])).rewrite_module(op)


#
# erase memreftype on all remaining values
# (runs after the patterns above consumed shape info)
#
# before:  %x : memref<4x8xf32>
# after:   %x : !llvm.ptr
#


class RewriteMemRefTypes(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()


# ===----------------------------------------------------------------------=== #
# exojit
# ===----------------------------------------------------------------------=== #


class IRGenerator:
    module: ModuleOp
    builder: Builder
    symbol_table: ScopedDict[str, SSAValue] | None
    seen_proc_names: set[str]
    seen_extern_decls: set[str]

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))
        self.symbol_table = None
        self.seen_proc_names = set()
        self.seen_extern_decls = set()

    @property
    def _syms(self) -> ScopedDict[str, SSAValue]:
        assert self.symbol_table is not None
        return self.symbol_table

    def _emit(self, op: Operation) -> SSAValue:
        self.builder.insert(op)
        assert op.results
        return op.results[0]

    def _int_const(self, value: int, int_type: IntegerType = i64) -> SSAValue:
        return self._emit(llvm.ConstantOp(IntegerAttr(value, int_type), int_type))

    @staticmethod
    def _void_fn_type(input_types: Sequence[Attribute]) -> llvm.LLVMFunctionType:
        # memrefs reach callees as bare pointers
        return llvm.LLVMFunctionType([llvm.LLVMPointerType() if isinstance(t, MemRefType) else t for t in input_types], llvm.LLVMVoidType())

    def _insert_at_module(self, op: Operation) -> None:
        Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0])).insert(op)

    @contextmanager
    def _scoped_state(self):
        # save and restore builder/symbol state across nested scopes
        parent_builder = self.builder
        parent_symbol_table = self.symbol_table
        try:
            yield
        finally:
            self.builder = parent_builder
            self.symbol_table = parent_symbol_table

    def _to_mlir_type(self, exo_type: object, mem_space: Attribute | None = None) -> Attribute:
        # map exo type (t.f32, t.tensor, etc.) to mlir type (f32, memref, etc.)
        match exo_type:
            case T.F16():
                return f16
            case T.F32() | T.Num():
                return f32
            case T.F64():
                return f64
            case T.INT8() | T.UINT8():
                return i8
            case T.UINT16():
                return i16
            case T.INT32():
                return i32
            case T.Index() | T.Size() | T.Int():
                return i64
            case T.Bool():
                return i1
            case T.Tensor():
                assert mem_space is not None
                inner = self._to_mlir_type(exo_type.type)
                assert inner in {f16, f32, f64, i8, i16, i32, i64}
                shape = [self._static_dim(dim) for dim in exo_type.shape()]
                return MemRefType(inner, shape, NoneAttr(), mem_space)
            case _:
                assert False

    @staticmethod
    def _static_dim(expr: LoopIR.expr) -> int:
        # variable/computed dims -> dynamic_index (-1), for memreftype declarations
        match expr:
            case LoopIR.Const():
                # literal (e.g. `f32[16, 16]`)
                assert isinstance(expr.val, int)
                return expr.val
            case LoopIR.Read() | LoopIR.BinOp():
                # variable (e.g. `f32[m, k]`) or computed (e.g. `f32[m+1, k*2]`)
                return DYNAMIC_INDEX
            case _:
                assert False

    def _emit_shape(self, tensor: T.Tensor) -> list[int | SSAValue]:
        # variable/computed dims -> live ssa values, for stride/offset arithmetic
        return [self._expr(dim) if isinstance(dim, (LoopIR.Read, LoopIR.BinOp)) else self._static_dim(dim) for dim in tensor.shape()]

    def _memref_load(self, memref_val: SSAValue, idx: list[SSAValue]) -> SSAValue:
        if len(idx) == 0:
            idx = [self._int_const(0)]
        indices = [self._emit(UnrealizedConversionCastOp.get([index], [IndexType()])) for index in idx]
        self.builder.insert(load := memref.LoadOp.get(memref_val, indices))
        return load.res

    def _memref_store(self, value: SSAValue, memref_val: SSAValue, idx: list[SSAValue]) -> None:
        # emit memref.store with i64->index casts, handling scalar memref cases
        if len(idx) == 0:
            assert isinstance(memref_val.type, MemRefType) and memref_val.type.get_shape() == (1,)
            idx = [self._int_const(0)]

        index_indices = [self._emit(UnrealizedConversionCastOp.get([index], [IndexType()])) for index in idx]

        # if value is a scalar memref, load it first
        if isinstance(value.type, MemRefType):
            assert value.type.get_shape() == (1,)
            value = self._memref_load(value, [])

        self.builder.insert(memref.StoreOp.get(value, memref_val, index_indices))

    def _expr_const(self, const: LoopIR.Const, expected_type: Attribute | None = None) -> SSAValue:
        # lower loopir literal to llvm.mlir.constant op
        mlir_type = expected_type if isinstance(const.type, T.Num) and expected_type is not None else self._to_mlir_type(const.type)
        assert isinstance(const.val, (int, float))
        if isinstance(mlir_type, AnyFloat):
            return self._emit(llvm.ConstantOp(FloatAttr(const.val, mlir_type), mlir_type))
        assert isinstance(mlir_type, IntegerType)
        return self._int_const(int(const.val), mlir_type)

    def _expr_read(self, read: LoopIR.Read) -> SSAValue:
        # lower loopir read to arith/memref ops
        idx = [self._expr(expr) for expr in read.idx]
        operand = self._syms[repr(read.name)]

        # only emit a load when the operand is a memref holding scalar elements
        # (not a window/tensor pass-through, and not a type-matched scalar already)
        needs_load = isinstance(operand.type, MemRefType) and not isinstance(read.type, (T.Window, T.Tensor)) and operand.type != self._to_mlir_type(read.type)
        return self._memref_load(operand, idx) if needs_load else operand

    def _expr_usub(self, usub: LoopIR.USub) -> SSAValue:
        # lower unary negation to llvm.fneg (float) or 0-x llvm.sub (int)
        expr = self._expr(usub.arg)
        if isinstance(expr.type, AnyFloat):
            return self._emit(FNegOp(expr, fast_math=llvm.FastMathAttr("fast")))
        assert isinstance(expr.type, IntegerType) and expr.type != i1
        return self._emit(llvm.SubOp(self._int_const(0, expr.type), expr))

    def _cmp_binop(self, lhs: SSAValue, rhs: SSAValue, op: str) -> SSAValue:
        P = llvm.ICmpPredicateFlag
        integer_cmp_table = {"==": P.EQ.to_int(), "!=": P.NE.to_int(), "<": P.SLT.to_int(), "<=": P.SLE.to_int(), ">": P.SGT.to_int(), ">=": P.SGE.to_int()}
        float_cmp_table = {"==": "oeq", "!=": "one", "<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}
        assert lhs.type == rhs.type
        if lhs.type == i1:
            bool_ops = {"and": llvm.AndOp, "or": llvm.OrOp}
            return self._emit(bool_ops[op](lhs, rhs))
        if isinstance(lhs.type, IntegerType):
            return self._emit(llvm.ICmpOp(lhs, rhs, IntegerAttr(integer_cmp_table[op], i64)))
        return self._emit(llvm.FCmpOp(lhs, rhs, float_cmp_table[op]))

    def _expr_binop(self, binop: LoopIR.BinOp) -> SSAValue:
        if not isinstance(binop.type, T.Num):
            mlir_type = self._to_mlir_type(binop.type)
            lhs = self._expr(binop.lhs, mlir_type)
            rhs = self._expr(binop.rhs, mlir_type)
        elif binop.op == "/" and isinstance(binop.lhs, LoopIR.Const):
            rhs = self._expr(binop.rhs)
            mlir_type = rhs.type
            lhs = self._expr(binop.lhs, mlir_type)
        else:
            lhs = self._expr(binop.lhs)
            rhs = self._expr(binop.rhs)
            mlir_type = lhs.type

        if mlir_type == i1:
            return self._cmp_binop(lhs, rhs, binop.op)

        float_ops = {"+": llvm.FAddOp, "-": llvm.FSubOp, "*": llvm.FMulOp, "/": llvm.FDivOp}
        int_ops = {"+": llvm.AddOp, "-": llvm.SubOp, "*": llvm.MulOp, "/": llvm.SDivOp, "%": llvm.SRemOp}
        if isinstance(mlir_type, AnyFloat):
            return self._emit(float_ops[binop.op](lhs, rhs, fast_math=llvm.FastMathAttr("fast")))
        assert isinstance(mlir_type, IntegerType)
        return self._emit(int_ops[binop.op](lhs, rhs))

    def _window_access(self, access: object) -> SSAValue:
        match access:
            case LoopIR.Point():
                return self._expr(access.pt)
            case LoopIR.Interval():
                return self._expr(access.lo)
            case _:
                assert False

    def _to_index_list(self, values: Sequence[SSAValue | int]) -> list:
        # cast i64 ssavalues to index type, pass through static ints as-is for subviewop
        static, dynamic = split_dynamic_index_list(values, DYNAMIC_INDEX)
        casted = [self._emit(UnrealizedConversionCastOp.get([value], [IndexType()])) for value in dynamic]
        return get_dynamic_index_list(static, casted, DYNAMIC_INDEX)

    def _expr_window(self, window: LoopIR.WindowExpr) -> SSAValue:
        # lower window expression to memref.subview
        indices = [self._window_access(access) for access in window.idx]
        source = self._syms[repr(window.name)]
        assert isinstance(source.type, MemRefType)
        assert isinstance(window.type, T.Window)
        dest_type = self._to_mlir_type(window.type.as_tensor, source.type.memory_space)
        output_sizes = self._emit_shape(window.type.as_tensor)

        offsets_idx = self._to_index_list(indices)
        sizes_idx = self._to_index_list(output_sizes)
        strides_idx = self._to_index_list([1] * len(indices))

        self.builder.insert(subview := memref.SubviewOp.get(source, offsets_idx, sizes_idx, strides_idx, dest_type))
        return subview.result

    def _expr_extern(self, extern: LoopIR.Extern) -> SSAValue:
        name = extern.f.name()
        if name == "select":
            arg_b = self._expr(extern.args[1])
            expected_type = arg_b.type
            arg_a = self._expr(extern.args[0], expected_type)
            arg_c = self._expr(extern.args[2], expected_type)
            arg_d = self._expr(extern.args[3], expected_type)
            cmp = self._emit(llvm.FCmpOp(arg_a, arg_b, "olt"))
            return self._emit(llvm.SelectOp(cmp, arg_c, arg_d))
        if name == "expf":
            x = self._expr(extern.args[0])
            x32 = x if x.type == f32 else self._emit(FPTruncOp(x, f32))
            r32 = self._emit(llvm.FExpOp(x32))
            return r32 if x.type == f32 else self._emit(llvm.FPExtOp(r32, x.type))
        unary_intrinsics = {"sqrt": llvm.FSqrtOp, "log": llvm.FLogOp, "exp": llvm.FExpOp, "sin": llvm.FSinOp, "cos": llvm.FCosOp, "floor": llvm.FFloorOp, "ceil": llvm.FCeilOp, "exp2": llvm.FExp2Op, "log2": llvm.FLog2Op}
        if (op_cls := unary_intrinsics.get(name)) is not None:
            return self._emit(op_cls(self._expr(extern.args[0])))
        args = [self._expr(arg) for arg in extern.args]
        output_type = self._to_mlir_type(extern.f.typecheck(extern.args))
        return self._emit(llvm.CallOp(name, *args, return_type=output_type))

    def _expr(self, expr: object, expected_type: Attribute | None = None) -> SSAValue:
        # dispatch loopir expression node to its typed lowering method
        match expr:
            case LoopIR.Read():
                return self._expr_read(expr)
            case LoopIR.Const():
                return self._expr_const(expr, expected_type)
            case LoopIR.USub():
                return self._expr_usub(expr)
            case LoopIR.BinOp():
                return self._expr_binop(expr)
            case LoopIR.WindowExpr():
                return self._expr_window(expr)
            case LoopIR.Extern():
                return self._expr_extern(expr)
            case _:
                assert False

    def _stmt_assign(self, stmt: LoopIR.Assign) -> None:
        # lower assignment to memref.store
        idx = [self._expr(expr) for expr in stmt.idx]
        memref_val = self._syms[repr(stmt.name)]
        expected_type = memref_val.type.element_type if isinstance(memref_val.type, MemRefType) else None
        value = self._expr(stmt.rhs, expected_type)
        self._memref_store(value, memref_val, idx)

    def _stmt_reduce(self, stmt: LoopIR.Reduce) -> None:
        # lower reduce to load + add + store (accumulate into buffer)
        idx = [self._expr(expr) for expr in stmt.idx]
        memref_val = self._syms[repr(stmt.name)]
        expected_type = memref_val.type.element_type if isinstance(memref_val.type, MemRefType) else None
        value = self._expr(stmt.rhs, expected_type)

        current = self._memref_load(memref_val, idx)
        result = self._emit(llvm.FAddOp(current, value, fast_math=llvm.FastMathAttr("fast")) if isinstance(value.type, AnyFloat) else llvm.AddOp(current, value))
        self._memref_store(result, memref_val, idx)

    def _stmt_if(self, if_stmt: LoopIR.If) -> None:
        # lower if/else to cf.cond_br with true, false, and merge blocks
        cond = self._expr(if_stmt.cond)

        region = self.builder.insertion_point.block.parent_region()
        assert region is not None
        true_block = Block()
        false_block = Block()
        merge_block = Block()
        region.add_block(true_block)
        region.add_block(false_block)

        self.builder.insert(llvm.CondBrOp(cond, true_block, [], false_block, []))

        # true branch
        self.builder = Builder(insertion_point=InsertPoint.at_end(true_block))
        for stmt in if_stmt.body:
            self._stmt(stmt)
        self.builder.insert(BrOp(merge_block))

        # false branch
        self.builder = Builder(insertion_point=InsertPoint.at_end(false_block))
        for stmt in if_stmt.orelse:
            self._stmt(stmt)
        self.builder.insert(BrOp(merge_block))

        # continue at merge
        region.add_block(merge_block)
        self.builder = Builder(insertion_point=InsertPoint.at_end(merge_block))

    def _stmt_for(self, for_stmt: LoopIR.For) -> None:
        lo = self._expr(for_stmt.lo)
        hi = self._expr(for_stmt.hi)
        assert lo.type == hi.type
        assert isinstance(lo.type, IntegerType)
        step = self._int_const(1, lo.type)

        region = self.builder.insertion_point.block.parent_region()
        assert region is not None
        header_block = Block(arg_types=[lo.type])
        body_block = Block()
        exit_block = Block()
        region.add_block(header_block)
        region.add_block(body_block)
        self.builder.insert(BrOp(header_block, lo))

        self.builder = Builder(insertion_point=InsertPoint.at_end(header_block))
        iv = header_block.args[0]
        cond = self._emit(llvm.ICmpOp(iv, hi, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64)))
        self.builder.insert(llvm.CondBrOp(cond, body_block, [], exit_block, []))

        with self._scoped_state():
            self.builder = Builder(insertion_point=InsertPoint.at_end(body_block))
            self.symbol_table = ScopedDict(self._syms)
            self._syms[repr(for_stmt.iter)] = iv
            for stmt in for_stmt.body:
                self._stmt(stmt)
            self.builder.insert(BrOp(header_block, self._emit(llvm.AddOp(iv, step))))

        region.add_block(exit_block)
        self.builder = Builder(insertion_point=InsertPoint.at_end(exit_block))

    def _stmt_alloc(self, alloc: LoopIR.Alloc) -> None:
        # lower alloc to llvm.call @malloc (dram) or llvm.alloca (stack)
        mem_name = alloc.mem.name()
        mem_space = StringAttr(mem_name)
        mlir_type = self._to_mlir_type(alloc.type, mem_space)

        # scalar allocs: wrap as memref<1x...>
        if not isinstance(mlir_type, MemRefType):
            mlir_type = MemRefType(mlir_type, [1], NoneAttr(), mem_space)

        shape = mlir_type.get_shape()
        assert all(dim != DYNAMIC_INDEX for dim in shape), "dynamic-sized allocs are not supported"
        total_elements = math.prod(shape)

        if mem_name == "DRAM":
            assert isinstance(mlir_type.element_type, FixedBitwidthType)
            size_val = self._int_const(total_elements * mlir_type.element_type.size)  # malloc takes bytes
            raw_ptr = self._emit(llvm.CallOp("malloc", size_val, return_type=llvm.LLVMPointerType()))
        else:
            size_val = self._int_const(total_elements)  # alloca takes element count
            raw_ptr = self._emit(llvm.AllocaOp(size_val, mlir_type.element_type))

        result = self._emit(UnrealizedConversionCastOp.get([raw_ptr], [mlir_type]))
        self._syms[repr(alloc.name)] = result

    def _stmt_free(self, free: LoopIR.Free) -> None:
        # lower free to llvm.call @free (dram) or no-op (stack)
        is_heap_mem = free.mem.name() == "DRAM"
        if not is_heap_mem:
            return
        memref_val = self._syms[repr(free.name)]
        cast = self._emit(UnrealizedConversionCastOp.get([memref_val], [llvm.LLVMPointerType()]))
        self.builder.insert(llvm.CallOp("free", cast))

    def _stmt_window(self, stmt: LoopIR.WindowStmt) -> None:
        # lower window statement to subview and bind result in symbol/type tables
        assert isinstance(stmt.rhs, LoopIR.WindowExpr) and isinstance(stmt.rhs.type, T.Window)
        result = self._expr_window(stmt.rhs)
        self._syms[repr(stmt.name)] = result

    def _arg_type(self, arg: LoopIR.fnarg, body: list[LoopIR.stmt]) -> Attribute:
        # scalars passed by reference (the body writes to them) must arrive as memref<1xt>
        mem_space = StringAttr(arg.mem.name()) if arg.mem is not None else None
        mlir_type = self._to_mlir_type(arg.type, mem_space)
        if not isinstance(mlir_type, MemRefType) and any(repr(sym) == repr(arg.name) for sym, _ in get_writes_of_stmts(body)):
            mlir_type = MemRefType(mlir_type, [1], NoneAttr())
        return mlir_type

    def _stmt_call(self, call: LoopIR.Call) -> None:
        # lower call to func.call. emit extern decl for intrinsics, recurse for procs
        if call.f.instr is None:
            self._generate_procedure(call.f)
            assert len(call.args) == len(call.f.args)
            args = []
            for arg, callee_arg in zip(call.args, call.f.args):
                callee_type = self._arg_type(callee_arg, call.f.body)
                if isinstance(callee_type, MemRefType) and not callee_arg.type.is_tensor_or_window():
                    assert isinstance(arg, LoopIR.Read) and not arg.idx, "writable scalar call arguments must be scalar lvalues"
                    arg_val = self._syms[repr(arg.name)]
                    assert isinstance(arg_val.type, MemRefType)
                else:
                    arg_val = self._expr(arg)
                # reconcile mlir type and shape mismatches (e.g. caller has memref<8xf32>, callee expects memref<?xf32>) via memref.cast
                if isinstance(arg_val.type, MemRefType) and isinstance(callee_type, MemRefType) and arg_val.type != callee_type:
                    arg_val = self._emit(memref.CastOp.get(arg_val, callee_type))
                args.append(arg_val)
        else:
            args = [self._expr(arg) for arg in call.args]

        if call.f.instr is not None and call.f.name not in self.seen_extern_decls:
            self.seen_extern_decls.add(call.f.name)
            self._insert_at_module(llvm.FuncOp(call.f.name, self._void_fn_type([SSAValue.get(arg).type for arg in args]), llvm.LinkageAttr("external")))

        self.builder.insert(llvm.CallOp(call.f.name, *args))

    def _stmt(self, stmt: object) -> None:
        # dispatch loopir statement node to its typed lowering method
        match stmt:
            case LoopIR.Assign():
                self._stmt_assign(stmt)
            case LoopIR.Reduce():
                self._stmt_reduce(stmt)
            case LoopIR.WriteConfig():
                assert False, "unsupported WriteConfig"
            case LoopIR.Pass():
                pass
            case LoopIR.If():
                self._stmt_if(stmt)
            case LoopIR.For():
                self._stmt_for(stmt)
            case LoopIR.Alloc():
                self._stmt_alloc(stmt)
            case LoopIR.Free():
                self._stmt_free(stmt)
            case LoopIR.Call():
                self._stmt_call(stmt)
            case LoopIR.WindowStmt():
                self._stmt_window(stmt)
            case _:
                assert False

    def _generate_procedure(self, procedure: LoopIR.proc) -> None:
        # lower loopir proc to llvm.func
        if procedure.name in self.seen_proc_names:
            return
        self.seen_proc_names.add(procedure.name)

        input_types = [self._arg_type(arg, procedure.body) for arg in procedure.args]
        func_type = self._void_fn_type(input_types)

        with self._scoped_state():
            block = Block(arg_types=input_types)
            func_region = Region(block)
            self.builder = Builder(insertion_point=InsertPoint.at_end(block))

            self.symbol_table = ScopedDict(local_scope={repr(arg.name): val for arg, val in zip(procedure.args, block.args)})

            for stmt in procedure.body:
                self._stmt(stmt)

            self.builder.insert(llvm.ReturnOp())

        noalias_attrs = ArrayAttr([DictionaryAttr({"llvm.noalias": UnitAttr()} if isinstance(t, LLVMPointerType) else {}) for t in input_types])
        self._insert_at_module(llvm.FuncOp(procedure.name, func_type, linkage=llvm.LinkageAttr("external"), body=func_region, other_props={"arg_attrs": noalias_attrs}))

    def generate(self, procs: list[LoopIR.proc]) -> ModuleOp:
        for proc in procs:
            self._generate_procedure(proc)
        # declare external malloc/free for dram alloc/free lowering
        self._insert_at_module(llvm.FuncOp("malloc", llvm.LLVMFunctionType([i64], llvm.LLVMPointerType()), llvm.LinkageAttr("external")))
        self._insert_at_module(llvm.FuncOp("free", llvm.LLVMFunctionType([llvm.LLVMPointerType()]), llvm.LinkageAttr("external")))
        return self.module


class LLVMBackend:
    @staticmethod
    @cache
    def _context() -> Context:
        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(llvm.LLVM)
        ctx.load_dialect(memref.MemRef)
        return ctx

    @staticmethod
    def _lower(procs: list[LoopIR.proc]) -> ModuleOp:
        ctx = LLVMBackend._context()
        module = IRGenerator().generate(procs)
        _simplify = lambda: (CanonicalizePass().apply(ctx, module), CommonSubexpressionElimination().apply(ctx, module), module.verify())

        _simplify()

        ExtendedConvertMemRefToPtr().apply(ctx, module)
        PatternRewriteWalker(GreedyRewritePatternApplier([RewriteMemRefTypes()])).rewrite_module(module)
        ReconcileUnrealizedCastsPass().apply(ctx, module)
        module.verify()

        _simplify()
        return module

    _jit_backend = LLVMJITBackend(lowering=(), opt_level=3)


class JITRuntime:
    @staticmethod
    def _eval_shape_expr(expr: LoopIR.expr, env: dict[object, int]) -> int:
        # resolve a dynamic tensor dimension against the size arguments seen so far
        bounds = constant_bound(expr, {sym: (value, value) for sym, value in env.items()})
        assert bounds is not None and bounds[0] is not None and bounds[0] == bounds[1], f"could not resolve dynamic tensor shape from {expr}"
        return bounds[0]

    @staticmethod
    def _tensor_converter(*, ffi: FFI, index: int, tensor_type: T.Tensor, writable: bool) -> Callable[[object, dict[object, int], list[object], list[Callable[[], None]]], object]:
        # build one argument converter for tensor or window inputs
        jit_tensor_c_types = {"f32": "float", "f64": "double", "i8": "int8_t", "ui8": "uint8_t", "ui16": "uint16_t", "i32": "int32_t", "index": "int64_t", "size": "int64_t", "bool": "_Bool"}
        shape = tensor_type.shape()
        basetype = str(tensor_type.basetype())
        assert basetype in jit_tensor_c_types, f"unsupported JIT tensor dtype: {basetype}"
        c_type = jit_tensor_c_types[basetype]

        def is_seq(x: object) -> TypeGuard[Sequence[object]]:
            return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray, memoryview))

        def linearize(value: Sequence[object], flat: list[object], leaves: list[tuple[MutableSequence[object], int]]) -> None:
            assert not writable or isinstance(value, MutableSequence), f"argument {index + 1}: writable tensor args passed as Python sequences must be mutable at every level"
            for i, item in enumerate(value):
                if is_seq(item):
                    linearize(item, flat, leaves)
                else:
                    flat.append(item)
                    if writable:
                        leaves.append((cast(MutableSequence[object], value), i))

        def convert(value: object, shape_env: dict[object, int], keepalive: list[object], syncbacks: list[Callable[[], None]]) -> object:
            assert not (isinstance(value, (bytes, bytearray, memoryview)) or (hasattr(value, "ndim") and hasattr(value, "dtype") and hasattr(value, "shape") and getattr(value, "ndim", 0) > 0)), f"argument {index + 1}: direct buffer inputs are not supported by jit(); pass Python lists/scalars or use jit(proc, raw=True)"
            numel = math.prod(JITRuntime._eval_shape_expr(expr, shape_env) for expr in shape)
            flat: list[object] = []
            leaves: list[tuple[MutableSequence[object], int]] = []
            if is_seq(value):
                linearize(value, flat, leaves)
            else:
                assert numel == 1, f"argument {index + 1}: expected {numel} values, got scalar {type(value).__name__}"
                assert not writable, f"argument {index + 1}: writable scalar tensor args require a mutable sequence"
                assert isinstance(value, numbers.Real), f"argument {index + 1}: expected scalar numeric data, got {type(value).__name__}"
                flat.append(value)
            assert len(flat) == numel, f"argument {index + 1}: expected {numel} values, got {len(flat)}"

            buf = ffi.new(f"{c_type}[{numel}]", flat)
            keepalive.append(buf)
            if writable:

                def sync() -> None:
                    for offset, (target, idx) in enumerate(leaves):
                        target[idx] = buf[offset]

                syncbacks.append(sync)
            return int(ffi.cast("uintptr_t", buf))

        return convert

    @staticmethod
    def compile(proc: Procedure, raw: bool = False) -> Callable[..., None]:
        mlir_module = to_mlir(proc)
        raw_jit = LLVMBackend._jit_backend.jit(mlir_module, proc.name(), LLVMBackend._context())
        fn = raw_jit.c_func

        ir_args = proc._loopir_proc.args
        for arg in ir_args:
            assert arg.type.is_tensor_or_window() or isinstance(arg.type, (LoopIR.Size, LoopIR.Index, LoopIR.Int, LoopIR.Bool, LoopIR.Stride)), f"unsupported JIT argument type for {arg.name}: {arg.type}"
        written = {sym for sym, _ in get_writes_of_stmts(proc._loopir_proc.body)}  # Exo resolves window and callee writes when classifying pointer mutability
        kinds = [arg.name in written if arg.type.is_tensor_or_window() else None for arg in ir_args]  # None: passed by value, False/True: pointer, writable or not

        ffi = FFI()

        def call(*args) -> None:
            fn(*[arg if kind is None else ffi.cast("void *", arg) if isinstance(arg, int) else ffi.from_buffer(cast(Any, arg), require_writable=kind) for arg, kind in zip(args, kinds, strict=True)])

        # mcjit owns the jitted code, so the raw_jit (which retains the engine) must outlive every call
        cast(Any, call)._raw_jit = raw_jit

        names = [re.sub(r"_\d+$", "", str(arg.name)) for arg in ir_args]
        if raw:
            wrapped = lambda *args, **kwargs: call(*(tuple(kwargs[name] for name in names) if kwargs else args))
            cast(Any, wrapped)._raw = call
            return wrapped

        converters = []
        for i, (arg, kind) in enumerate(zip(ir_args, kinds, strict=True)):
            if kind is None:

                def convert(value: SupportsInt | str, shape_env: dict[object, int], _keepalive: list[object], _syncbacks: list[Callable[[], None]], name=arg.name) -> int:
                    shape_env[name] = converted = int(value)
                    return converted

                converters.append(convert)
            else:
                converters.append(JITRuntime._tensor_converter(ffi=ffi, index=i, tensor_type=arg.type.as_tensor if isinstance(arg.type, T.Window) else arg.type, writable=kind))

        def wrapped(*args, **kwargs):
            args = tuple(kwargs[name] for name in names) if kwargs else args
            shape_env, keepalive, syncbacks = {}, [], []
            call(*[conv(arg, shape_env, keepalive, syncbacks) for conv, arg in zip(converters, args, strict=True)])
            for sync in syncbacks:
                sync()

        cast(Any, wrapped)._raw = call
        return wrapped


def to_mlir(library: Procedure | Sequence[Procedure]) -> ModuleOp:
    # exo procedures -> xdsl mlir (llvm dialect)
    if isinstance(library, Procedure):
        library = [library]
    compilable = [proc._loopir_proc for proc in library if not proc.is_instr()]
    all_procs = sorted(find_all_subprocs(compilable), key=lambda proc: proc.name)
    unique_procs = list({proc.name: proc for proc in all_procs if proc.instr is None}.values())

    def exo_analyze(proc: LoopIR.proc) -> LoopIR.proc:
        proc = PrecisionAnalysis().run(proc)
        proc = WindowAnalysis().apply_proc(proc)
        return MemoryAnalysis().run(proc)

    return LLVMBackend._lower([exo_analyze(proc) for proc in unique_procs])


def jit(proc=None, *, raw: bool = False, optimize: Callable[[Procedure], Procedure] | None = None):
    # call directly: `jit(proc)(...)`
    # call as a decorator: `@jit(optimize=fn)`
    if proc is None:
        return lambda fn: jit(fn, raw=raw, optimize=optimize)
    if callable(proc) and not isinstance(proc, Procedure):
        body, src_info = get_ast_from_python(proc)
        parser = Parser(body, src_info, parent_scope=DummyScope(proc.__globals__, {}), as_func=True)
        proc = Procedure(parser.result())
    if optimize:
        proc = optimize(proc)
    return JITRuntime.compile(proc, raw=raw)


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--c", "fmt", flag_value="c", help="Output C source")
@click.option("--mlir", "fmt", flag_value="mlir", help="Output MLIR")
def cli(source: Path, fmt: Literal["c", "mlir"] | None):
    assert fmt, "choose --c or --mlir"
    mod = load_user_code(source)
    procs = list({v.name(): v for v in mod.__dict__.values() if isinstance(v, Procedure) and not v.is_instr()}.values())

    match fmt:
        case "c":
            tmpdir = Path(tempfile.mkdtemp())
            exo_compile_procs(procs, tmpdir, "o.c", "o.h")
            print((tmpdir / "o.c").read_text())
        case "mlir":
            print(to_mlir(procs))
