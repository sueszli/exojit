from __future__ import annotations

import hashlib
import math
import numbers
import re
import sys
import tempfile
from collections.abc import Callable, MutableSequence, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Literal, SupportsInt, TypeGuard, cast

import click
import llvmlite.binding
from cffi import FFI
from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from exo.backend.LoopIR_compiler import find_all_subprocs
from exo.backend.mem_analysis import MemoryAnalysis
from exo.backend.parallel_analysis import ParallelAnalysis
from exo.backend.prec_analysis import PrecisionAnalysis
from exo.backend.win_analysis import WindowAnalysis
from exo.core.LoopIR import LoopIR, T, get_writes_of_stmts
from exo.main import load_user_code
from exo.rewrite.range_analysis import constant_bound
from xdsl.backend.llvm.convert import convert_module
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import llvm, memref
from xdsl.dialects.builtin import DYNAMIC_INDEX, AnyFloat, ArrayAttr, BoolAttr, Builtin, DictionaryAttr, FloatAttr, IndexType, IntAttr, IntegerAttr, IntegerType, MemRefType, ModuleOp, NoneAttr, StringAttr, UnitAttr, UnrealizedConversionCastOp, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.llvm import BrOp, FNegOp
from xdsl.dialects.utils import get_dynamic_index_list, split_dynamic_index_list
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.jit.c_type_context import CTypeContext, register_builtin_types
from xdsl.jit.llvm.c_type_context import register_llvm_types, to_c_func_type
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.scoped_dict import ScopedDict

import exojit.patches_exo  # noqa: F401
from exojit.patches_xdsl_intrinsics import ConvertVecIntrinsic
from exojit.patches_xdsl_llvm import ExtendedConvertMemRefToPtr, FPTruncOp, RewriteMemRefTypes


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
        self._par_counter = 0  # for naming

    @property
    def _syms(self) -> ScopedDict[str, SSAValue]:
        assert self.symbol_table is not None
        return self.symbol_table

    def _emit(self, op: Operation) -> SSAValue:
        self.builder.insert(op)
        assert op.results
        return op.results[0]

    def _insert_at_module(self, op: Operation) -> None:
        Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0])).insert(op)

    @contextmanager
    def _scoped_state(self, *, inherit: bool = True):
        # save and restore builder/symbol/type state across nested scopes
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
            case SSAValue():
                return exo_type.type
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
                shape = self._shape(exo_type)
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

    def _shape(self, tensor: T.Tensor) -> list[int]:
        return [self._static_dim(expr) for expr in tensor.shape()]

    def _emit_shape(self, tensor: T.Tensor) -> list[int | SSAValue]:
        # variable/computed dims -> live ssa values, for stride/offset arithmetic

        def from_expr(expr: LoopIR.expr) -> int | SSAValue:
            match expr:
                case LoopIR.Read():
                    return self._syms[repr(expr.name)]
                case LoopIR.BinOp():
                    return self._expr_binop(expr)
                case _:
                    return self._static_dim(expr)

        return [from_expr(expr) for expr in tensor.shape()]

    def _zero_index(self) -> list[SSAValue]:
        return [self._emit(llvm.ConstantOp(IntegerAttr(0, i64), i64))]

    def _memref_load(self, memref_val: SSAValue, idx: list[SSAValue]) -> SSAValue:
        if len(idx) == 0:
            idx = self._zero_index()
        indices = [self._emit(UnrealizedConversionCastOp.get([index], [IndexType()])) for index in idx]
        self.builder.insert(load := memref.LoadOp.get(memref_val, indices))
        return load.res

    def _memref_store(self, value: SSAValue, memref_val: SSAValue, idx: list[SSAValue]) -> None:
        # emit memref.store with i64->index casts, handling scalar memref cases
        if len(idx) == 0:
            assert isinstance(memref_val.type, MemRefType) and memref_val.type.get_shape() == (1,)
            idx = self._zero_index()

        index_indices = [self._emit(UnrealizedConversionCastOp.get([index], [IndexType()])) for index in idx]

        # if value is a scalar memref, load it first
        if isinstance(value.type, MemRefType):
            assert value.type.get_shape() == (1,)
            value = self._memref_load(value, [])

        self.builder.insert(memref.StoreOp.get(value, memref_val, index_indices))

    def _expr_const(self, const: LoopIR.Const, expected_type: Attribute | None = None) -> SSAValue:
        # lower loopir literal to llvm.mlir.constant op
        if isinstance(const.type, T.Num) and expected_type is not None:
            mlir_type = expected_type
        else:
            mlir_type = self._to_mlir_type(const.type)
        assert isinstance(const.val, (int, float))

        if mlir_type in [f16, f32, f64]:
            assert isinstance(mlir_type, AnyFloat)
            attr = FloatAttr(const.val, mlir_type)
        elif mlir_type in [i8, i16, i32, i64]:
            assert isinstance(mlir_type, IntegerType)
            attr = IntegerAttr(IntAttr(int(const.val)), mlir_type)
        elif mlir_type == i1:
            assert isinstance(const.val, int)
            attr = BoolAttr(const.val, i1)
        else:
            assert False

        return self._emit(llvm.ConstantOp(attr, mlir_type))

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
        is_num_type = isinstance(usub.type, T.Num)
        mlir_type = expr.type if is_num_type else self._to_mlir_type(usub.type)

        if mlir_type in [f16, f32, f64]:
            return self._emit(FNegOp(expr, fast_math=llvm.FastMathAttr("fast")))
        elif mlir_type in [i8, i16, i32, i64]:
            assert isinstance(mlir_type, IntegerType)
            zero = self._emit(llvm.ConstantOp(IntegerAttr(0, mlir_type), mlir_type))
            return self._emit(llvm.SubOp(zero, expr))
        else:
            assert False

    @staticmethod
    def _cmp_binop(lhs: SSAValue, rhs: SSAValue, op: str, emit: Callable[[Operation], SSAValue]) -> SSAValue:
        P = llvm.ICmpPredicateFlag
        integer_cmp_table = {"==": P.EQ.to_int(), "!=": P.NE.to_int(), "<": P.SLT.to_int(), "<=": P.SLE.to_int(), ">": P.SGT.to_int(), ">=": P.SGE.to_int()}
        fcmp_predicates: dict[str, tuple[str, bool]] = {"oeq": ("==", True), "ogt": (">", True), "oge": (">=", True), "olt": ("<", True), "ole": ("<=", True), "one": ("!=", True), "ord": ("ord", True), "ueq": ("==", False), "ugt": (">", False), "uge": (">=", False), "ult": ("<", False), "ule": ("<=", False), "une": ("!=", False), "uno": ("uno", False)}
        float_cmp_table = {op: pred for pred, (op, ordered) in fcmp_predicates.items() if ordered and op not in ("ord", "uno")}
        assert lhs.type == rhs.type
        if lhs.type == i1:
            bool_ops = {"and": llvm.AndOp, "or": llvm.OrOp}
            return emit(bool_ops[op](lhs, rhs))
        if lhs.type in [i8, i16, i32, i64]:
            return emit(llvm.ICmpOp(lhs, rhs, IntegerAttr(integer_cmp_table[op], i64)))
        return emit(llvm.FCmpOp(lhs, rhs, float_cmp_table[op]))

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
            return self._cmp_binop(lhs, rhs, binop.op, self._emit)

        float_ops = {"+": llvm.FAddOp, "-": llvm.FSubOp, "*": llvm.FMulOp, "/": llvm.FDivOp}
        int_ops = {"+": llvm.AddOp, "-": llvm.SubOp, "*": llvm.MulOp, "/": llvm.SDivOp, "%": llvm.SRemOp}
        if mlir_type in [f16, f32, f64]:
            return self._emit(float_ops[binop.op](lhs, rhs, fast_math=llvm.FastMathAttr("fast")))
        if mlir_type in [i8, i16, i32, i64]:
            return self._emit(int_ops[binop.op](lhs, rhs))
        assert False

    @staticmethod
    def _window_access(access: object, expr_fn: Callable[[object], SSAValue]) -> SSAValue:
        match access:
            case LoopIR.Point():
                return expr_fn(access.pt)
            case LoopIR.Interval():
                return expr_fn(access.lo)
            case _:
                assert False

    @staticmethod
    def _to_index_list(values: Sequence[SSAValue | int], emit: Callable[[Operation], SSAValue]) -> list:
        # cast i64 ssavalues to index type, pass through static ints as-is for subviewop
        static, dynamic = split_dynamic_index_list(values, DYNAMIC_INDEX)
        casted = [emit(UnrealizedConversionCastOp.get([value], [IndexType()])) for value in dynamic]
        return get_dynamic_index_list(static, casted, DYNAMIC_INDEX)

    def _expr_window(self, window: LoopIR.WindowExpr) -> SSAValue:
        # lower window expression to memref.subview
        indices = [self._window_access(access, self._expr) for access in window.idx]
        source = self._syms[repr(window.name)]
        assert isinstance(source.type, MemRefType)
        assert isinstance(window.type, T.Window)
        dest_type = self._to_mlir_type(window.type.as_tensor, source.type.memory_space)
        output_sizes = self._emit_shape(window.type.as_tensor)

        offsets_idx = self._to_index_list(indices, self._emit)
        sizes_idx = self._to_index_list(output_sizes, self._emit)
        strides_idx = self._to_index_list([1] * len(indices), self._emit)

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

    def _expr(self, expr: object, expected_type: Attribute | None = None) -> OpResult | SSAValue:
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
        if value.type in [f16, f32, f64]:
            result = self._emit(llvm.FAddOp(current, value, fast_math=llvm.FastMathAttr("fast")))
        else:
            result = self._emit(llvm.AddOp(current, value))
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

    def _stmt_for_par(self, s: LoopIR.For) -> None:
        # par() loop -> __kmpc_fork_call(@outlined, lo, hi, ...shared)
        # outlined fn: static_init_8 -> loop [adj_lo, adj_hi] -> static_fini
        lo = self._expr(s.lo)
        hi = self._expr(s.hi)
        ptr = llvm.LLVMPointerType()

        def c(v: int, t: IntegerType = i64) -> SSAValue:
            return self._emit(llvm.ConstantOp(IntegerAttr(v, t), t))

        st = lambda v, p: self.builder.insert(llvm.StoreOp(v, p))
        ext = lambda v: v if v.type == i64 else self._emit(llvm.SExtOp(v, i64))
        alloc = lambda t: self._emit(llvm.AllocaOp(c(1), t))

        def flat(sd):  # flatten ScopedDict parent chain
            d = flat(sd.parent) if sd.parent else {}
            d.update(sd.local_scope)
            return d

        # shared captures: all live vars passed to outlined fn
        syms = flat(self._syms)
        names = list(syms.keys())

        # bounds passed by pointer
        lo_p = alloc(lo.type)
        st(lo, lo_p)
        hi_p = alloc(hi.type)
        st(hi, hi_p)

        # outlined fn: void @__omp_outlined_N(i32* gtid, i32* tid, T* lo, T* hi, ...shared)
        oname = f"__omp_outlined_{self._par_counter}"
        self._par_counter += 1
        atypes = [ptr] * 4 + [syms[n].type for n in names]
        ftype = llvm.LLVMFunctionType([llvm.LLVMPointerType() if isinstance(t, MemRefType) else t for t in atypes], llvm.LLVMVoidType())
        with self._scoped_state(inherit=False):
            blk = Block(arg_types=atypes)
            region = Region(blk)
            self.builder = Builder(insertion_point=InsertPoint.at_end(blk))
            self.symbol_table = ScopedDict()
            for i, n in enumerate(names):  # bind shared captures (args[4:])
                self._syms[n] = blk.args[4 + i]
            gtid = self._emit(llvm.LoadOp(blk.args[0], i32))
            lo_v = self._emit(llvm.LoadOp(blk.args[2], lo.type))
            hi_v = self._emit(llvm.LoadOp(blk.args[3], hi.type))

            # static_init_8 out-params: is_last, lower, upper, stride
            is_last_p = alloc(i32)
            lower_p = alloc(i64)
            upper_p = alloc(i64)
            stride_p = alloc(i64)
            st(c(0, i32), is_last_p)
            lo64 = ext(lo_v)
            hi_incl = self._emit(llvm.SubOp(ext(hi_v), c(1)))  # [lo, hi) -> [lo, hi-1]
            st(lo64, lower_p)
            st(hi_incl, upper_p)
            st(c(1), stride_p)

            # partition [lo, hi-1] across threads (schedule 34 = static)
            null = self._emit(llvm.ZeroOp(result_types=[ptr]))
            self.builder.insert(llvm.CallOp("__kmpc_for_static_init_8", null, gtid, c(34, i32), is_last_p, lower_p, upper_p, stride_p, c(1), c(1)))

            # this thread's chunk; clamp upper to original hi-1
            adj_lo = self._emit(llvm.LoadOp(lower_p, i64))
            adj_hi_raw = self._emit(llvm.LoadOp(upper_p, i64))
            adj_hi = self._emit(llvm.SelectOp(self._emit(llvm.ICmpOp(adj_hi_raw, hi_incl, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))), adj_hi_raw, hi_incl))

            # loop: header(iv) -> body -> back-edge
            r = blk.parent_region()
            assert r is not None
            hdr = Block(arg_types=[i64])
            body = Block()
            exit_ = Block()
            r.add_block(hdr)
            r.add_block(body)
            self.builder.insert(BrOp(hdr, adj_lo))
            self.builder = Builder(insertion_point=InsertPoint.at_end(hdr))
            iv = hdr.args[0]
            self.builder.insert(llvm.CondBrOp(self._emit(llvm.ICmpOp(iv, adj_hi, IntegerAttr(llvm.ICmpPredicateFlag.SLE.to_int(), i64))), body, [], exit_, []))
            with self._scoped_state():  # body: bind iter, emit stmts, iv++
                self.builder = Builder(insertion_point=InsertPoint.at_end(body))
                self.symbol_table = ScopedDict(self._syms)
                self._syms[repr(s.iter)] = self._emit(llvm.TruncOp(iv, lo.type)) if i64 != lo.type else iv
                for stmt in s.body:
                    self._stmt(stmt)
                self.builder.insert(BrOp(hdr, self._emit(llvm.AddOp(iv, c(1)))))
            r.add_block(exit_)  # exit: static_fini + ret
            self.builder = Builder(insertion_point=InsertPoint.at_end(exit_))
            self.builder.insert(llvm.CallOp("__kmpc_for_static_fini", self._emit(llvm.ZeroOp(result_types=[ptr])), self._emit(llvm.LoadOp(blk.args[0], i32))))
            self.builder.insert(llvm.ReturnOp())
        self._insert_at_module(llvm.FuncOp(oname, ftype, linkage=llvm.LinkageAttr("external"), body=region))

        # caller: fork_call(loc=null, argc, @outlined, lo*, hi*, ...shared_as_ptr)
        args = [self._emit(llvm.ZeroOp(result_types=[ptr])), c(len(names) + 2, i32), self._emit(llvm.AddressOfOp(oname, ptr)), lo_p, hi_p]
        args += [self._emit(UnrealizedConversionCastOp.get([syms[n]], [ptr])) if syms[n].type != ptr else syms[n] for n in names]
        self.builder.insert(llvm.CallOp("__kmpc_fork_call", *args))

    def _stmt_for(self, for_stmt: LoopIR.For) -> None:
        if isinstance(for_stmt.loop_mode, LoopIR.Par):
            return self._stmt_for_par(for_stmt)

        # lower for loop to cf.br/cond_br with header, body, and exit blocks
        lo = self._expr(for_stmt.lo)
        hi = self._expr(for_stmt.hi)
        assert lo.type == hi.type
        assert isinstance(lo.type, IntegerType)
        step = self._emit(llvm.ConstantOp(IntegerAttr(1, lo.type), lo.type))

        region = self.builder.insertion_point.block.parent_region()
        assert region is not None
        header_block = Block(arg_types=[lo.type])
        body_block = Block()
        exit_block = Block()
        region.add_block(header_block)
        region.add_block(body_block)

        # branch from current block to header with initial iv
        self.builder.insert(BrOp(header_block, lo))

        # header: condition check
        self.builder = Builder(insertion_point=InsertPoint.at_end(header_block))
        iv = header_block.args[0]
        cond = self._emit(llvm.ICmpOp(iv, hi, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64)))
        self.builder.insert(llvm.CondBrOp(cond, body_block, [], exit_block, []))

        # body: emit loop body in a child symbol scope
        with self._scoped_state():
            self.builder = Builder(insertion_point=InsertPoint.at_end(body_block))
            self.symbol_table = ScopedDict(self._syms)
            self._syms[repr(for_stmt.iter)] = iv

            for stmt in for_stmt.body:
                self._stmt(stmt)

            # after body: increment iv and branch back to header
            next_iv = self._emit(llvm.AddOp(iv, step))
            self.builder.insert(BrOp(header_block, next_iv))

        # continue at exit block
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
            elem_bytes = {f16: 2, f32: 4, f64: 8, i8: 1, i16: 2, i32: 4, i64: 8}[mlir_type.element_type]
            size_val = self._emit(llvm.ConstantOp(IntegerAttr(total_elements * elem_bytes, i64), i64))  # malloc takes bytes
            raw_ptr = self._emit(llvm.CallOp("malloc", size_val, return_type=llvm.LLVMPointerType()))
        else:
            size_val = self._emit(llvm.ConstantOp(IntegerAttr(total_elements, i64), i64))  # alloca takes element count
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

    @staticmethod
    def _is_mutated(name: str, body: list[LoopIR.stmt]) -> bool:
        return any(repr(sym) == name for sym, _ in get_writes_of_stmts(body))

    @staticmethod
    def _coerce_arg(arg_val: SSAValue, callee_arg: LoopIR.fnarg, callee_body: list[LoopIR.stmt], type_fn: Callable[[object, Attribute | None], Attribute], emit_fn: Callable[[Operation], SSAValue]) -> SSAValue:
        # reconcile mlir type and shape mismatches (e.g. caller has memref<8xf32>, callee expects memref<?xf32>) via memref.cast
        mem_space = StringAttr(callee_arg.mem.name()) if callee_arg.mem is not None else None
        callee_type = type_fn(callee_arg.type, mem_space)

        # scalars passed by reference (callee writes to them) must arrive as memref<1xt>
        scalar_passed_by_ref = not isinstance(callee_type, MemRefType) and IRGenerator._is_mutated(repr(callee_arg.name), callee_body)
        if scalar_passed_by_ref:
            callee_type = MemRefType(callee_type, [1], NoneAttr())

        if isinstance(arg_val.type, MemRefType) and isinstance(callee_type, MemRefType) and arg_val.type != callee_type:
            return emit_fn(memref.CastOp.get(arg_val, callee_type))

        return arg_val

    def _stmt_call(self, call: LoopIR.Call) -> None:
        # lower call to func.call. emit extern decl for intrinsics, recurse for procs
        if call.f.instr is None:
            self._generate_procedure(call.f)
            assert len(call.args) == len(call.f.args)
            args = []
            for arg, callee_arg in zip(call.args, call.f.args):
                mem_space = StringAttr(callee_arg.mem.name()) if callee_arg.mem is not None else None
                callee_type = self._to_mlir_type(callee_arg.type, mem_space)
                scalar_passed_by_ref = not isinstance(callee_type, MemRefType) and self._is_mutated(repr(callee_arg.name), call.f.body)
                if scalar_passed_by_ref:
                    assert isinstance(arg, LoopIR.Read) and not arg.idx, "writable scalar call arguments must be scalar lvalues"
                    arg_val = self._syms[repr(arg.name)]
                    assert isinstance(arg_val.type, MemRefType)
                else:
                    arg_val = self._expr(arg)
                args.append(self._coerce_arg(arg_val, callee_arg, call.f.body, self._to_mlir_type, self._emit))
        else:
            args = [self._expr(arg) for arg in call.args]

        if call.f.instr is not None and call.f.name not in self.seen_extern_decls:
            self.seen_extern_decls.add(call.f.name)
            input_types = [SSAValue.get(arg).type for arg in args]
            self._insert_at_module(
                llvm.FuncOp(
                    call.f.name,
                    llvm.LLVMFunctionType([llvm.LLVMPointerType() if isinstance(t, MemRefType) else t for t in input_types], llvm.LLVMVoidType()),
                    llvm.LinkageAttr("external"),
                )
            )

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

        # build func signature: map each arg to its mlir type, wrapping mutated scalars in memref<1x>
        input_types = []
        for arg in procedure.args:
            mem = StringAttr(arg.mem.name()) if arg.mem is not None else None
            mlir_type = self._to_mlir_type(arg.type, mem)
            if not isinstance(mlir_type, MemRefType) and self._is_mutated(repr(arg.name), procedure.body):
                mlir_type = MemRefType(mlir_type, [1], NoneAttr())
            input_types.append(mlir_type)
        func_type = llvm.LLVMFunctionType([llvm.LLVMPointerType() if isinstance(t, MemRefType) else t for t in input_types], llvm.LLVMVoidType())

        with self._scoped_state(inherit=False):
            block = Block(arg_types=input_types)
            func_region = Region(block)
            self.builder = Builder(insertion_point=InsertPoint.at_end(block))

            self.symbol_table = ScopedDict(local_scope={repr(arg.name): val for arg, val in zip(procedure.args, block.args)})

            for stmt in procedure.body:
                self._stmt(stmt)

            self.builder.insert(llvm.ReturnOp())

        self._insert_at_module(llvm.FuncOp(procedure.name, func_type, linkage=llvm.LinkageAttr("external"), body=func_region))

    def generate(self, procs: list[LoopIR.proc]) -> ModuleOp:
        for proc in procs:
            self._generate_procedure(proc)
        # declare external malloc/free for dram alloc/free lowering
        self._insert_at_module(llvm.FuncOp("malloc", llvm.LLVMFunctionType([i64], llvm.LLVMPointerType()), llvm.LinkageAttr("external")))
        self._insert_at_module(llvm.FuncOp("free", llvm.LLVMFunctionType([llvm.LLVMPointerType()]), llvm.LinkageAttr("external")))
        if self._par_counter:
            ptr = llvm.LLVMPointerType()
            self._insert_at_module(llvm.FuncOp("__kmpc_fork_call", llvm.LLVMFunctionType([ptr, i32, ptr], is_variadic=True), llvm.LinkageAttr("external")))
            self._insert_at_module(llvm.FuncOp("__kmpc_for_static_init_8", llvm.LLVMFunctionType([ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64]), llvm.LinkageAttr("external")))
            self._insert_at_module(llvm.FuncOp("__kmpc_for_static_fini", llvm.LLVMFunctionType([ptr, i32]), llvm.LinkageAttr("external")))
        return self.module


@cache
def _context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(llvm.LLVM)
    ctx.load_dialect(memref.MemRef)
    return ctx


def _lower(procs: list[LoopIR.proc]) -> ModuleOp:
    ctx = _context()

    module = IRGenerator().generate(procs)

    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    # full lowering to llvm dialect
    _rewrite = lambda patterns: PatternRewriteWalker(GreedyRewritePatternApplier(patterns)).rewrite_module(module)
    ExtendedConvertMemRefToPtr().apply(ctx, module)  # memref.{load,store,subview,cast} -> llvm
    _rewrite([RewriteMemRefTypes()])  # memreftype -> llvm.ptr on all values
    _rewrite([ConvertVecIntrinsic()])  # vec_*/neon_* calls -> llvm ops
    ReconcileUnrealizedCastsPass().apply(ctx, module)  # fold paired unrealized casts
    module.verify()

    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    return module


def to_mlir(library: Procedure | Sequence[Procedure]) -> ModuleOp:
    # exo procedures -> xdsl mlir (llvm dialect)
    if isinstance(library, Procedure):
        library = [library]
    compilable = [proc._loopir_proc for proc in library if not proc.is_instr()]
    all_procs = sorted(find_all_subprocs(compilable), key=lambda proc: proc.name)
    unique_procs = list({proc.name: proc for proc in all_procs if proc.instr is None}.values())

    def exo_analyze(proc: LoopIR.proc) -> LoopIR.proc:
        proc = ParallelAnalysis().run(proc)
        proc = PrecisionAnalysis().run(proc)
        proc = WindowAnalysis().apply_proc(proc)
        return MemoryAnalysis().run(proc)

    return _lower([exo_analyze(proc) for proc in unique_procs])


llvmlite.binding.initialize_native_target()
llvmlite.binding.initialize_native_asmprinter()


def _target_machine() -> llvmlite.binding.TargetMachine:
    # llvmlite target machines are not safe to reuse after MCJIT compilation.
    # do not cache to avoid stale target-data pointers during later `--asm` runs.
    return llvmlite.binding.Target.from_default_triple().create_target_machine(cpu=llvmlite.binding.get_host_cpu_name(), features=llvmlite.binding.get_host_cpu_features().flatten(), opt=3)


def _to_llvm_ir(module: ModuleOp) -> str:
    # xdsl mlir (llvm dialect) -> llvm ir text, pinned to the host target
    module = module.clone()
    for func_op in module.ops:
        # noalias lets llvm's loop vectorizer assume buffers do not overlap
        assert isinstance(func_op, llvm.FuncOp)
        func_op.arg_attrs = ArrayAttr([DictionaryAttr({"llvm.noalias": UnitAttr()} if isinstance(input_type, llvm.LLVMPointerType) else {}) for input_type in func_op.function_type.inputs])
    tm = _target_machine()
    return str(convert_module(module, fallback_target_triple=tm.triple, data_layout=str(tm.target_data)))


def _to_llvmlite_moduleref(ir: str) -> tuple[llvmlite.binding.ModuleRef, llvmlite.binding.TargetMachine]:
    mod_ref = llvmlite.binding.parse_assembly(ir)
    tm = _target_machine()
    pto = llvmlite.binding.PipelineTuningOptions(speed_level=3)
    pto.slp_vectorization = True
    pb = llvmlite.binding.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(mod_ref, pb)
    return mod_ref, tm


def to_asm(module: ModuleOp) -> str:
    # xdsl mlir -> native assembly text
    mod_ref, tm = _to_llvmlite_moduleref(_to_llvm_ir(module))
    return tm.emit_assembly(mod_ref)


def _disk_cache(name: object, generate: Callable[[], str]) -> str:
    # hash all compiler sources -> .cache/exojit/{hash}/. auto-invalidates when compiler code changes.
    src_dir = Path(__file__).resolve().parent
    cache_dir = src_dir.parent / ".cache" / "exojit" / hashlib.sha256(b"".join(f.read_bytes() for f in sorted(src_dir.glob("*.py")))).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.ll"
    if not path.exists():
        path.write_text(generate())
    return path.read_text()


@cache
def _load_libomp() -> None:
    if sys.platform != "darwin":
        return llvmlite.binding.load_library_permanently("libgomp.so.1")
    candidates = [f"{prefix}/opt/{pkg}/lib/libomp.dylib" for prefix in ("/opt/homebrew", "/usr/local") for pkg in ("libomp", "llvm")]
    lib = next((c for c in candidates if Path(c).exists()), None)
    assert lib, f"libomp not found; tried {candidates}"
    llvmlite.binding.load_library_permanently(lib)


@cache
def _c_type_context() -> CTypeContext:
    ctx = CTypeContext()
    register_builtin_types(ctx)
    register_llvm_types(ctx)
    return ctx


def _c_signature(module: ModuleOp, name: str) -> str:
    # cffi abi declaration for the entry point, derived from its llvm function type
    func_op = SymbolTable.lookup_symbol(module, name)
    assert isinstance(func_op, llvm.FuncOp)
    signature = to_c_func_type(_c_type_context(), func_op.function_type)
    return f"{signature.output}(*)({', '.join(signature.inputs) or 'void'})"


def _jit_eval_shape_expr(expr: LoopIR.expr, env: dict[object, int]) -> int:
    # resolve a dynamic tensor dimension against the size arguments seen so far
    bounds = constant_bound(expr, {sym: (value, value) for sym, value in env.items()})
    assert bounds is not None and bounds[0] is not None and bounds[0] == bounds[1], f"could not resolve dynamic tensor shape from {expr}"
    return bounds[0]


def _jit_tensor_converter(*, ffi: FFI, index: int, tensor_type: T.Tensor, writable: bool) -> Callable[[object, dict[object, int], list[object], list[Callable[[], None]]], object]:
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
        numel = math.prod(_jit_eval_shape_expr(expr, shape_env) for expr in shape)
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


def _jit_compile(proc: Procedure, raw: bool = False) -> Callable[..., None]:
    mlir_module = to_mlir(proc)
    ir_text = _disk_cache(hashlib.sha256(str(mlir_module).encode()).hexdigest()[:16], lambda: _to_llvm_ir(mlir_module))
    if "__kmpc_fork_call" in ir_text:  # see https://openmp.llvm.org/doxygen/group__THREADPRIVATE.html
        _load_libomp()

    mod_ref, tm = _to_llvmlite_moduleref(ir_text)
    engine = llvmlite.binding.create_mcjit_compiler(mod_ref, tm)
    engine.finalize_object()
    engine.run_static_constructors()

    ir_args = proc._loopir_proc.args
    for arg in ir_args:
        assert arg.type.is_tensor_or_window() or isinstance(arg.type, (LoopIR.Size, LoopIR.Index, LoopIR.Int, LoopIR.Bool, LoopIR.Stride)), f"unsupported JIT argument type for {arg.name}: {arg.type}"
    written = {sym for sym, _ in get_writes_of_stmts(proc._loopir_proc.body)}  # Exo resolves window and callee writes when classifying pointer mutability
    kinds = [arg.name in written if arg.type.is_tensor_or_window() else None for arg in ir_args]  # None: passed by value, False/True: pointer, writable or not

    ffi = FFI()
    address = engine.get_function_address(proc.name())
    assert address, f"no address for symbol after compilation: {proc.name()}"
    fn = ffi.cast(_c_signature(mlir_module, proc.name()), address)

    def call(*args) -> None:
        fn(*[arg if kind is None else ffi.cast("void *", arg) if isinstance(arg, int) else ffi.from_buffer(cast(Any, arg), require_writable=kind) for arg, kind in zip(args, kinds, strict=True)])

    # mcjit owns the jitted code, so the engine has to outlive every call into it
    cast(Any, call)._engine = engine

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
            converters.append(_jit_tensor_converter(ffi=ffi, index=i, tensor_type=arg.type.as_tensor if isinstance(arg.type, T.Window) else arg.type, writable=kind))

    def wrapped(*args, **kwargs):
        args = tuple(kwargs[name] for name in names) if kwargs else args
        shape_env, keepalive, syncbacks = {}, [], []
        call(*[conv(arg, shape_env, keepalive, syncbacks) for conv, arg in zip(converters, args, strict=True)])
        for sync in syncbacks:
            sync()

    cast(Any, wrapped)._raw = call
    return wrapped


def jit(proc=None, *, raw: bool = False, optimize: Callable[[Procedure], Procedure] | None = None):
    # call directly: `jit(proc)(...)`
    # call as a decorator: `@jit(optimize=fn)`
    if proc is None:
        return lambda fn: jit(fn, raw=raw, optimize=optimize)
    if callable(proc) and not isinstance(proc, Procedure):
        from exo.frontend.pyparser import DummyScope, Parser, get_ast_from_python

        body, src_info = get_ast_from_python(proc)
        parser = Parser(body, src_info, parent_scope=DummyScope(proc.__globals__, {}), as_func=True)
        proc = Procedure(parser.result())
    if optimize:
        proc = optimize(proc)
    return _jit_compile(proc, raw=raw)


def _dedup_proc_names(user_module: object) -> list[Procedure]:
    exported = getattr(user_module, "__all__", None)
    symbols = user_module.__dict__.items() if exported is None else ((name, getattr(user_module, name)) for name in exported)
    return list({proc.name(): proc for name, proc in symbols if not name.startswith("_") and isinstance(proc, Procedure) and not proc.is_instr()}.values())


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--c", "fmt", flag_value="c", help="Output C source")
@click.option("--mlir", "fmt", flag_value="mlir", help="Output MLIR")
@click.option("--asm", "fmt", flag_value="asm", help="Output assembly")
def cli(source: Path, fmt: Literal["c", "mlir", "asm"] | None):
    assert fmt, "choose --c, --mlir, or --asm"
    procs = _dedup_proc_names(load_user_code(source))

    match fmt:
        case "c":
            tmpdir = Path(tempfile.mkdtemp())
            exo_compile_procs(procs, tmpdir, "o.c", "o.h")
            text = (tmpdir / "o.c").read_text()
        case "mlir":
            text = str(to_mlir(procs))
        case "asm":
            text = to_asm(to_mlir(procs))

    click.echo(text)
