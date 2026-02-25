from __future__ import annotations

from typing import TypeAlias

from exo.API import Sym
from exo.core.LoopIR import LoopIR, T

from xdsl.builder import Builder
from xdsl.dialects.arith import AddfOp, AddiOp, AndIOp, CmpfOp, CmpiOp, ConstantOp, DivfOp, DivSIOp, FastMathFlagsAttr, MulfOp, MuliOp, NegfOp, OrIOp, RemSIOp, SubfOp, SubiOp
from xdsl.dialects.builtin import I8, I16, I32, BoolAttr, Float16Type, Float32Type, Float64Type, FloatAttr, FunctionType, IndexType, IntAttr, IntegerAttr, MemRefType, ModuleOp, NoneAttr, StringAttr, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.memref import CastOp as MemrefCastOp
from xdsl.dialects.scf import ForOp, IfOp, YieldOp
from xdsl.dialects.test import TestOp
from xdsl.ir import Attribute, Block, BlockArgument, OpResult, Region, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict
from xdsl_exo.dialects.exo import AllocOp, AssignOp, ExternOp, FreeOp, InstrOp, IntervalOp, ReadOp, ReduceOp, WindowOp
from xdsl_exo.dialects.extra import CastsOp

MemRefTypeI8: TypeAlias = MemRefType[I8]
MemRefTypeI16: TypeAlias = MemRefType[I16]
MemRefTypeI32: TypeAlias = MemRefType[I32]

MemRefTypeF16: TypeAlias = MemRefType[Float16Type]
MemRefTypeF32: TypeAlias = MemRefType[Float32Type]
MemRefTypeF64: TypeAlias = MemRefType[Float64Type]


INTEGER_CMP_TABLE = {
    "==": "eq",
    "!=": "ne",
    "<": "slt",
    "<=": "sle",
    ">": "sgt",
    ">=": "sge",
}

FLOAT_CMP_TABLE = {
    "==": "oeq",
    "!=": "one",
    "<": "olt",
    "<=": "ole",
    ">": "ogt",
    ">=": "oge",
}


class IRGeneratorError(Exception):
    pass


class IRGenerator:
    module: ModuleOp
    builder: Builder

    symbol_table: ScopedDict[str, SSAValue] | None = None
    type_table: ScopedDict[str, Attribute] | None = None

    seen_procs: set[str] = set()
    seen_externs: set[str] = set()

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))

    def with_empty_scope(self):
        """
        Return this IRGenerator with an empty symbol table.
        """
        self.symbol_table = ScopedDict()
        self.type_table = ScopedDict()
        return self

    def declare_arg(self, sym: Sym, arg: BlockArgument) -> BlockArgument:
        """
        Declare a symbol in the symbol table.
        """
        assert self.symbol_table is not None
        self.declare_value(sym, arg)
        return arg

    def declare_value(self, sym: Sym, value: SSAValue) -> SSAValue:
        """
        Declare a value in the symbol table.
        """
        assert self.symbol_table is not None
        self.symbol_table[sym.__repr__()] = value
        return value

    def _with_test_op(self, sym: Sym, type):
        assert self.symbol_table is not None
        op = TestOp(result_types=[self.get_type(type)])
        self.builder.insert(op)
        self.symbol_table[sym.__repr__()] = op.res[0]
        if self.type_table is not None:
            self.type_table[sym.__repr__()] = type
        return self

    def get_sym(self, sym: Sym) -> SSAValue:
        """Get the SSAValue for a symbol."""
        assert self.symbol_table is not None

        if sym.__repr__() not in self.symbol_table:
            raise IRGeneratorError(f"Unknown symbol {sym.__repr__()}")

        return self.symbol_table[sym.__repr__()]

    def declare_sym_exo_type(self, sym: Sym, type):
        """
        Declare a type for a symbol in the type table.
        """
        assert self.type_table is not None
        self.type_table[sym.__repr__()] = type
        return type

    def get_sym_exo_type(self, sym: Sym):
        """Get the type for a symbol."""
        assert self.type_table is not None

        if sym.__repr__() not in self.type_table:
            raise IRGeneratorError(f"Unknown symbol {sym.__repr__()}")

        return self.type_table[sym.__repr__()]

    def cast_to_index(self, value: SSAValue) -> SSAValue:
        # must not cast if already an index
        if isinstance(value.type, IndexType):
            return value
        cast = CastsOp(value, IndexType())
        self.builder.insert(cast)
        return cast.result

    def cast_to(self, value: SSAValue, type: Attribute) -> SSAValue:
        # no need to cast if types match
        if value.type == type:
            return value

        if isinstance(type, IndexType) ^ isinstance(value.type, IndexType):
            cast = CastsOp(value, type)
            result = cast.result

        elif isinstance(type, MemRefType) and isinstance(value.type, MemRefType):
            # check inner types are equal
            if type.element_type != value.type.element_type:
                raise IRGeneratorError(f"Cannot cast from {value.type} to {type} as inner types do not match")

            cast = MemrefCastOp.get(value, type)
            result = cast.results[0]
        else:
            raise IRGeneratorError(f"Unknown cast from {value.type} to {type}")

        self.builder.insert(cast)
        return result

    def generate(self, procs) -> ModuleOp:
        """
        Generate the MLIR module from the given procedures and verify it.
        """
        for proc in procs:
            self.generate_procedure(proc)

        # verify module
        # TODO: none of the operations actually implement verify_()
        try:
            self.module.verify()
        except Exception as e:
            print("module verification failed: ", e)
            raise

        return self.module

    def generate_procedure(self, procedure):
        """Generate a procedure."""

        if procedure.name in self.seen_procs:
            return

        self.seen_procs.add(procedure.name)

        input_types = [self.get_type(arg.type) for arg in procedure.args]
        input_types = [
            (
                MemRefType(
                    ty.element_type,
                    ty.shape,
                    ty.layout,
                    StringAttr(arg.mem.name()),
                )
                if isinstance(ty, MemRefType)
                else ty
            )
            for (ty, arg) in zip(input_types, procedure.args)
        ]

        func_type = FunctionType.from_lists(input_types, [])

        # instantiate builder at module level
        parent_builder = self.builder
        module_builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))

        # generate private funcs for instruction procedures
        if procedure.instr is not None:
            return

        parent_symbol_table = self.symbol_table
        parent_type_table = self.type_table
        self.symbol_table = ScopedDict[str, SSAValue]()
        self.type_table = ScopedDict[str, Attribute]()

        # initialise function block
        block = Block(arg_types=input_types)
        self.builder = Builder(insertion_point=InsertPoint.at_end(block))

        # add arguments to symbol table
        for proc_arg, block_arg in zip(procedure.args, block.args):
            self.declare_arg(proc_arg.name, block_arg)
            self.declare_sym_exo_type(proc_arg.name, proc_arg.type)

        # generate function body
        self.generate_stmt_list(procedure.body)
        self.builder.insert(ReturnOp())

        # cleanup
        self.symbol_table = parent_symbol_table
        self.type_table = parent_type_table
        self.builder = parent_builder

        # insert procedure into module
        module_builder.insert(FuncOp(procedure.name, func_type, Region(block)))

    def generate_stmt_list(self, stmts):
        """Generate a list of statements."""
        for stmt in stmts:
            self.generate_stmt(stmt)

    def generate_stmt(self, stmt):
        if isinstance(stmt, LoopIR.Assign):
            self.generate_assign_stmt(stmt)
        elif isinstance(stmt, LoopIR.Reduce):
            self.generate_reduce_stmt(stmt)
        elif isinstance(stmt, LoopIR.WriteConfig):
            self.generate_write_config_stmt(stmt)
        elif isinstance(stmt, LoopIR.Pass):
            # do nothing!!
            pass
        elif isinstance(stmt, LoopIR.If):
            self.generate_if_stmt(stmt)
        elif isinstance(stmt, LoopIR.For):
            self.generate_for_stmt(stmt)
        elif isinstance(stmt, LoopIR.Alloc):
            self.generate_alloc_stmt(stmt)
        elif isinstance(stmt, LoopIR.Free):
            self.generate_free_stmt(stmt)
        elif isinstance(stmt, LoopIR.Call):
            self.generate_call_stmt(stmt)
        elif isinstance(stmt, LoopIR.Window):
            raise IRGeneratorError("Window statements are not supported")
        else:
            raise IRGeneratorError(f"Unknown statement {stmt}")

    def generate_assign_stmt(self, assign):
        idx = self.generate_expr_list(assign.idx)
        value = self.generate_expr(assign.rhs)
        memref = self.get_sym(assign.name)

        exo_type = self.get_sym_exo_type(assign.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self.get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(AssignOp(value, memref, idx, sizes))

    def generate_reduce_stmt(self, reduce):
        memref = self.get_sym(reduce.name)
        idx = self.generate_expr_list(reduce.idx)
        value = self.generate_expr(reduce.rhs)

        exo_type = self.get_sym_exo_type(reduce.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self.get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(ReduceOp(value, memref, idx, sizes))

    def generate_write_config_stmt(self, write_config):
        # rhs = self.generate_expr(write_config.rhs)
        # self.builder.insert(WriteConfigOp(write_config.name, write_config.field, rhs))
        raise NotImplementedError

    def generate_if_stmt(self, if_stmt):
        cond = self.generate_expr(if_stmt.cond)

        parent_builder = self.builder

        # construct true_block
        true_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(true_block))
        self.generate_stmt_list(if_stmt.body)
        self.builder.insert(YieldOp())

        # construct false_block
        false_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(false_block))
        self.generate_stmt_list(if_stmt.orelse)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.builder = parent_builder
        self.builder.insert(IfOp(cond, [], Region(true_block), Region(false_block)))

    def generate_for_stmt(self, for_stmt):
        lo = self.generate_expr(for_stmt.lo)
        hi = self.generate_expr(for_stmt.hi)
        step = ConstantOp(IntegerAttr(1, i64))
        self.builder.insert(step)

        parent_builder = self.builder
        parent_scope = self.symbol_table

        # construct loop block
        loop_block = Block(
            # TODO: this should be inferred from lo and hi
            arg_types=[i64],
        )
        self.builder = Builder(insertion_point=InsertPoint.at_end(loop_block))
        self.symbol_table = ScopedDict(parent_scope)

        # add loop variable to symbol table
        self.declare_arg(for_stmt.iter, loop_block.args[0])
        self.declare_sym_exo_type(for_stmt.iter, T.Index)

        # generate loop body
        self.generate_stmt_list(for_stmt.body)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.builder.insert(ForOp(lo, hi, step.result, [], Region(loop_block)))

    def generate_alloc_stmt(self, alloc):
        type = self.get_type(alloc.type, StringAttr(alloc.mem.name()))
        self.builder.insert(op := AllocOp(alloc.mem.name(), type))
        self.declare_value(alloc.name, op.results[0])
        self.declare_sym_exo_type(alloc.name, alloc.type)
        return op.result

    def generate_free_stmt(self, free):
        self.builder.insert(FreeOp(self.get_sym(free.name), free.mem.name()))

    def generate_call_stmt(self, call):
        # build arguments
        args = [self.generate_expr(arg) for arg in call.args]

        if call.f.instr is not None:
            self.builder.insert(InstrOp(call.f.name, args))
            return

        self.generate_procedure(call.f)

        # ensure arg lengths match
        if len(call.args) != len(call.f.args):
            raise IRGeneratorError(f"Call to '{call.f.name}' has {len(call.args)} arguments, expected {len(call.f.args)}")

        self.builder.insert(CallOp(call.f.name, args, []))

    # def generate_window_stmt(self, window):
    #     rhs = self.generate_expr(window.rhs)
    #     self.builder.insert(WindowStmtOp(self.symbol(window.name), rhs))

    def generate_expr_list(self, exprs) -> list[OpResult | SSAValue]:
        return [self.generate_expr(expr) for expr in exprs]

    def generate_expr(self, expr) -> OpResult | SSAValue:
        if isinstance(expr, LoopIR.Read):
            return self.generate_read_expr(expr)
        elif isinstance(expr, LoopIR.Const):
            return self.generate_const_expr(expr)
        elif isinstance(expr, LoopIR.USub):
            return self.generate_usub_expr(expr)
        elif isinstance(expr, LoopIR.BinOp):
            return self.generate_binop_expr(expr)
        elif isinstance(expr, LoopIR.WindowExpr):
            return self.generate_window_expr(expr)
        elif isinstance(expr, LoopIR.Extern):
            return self.generate_extern_expr(expr)
        else:
            raise IRGeneratorError(f"Unknown expression type '{type(expr)}' for expression '{expr}'")

    def generate_read_expr(self, read):
        idx = self.generate_expr_list(read.idx)

        operand = self.get_sym(read.name)

        exo_type = self.get_sym_exo_type(read.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self.get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(op := ReadOp(operand, idx, sizes, result_type=self.get_type(read.type)))

        return op.result

    def generate_const_expr(self, const):
        type = self.get_type(const.type)

        # construct attribute depending on type
        if type in [f16, f32, f64]:
            attr = FloatAttr(const.val, type)
        elif type in [i8, i16, i32, i64]:
            attr = IntegerAttr(IntAttr(const.val), type)
        elif type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            raise IRGeneratorError(f"Unknown type {type} passed to Const")

        const = ConstantOp(attr, self.get_type(const.type))
        self.builder.insert(const)
        return const.result

    def generate_usub_expr(self, usub):
        """
        Generate a unary negation expression.
        """

        expr = self.generate_expr(usub.arg)
        # float case
        if self.get_type(usub.type) in [f16, f32, f64]:
            usub = NegfOp(expr)
        # integer case
        elif self.get_type(usub.type) in [i8, i16, i32, i64]:
            zero = ConstantOp(IntegerAttr(0, self.get_type(usub.type)))
            usub = SubiOp(zero.result, expr, result_type=self.get_type(usub.type))
            self.builder.insert(zero)
        else:
            raise IRGeneratorError(f"Bad type {type} passed to USub")

        self.builder.insert(usub)
        return usub.result

    def generate_binop_expr(self, binop):
        """
        Generate a binary operation expression.
        """

        type = self.get_type(binop.type)

        if type in [f16, f32, f64]:
            return self.generate_binop_expr_float(binop)
        elif type in [i8, i16, i32, i64]:
            return self.generate_binop_expr_int(binop)
        elif type == i1:
            return self.generate_binop_expr_cmp(binop)
        else:
            raise IRGeneratorError(f"Unknown type '{type.name}'")

    def generate_binop_expr_float(self, binop):
        """
        Generate a floating point binary operation expression.
        """

        type = self.get_type(binop.type)
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)

        if binop.op == "+":
            binop = AddfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        elif binop.op == "-":
            binop = SubfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        elif binop.op == "*":
            binop = MulfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        elif binop.op == "/":
            binop = DivfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        else:
            raise IRGeneratorError(f"Unknown binop {binop.op}")

        self.builder.insert(binop)
        return binop.result

    def generate_binop_expr_int(self, binop):
        """
        Generate an integer binary operation expression.
        """

        type = self.get_type(binop.type)
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)

        if binop.op == "+":
            binop = AddiOp(lhs, rhs, result_type=type)
        elif binop.op == "-":
            binop = SubiOp(lhs, rhs, result_type=type)
        elif binop.op == "*":
            binop = MuliOp(lhs, rhs, result_type=type)
        elif binop.op == "/":
            binop = DivSIOp(lhs, rhs, result_type=type)
        elif binop.op == "%":
            binop = RemSIOp(lhs, rhs, result_type=type)
        else:
            raise IRGeneratorError(f"Unknown binop {binop.op}")

        self.builder.insert(binop)
        return binop.result

    def generate_binop_expr_cmp(self, binop):
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)

        assert lhs.type == rhs.type, f"Cannot compare {lhs.type} and {rhs.type} with operator '{binop.op}'"

        # boolean operations
        if lhs.type == i1:
            if binop.op == "and":
                binop = AndIOp(lhs, rhs)
            elif binop.op == "or":
                binop = OrIOp(lhs, rhs)
            else:
                raise IRGeneratorError(f"Unknown boolean operator '{binop.op}'")
        # cmpi
        elif lhs.type in [i8, i16, i32, i64]:
            op = INTEGER_CMP_TABLE[binop.op]
            if op is None:
                raise IRGeneratorError(f"Unknown integer comparison operator '{binop.op}'")

            binop = CmpiOp(lhs, rhs, op)
        # cmpf
        else:
            op = FLOAT_CMP_TABLE[binop.op]
            if op is None:
                raise IRGeneratorError(f"Unknown float comparison operator '{binop.op}'")

            binop = CmpfOp(lhs, rhs, op)

        self.builder.insert(binop)
        return binop.result

    def generate_window_expr(self, window):
        # compute indices and result type
        idx = [self.generate_w_access(w_access) for w_access in window.idx]

        input = self.get_sym(window.name)
        dest_type = self.get_type(window.type.as_tensor, input.type.memory_space)

        input_sizes = self.get_dynamic_shape(self.get_sym_exo_type(window.name))
        output_sizes = self.get_dynamic_shape(window.type.as_tensor)

        self.builder.insert(op := WindowOp(self.get_sym(window.name), idx, input_sizes, output_sizes, dest_type))

        return op.result

    def generate_w_access(self, w_access):
        if isinstance(w_access, LoopIR.Point):
            return self.generate_expr(w_access.pt)

        assert isinstance(w_access, LoopIR.Interval), f"Unknown window access type '{type(w_access)}' for '{w_access}'"

        lo = self.generate_expr(w_access.lo)
        hi = self.generate_expr(w_access.hi)

        self.builder.insert(op := IntervalOp(lo, hi))

        return op.result

    def generate_stride_expr(self, stride):
        raise NotImplementedError("stride expressions are not yet supported")

    def generate_extern_expr(self, extern):
        # query exo for the type of the result
        output_type = self.get_type(extern.f.typecheck(extern.args))
        args = self.generate_expr_list(extern.args)
        self.builder.insert(op := ExternOp(extern.f.name(), args, output_type))
        return op.result

    def generate_read_config_expr(self, read_config):
        raise NotImplementedError()

    def get_type(self, t, mem_space=StringAttr("DRAM")) -> Attribute:
        """
        Get the type of a LoopIR type as an MLIR type.
        """

        # mlir
        if isinstance(t, SSAValue):
            return t.type
        # exo
        if isinstance(t, T.F16):
            return f16
        elif isinstance(t, T.F32) or isinstance(t, T.Num):
            return f32
        elif isinstance(t, T.F64):
            return f64
        elif isinstance(t, T.INT8) or isinstance(t, T.UINT8):
            return i8
        elif isinstance(t, T.UINT16):
            return i16
        elif isinstance(t, T.INT32):
            return i32
        elif isinstance(t, T.Index) or isinstance(t, T.Size) or isinstance(t, T.Int):
            return i64
        elif isinstance(t, T.Bool):
            return i1
        elif isinstance(t, T.Tensor):
            inner = self.get_type(t.type)

            if inner not in [f16, f32, f64, i8, i16, i32]:
                raise IRGeneratorError(f"Unknown tensor inner type '{inner}'")

            # compute shape and strides
            shape = self.get_static_shape(t)

            if inner == f16:
                return MemRefTypeF16(f16, shape, NoneAttr(), mem_space)
            elif inner == f32:
                return MemRefTypeF32(f32, shape, NoneAttr(), mem_space)
            elif inner == f64:
                return MemRefTypeF64(f64, shape, NoneAttr(), mem_space)
            elif inner == i8:
                return MemRefTypeI8(i8, shape, NoneAttr(), mem_space)
            elif inner == i16:
                return MemRefTypeI16(i16, shape, NoneAttr(), mem_space)
            elif inner == i32:
                return MemRefTypeI32(i32, shape, NoneAttr(), mem_space)
            else:
                raise IRGeneratorError("Entered unreachable code")

        else:
            raise IRGeneratorError(f"Unknown type '{t}'")

    def get_shape(self, type) -> tuple[list[IntegerAttr], list[SSAValue]]:
        """
        Get the shape of a tensor type as a list of integer attributes.
        """
        assert isinstance(type, T.Tensor)

        dynamic_shapes = []

        def attr_from_expr(expr):
            if isinstance(expr, LoopIR.Const):
                return IntAttr(expr.val)
            elif isinstance(expr, LoopIR.Read):
                if self.symbol_table is not None:
                    dynamic_shapes.append(self.get_sym(expr.name))
                return IntAttr(-1)
            elif isinstance(expr, LoopIR.BinOp):
                if self.symbol_table is not None:
                    dynamic_shapes.append(self.generate_binop_expr(expr))
                return IntAttr(-1)
            else:
                raise IRGeneratorError(f"Invalid shape argument {expr}")

        return ([attr_from_expr(expr) for expr in type.shape()], dynamic_shapes)

    def get_static_shape(self, type) -> list[int]:
        """
        Get the shape of a tensor type as a list of integer attributes.
        """
        assert isinstance(type, T.Tensor)

        def attr_from_expr(expr):
            if isinstance(expr, LoopIR.Const):
                return expr.val
            elif isinstance(expr, LoopIR.Read):
                return -1
            elif isinstance(expr, LoopIR.BinOp):
                return -1
            else:
                raise IRGeneratorError(f"Invalid shape argument {expr}")

        return [attr_from_expr(expr) for expr in type.shape()]

    def get_dynamic_shape(self, type) -> list[SSAValue[Attribute] | int]:
        """
        Get the shape of a tensor type as a list of integer attributes.
        """
        assert isinstance(type, T.Tensor)

        def attr_from_expr(expr):
            if isinstance(expr, LoopIR.Const):
                return expr.val
            elif isinstance(expr, LoopIR.Read):
                return self.get_sym(expr.name)
            elif isinstance(expr, LoopIR.BinOp):
                return self.generate_binop_expr(expr)
            else:
                raise IRGeneratorError(f"Invalid shape argument {expr}")

        return [attr_from_expr(expr) for expr in type.shape()]
