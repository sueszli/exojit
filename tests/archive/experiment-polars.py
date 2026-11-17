# /// script
# requires-python = ">=3.14"
# dependencies = ["xdsl==0.75.0"]
# ///
from typing import Any, ClassVar

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, PythonValues, impl, register_impls
from xdsl.interpreters.func import FuncFunctions
from xdsl.ir import Block, ParametrizedAttribute, Region, TypeAttribute
from xdsl.irdl import IRDLOperation, VarConstraint, base, eq, irdl_attr_definition, irdl_op_definition, operand_def, result_def

# ===----------------------------------------------------------------------=== #
# lazy series and dataframe
# ===----------------------------------------------------------------------=== #


class Series:
    def __init__(self, values: list[Any], dtype: type) -> None:
        # validate the values and store a copy as an input node
        # int is unbounded, not i64
        assert dtype in (str, bool, int, float), f"bad dtype {dtype!r}"
        # exact types: bool is not int, int is not float
        assert isinstance(values, list) and all(v is None or type(v) is dtype for v in values), f"expected list[{dtype.__name__} | None]"
        self.dtype = dtype
        self._len = len(values)
        self._node = list(values)

    @staticmethod
    def _lazy(op: type[IRDLOperation], dtype: type, length: int | None, *args: Series) -> Series:
        # a series that computes op(*args) when observed
        # length is None when it depends on data, e.g. after a filter
        out = Series.__new__(Series)
        out.dtype = dtype
        out._len = length
        out._node = (op, *args)
        return out

    def _binary(self, op: type[IRDLOperation], other: object, dtype: type) -> Series:
        # validate now, compute when observed. null if either side is null, except & and |
        match other:
            case Series():
                assert other.dtype is self.dtype, f"dtype {self.dtype.__name__} != {other.dtype.__name__}"
                rhs = other
            case _:
                Series([other], self.dtype)  # validate even if lhs is empty
                # TODO: len() runs a filtered lhs early. fix: a polars.broadcast op sized at runtime
                rhs = Series([other] * len(self), self.dtype)
        # unknown lengths are checked at runtime
        assert None in (self._len, rhs._len) or self._len == rhs._len, f"length {self._len} != {rhs._len}"
        return Series._lazy(op, dtype, rhs._len if self._len is None else self._len, self, rhs)

    def __add__(self, other: object) -> Series:
        assert self.dtype in (int, float), f"+ on {self.dtype.__name__}"
        return self._binary(AddOp, other, self.dtype)

    def __sub__(self, other: object) -> Series:
        assert self.dtype in (int, float), f"- on {self.dtype.__name__}"
        return self._binary(SubOp, other, self.dtype)

    def __mul__(self, other: object) -> Series:
        assert self.dtype in (int, float), f"* on {self.dtype.__name__}"
        return self._binary(MulOp, other, self.dtype)

    def __truediv__(self, other: object) -> Series:
        # true division: int / int is float
        assert self.dtype in (int, float), f"/ on {self.dtype.__name__}"
        return self._binary(DivOp, other, float)

    def __lt__(self, other: object) -> Series:
        assert self.dtype in (int, float), f"< on {self.dtype.__name__}"
        return self._binary(LtOp, other, bool)

    def __le__(self, other: object) -> Series:
        assert self.dtype in (int, float), f"<= on {self.dtype.__name__}"
        return self._binary(LeOp, other, bool)

    def __gt__(self, other: object) -> Series:
        assert self.dtype in (int, float), f"> on {self.dtype.__name__}"
        return self._binary(GtOp, other, bool)

    def __ge__(self, other: object) -> Series:
        assert self.dtype in (int, float), f">= on {self.dtype.__name__}"
        return self._binary(GeOp, other, bool)

    def __eq__(self, other: object) -> Series:
        # any dtype. returns a series, not a python bool
        return self._binary(EqOp, other, bool)

    def __ne__(self, other: object) -> Series:
        # explicit, or python would negate __eq__ through __bool__
        return self._binary(NeOp, other, bool)

    def __and__(self, other: object) -> Series:
        # kleene: false wins over null
        assert self.dtype is bool, f"& on {self.dtype.__name__}"
        return self._binary(AndOp, other, bool)

    def __or__(self, other: object) -> Series:
        # kleene: true wins over null
        assert self.dtype is bool, f"| on {self.dtype.__name__}"
        return self._binary(OrOp, other, bool)

    def __xor__(self, other: object) -> Series:
        assert self.dtype is bool, f"^ on {self.dtype.__name__}"
        return self._binary(XorOp, other, bool)

    def __invert__(self) -> Series:
        assert self.dtype is bool, f"~ on {self.dtype.__name__}"
        return Series._lazy(NotOp, bool, self._len, self)

    def __getitem__(self, key: int | Series) -> Any:
        # an int returns one value and runs the graph
        # a bool series keeps the rows where it is true, without running
        match key:
            case Series():
                assert key.dtype is bool, f"mask of {key.dtype.__name__}"
                assert None in (self._len, key._len) or self._len == key._len, f"length {self._len} != {key._len}"
                return Series._lazy(FilterOp, self.dtype, None, self, key)
            case _:
                assert type(key) is int, f"bad index {key!r}"
                return self._values()[key]

    def __len__(self) -> int:
        # known without running unless a filter is involved
        return len(self._values()) if self._len is None else self._len

    def __bool__(self) -> bool:
        # forbid `if`, `and`, `or`, `not` on a series
        assert False, "ambiguous truth value, use &, | and ~"

    def __repr__(self) -> str:
        # format like the Polars library. runs the graph
        values = self._values()
        return "\n".join([f"shape: ({len(values)},)", f"Series: [{self.dtype.__name__}]", "[", *(f"\t{'null' if v is None else repr(v)}" for v in values), "]"])

    def to_list(self) -> list[Any]:
        # copy of the values. runs the graph
        return list(self._values())

    def explain(self) -> str:
        # the ir that would run, without running it
        return str(self._module()[0])

    def _module(self) -> tuple[ModuleOp, list[list[Any]]]:
        # emit the graph reaching self as `func.func @query`. returns it with the input lists for its block arguments
        block = Block()
        inputs = []
        ssa = {}

        # post-order walk with an explicit stack, so deep chains do not hit the recursion limit
        # memoized by id, so a shared node is emitted once
        stack = [self]
        while stack:
            s = stack[-1]
            if id(s) in ssa:
                stack.pop()
                continue
            match s._node:
                case list():
                    inputs.append(s._node)
                    ssa[id(s)] = block.insert_arg(SeriesType(s.dtype), len(block.args))
                case (op, *args):
                    pending = [a for a in args if id(a) not in ssa]
                    if pending:
                        stack.extend(reversed(pending))
                        continue
                    block.add_op(new := op(operands=[ssa[id(a)] for a in args], result_types=[SeriesType(s.dtype)]))
                    ssa[id(s)] = new.results[0]
            stack.pop()

        block.add_op(ReturnOp(result := ssa[id(self)]))
        module = ModuleOp([FuncOp("query", (block.arg_types, [result.type]), Region(block))])
        module.verify()
        return module, inputs

    def _values(self) -> list[Any]:
        # run the graph once, then keep the result as an input node
        if type(self._node) is tuple:
            module, inputs = self._module()
            interpreter = Interpreter(module)
            interpreter.register_implementations(FuncFunctions())
            interpreter.register_implementations(PolarsFunctions())
            (self._node,) = interpreter.call_op(module.ops.first, tuple(inputs))
            self._len = len(self._node)
        return self._node


class DataFrame:
    def __init__(self, columns: dict[str, Series]) -> None:
        # validate names, series and equal lengths. copies the dict, shares the series
        assert isinstance(columns, dict) and all(type(k) is str and isinstance(v, Series) for k, v in columns.items()), "expected dict[str, Series]"
        assert len({len(s) for s in columns.values()}) <= 1, "unequal column lengths"
        self._columns = dict(columns)

    def __getitem__(self, key: str | Series) -> Series | DataFrame:
        # a name returns its column
        # a bool series filters every column lazily, so unused columns are never computed
        match key:
            case Series():
                out = DataFrame.__new__(DataFrame)  # filtered lengths are unknown until run, so skip the length check
                out._columns = {name: s[key] for name, s in self._columns.items()}
                return out
            case _:
                assert type(key) is str, f"bad column {key!r}"
                return self._columns[key]

    def __repr__(self) -> str:
        # header, dtype row and one row per record. runs every column
        columns = [s._values() for s in self._columns.values()]
        rows = ["\t".join("null" if v is None else repr(v) for v in row) for row in zip(*columns)]
        shape = f"shape: ({len(next(iter(columns), []))}, {len(columns)})"
        return "\n".join([shape, "\t".join(self._columns), "\t".join(s.dtype.__name__ for s in self._columns.values()), *rows])


# ===----------------------------------------------------------------------=== #
# polars dialect and list interpreter
# ===----------------------------------------------------------------------=== #


@irdl_attr_definition
class SeriesType(ParametrizedAttribute, TypeAttribute):
    name = "polars.series"
    dtype_name: StringAttr

    def __init__(self, dtype: type) -> None:
        # spelled by its python name, e.g. !polars.series<"int">
        super().__init__(StringAttr(dtype.__name__))


# types are checked declaratively: T binds every use to one type
# not Pure: ops may fail at runtime
@irdl_op_definition
class AddOp(IRDLOperation):
    name = "polars.add"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)


@irdl_op_definition
class SubOp(IRDLOperation):
    name = "polars.sub"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)


@irdl_op_definition
class MulOp(IRDLOperation):
    name = "polars.mul"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)


@irdl_op_definition
class DivOp(IRDLOperation):
    name = "polars.div"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(SeriesType(float))


@irdl_op_definition
class LtOp(IRDLOperation):
    name = "polars.lt"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(SeriesType(bool))


@irdl_op_definition
class LeOp(IRDLOperation):
    name = "polars.le"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(SeriesType(bool))


@irdl_op_definition
class GtOp(IRDLOperation):
    name = "polars.gt"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(SeriesType(bool))


@irdl_op_definition
class GeOp(IRDLOperation):
    name = "polars.ge"
    T: ClassVar = VarConstraint("T", eq(SeriesType(int)) | eq(SeriesType(float)))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(SeriesType(bool))


@irdl_op_definition
class EqOp(IRDLOperation):
    name = "polars.eq"
    T: ClassVar = VarConstraint("T", base(SeriesType))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(SeriesType(bool))


@irdl_op_definition
class NeOp(IRDLOperation):
    name = "polars.ne"
    T: ClassVar = VarConstraint("T", base(SeriesType))
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(SeriesType(bool))


@irdl_op_definition
class AndOp(IRDLOperation):
    name = "polars.and"
    lhs = operand_def(SeriesType(bool))
    rhs = operand_def(SeriesType(bool))
    result = result_def(SeriesType(bool))


@irdl_op_definition
class OrOp(IRDLOperation):
    name = "polars.or"
    lhs = operand_def(SeriesType(bool))
    rhs = operand_def(SeriesType(bool))
    result = result_def(SeriesType(bool))


@irdl_op_definition
class XorOp(IRDLOperation):
    name = "polars.xor"
    lhs = operand_def(SeriesType(bool))
    rhs = operand_def(SeriesType(bool))
    result = result_def(SeriesType(bool))


@irdl_op_definition
class NotOp(IRDLOperation):
    name = "polars.not"
    input = operand_def(SeriesType(bool))
    result = result_def(SeriesType(bool))


@irdl_op_definition
class FilterOp(IRDLOperation):
    name = "polars.filter"
    T: ClassVar = VarConstraint("T", base(SeriesType))
    values = operand_def(T)
    mask = operand_def(SeriesType(bool))
    result = result_def(T)


# strict zip catches lengths that only differ at runtime, e.g. after two different filters
# TODO: perf. per-row python loop over boxed objects. use a contiguous typed buffer + null bitmap and a native kernel
@register_impls
class PolarsFunctions(InterpreterFunctions):
    @impl(AddOp)
    def run_add(self, _interpreter: Interpreter, _op: AddOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a + b for a, b in zip(*args, strict=True)],)

    @impl(SubOp)
    def run_sub(self, _interpreter: Interpreter, _op: SubOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a - b for a, b in zip(*args, strict=True)],)

    @impl(MulOp)
    def run_mul(self, _interpreter: Interpreter, _op: MulOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a * b for a, b in zip(*args, strict=True)],)

    @impl(DivOp)
    def run_div(self, _interpreter: Interpreter, _op: DivOp, args: PythonValues) -> PythonValues:
        # null check first, so None / 0 is None
        return ([None if a is None or b is None else a / b for a, b in zip(*args, strict=True)],)

    @impl(LtOp)
    def run_lt(self, _interpreter: Interpreter, _op: LtOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a < b for a, b in zip(*args, strict=True)],)

    @impl(LeOp)
    def run_le(self, _interpreter: Interpreter, _op: LeOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a <= b for a, b in zip(*args, strict=True)],)

    @impl(GtOp)
    def run_gt(self, _interpreter: Interpreter, _op: GtOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a > b for a, b in zip(*args, strict=True)],)

    @impl(GeOp)
    def run_ge(self, _interpreter: Interpreter, _op: GeOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a >= b for a, b in zip(*args, strict=True)],)

    @impl(EqOp)
    def run_eq(self, _interpreter: Interpreter, _op: EqOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a == b for a, b in zip(*args, strict=True)],)

    @impl(NeOp)
    def run_ne(self, _interpreter: Interpreter, _op: NeOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a != b for a, b in zip(*args, strict=True)],)

    @impl(AndOp)
    def run_and(self, _interpreter: Interpreter, _op: AndOp, args: PythonValues) -> PythonValues:
        return ([False if a is False or b is False else None if a is None or b is None else True for a, b in zip(*args, strict=True)],)

    @impl(OrOp)
    def run_or(self, _interpreter: Interpreter, _op: OrOp, args: PythonValues) -> PythonValues:
        return ([True if a is True or b is True else None if a is None or b is None else False for a, b in zip(*args, strict=True)],)

    @impl(XorOp)
    def run_xor(self, _interpreter: Interpreter, _op: XorOp, args: PythonValues) -> PythonValues:
        return ([None if a is None or b is None else a != b for a, b in zip(*args, strict=True)],)

    @impl(NotOp)
    def run_not(self, _interpreter: Interpreter, _op: NotOp, args: PythonValues) -> PythonValues:
        return ([None if a is None else not a for a in args[0]],)

    @impl(FilterOp)
    def run_filter(self, _interpreter: Interpreter, _op: FilterOp, args: PythonValues) -> PythonValues:
        # keeps rows whose mask is exactly True
        return ([v for v, m in zip(*args, strict=True) if m is True],)
