import builtins
from typing import Any, List, Self

from exo.core.configs import Config
from exo.core.extern import Extern
from exo.core.memory import Memory
from exo.core.prelude import Sym, SrcInfo

class Identifier(str):
    _valid_re = ...
    def __new__(cls, name) -> Self: ...

class IdentifierOrHole(str):
    _valid_re = ...
    def __new__(cls, name) -> Self: ...

comparision_ops: set[str]
arithmetic_ops: set[str]
logical_ops: set[str]
front_ops: set[str]

class Operator(str):
    def __new__(cls, op) -> Self: ...

class _ADT:
    def update(self, **kwargs) -> Self: ...

class LoopIR:
    class type(_ADT):
        def is_real_scalar(self) -> bool: ...
        def is_tensor_or_window(self) -> bool: ...
        def is_win(self) -> bool: ...
        def is_dense_tensor(self) -> bool: ...
        def is_numeric(self) -> bool: ...
        def is_bool(self) -> bool: ...
        def is_indexable(self) -> bool: ...
        def is_stridable(self) -> bool: ...
        def basetype(self) -> LoopIR.type: ...

    class Num(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class F16(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class F32(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class F64(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class INT8(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class UINT8(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class UINT16(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class INT32(type):
        def __init__(self) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...
        def ctype(self) -> str: ...

    class Bool(type):
        def __init__(self) -> None: ...
        def ctype(self) -> str: ...

    class Int(type):
        def __init__(self) -> None: ...
        def ctype(self) -> str: ...

    class Index(type):
        def __init__(self) -> None: ...
        def ctype(self) -> str: ...

    class Size(type):
        def __init__(self) -> None: ...
        def ctype(self) -> str: ...

    class Stride(type):
        def __init__(self) -> None: ...
        def ctype(self) -> str: ...

    class Error(type):
        def __init__(self) -> None: ...

    class Tensor(type):
        hi: list[LoopIR.expr]
        is_window: bool
        type: LoopIR.type
        def __init__(
            self, hi: list[LoopIR.expr], is_window: bool, type: LoopIR.type
        ) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...

    class WindowType(type):
        src_type: LoopIR.type
        as_tensor: LoopIR.Tensor
        src_buf: Sym
        idx: list[LoopIR.w_access]
        def __init__(
            self,
            src_type: LoopIR.type,
            as_tensor: LoopIR.Tensor,
            src_buf: Sym,
            idx: list[LoopIR.w_access],
        ) -> None: ...
        def shape(self) -> list[LoopIR.expr]: ...

    class expr(_ADT):
        type: LoopIR.type
        srcinfo: SrcInfo

    class Read(expr):
        name: Sym
        idx: list[LoopIR.expr]
        def __init__(
            self,
            name: Sym,
            idx: list[LoopIR.expr],
            type: LoopIR.type,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Const(expr):
        val: object
        def __init__(
            self, val: object, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class USub(expr):
        arg: LoopIR.expr
        def __init__(
            self, arg: LoopIR.expr, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class BinOp(expr):
        op: Operator
        lhs: LoopIR.expr
        rhs: LoopIR.expr
        def __init__(
            self,
            op: Operator | str,
            lhs: LoopIR.expr,
            rhs: LoopIR.expr,
            type: LoopIR.type,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Extern(expr):
        f: Extern
        args: list[LoopIR.expr]
        def __init__(
            self,
            f: Extern,
            args: list[LoopIR.expr],
            type: LoopIR.type,
            srcinfo: SrcInfo,
        ) -> None: ...

    class WindowExpr(expr):
        name: Sym
        idx: list[LoopIR.w_access]
        def __init__(
            self,
            name: Sym,
            idx: list[LoopIR.w_access],
            type: LoopIR.type,
            srcinfo: SrcInfo,
        ) -> None: ...

    class StrideExpr(expr):
        name: Sym
        dim: int
        def __init__(
            self, name: Sym, dim: int, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class ReadConfig(expr):
        config: Config
        field: str
        def __init__(
            self, config: Config, field: str, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class w_access(_ADT):
        srcinfo: SrcInfo

    class Interval(w_access):
        lo: LoopIR.expr
        hi: LoopIR.expr
        def __init__(
            self, lo: LoopIR.expr, hi: LoopIR.expr, srcinfo: SrcInfo
        ) -> None: ...

    class Point(w_access):
        pt: LoopIR.expr
        def __init__(self, pt: LoopIR.expr, srcinfo: SrcInfo) -> None: ...

    class loop_mode(_ADT): ...

    class Seq(loop_mode):
        def __init__(self) -> None: ...

    class Par(loop_mode):
        def __init__(self) -> None: ...

    class instr(_ADT):
        c_instr: str
        c_global: str
        def __init__(self, c_instr: str, c_global: str) -> None: ...

    class fnarg(_ADT):
        name: Sym
        type: LoopIR.type
        mem: builtins.type[Memory] | None
        srcinfo: SrcInfo
        def __init__(
            self,
            name: Sym,
            type: LoopIR.type,
            mem: builtins.type[Memory] | None,
            srcinfo: SrcInfo,
        ) -> None: ...

    class stmt(_ADT):
        srcinfo: SrcInfo

    class Assign(stmt):
        name: Sym
        type: LoopIR.type
        idx: list[LoopIR.expr]
        rhs: LoopIR.expr
        def __init__(
            self,
            name: Sym,
            type: LoopIR.type,
            idx: list[LoopIR.expr],
            rhs: LoopIR.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Reduce(stmt):
        name: Sym
        type: LoopIR.type
        idx: list[LoopIR.expr]
        rhs: LoopIR.expr
        def __init__(
            self,
            name: Sym,
            type: LoopIR.type,
            idx: list[LoopIR.expr],
            rhs: LoopIR.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class WriteConfig(stmt):
        config: Config
        field: str
        rhs: LoopIR.expr
        def __init__(
            self, config: Config, field: str, rhs: LoopIR.expr, srcinfo: SrcInfo
        ) -> None: ...

    class Pass(stmt):
        def __init__(self, srcinfo: SrcInfo) -> None: ...

    class If(stmt):
        cond: LoopIR.expr
        body: list[LoopIR.stmt]
        orelse: list[LoopIR.stmt]
        def __init__(
            self,
            cond: LoopIR.expr,
            body: list[LoopIR.stmt],
            orelse: list[LoopIR.stmt],
            srcinfo: SrcInfo,
        ) -> None: ...

    class For(stmt):
        iter: Sym
        lo: LoopIR.expr
        hi: LoopIR.expr
        body: list[LoopIR.stmt]
        loop_mode: LoopIR.loop_mode
        def __init__(
            self,
            iter: Sym,
            lo: LoopIR.expr,
            hi: LoopIR.expr,
            body: list[LoopIR.stmt],
            loop_mode: LoopIR.loop_mode,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Alloc(stmt):
        name: Sym
        type: LoopIR.type
        mem: builtins.type[Memory]
        def __init__(
            self,
            name: Sym,
            type: LoopIR.type,
            mem: builtins.type[Memory],
            srcinfo: SrcInfo,
        ) -> None: ...

    class Free(stmt):
        name: Sym
        type: LoopIR.type
        mem: builtins.type[Memory]
        def __init__(
            self,
            name: Sym,
            type: LoopIR.type,
            mem: builtins.type[Memory],
            srcinfo: SrcInfo,
        ) -> None: ...

    class Call(stmt):
        f: LoopIR.proc
        args: list[LoopIR.expr]
        def __init__(
            self, f: LoopIR.proc, args: list[LoopIR.expr], srcinfo: SrcInfo
        ) -> None: ...

    class WindowStmt(stmt):
        name: Sym
        rhs: LoopIR.expr
        def __init__(
            self, name: Sym, rhs: LoopIR.expr, srcinfo: SrcInfo
        ) -> None: ...

    class proc(_ADT):
        name: Identifier
        args: list[LoopIR.fnarg]
        preds: list[LoopIR.expr]
        body: list[LoopIR.stmt]
        instr: LoopIR.instr | None
        srcinfo: SrcInfo
        def __init__(
            self,
            name: Identifier | str,
            args: list[LoopIR.fnarg],
            preds: list[LoopIR.expr],
            body: list[LoopIR.stmt],
            instr: LoopIR.instr | None,
            srcinfo: SrcInfo,
        ) -> None: ...
        def __hash__(self) -> int: ...

class UAST:
    class type(_ADT):
        def shape(self) -> list[UAST.expr]: ...
        def basetype(self) -> UAST.type: ...

    class Num(type):
        def __init__(self) -> None: ...

    class F16(type):
        def __init__(self) -> None: ...

    class F32(type):
        def __init__(self) -> None: ...

    class F64(type):
        def __init__(self) -> None: ...

    class INT8(type):
        def __init__(self) -> None: ...

    class UINT8(type):
        def __init__(self) -> None: ...

    class UINT16(type):
        def __init__(self) -> None: ...

    class INT32(type):
        def __init__(self) -> None: ...

    class Bool(type):
        def __init__(self) -> None: ...

    class Int(type):
        def __init__(self) -> None: ...

    class Size(type):
        def __init__(self) -> None: ...

    class Index(type):
        def __init__(self) -> None: ...

    class Stride(type):
        def __init__(self) -> None: ...

    class Tensor(type):
        hi: list[UAST.expr]
        is_window: bool
        type: UAST.type
        def __init__(
            self, hi: list[UAST.expr], is_window: bool, type: UAST.type
        ) -> None: ...

    class expr(_ADT):
        srcinfo: SrcInfo

    class Read(expr):
        name: Sym
        idx: list[UAST.expr]
        def __init__(
            self, name: Sym, idx: list[UAST.expr], srcinfo: SrcInfo
        ) -> None: ...

    class Const(expr):
        val: object
        def __init__(self, val: object, srcinfo: SrcInfo) -> None: ...

    class USub(expr):
        arg: UAST.expr
        def __init__(self, arg: UAST.expr, srcinfo: SrcInfo) -> None: ...

    class BinOp(expr):
        op: Operator
        lhs: UAST.expr
        rhs: UAST.expr
        def __init__(
            self,
            op: Operator | str,
            lhs: UAST.expr,
            rhs: UAST.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Extern(expr):
        f: Extern
        args: list[UAST.expr]
        def __init__(
            self, f: Extern, args: list[UAST.expr], srcinfo: SrcInfo
        ) -> None: ...

    class WindowExpr(expr):
        name: Sym
        idx: list[UAST.w_access]
        def __init__(
            self, name: Sym, idx: list[UAST.w_access], srcinfo: SrcInfo
        ) -> None: ...

    class StrideExpr(expr):
        name: Sym
        dim: int
        def __init__(self, name: Sym, dim: int, srcinfo: SrcInfo) -> None: ...

    class ParRange(expr):
        lo: UAST.expr
        hi: UAST.expr
        def __init__(
            self, lo: UAST.expr, hi: UAST.expr, srcinfo: SrcInfo
        ) -> None: ...

    class SeqRange(expr):
        lo: UAST.expr
        hi: UAST.expr
        def __init__(
            self, lo: UAST.expr, hi: UAST.expr, srcinfo: SrcInfo
        ) -> None: ...

    class ReadConfig(expr):
        config: Config
        field: str
        def __init__(
            self, config: Config, field: str, srcinfo: SrcInfo
        ) -> None: ...

    class w_access(_ADT):
        srcinfo: SrcInfo

    class Interval(w_access):
        lo: UAST.expr | None
        hi: UAST.expr | None
        def __init__(
            self, lo: UAST.expr | None, hi: UAST.expr | None, srcinfo: SrcInfo
        ) -> None: ...

    class Point(w_access):
        pt: UAST.expr
        def __init__(self, pt: UAST.expr, srcinfo: SrcInfo) -> None: ...

    class instr(_ADT):
        c_instr: str
        c_global: str
        def __init__(self, c_instr: str, c_global: str) -> None: ...

    class fnarg(_ADT):
        name: Sym
        type: UAST.type
        mem: builtins.type[Memory] | None
        srcinfo: SrcInfo
        def __init__(
            self,
            name: Sym,
            type: UAST.type,
            mem: builtins.type[Memory] | None,
            srcinfo: SrcInfo,
        ) -> None: ...

    class stmt(_ADT):
        srcinfo: SrcInfo

    class Assign(stmt):
        name: Sym
        idx: list[UAST.expr]
        rhs: UAST.expr
        def __init__(
            self,
            name: Sym,
            idx: list[UAST.expr],
            rhs: UAST.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Reduce(stmt):
        name: Sym
        idx: list[UAST.expr]
        rhs: UAST.expr
        def __init__(
            self,
            name: Sym,
            idx: list[UAST.expr],
            rhs: UAST.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class WriteConfig(stmt):
        config: Config
        field: str
        rhs: UAST.expr
        def __init__(
            self, config: Config, field: str, rhs: UAST.expr, srcinfo: SrcInfo
        ) -> None: ...

    class FreshAssign(stmt):
        name: Sym
        rhs: UAST.expr
        def __init__(self, name: Sym, rhs: UAST.expr, srcinfo: SrcInfo) -> None: ...

    class Pass(stmt):
        def __init__(self, srcinfo: SrcInfo) -> None: ...

    class If(stmt):
        cond: UAST.expr
        body: list[UAST.stmt]
        orelse: list[UAST.stmt]
        def __init__(
            self,
            cond: UAST.expr,
            body: list[UAST.stmt],
            orelse: list[UAST.stmt],
            srcinfo: SrcInfo,
        ) -> None: ...

    class For(stmt):
        iter: Sym
        cond: UAST.expr
        body: list[UAST.stmt]
        def __init__(
            self,
            iter: Sym,
            cond: UAST.expr,
            body: list[UAST.stmt],
            srcinfo: SrcInfo,
        ) -> None: ...

    class Alloc(stmt):
        name: Sym
        type: UAST.type
        mem: builtins.type[Memory] | None
        def __init__(
            self,
            name: Sym,
            type: UAST.type,
            mem: builtins.type[Memory] | None,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Call(stmt):
        f: LoopIR.proc
        args: list[UAST.expr]
        def __init__(
            self, f: LoopIR.proc, args: list[UAST.expr], srcinfo: SrcInfo
        ) -> None: ...

    class proc(_ADT):
        name: Identifier | None
        args: list[UAST.fnarg]
        preds: list[UAST.expr]
        body: list[UAST.stmt]
        instr: UAST.instr | None
        srcinfo: SrcInfo
        def __init__(
            self,
            name: Identifier | str | None,
            args: list[UAST.fnarg],
            preds: list[UAST.expr],
            body: list[UAST.stmt],
            instr: UAST.instr | None,
            srcinfo: SrcInfo,
        ) -> None: ...

class PAST:
    class expr(_ADT):
        srcinfo: SrcInfo

    class Read(expr):
        name: IdentifierOrHole
        idx: list[PAST.expr]
        def __init__(
            self,
            name: IdentifierOrHole | str,
            idx: list[PAST.expr],
            srcinfo: SrcInfo,
        ) -> None: ...

    class StrideExpr(expr):
        name: IdentifierOrHole
        dim: int | None
        def __init__(
            self, name: IdentifierOrHole | str, dim: int | None, srcinfo: SrcInfo
        ) -> None: ...

    class E_Hole(expr):
        def __init__(self, srcinfo: SrcInfo) -> None: ...

    class Const(expr):
        val: object
        def __init__(self, val: object, srcinfo: SrcInfo) -> None: ...

    class USub(expr):
        arg: PAST.expr
        def __init__(self, arg: PAST.expr, srcinfo: SrcInfo) -> None: ...

    class BinOp(expr):
        op: Operator
        lhs: PAST.expr
        rhs: PAST.expr
        def __init__(
            self,
            op: Operator | str,
            lhs: PAST.expr,
            rhs: PAST.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Extern(expr):
        f: IdentifierOrHole
        args: list[PAST.expr]
        def __init__(
            self,
            f: IdentifierOrHole | str,
            args: list[PAST.expr],
            srcinfo: SrcInfo,
        ) -> None: ...

    class ReadConfig(expr):
        config: str
        field: str
        def __init__(self, config: str, field: str, srcinfo: SrcInfo) -> None: ...

    class stmt(_ADT):
        srcinfo: SrcInfo

    class Assign(stmt):
        name: IdentifierOrHole
        idx: list[PAST.expr]
        rhs: PAST.expr
        def __init__(
            self,
            name: IdentifierOrHole | str,
            idx: list[PAST.expr],
            rhs: PAST.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Reduce(stmt):
        name: IdentifierOrHole
        idx: list[PAST.expr]
        rhs: PAST.expr
        def __init__(
            self,
            name: IdentifierOrHole | str,
            idx: list[PAST.expr],
            rhs: PAST.expr,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Pass(stmt):
        def __init__(self, srcinfo: SrcInfo) -> None: ...

    class If(stmt):
        cond: PAST.expr
        body: list[PAST.stmt]
        orelse: list[PAST.stmt]
        def __init__(
            self,
            cond: PAST.expr,
            body: list[PAST.stmt],
            orelse: list[PAST.stmt],
            srcinfo: SrcInfo,
        ) -> None: ...

    class For(stmt):
        iter: IdentifierOrHole
        lo: PAST.expr
        hi: PAST.expr
        body: list[PAST.stmt]
        def __init__(
            self,
            iter: IdentifierOrHole | str,
            lo: PAST.expr,
            hi: PAST.expr,
            body: list[PAST.stmt],
            srcinfo: SrcInfo,
        ) -> None: ...

    class Alloc(stmt):
        name: IdentifierOrHole
        sizes: list[PAST.expr]
        def __init__(
            self,
            name: IdentifierOrHole | str,
            sizes: list[PAST.expr],
            srcinfo: SrcInfo,
        ) -> None: ...

    class Call(stmt):
        f: IdentifierOrHole
        args: list[PAST.expr]
        def __init__(
            self,
            f: IdentifierOrHole | str,
            args: list[PAST.expr],
            srcinfo: SrcInfo,
        ) -> None: ...

    class WriteConfig(stmt):
        config: IdentifierOrHole
        field: IdentifierOrHole
        def __init__(
            self,
            config: IdentifierOrHole | str,
            field: IdentifierOrHole | str,
            srcinfo: SrcInfo,
        ) -> None: ...

    class S_Hole(stmt):
        def __init__(self, srcinfo: SrcInfo) -> None: ...

class CIR:
    class expr(_ADT): ...

    class Read(expr):
        name: Sym
        is_non_neg: bool
        def __init__(self, name: Sym, is_non_neg: bool) -> None: ...

    class Stride(expr):
        name: Sym
        dim: int
        def __init__(self, name: Sym, dim: int) -> None: ...

    class Const(expr):
        val: object
        def __init__(self, val: object) -> None: ...

    class BinOp(expr):
        op: Operator
        lhs: CIR.expr
        rhs: CIR.expr
        is_non_neg: bool
        def __init__(
            self,
            op: Operator | str,
            lhs: CIR.expr,
            rhs: CIR.expr,
            is_non_neg: bool,
        ) -> None: ...

    class USub(expr):
        arg: CIR.expr
        is_non_neg: bool
        def __init__(self, arg: CIR.expr, is_non_neg: bool) -> None: ...

class T:
    Num = LoopIR.Num
    F16 = LoopIR.F16
    F32 = LoopIR.F32
    F64 = LoopIR.F64
    INT8 = LoopIR.INT8
    UINT8 = LoopIR.UINT8
    UINT16 = LoopIR.UINT16
    INT32 = LoopIR.INT32
    Bool = LoopIR.Bool
    Int = LoopIR.Int
    Index = LoopIR.Index
    Size = LoopIR.Size
    Stride = LoopIR.Stride
    Error = LoopIR.Error
    Tensor = LoopIR.Tensor
    Window = LoopIR.WindowType
    type = LoopIR.type
    R: LoopIR.Num
    f16: LoopIR.F16
    f32: LoopIR.F32
    int8: LoopIR.INT8
    uint8: LoopIR.UINT8
    uint16: LoopIR.UINT16
    i8: LoopIR.INT8
    ui8: LoopIR.UINT8
    ui16: LoopIR.UINT16
    int32: LoopIR.INT32
    i32: LoopIR.INT32
    f64: LoopIR.F64
    bool: LoopIR.Bool
    int: LoopIR.Int
    index: LoopIR.Index
    size: LoopIR.Size
    stride: LoopIR.Stride
    err: LoopIR.Error

def is_stridable(t: LoopIR.type) -> bool: ...
def chain_window_idx(
    idx0: list[LoopIR.w_access], idx1: list[LoopIR.w_access]
) -> list[LoopIR.w_access]: ...
def build_window_shape(ws: List[LoopIR.w_access]) -> list[LoopIR.expr]: ...
def create_window_type(
    in_name: Sym, in_typ: LoopIR.type, idx: list[LoopIR.w_access]
) -> LoopIR.WindowType:
    """Construct a derived window type from any tensor or window type"""
    ...

class LoopIR_Rewrite:
    def apply_proc(self, old):
        ...
    
    def apply_fnarg(self, old):
        ...
    
    def apply_stmts(self, old): # -> list[Any]:
        ...
    
    def apply_exprs(self, old): # -> list[Any]:
        ...
    
    def apply_s(self, old): # -> list[Any]:
        ...
    
    def apply_e(self, old):
        ...
    
    def apply_w_access(self, old):
        ...
    
    def apply_t(self, old):
        ...
    
    def map_proc(self, p): # -> None:
        ...
    
    def map_fnarg(self, a): # -> None:
        ...
    
    def map_stmts(self, stmts): # -> list[Any] | None:
        ...
    
    def map_exprs(self, exprs): # -> list[Any] | None:
        ...
    
    def map_s(self, s):
        ...
    
    def map_e(self, e):
        ...
    
    def map_w_access(self, w): # -> None:
        ...
    
    def map_t(self, t): # -> None:
        ...
    


class LoopIR_Do:
    def __init__(self, proc, *args, **kwargs) -> None:
        ...
    
    def do_stmts(self, stmts): # -> None:
        ...
    
    def do_s(self, s): # -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    
    def do_w_access(self, w): # -> None:
        ...
    
    def do_t(self, t): # -> None:
        ...
    


class LoopIR_Compare:
    def __init__(self) -> None:
        ...
    
    def match_stmts(self, stmts1, stmts2): # -> bool:
        ...
    
    def match_s(self, s1, s2): # -> bool:
        ...
    
    def match_e(self, e1, e2): # -> bool:
        ...
    
    def match_name(self, n1, n2):
        ...
    
    def match_w_access(self, w1, w2):
        ...
    
    def match_t(self, t1, t2): # -> bool:
        ...
    


class GetReads(LoopIR_Do):
    def __init__(self) -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    


class GetReadConfigs(LoopIR_Do):
    def __init__(self) -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    


def get_reads_of_expr(e): # -> list[Any]:
    ...

def get_reads_of_stmts(stmts): # -> list[Any]:
    ...

def get_readconfigs(stmts): # -> list[Any]:
    ...

class GetWrites(LoopIR_Do):
    def __init__(self) -> None:
        ...
    
    def do_s(self, s): # -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    


def get_writes_of_stmts(stmts): # -> list[Any]:
    ...

class GetWriteConfigs(LoopIR_Do):
    def __init__(self) -> None:
        ...
    
    def do_s(self, s): # -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    


def get_writeconfigs(stmts): # -> list[Any]:
    ...

class GetLoopIters(LoopIR_Do):
    def __init__(self) -> None:
        ...
    
    def do_s(self, s): # -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    


def get_loop_iters(stmts): # -> list[Any]:
    ...

def is_const_zero(e): # -> Literal[False]:
    ...

class FreeVars(LoopIR_Do):
    def __init__(self, node) -> None:
        ...
    
    def result(self): # -> set[Any]:
        ...
    
    def push(self): # -> None:
        ...
    
    def pop(self): # -> None:
        ...
    
    def do_s(self, s): # -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    
    def do_t(self, t): # -> None:
        ...
    


class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node) -> None:
        ...
    
    def result(self): # -> Any | list[Any]:
        ...
    
    def push(self): # -> None:
        ...
    
    def pop(self): # -> None:
        ...
    
    def map_fnarg(self, fa):
        ...
    
    def map_s(self, s): # -> list[Any]:
        ...
    
    def map_e(self, e):
        ...
    
    def map_t(self, t): # -> None:
        ...
    


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, nodes, binding) -> None:
        ...
    
    def result(self): # -> list[Any]:
        ...
    
    def map_s(self, s): # -> list[Any] | None:
        ...
    
    def map_e(self, e):
        ...
    
    def map_t(self, t): # -> None:
        ...
    


class LoopIR_Dependencies(LoopIR_Do):
    def __init__(self, buf_sym, stmts) -> None:
        ...
    
    def result(self): # -> set[Any]:
        ...
    
    def do_s(self, s): # -> None:
        ...
    
    def do_e(self, e): # -> None:
        ...
    
    def do_t(self, t): # -> None:
        ...
    


