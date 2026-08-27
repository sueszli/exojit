from typing import Self

from exo.core.LoopIR import LoopIR, Operator
from exo.core.configs import Config
from exo.core.prelude import Sym, SrcInfo

class _ADT:
    def update(self, **kwargs) -> Self: ...

class E:
    class expr(_ADT):
        type: LoopIR.type
        srcinfo: SrcInfo
        def negate(self) -> E.expr: ...
        def subst(self, env: dict[Sym, E.expr]) -> E.expr: ...
        def config_subst(self, env: dict[tuple[Config, str], E.expr]) -> E.expr: ...

    class Var(expr):
        name: Sym
        def __init__(
            self, name: Sym, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class Not(expr):
        arg: E.expr
        def __init__(
            self, arg: E.expr, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class Const(expr):
        val: object
        def __init__(
            self, val: object, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class BinOp(expr):
        op: Operator
        lhs: E.expr
        rhs: E.expr
        def __init__(
            self,
            op: Operator | str,
            lhs: E.expr,
            rhs: E.expr,
            type: LoopIR.type,
            srcinfo: SrcInfo,
        ) -> None: ...

    class Stride(expr):
        name: Sym
        dim: int
        def __init__(
            self, name: Sym, dim: int, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class Select(expr):
        cond: E.expr
        tcase: E.expr
        fcase: E.expr
        def __init__(
            self,
            cond: E.expr,
            tcase: E.expr,
            fcase: E.expr,
            type: LoopIR.type,
            srcinfo: SrcInfo,
        ) -> None: ...

    class ConfigField(expr):
        config: Config
        field: str
        def __init__(
            self, config: Config, field: str, type: LoopIR.type, srcinfo: SrcInfo
        ) -> None: ...

    class effset(_ADT):
        buffer: Sym
        loc: list[E.expr]
        names: list[Sym]
        pred: E.expr | None
        srcinfo: SrcInfo
        def __init__(
            self,
            buffer: Sym,
            loc: list[E.expr],
            names: list[Sym],
            pred: E.expr | None,
            srcinfo: SrcInfo,
        ) -> None: ...
        def subst(self, env: dict[Sym, E.expr]) -> E.effset: ...
        def config_subst(self, env: dict[tuple[Config, str], E.expr]) -> E.effset: ...

    class config_eff(_ADT):
        config: Config
        field: str
        value: E.expr | None
        pred: E.expr | None
        srcinfo: SrcInfo
        def __init__(
            self,
            config: Config,
            field: str,
            value: E.expr | None,
            pred: E.expr | None,
            srcinfo: SrcInfo,
        ) -> None: ...

    class effect(_ADT):
        reads: list[E.effset]
        writes: list[E.effset]
        reduces: list[E.effset]
        config_reads: list[E.config_eff]
        config_writes: list[E.config_eff]
        srcinfo: SrcInfo
        def __init__(
            self,
            reads: list[E.effset],
            writes: list[E.effset],
            reduces: list[E.effset],
            config_reads: list[E.config_eff],
            config_writes: list[E.config_eff],
            srcinfo: SrcInfo,
        ) -> None: ...
        def subst(self, env: dict[Sym, E.expr]) -> E.effect: ...
        def config_subst(self, env: dict[tuple[Config, str], E.expr]) -> E.effect: ...

def lift_expr(e): ...
def negate_expr(e): ...

def eff_subst(env, eff): # -> Any:
    ...

def eff_null(srcinfo=...): # -> Any:
    ...

def eff_read(buf, loc, srcinfo=...): # -> Any:
    ...

def eff_write(buf, loc, srcinfo=...): # -> Any:
    ...

def eff_reduce(buf, loc, srcinfo=...): # -> Any:
    ...

def eff_config_read(config, field, srcinfo=...): # -> Any:
    ...

def eff_config_write(config, field, value, srcinfo=...): # -> Any:
    ...

def eff_union(e1, e2, srcinfo=...): # -> Any:
    ...

def eff_concat(e1, e2, srcinfo=...): # -> Any:
    ...

def eff_remove_buf(buf, e): # -> Any:
    ...

def eff_filter(pred, e): # -> Any:
    ...

def eff_bind(bind_name, e, pred=..., config_pred=...): # -> Any:
    ...

def loopir_subst(e, subst): # -> Any:
    ...

class CheckBounds:
    def __init__(self, proc) -> None:
        ...
    
    def rec_proc_types(self, proc): # -> dict[Any, Any]:
        ...
    
    def rec_stmts_types(self, body): # -> dict[Any, Any]:
        ...
    
    def rec_s_types(self, stmt, type_env): # -> None:
        ...
    
    def counter_example(self): # -> LiteralString:
        ...
    
    def push(self): # -> None:
        ...
    
    def pop(self): # -> None:
        ...
    
    def err(self, node, msg): # -> None:
        ...
    
    def sym_to_smt(self, sym, typ=...):
        ...
    
    def config_to_smt(self, config, field, typ):
        ...
    
    def expr_to_smt(self, expr): # -> None:
        ...
    
    def assume_tensor_strides(self, node, name, shape): # -> None:
        ...
    
    def check_in_bounds(self, sym, shape, eff, eff_str): # -> None:
        ...
    
    def check_bounds(self, sym, shape, eff): # -> None:
        ...
    
    def check_pos_size(self, expr): # -> None:
        ...
    
    def check_non_negative(self, expr): # -> None:
        ...
    
    def check_call_shape_eqv(self, argshp, sigshp, node): # -> None:
        ...
    
    def preprocess_stmts(self, body): # -> None:
        ...
    
    def map_stmts(self, body, type_env): # -> Any:
        """
        Returns an effect for the argument `body`
        And also checks bounds/parallelism for any
        allocations/loops within `body`
        """
        ...
    
    def eff_e(self, e, type_env): # -> Any:
        ...
    
    def translate_eff(self, eff, buf_name, win_typ, type_env): # -> Any:
        ...
    


