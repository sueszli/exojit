from __future__ import annotations

import exo.frontend.boundscheck as _boundscheck
import exo.frontend.pyparser as _pyparser
from exo.core.LoopIR import UAST, LoopIR
from exo.core.prelude import Sym

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
