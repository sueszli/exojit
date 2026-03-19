from __future__ import annotations

import ast
import inspect
from collections import ChainMap

import exo.API as _exo_api
import exo.frontend.boundscheck as _boundscheck
import exo.frontend.pyparser as _pyparser
from exo.core.LoopIR import LoopIR
from exo.core.memory import MemGenError, Memory
from exo.core.prelude import Sym


# exo does `assert isinstance(frame.f_locals, dict)` but PEP 667 (Python 3.14+)
# returns FrameLocalsProxy instead. Patch both import sites to wrap in dict().
# see: https://peps.python.org/pep-0667/
def patched_get_src_locals(*, depth):
    frames = inspect.stack()
    assert len(frames) >= depth
    return ChainMap(dict(frames[depth].frame.f_locals))


ORIGINAL_LIFT_EXPR = _boundscheck.lift_expr
ORIGINAL_PARSE_STMT_BLOCK = _pyparser.Parser.parse_stmt_block


# let exo parse tensor types like `size[n]` and `index[n]`
_pyparser._prim_types["size"] = _pyparser.UAST.Size()
_pyparser._prim_types["index"] = _pyparser.UAST.Index()


# treat size/index tensor reads as symbolic scalars during bounds checking
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
                raise TypeError(f"unsupported lifted index expression: {type(e).__name__}")

    if not (isinstance(e, LoopIR.Read) and e.idx and e.type.is_indexable()):
        return ORIGINAL_LIFT_EXPR(e)
    key = expr_key(e)
    sym = LIFTED_INDEX_SYMS.get(key)
    if sym is None:
        sym = Sym(f"lifted_index_{len(LIFTED_INDEX_SYMS)}")
        LIFTED_INDEX_SYMS[key] = sym
    return _boundscheck.E.Var(sym, e.type, e.srcinfo)


def patched_parse_stmt_block(self, stmts):
    rewritten = []
    for stmt in stmts:
        if not self.is_fragment and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute):
            name = f"liftedAttrCall{len(self.globals)}"
            while name in self.locals or name in self.globals:
                name += "x"
            self.globals[name] = self.eval_expr(stmt.value.func)
            call = ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()),
                args=stmt.value.args,
                keywords=stmt.value.keywords,
            )
            stmt = ast.fix_missing_locations(ast.copy_location(ast.Expr(value=ast.copy_location(call, stmt.value)), stmt))
        rewritten.append(stmt)
    return ORIGINAL_PARSE_STMT_BLOCK(self, rewritten)


_pyparser.get_src_locals = patched_get_src_locals
_exo_api.get_src_locals = patched_get_src_locals
_boundscheck.lift_expr = patched_lift_expr
_pyparser.Parser.parse_stmt_block = patched_parse_stmt_block


class Stack(Memory):
    # stack-allocated memory (uses alloca instead of malloc)
    # use for small temporaries that don't need heap allocation

    @classmethod
    def alloc(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        c_types = {"float": "float", "double": "double", "int8_t": "int8_t", "int32_t": "int32_t"}
        c_type = c_types.get(prim_type, prim_type)
        if not shape:
            return f"{c_type} {new_name};"
        return f'{c_type} {new_name}[{"][".join(shape)}];'

    @classmethod
    def can_read(cls) -> bool:
        return True

    @classmethod
    def write(cls, s, lhs: str, rhs: str) -> str:
        return f"{lhs} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs: str, rhs: str) -> str:
        return f"{lhs} += {rhs};"

    @classmethod
    def free(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        return ""


class NEON(Memory):
    # arm neon 128-bit vector memory (4×f32, 2×f64)

    @classmethod
    def global_(cls) -> str:
        return "#include <arm_neon.h>"

    @classmethod
    def alloc(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        if not shape:
            raise MemGenError(f"{srcinfo}: NEON vectors are not scalar values")

        vec_types: dict[str, tuple[int, str]] = {"float": (4, "float32x4_t"), "double": (2, "float64x2_t")}

        if prim_type not in vec_types:
            raise MemGenError(f"{srcinfo}: NEON vectors must be f32/f64, got {prim_type}")

        reg_width, c_type = vec_types[prim_type]
        if not (shape[-1].isdecimal() and int(shape[-1]) == reg_width):
            raise MemGenError(f"{srcinfo}: NEON vectors of type {prim_type} must be {reg_width}-wide, got {shape}")
        remaining = shape[:-1]
        if remaining:
            result = f'{c_type} {new_name}[{"][".join(map(str, remaining))}];'
        else:
            result = f"{c_type} {new_name};"
        return result

    @classmethod
    def can_read(cls) -> bool:
        return False

    @classmethod
    def free(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        return ""
