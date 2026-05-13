from collections.abc import Callable
from typing import ClassVar, TypeAlias

from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseArrayBase, DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr, VectorType, f32, f64, i32, i64
from xdsl.dialects.llvm import FAbsOp, FNegOp, FSqrtOp, MaskedStoreOp, VectorFMaxOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern

# vec_<op>_<type>      — plain: all lanes written
# vec_<op>_<type>_pfx  — prefix: only lanes 0..n-1 written (masked store for loop tails)
# neon_<op>_<type>     — arm neon intrinsics

BuildResult: TypeAlias = tuple[list[Operation], SSAValue]
BuilderFn: TypeAlias = Callable[..., BuildResult]
Handler: TypeAlias = Callable[[list[SSAValue]], tuple[Operation, ...]]

F32X4 = VectorType(f32, [4])
F64X2 = VectorType(f64, [2])


def _broadcast(scalar: SSAValue, vec_type: VectorType) -> tuple[list[Operation], SSAValue]:
    n_lanes = vec_type.get_shape()[0]
    undef = llvm.UndefOp(vec_type)
    idx = llvm.ConstantOp(IntegerAttr(0, i64), i64)
    inserted = llvm.InsertElementOp(undef, scalar, idx)
    shuffled = llvm.ShuffleVectorOp(inserted.res, undef.res, DenseArrayBase.from_list(i32, [0] * n_lanes), vec_type)
    return [undef, idx, inserted, shuffled], shuffled.res


def _make_mask(lane_count: SSAValue, n_lanes: int, *, extend: bool = False) -> tuple[list[Operation], SSAValue]:
    ops: list[Operation] = []
    indices = llvm.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [n_lanes]), list(range(n_lanes))), VectorType(i64, [n_lanes]))
    ops.append(indices)
    if extend:
        ext = llvm.SExtOp(lane_count, i64)
        ops.append(ext)
        lane_count = ext.res
    bc_ops, bc_val = _broadcast(lane_count, VectorType(i64, [n_lanes]))
    mask = llvm.ICmpOp(indices.result, bc_val, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return ops + bc_ops + [mask], mask.res


# _builder(op, i, j, ...): load args[i], args[j], ..., apply op, return (ops, result)
# arg index 0 = dst, 1 = first src, etc.
def _builder(op_fn: Callable[..., Operation] | None, *arg_indices: int) -> BuilderFn:
    def builder(dst: SSAValue, *srcs: SSAValue, vec_type: VectorType) -> BuildResult:
        all_args = (dst, *srcs)
        loads = [llvm.LoadOp(all_args[i], vec_type) for i in arg_indices]
        vals = [ld.dereferenced_value for ld in loads]
        if op_fn is None:
            return list(loads), vals[0]
        result_op = op_fn(*vals)
        return [*loads, result_op], result_op.results[0]
    return builder


def _abs_pfx(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # pre-store src to all lanes so inactive lanes keep original values after masked store
    load = llvm.LoadOp(src, vec_type)
    fabs = FAbsOp(load.dereferenced_value)
    return [load, fabs, llvm.StoreOp(load.dereferenced_value, dst)], fabs.results[0]


def _bcast(dst: SSAValue, scalar: SSAValue, *, vec_type: VectorType) -> BuildResult:
    return _broadcast(scalar, vec_type)


def _zero(dst: SSAValue, *, vec_type: VectorType) -> BuildResult:
    zero = llvm.ConstantOp(DenseIntOrFPElementsAttr.from_list(vec_type, [0.0] * vec_type.get_shape()[0]), vec_type)
    return [zero], zero.result


def _plain(builder: BuilderFn, vec_type: VectorType) -> Handler:
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        dst, *srcs = args
        ops, result = builder(dst, *srcs, vec_type=vec_type)
        return (*ops, llvm.StoreOp(result, dst))
    return handle


def _pfx(builder: BuilderFn, vec_type: VectorType, n_lanes: int, *, extend: bool = False) -> Handler:
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        lane_count, dst, *srcs = args
        mask_ops, mask = _make_mask(lane_count, n_lanes, extend=extend)
        core_ops, result = builder(dst, *srcs, vec_type=vec_type)
        return (*mask_ops, *core_ops, MaskedStoreOp(result, dst, mask))
    return handle


def _reduce(vec_type: VectorType) -> Handler:
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        acc_val, src_ptr = args[0], args[1]
        assert isinstance(acc_val.owner, llvm.LoadOp)
        src_load = llvm.LoadOp(src_ptr, vec_type)
        reduce = llvm.CallIntrinsicOp("llvm.vector.reduce.fadd", [acc_val, src_load.dereferenced_value], [vec_type.element_type])
        return (src_load, reduce, llvm.StoreOp(reduce.ress, acc_val.owner.ptr))
    return handle


def _neon_broadcast(dst: SSAValue, scalar_ptr: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    load = llvm.LoadOp(scalar_ptr, vec_type.element_type)
    bc_ops, bc_val = _broadcast(load.dereferenced_value, vec_type)
    return (load, *bc_ops, llvm.StoreOp(bc_val, dst))


def _neon_zero(dst: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    zero = llvm.ConstantOp(FloatAttr(0.0, vec_type.element_type), vec_type.element_type)
    bc_ops, bc_val = _broadcast(zero.result, vec_type)
    return (zero, *bc_ops, llvm.StoreOp(bc_val, dst))


def _make_intrinsics() -> dict[str, Handler]:
    entries: dict[str, Handler] = {}

    # vec_* intrinsics: plain and prefix variants for f32x4 and f64x2
    _VEC_OPS: list[tuple[str, BuilderFn, BuilderFn | None, bool]] = [
        ("abs",       _builder(FAbsOp, 1),          _abs_pfx, True),
        ("add_red",   _builder(llvm.FAddOp, 0, 1),  None,     True),
        ("copy",      _builder(None, 1),             None,     True),
        ("load",      _builder(None, 1),             None,     True),
        ("store",     _builder(None, 1),             None,     False),
        ("add",       _builder(llvm.FAddOp, 1, 2),   None,     False),
        ("mul",       _builder(llvm.FMulOp, 1, 2),   None,     False),
        ("neg",       _builder(FNegOp, 1),           None,     False),
        ("brdcst_scl", _bcast,                       None,     False),
        ("fmadd2",    _builder(llvm.FMAOp, 1, 2, 3), None,     False),
        ("fmadd1",    _builder(llvm.FMAOp, 1, 2, 3), None,     False),
        ("fmadd_red", _builder(llvm.FMAOp, 1, 2, 0), None,     False),
        ("zero",      _zero,                         None,     False),
    ]
    for name, b, pfx_b, ext in _VEC_OPS:
        pb = pfx_b if pfx_b is not None else b
        entries[f"vec_{name}_f32x4"]     = _plain(b, F32X4)
        entries[f"vec_{name}_f32x4_pfx"] = _pfx(pb, F32X4, 4)
        entries[f"vec_{name}_f64x2"]     = _plain(b, F64X2)
        entries[f"vec_{name}_f64x2_pfx"] = _pfx(pb, F64X2, 2, extend=ext)

    for sfx, vt in [("f32x4", F32X4), ("f64x2", F64X2)]:
        entries[f"vec_reduce_add_scl_{sfx}"] = _reduce(vt)

    # neon ops: reuse _plain + _builder for load/store/arith patterns
    for name, op in [("add", llvm.FAddOp), ("sub", llvm.FSubOp), ("mul", llvm.FMulOp), ("div", llvm.FDivOp),
                     ("vadd", llvm.FAddOp), ("vsub", llvm.FSubOp), ("vmul", llvm.FMulOp)]:
        entries[f"neon_{name}_f32x4"] = _plain(_builder(op, 1, 2), F32X4)
    for name, op in [("add_acc", llvm.FAddOp), ("fmax_acc", VectorFMaxOp), ("mul_acc", llvm.FMulOp),
                     ("sub_acc", llvm.FSubOp), ("div_acc", llvm.FDivOp)]:
        entries[f"neon_{name}_f32x4"] = _plain(_builder(op, 0, 1), F32X4)
    for name, op in [("neg", FNegOp), ("vneg", FNegOp), ("sqrt", FSqrtOp)]:
        entries[f"neon_{name}_f32x4"] = _plain(_builder(op, 1), F32X4)
    for sfx, vt in [("f32x4", F32X4), ("f64x2", F64X2)]:
        entries[f"neon_storeu_{sfx}"]    = _plain(_builder(None, 1), vt)
        entries[f"neon_loadu_{sfx}"]     = _plain(_builder(None, 1), vt)
        entries[f"neon_fmadd_{sfx}"]     = _plain(_builder(llvm.FMAOp, 1, 2, 0), vt)
        entries[f"neon_broadcast_{sfx}"] = lambda args, v=vt: _neon_broadcast(*args, vec_type=v)
    entries["neon_zero_f32x4"]   = lambda args: _neon_zero(args[0], vec_type=F32X4)
    entries["neon_square_f32x4"] = _plain(_builder(llvm.FMulOp, 1, 1), F32X4)

    return entries


class ConvertVecIntrinsic(RewritePattern):
    _INTRINSICS: ClassVar[dict[str, Handler]] = _make_intrinsics()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.CallOp, rewriter: PatternRewriter) -> None:
        if op.callee is None:
            return
        handler = self._INTRINSICS.get(op.callee.root_reference.data)
        if handler is None:
            return
        rewriter.replace_op(op, handler(list(op.args)))
