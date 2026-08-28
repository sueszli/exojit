from collections.abc import Callable
from typing import ClassVar, TypeAlias

from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, VectorType, f32, f64, i32, i64
from xdsl.dialects.llvm import FNegOp, FSqrtOp, VectorFMaxOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern

# `vec_*`/`neon_*` intrinsic lowering: `llvm.CallOp` -> LLVM dialect ops
#
#     llvm.call @vec_add_f32x4(%dst, %a, %b)
#     =>
#     %v0 = llvm.load %a : vector<4xf32>
#     %v1 = llvm.load %b : vector<4xf32>
#     %r  = llvm.fadd %v0, %v1
#           llvm.store %r, %dst

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


def _build_binop(op_fn: Callable[..., Operation] | None, *ptrs: SSAValue, vec_type: VectorType) -> BuildResult:
    # load all ptrs, apply op_fn to their values (or just return first value if op_fn is None)
    loads = [llvm.LoadOp(p, vec_type) for p in ptrs]
    vals = [ld.dereferenced_value for ld in loads]
    if op_fn is None:
        return list(loads), vals[0]
    result_op = op_fn(*vals)
    return [*loads, result_op], result_op.results[0]


def _builder(op_fn: Callable[..., Operation] | None, *arg_indices: int) -> BuilderFn:
    # shorthand for creating a BuilderFn that calls _build_binop.
    # arg_indices select which call args to load: 0 = dst, 1 = first src, 2 = second src, ...
    #
    # e.g. _builder(llvm.FAddOp, 1, 2)  creates  (dst, a, b)  -> load a, load b, fadd
    #      _builder(llvm.FAddOp, 0, 1)  creates  (dst, src)   -> load dst, load src, fadd
    #      _builder(None, 1)            creates  (dst, src)   -> load src (copy)
    def builder(dst: SSAValue, *srcs: SSAValue, vec_type: VectorType) -> BuildResult:
        all_args = (dst, *srcs)
        return _build_binop(op_fn, *(all_args[i] for i in arg_indices), vec_type=vec_type)

    return builder


def _plain_handler(builder: BuilderFn, vec_type: VectorType) -> Handler:
    # build ops then store result to dst
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        dst, *srcs = args
        ops, result = builder(dst, *srcs, vec_type=vec_type)
        return (*ops, llvm.StoreOp(result, dst))

    return handle


def _build_neon_storeu(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = src[:]
    load = llvm.LoadOp(src, vec_type)
    return (load, llvm.StoreOp(load.dereferenced_value, dst))


def _build_neon_fmadd(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = dst[:] + src_a[:] * src_b[:]
    load_acc = llvm.LoadOp(dst, vec_type)
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    fma = llvm.FMAOp(load_a.dereferenced_value, load_b.dereferenced_value, load_acc.dereferenced_value)
    return (load_acc, load_a, load_b, fma, llvm.StoreOp(fma.res, dst))


def _build_neon_broadcast(dst: SSAValue, scalar_ptr: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = [*scalar_ptr] * n_lanes  (scalar_ptr is already !llvm.ptr at this stage of the pipeline)
    elem_type = vec_type.element_type
    load = llvm.LoadOp(scalar_ptr, elem_type)
    bc_ops, bc_val = _broadcast(load.dereferenced_value, vec_type)
    return (load, *bc_ops, llvm.StoreOp(bc_val, dst))


def _build_neon_binop(op_cls: type, dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = op(src_a[:], src_b[:])
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    result = op_cls(load_a.dereferenced_value, load_b.dereferenced_value)
    return (load_a, load_b, result, llvm.StoreOp(result.res, dst))


def _build_neon_square(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = src[:] * src[:]
    load = llvm.LoadOp(src, vec_type)
    result = llvm.FMulOp(load.dereferenced_value, load.dereferenced_value)
    return (load, result, llvm.StoreOp(result.res, dst))


def _build_neon_unop(op_cls: type, dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = op(src[:])
    load = llvm.LoadOp(src, vec_type)
    result = op_cls(load.dereferenced_value)
    return (load, result, llvm.StoreOp(result.res, dst))


def _make_intrinsics() -> dict[str, Handler]:
    # callee name -> handler(args: list[SSAValue]) -> tuple[Operation, ...]
    entries: dict[str, Handler] = {}

    # vec_*
    for op_name, op_fn in [("add", llvm.FAddOp), ("mul", llvm.FMulOp)]:
        entries[f"vec_{op_name}_f32x4"] = _plain_handler(_builder(op_fn, 1, 2), F32X4)  # dst = a op b
        entries[f"vec_{op_name}_f64x2"] = _plain_handler(_builder(op_fn, 1, 2), F64X2)
    entries["vec_add_red_f32x4"] = _plain_handler(_builder(llvm.FAddOp, 0, 1), F32X4)  # dst += src
    entries["vec_copy_f32x4"] = _plain_handler(_builder(None, 1), F32X4)  # dst = src
    entries["vec_neg_f32x4"] = _plain_handler(_builder(FNegOp, 1), F32X4)  # dst = -src

    # neon binops: dst = op(a, b)
    _NEON_BINOPS: list[tuple[str, type]] = [
        ("add", llvm.FAddOp),
        ("sub", llvm.FSubOp),
        ("mul", llvm.FMulOp),
        ("div", llvm.FDivOp),
    ]
    for op_name, op_cls in _NEON_BINOPS:
        entries[f"neon_{op_name}_f32x4"] = lambda args, o=op_cls: _build_neon_binop(o, *args, vec_type=F32X4)

    # neon acc binops: acc = op(acc, src)
    _NEON_ACC_OPS: list[tuple[str, type]] = [
        ("add_acc", llvm.FAddOp),
        ("fmax_acc", VectorFMaxOp),
        ("mul_acc", llvm.FMulOp),
        ("sub_acc", llvm.FSubOp),
        ("div_acc", llvm.FDivOp),
    ]
    for op_name, op_cls in _NEON_ACC_OPS:
        entries[f"neon_{op_name}_f32x4"] = lambda args, o=op_cls: _build_neon_binop(o, args[0], args[0], args[1], vec_type=F32X4)

    # neon load/store/fmadd/broadcast (both types)
    for suffix, vt in [
        ("f32x4", F32X4),
        ("f64x2", F64X2),
    ]:
        entries[f"neon_storeu_{suffix}"] = lambda args, v=vt: _build_neon_storeu(*args, vec_type=v)
        entries[f"neon_loadu_{suffix}"] = lambda args, v=vt: _build_neon_storeu(*args, vec_type=v)
        entries[f"neon_fmadd_{suffix}"] = lambda args, v=vt: _build_neon_fmadd(*args, vec_type=v)
        entries[f"neon_broadcast_{suffix}"] = lambda args, v=vt: _build_neon_broadcast(*args, vec_type=v)

    # neon misc
    entries["neon_sqrt_f32x4"] = lambda args: _build_neon_unop(FSqrtOp, *args, vec_type=F32X4)
    entries["neon_square_f32x4"] = lambda args: _build_neon_square(*args, vec_type=F32X4)

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
