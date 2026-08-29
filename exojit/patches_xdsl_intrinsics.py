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

Handler: TypeAlias = Callable[[list[SSAValue]], tuple[Operation, ...]]
OpFn: TypeAlias = Callable[..., Operation]

F32X4 = VectorType(f32, [4])
F64X2 = VectorType(f64, [2])


def _handler(op_fn: OpFn | None, vec_type: VectorType, *operands: int, load_order: tuple[int, ...] = ()) -> Handler:
    # arg 0 is dst, 1.. are srcs: load `load_order` (default `operands`), apply `op_fn` over `operands`, store to arg 0
    arity = max((*operands, *load_order)) + 1

    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        assert len(args) == arity, f"expected {arity} operands, got {len(args)}"
        loads: dict[int, llvm.LoadOp] = {}
        for i in load_order or operands:
            if i not in loads:
                loads[i] = llvm.LoadOp(args[i], vec_type)
        ops: list[Operation] = list(loads.values())
        result = loads[operands[0]].dereferenced_value
        if op_fn is not None:
            op = op_fn(*(loads[i].dereferenced_value for i in operands))
            ops.append(op)
            result = op.results[0]
        return (*ops, llvm.StoreOp(result, args[0]))

    return handle


def _broadcast_handler(vec_type: VectorType) -> Handler:
    # dst[:] = [*scalar_ptr] * n_lanes; scalar_ptr is already !llvm.ptr here
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        assert len(args) == 2, f"expected 2 operands, got {len(args)}"
        load = llvm.LoadOp(args[1], vec_type.element_type)
        undef = llvm.UndefOp(vec_type)
        idx = llvm.ConstantOp(IntegerAttr(0, i64), i64)
        inserted = llvm.InsertElementOp(undef, load.dereferenced_value, idx)
        lanes = DenseArrayBase.from_list(i32, [0] * vec_type.get_shape()[0])
        shuffled = llvm.ShuffleVectorOp(inserted.res, undef.res, lanes, vec_type)
        return (load, undef, idx, inserted, shuffled, llvm.StoreOp(shuffled.res, args[0]))

    return handle


def _make_intrinsics() -> dict[str, Handler]:
    # callee name -> handler(args: list[SSAValue]) -> tuple[Operation, ...]
    binops: list[tuple[str, OpFn]] = [("add", llvm.FAddOp), ("sub", llvm.FSubOp), ("mul", llvm.FMulOp), ("div", llvm.FDivOp)]
    entries: dict[str, Handler] = {
        **{f"neon_{name}_f32x4": _handler(op_fn, F32X4, 1, 2) for name, op_fn in binops},  # dst = a op b
        **{f"neon_{name}_acc_f32x4": _handler(op_fn, F32X4, 0, 1) for name, op_fn in binops},  # dst = dst op src
        "neon_fmax_acc_f32x4": _handler(VectorFMaxOp, F32X4, 0, 1),
        "neon_sqrt_f32x4": _handler(FSqrtOp, F32X4, 1),
        "neon_square_f32x4": _handler(llvm.FMulOp, F32X4, 1, 1),
        "vec_add_f32x4": _handler(llvm.FAddOp, F32X4, 1, 2),
        "vec_add_f64x2": _handler(llvm.FAddOp, F64X2, 1, 2),
        "vec_mul_f32x4": _handler(llvm.FMulOp, F32X4, 1, 2),
        "vec_mul_f64x2": _handler(llvm.FMulOp, F64X2, 1, 2),
        "vec_add_red_f32x4": _handler(llvm.FAddOp, F32X4, 0, 1),  # dst += src
        "vec_neg_f32x4": _handler(FNegOp, F32X4, 1),
        "vec_copy_f32x4": _handler(None, F32X4, 1),  # dst = src
    }
    for suffix, vt in [("f32x4", F32X4), ("f64x2", F64X2)]:
        entries[f"neon_loadu_{suffix}"] = _handler(None, vt, 1)
        entries[f"neon_storeu_{suffix}"] = _handler(None, vt, 1)
        entries[f"neon_fmadd_{suffix}"] = _handler(llvm.FMAOp, vt, 1, 2, 0, load_order=(0, 1, 2))  # dst += a * b
        entries[f"neon_broadcast_{suffix}"] = _broadcast_handler(vt)
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
