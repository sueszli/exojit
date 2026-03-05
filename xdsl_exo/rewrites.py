from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from xdsl.dialects import arith, func, llvm, memref, vector
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IndexType, IntegerAttr, MemRefType, VectorType, f32, f64, i64
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern

from xdsl_exo import patches as llvm_extra

# `vec_*` intrinsic lowering: `func.CallOp` -> LLVM/vector dialect ops
#
# Naming:
# -------
#     vec_<op>_<type>          - plain version. all lanes written
#     vec_<op>_<type>_pfx      - prefix version. only lanes 0..n-1 written (loop tails)
#
#     <type> = f32x8 | f64x4
#     <op>  = add, mul, neg, abs, fmadd1, fmadd2, fmadd_red, zero, ...
#
# Plain variant:
# --------------
#     func.call @vec_add_f32x8(%dst, %a, %b)
#     =>
#     %v0  = llvm.load %a : vector<8xf32>
#     %v1  = llvm.load %b : vector<8xf32>
#     %r   = llvm.fadd %v0, %v1
#            llvm.store %r, %dst
#
# Prefix (_pfx) variant:
# ----------------------
# First arg is a lane-count `n`. A boolean mask selects which lanes get written.
#
#     func.call @vec_add_f32x8_pfx(%n, %dst, %a, %b)      e.g. n=3
#     =>
#     %idx  = arith.constant   [0, 1, 2, 3, 4, 5, 6, 7]
#     %bc   = vector.broadcast [3, 3, 3, 3, 3, 3, 3, 3]   (n splatted to all lanes)
#     %mask = llvm.icmp "slt"  [T, T, T, F, F, F, F, F]   (idx < bc)
#     %v0   = llvm.load %a : vector<8xf32>
#     %v1   = llvm.load %b : vector<8xf32>
#     %r    = llvm.fadd %v0, %v1
#             llvm.masked_store %r, %dst, %mask


MaskResult: TypeAlias = tuple[list[Operation], SSAValue]
BuildResult: TypeAlias = tuple[list[Operation], SSAValue, SSAValue]
Builder: TypeAlias = Callable[[list[SSAValue], VectorType], BuildResult]
MaskFn: TypeAlias = Callable[[SSAValue], MaskResult]


def _make_mask(m: SSAValue, n_lanes: int, *, extend: bool = False) -> MaskResult:
    # lane i is active if i < m; extend=True sign-extends m from i32 to i64
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [n_lanes]), list(range(n_lanes))))
    ops = [indices]
    if extend:
        ext = arith.ExtSIOp(m, i64)
        ops.append(ext)
        m = ext.result
    broadcast = vector.BroadcastOp(operands=[m], result_types=[VectorType(i64, [n_lanes])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return ops + [broadcast, mask], mask.res


def _mask_f32x8(m: SSAValue) -> MaskResult:
    return _make_mask(m, 8)


def _mask_f64x4(m: SSAValue) -> MaskResult:
    return _make_mask(m, 4)


def _mask_f64x4_ext(m: SSAValue) -> MaskResult:
    return _make_mask(m, 4, extend=True)  # m is i32 at call site


def _build_identity(args: list[SSAValue], vt: VectorType) -> BuildResult:
    load = llvm.LoadOp(args[1], vt)
    return [load], load.dereferenced_value, args[0]


def _build_abs(args: list[SSAValue], vt: VectorType) -> BuildResult:
    dst, src = args[0], args[1]
    load = llvm.LoadOp(src, vt)
    fabs = llvm_extra.FAbsOp(load.dereferenced_value, vt)
    return [load, fabs], fabs.result, dst


def _build_abs_pfx(args: list[SSAValue], vt: VectorType) -> BuildResult:
    # Pre-store raw src to dst so non-masked lanes retain src values (not garbage)
    # when the caller's masked store writes only lanes 0..n-1 with the fabs result.
    dst, src = args[0], args[1]
    load = llvm.LoadOp(src, vt)
    fabs = llvm_extra.FAbsOp(load.dereferenced_value, vt)
    return [load, fabs, llvm.StoreOp(load.dereferenced_value, dst)], fabs.result, dst


def _build_neg(args: list[SSAValue], vt: VectorType) -> BuildResult:
    load = llvm.LoadOp(args[1], vt)
    neg = llvm_extra.FNegOp(load.dereferenced_value)
    return [load, neg], neg.res, args[0]


def _build_binary(binop_cls: Callable[[SSAValue, SSAValue], Operation]) -> Builder:
    def build(args: list[SSAValue], vt: VectorType) -> BuildResult:
        l1 = llvm.LoadOp(args[1], vt)
        l2 = llvm.LoadOp(args[2], vt)
        result = binop_cls(l1.dereferenced_value, l2.dereferenced_value)
        return [l1, l2, result], result.res, args[0]

    return build


def _build_add_red(args: list[SSAValue], vt: VectorType) -> BuildResult:
    dst = args[0]
    l_dst = llvm.LoadOp(dst, vt)
    l_src = llvm.LoadOp(args[1], vt)
    add = llvm.FAddOp(l_dst.dereferenced_value, l_src.dereferenced_value)
    return [l_dst, l_src, add], add.res, dst


def _build_fma(args: list[SSAValue], vt: VectorType) -> BuildResult:
    l1 = llvm.LoadOp(args[1], vt)
    l2 = llvm.LoadOp(args[2], vt)
    l3 = llvm.LoadOp(args[3], vt)
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, l3.dereferenced_value], result_types=[vt])
    return [l1, l2, l3, fma], fma.res, args[0]


def _build_fma_red(args: list[SSAValue], vt: VectorType) -> BuildResult:
    dst = args[0]
    ld = llvm.LoadOp(dst, vt)
    l1 = llvm.LoadOp(args[1], vt)
    l2 = llvm.LoadOp(args[2], vt)
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, ld.dereferenced_value], result_types=[vt])
    return [ld, l1, l2, fma], fma.res, dst


def _build_broadcast(args: list[SSAValue], vt: VectorType) -> BuildResult:
    bcast = vector.BroadcastOp(operands=[args[1]], result_types=[vt])
    return [bcast], bcast.vector, args[0]


def _build_zero(args: list[SSAValue], vt: VectorType) -> BuildResult:
    zero = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(vt, [0.0] * vt.get_shape()[0]))
    return [zero], zero.result, args[0]


# name -> (builder_fn, vector_type, mask_fn | None)
_VEC_INTRINSICS: dict[str, tuple[Builder, VectorType, MaskFn | None]] = {}
for _name, _builder, _pfx_builder, _f64_mask in [
    # name          plain                         pfx (None = same as plain)  f64 mask
    ("abs", _build_abs, _build_abs_pfx, _mask_f64x4_ext),
    ("add_red", _build_add_red, None, _mask_f64x4_ext),
    ("copy", _build_identity, None, _mask_f64x4_ext),
    ("load", _build_identity, None, _mask_f64x4_ext),
    ("store", _build_identity, None, _mask_f64x4),
    ("add", _build_binary(llvm.FAddOp), None, _mask_f64x4),
    ("mul", _build_binary(llvm.FMulOp), None, _mask_f64x4),
    ("neg", _build_neg, None, _mask_f64x4),
    ("brdcst_scl", _build_broadcast, None, _mask_f64x4),
    ("fmadd2", _build_fma, None, _mask_f64x4),
    ("fmadd1", _build_fma, None, _mask_f64x4),
    ("fmadd_red", _build_fma_red, None, _mask_f64x4),
    ("zero", _build_zero, None, _mask_f64x4),
]:
    _pfx_b = _pfx_builder or _builder
    _VEC_INTRINSICS[f"vec_{_name}_f32x8"] = (_builder, VectorType(f32, [8]), None)
    _VEC_INTRINSICS[f"vec_{_name}_f32x8_pfx"] = (_pfx_b, VectorType(f32, [8]), _mask_f32x8)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4"] = (_builder, VectorType(f64, [4]), None)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4_pfx"] = (_pfx_b, VectorType(f64, [4]), _f64_mask)


def _build_mm256_storeu_ps(args: list[SSAValue]) -> tuple[Operation, ...]:
    load = llvm.LoadOp(args[1], VectorType(f32, [8]))
    return (load, llvm.StoreOp(load.dereferenced_value, args[0]))


def _build_mm256_fmadd_ps(args: list[SSAValue]) -> tuple[Operation, ...]:
    # accumulates: args[0] = args[1]*args[2] + args[0]
    zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
    l0 = llvm.LoadOp(args[0], VectorType(f32, [8]))
    l1 = llvm.LoadOp(args[1], VectorType(f32, [8]))
    l2 = llvm.LoadOp(args[2], VectorType(f32, [8]))
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, l0.dereferenced_value], result_types=[VectorType(f32, [8])])
    return (zero, l0, l1, l2, fma, llvm.StoreOp(fma.res, args[0], [zero.result]))


def _build_mm256_broadcast_ss(args: list[SSAValue]) -> tuple[Operation, ...]:
    # args[1] is a memref (scalar pointer), unlike the llvm ptrs used elsewhere
    zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
    load = memref.LoadOp.get(args[1], [zero.result])
    bcast = vector.BroadcastOp(operands=[load.results[0]], result_types=[VectorType(f32, [8])])
    return (zero, load, bcast, llvm.StoreOp(bcast.results[0], args[0], [zero.result]))


_MM256_INTRINSICS: dict[str, Callable[[list[SSAValue]], tuple[Operation, ...]]] = {
    "mm256_storeu_ps": _build_mm256_storeu_ps,
    "mm256_fmadd_ps": _build_mm256_fmadd_ps,
    "mm256_broadcast_ss": _build_mm256_broadcast_ss,
    "mm256_loadu_ps": _build_mm256_storeu_ps,  # same load-then-store operation
}


def _reduce(op: func.CallOp, rewriter: PatternRewriter, vt: VectorType) -> None:
    # acc_scalar += sum(src_vector); result written back through acc's originating load ptr
    assert isinstance(op.arguments[0].owner, llvm.LoadOp)
    acc_load = op.arguments[0].owner
    load = llvm.LoadOp(op.arguments[1], vt)
    reduce = vector.ReductionOp(load.dereferenced_value, vector.CombiningKindAttr([vector.CombiningKindFlag.ADD]), acc=op.arguments[0])
    rewriter.replace_matched_op((load, reduce, llvm.StoreOp(reduce.dest, acc_load.ptr)))


class ConvertVecIntrinsic(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter) -> None:
        callee = op.callee.root_reference.data

        if callee == "vec_reduce_add_scl_f32x8":
            return _reduce(op, rewriter, VectorType(f32, [8]))
        if callee == "vec_reduce_add_scl_f64x4":
            return _reduce(op, rewriter, VectorType(f64, [4]))

        mm256_builder = _MM256_INTRINSICS.get(callee)
        if mm256_builder is not None:
            rewriter.replace_matched_op(mm256_builder(list(op.arguments)))
            return

        entry = _VEC_INTRINSICS.get(callee)
        if entry is None:
            return

        builder, vt, mask_fn = entry
        pfx = mask_fn is not None
        # prefixed variants pass lane-count as args[0]; strip before forwarding to builder
        args = list(op.arguments[1:]) if pfx else list(op.arguments)
        core_ops, result, dst = builder(args, vt)

        if pfx:
            mask_ops, mask = mask_fn(op.arguments[0])
            rewriter.replace_matched_op((*mask_ops, *core_ops, llvm_extra.MaskedStoreOp(result, dst, mask)))
        else:
            rewriter.replace_matched_op((*core_ops, llvm.StoreOp(result, dst)))


@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    # must run last: after shape/element info is consumed by earlier passes into ops
    recursive: bool = True

    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()
