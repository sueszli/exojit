from dataclasses import dataclass
from functools import cache

from xdsl.dialects import arith, func, llvm, memref, vector
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IndexType, IntegerAttr, MemRefType, VectorType, f32, f64, i32, i64
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern

from xdsl_exo import patches as llvm_extra

# `vec_*` intrinsic lowering: `func.CallOp` -> LLVM/vector dialect ops
#
# Naming:
# -------
#     vec_<op>_<type>      plain variant  – all lanes written
#     vec_<op>_<type>_pfx  prefix variant – only lanes 0..n-1 written (loop tails)
#
#     <type>  =  f32x8  |  f64x4
#     <op>    =  add, mul, neg, abs, fmadd1, fmadd2, fmadd_red, zero, ...
#
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


def _mask_f32x8(m):
    # prefix mask for f32x8: lane i is active if i < m (i64 indices vs i64 broadcast)
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [8]), list(range(8))))
    broadcast = vector.BroadcastOp(operands=[m], result_types=[VectorType(i64, [8])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return [indices, broadcast, mask], mask.res


def _mask_f64x4_ext(m):
    # prefix mask for f64x4 when m is i32: sign-extend to i64 before broadcast
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [4]), list(range(4))))
    ext = arith.ExtSIOp(m, i64)
    broadcast = vector.BroadcastOp(operands=[ext.result], result_types=[VectorType(i64, [4])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return [indices, ext, broadcast, mask], mask.res


def _mask_f64x4(m):
    # prefix mask for f64x4 when m is already i64: broadcast as i32 vector, compare against i64 indices
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [4]), list(range(4))))
    broadcast = vector.BroadcastOp(operands=[m], result_types=[VectorType(i32, [4])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return [indices, broadcast, mask], mask.res


def _build_identity(args, vt):
    # load src into a vector; result is the loaded value written to dst
    load = llvm.LoadOp(args[1], vt)
    return [load], load.dereferenced_value, args[0]


def _build_abs(args, vt, *, prefixed=False):
    # element-wise fabs; prefixed variant also pre-stores src into dst before masking
    dst, src = args[0], args[1]
    load = llvm.LoadOp(src, vt)
    fabs = llvm_extra.FAbsOp(load.dereferenced_value, vt)
    if prefixed:
        return [load, fabs, llvm.StoreOp(load.dereferenced_value, dst)], fabs.result, dst
    return [load, fabs], fabs.result, dst


def _build_neg(args, vt):
    # element-wise fneg: load src, negate, write to dst
    load = llvm.LoadOp(args[1], vt)
    neg = llvm_extra.FNegOp(load.dereferenced_value)
    return [load, neg], neg.res, args[0]


@cache
def _build_binary(binop_cls):
    # factory: returns a builder that loads two operands and applies binop_cls
    def build(args, vt):
        l1 = llvm.LoadOp(args[1], vt)
        l2 = llvm.LoadOp(args[2], vt)
        result = binop_cls(l1.dereferenced_value, l2.dereferenced_value)
        return [l1, l2, result], result.res, args[0]

    return build


def _build_add_red(args, vt):
    # reduction add: dst += src (loads dst first, then adds src into it)
    dst = args[0]
    l_dst = llvm.LoadOp(dst, vt)
    l_src = llvm.LoadOp(args[1], vt)
    add = llvm.FAddOp(l_dst.dereferenced_value, l_src.dereferenced_value)
    return [l_dst, l_src, add], add.res, dst


def _build_fma(args, vt):
    # fused multiply-add: dst = a*b + c (all three operands loaded from memory)
    l1 = llvm.LoadOp(args[1], vt)
    l2 = llvm.LoadOp(args[2], vt)
    l3 = llvm.LoadOp(args[3], vt)
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, l3.dereferenced_value], result_types=[vt])
    return [l1, l2, l3, fma], fma.res, args[0]


def _build_fma_red(args, vt):
    # fma reduction: dst = a*b + dst (accumulates into existing dst value)
    dst = args[0]
    ld = llvm.LoadOp(dst, vt)
    l1 = llvm.LoadOp(args[1], vt)
    l2 = llvm.LoadOp(args[2], vt)
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, ld.dereferenced_value], result_types=[vt])
    return [ld, l1, l2, fma], fma.res, dst


def _build_broadcast(args, vt):
    # splat scalar args[1] to all lanes of a vector of type vt
    bcast = vector.BroadcastOp(operands=[args[1]], result_types=[vt])
    return [bcast], bcast.vector, args[0]


def _build_zero(args, vt):
    # zero-initialize a vector constant of type vt
    zero = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(vt, [0.0] * vt.get_shape()[0]))
    return [zero], zero.result, args[0]


# registry mapping intrinsic name -> (builder_fn, vector_type, optional_prefix_mask_fn)
_VEC_INTRINSICS: dict = {}
for _name, _builder, _f64_mask in [
    ("abs", _build_abs, _mask_f64x4_ext),
    ("add_red", _build_add_red, _mask_f64x4_ext),
    ("copy", _build_identity, _mask_f64x4_ext),
    ("load", _build_identity, _mask_f64x4_ext),
    ("store", _build_identity, _mask_f64x4),
    ("add", _build_binary(llvm.FAddOp), _mask_f64x4),
    ("mul", _build_binary(llvm.FMulOp), _mask_f64x4),
    ("neg", _build_neg, _mask_f64x4),
    ("brdcst_scl", _build_broadcast, _mask_f64x4),
    ("fmadd2", _build_fma, _mask_f64x4),
    ("fmadd1", _build_fma, _mask_f64x4),
    ("fmadd_red", _build_fma_red, _mask_f64x4),
    ("zero", _build_zero, _mask_f64x4),
]:
    # register plain and prefixed (masked) variants for both f32x8 and f64x4
    _VEC_INTRINSICS[f"vec_{_name}_f32x8"] = (_builder, VectorType(f32, [8]), None)
    _VEC_INTRINSICS[f"vec_{_name}_f32x8_pfx"] = (_builder, VectorType(f32, [8]), _mask_f32x8)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4"] = (_builder, VectorType(f64, [4]), None)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4_pfx"] = (_builder, VectorType(f64, [4]), _f64_mask)


def _build_mm256_storeu_ps(args):
    # _mm256_storeu_ps: load f32x8 from args[1], store to args[0]
    load = llvm.LoadOp(args[1], VectorType(f32, [8]))
    return (load, llvm.StoreOp(load.dereferenced_value, args[0]))


def _build_mm256_fmadd_ps(args):
    # _mm256_fmadd_ps: args[0] = args[1]*args[2] + args[0] (in-place accumulate)
    zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
    l0 = llvm.LoadOp(args[0], VectorType(f32, [8]))
    l1 = llvm.LoadOp(args[1], VectorType(f32, [8]))
    l2 = llvm.LoadOp(args[2], VectorType(f32, [8]))
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, l0.dereferenced_value], result_types=[VectorType(f32, [8])])
    return (zero, l0, l1, l2, fma, llvm.StoreOp(fma.res, args[0], [zero.result]))


def _build_mm256_broadcast_ss(args):
    # _mm256_broadcast_ss: load scalar from memref args[1][0], splat to f32x8, store to args[0]
    zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
    load = memref.LoadOp.get(args[1], [zero.result])
    bcast = vector.BroadcastOp(operands=[load.results[0]], result_types=[VectorType(f32, [8])])
    return (zero, load, bcast, llvm.StoreOp(bcast.results[0], args[0], [zero.result]))


def _build_mm256_loadu_ps(args):
    # _mm256_loadu_ps: load f32x8 from args[1] and store back (materializes the pointer as a vector)
    zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
    load = llvm.LoadOp(args[1], VectorType(f32, [8]))
    return (zero, load, llvm.StoreOp(load.dereferenced_value, args[1], [zero.result]))


_MM256_INTRINSICS: dict = {
    "mm256_storeu_ps": _build_mm256_storeu_ps,
    "mm256_fmadd_ps": _build_mm256_fmadd_ps,
    "mm256_broadcast_ss": _build_mm256_broadcast_ss,
    "mm256_loadu_ps": _build_mm256_loadu_ps,
}


def _reduce(op, rewriter, vt):
    # horizontal add-reduction: acc_scalar += sum(src_vector); writes result back through acc's load ptr
    assert isinstance(op.arguments[0].owner, llvm.LoadOp)
    acc_load = op.arguments[0].owner
    load = llvm.LoadOp(op.arguments[1], vt)
    reduce = vector.ReductionOp(load.dereferenced_value, vector.CombiningKindAttr([vector.CombiningKindFlag.ADD]), acc=op.arguments[0])
    rewriter.replace_matched_op((load, reduce, llvm.StoreOp(reduce.dest, acc_load.ptr)))


class ConvertVecIntrinsic(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
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
        # prefixed variants pass the lane-count as args[0]; strip it before forwarding to builder
        args = list(op.arguments[1:]) if pfx else list(op.arguments)
        if builder is _build_abs and pfx:
            core_ops, result, dst = builder(args, vt, prefixed=True)
        else:
            core_ops, result, dst = builder(args, vt)

        if pfx:
            # emit mask ops + core ops + a masked store to apply the prefix predicate
            mask_ops, mask = mask_fn(op.arguments[0])
            rewriter.replace_matched_op((*mask_ops, *core_ops, llvm_extra.MaskedStoreOp(result, dst, mask)))
        else:
            rewriter.replace_matched_op((*core_ops, llvm.StoreOp(result, dst)))


@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    # must run last: after shape/element info is consumed by earlier passes into ops.
    recursive: bool = True

    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType):
        return llvm.LLVMPointerType()
