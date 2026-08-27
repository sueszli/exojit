from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, TypeAlias

from xdsl.context import Context
from xdsl.dialects import builtin, llvm, memref
from xdsl.dialects.builtin import DYNAMIC_INDEX, IntegerAttr, MemRefType, UnrealizedConversionCastOp, i64
from xdsl.dialects.llvm import GEP_USE_SSA_VAL, GenericCastOp, LLVMPointerType
from xdsl.ir import OpResult, SSAValue
from xdsl.irdl import irdl_op_definition
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.backend.llvm.convert_op import _CAST_OP_NAMES
from xdsl.transforms.convert_memref_to_ptr import ConvertCastOp
from xdsl.utils.hints import isa

DimSizes: TypeAlias = "Mapping[SSAValue, Sequence[int | SSAValue]]"


@irdl_op_definition
class FPTruncOp(GenericCastOp):
    name = "llvm.fptrunc"


# xdsl has no llvm.fptrunc op, and convert_module offers no hook for custom ops
_CAST_OP_NAMES[FPTruncOp] = "fptrunc"

# `memref` -> `llvm.ptr` lowering: replace structured memory ops with raw pointer arithmetic
#
# standard mlir lowers memref through a "descriptor" struct (base ptr, offset, sizes, strides).
# we skip that and go straight to flat pointer math because exo only emits statically-shaped,
# row-major memrefs with no affine maps. the descriptor is unnecessary overhead.
#
# pipeline order matters:
# ----------------------
#     1. extendedconvertmemreftoptr   — rewrite load/store/subview while shape info is still on the memreftype
#     2. rewritememreftypes           — erase memreftype -> llvm.ptr everywhere
#     3. reconcile-unrealized-casts   — clean up identity casts left behind
#
# example (convertloadpattern):
# -----------------------------
#     memref.load %buf[%i, %j] : memref<4x8xf32>
#     =>
#     %stride = llvm.mul %1, 8          ; stride[0] = dim[1] = 8
#     %off0   = llvm.mul %i, %stride    ; i * 8
#     %off1   = llvm.mul %j, %1         ; j * 1
#     %flat   = llvm.add %off0, %off1   ; i*8 + j
#     %bytes  = llvm.mul %flat, 4       ; * sizeof(f32)
#     %ptr    = llvm.inttoptr ...       ; base + bytes
#     %val    = llvm.load %ptr : f32


def _unwrap_i64(val: SSAValue) -> SSAValue:
    # peek through unrealized_cast(x:i64 -> index) to recover the original i64
    if isinstance(val, OpResult) and isinstance(val.op, UnrealizedConversionCastOp):
        inputs = list(val.op.operands)
        if len(inputs) == 1 and inputs[0].type == i64:
            return inputs[0]
    return val


def _iconst(ins, n: int) -> SSAValue:
    return ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result


def _flat_offset(indices: Sequence[SSAValue], rank: int, dim_size_fn, ins) -> SSAValue | None:
    # row-major strides: stride[last]=1, stride[i]=stride[i+1]*dim[i+1]
    strides: list[SSAValue] = [_iconst(ins, 1)] * rank
    for i in range(rank - 2, -1, -1):
        strides[i] = ins(llvm.MulOp(strides[i + 1], dim_size_fn(i + 1))).res

    # flat element offset = sum(index_i * stride_i)
    flat: SSAValue | None = None
    for idx, stride in zip(indices, strides):
        term = ins(llvm.MulOp(_unwrap_i64(idx), stride)).res
        flat = term if flat is None else ins(llvm.AddOp(flat, term)).res
    return flat


def _offset_ptr_gep(base: SSAValue, indices: Sequence[SSAValue], rank: int, dim_size_fn, elem_type, ins) -> SSAValue:
    # compute &base[indices] using gep with element type (for scalar load/store, enables llvm vectorization)
    flat = _flat_offset(indices, rank, dim_size_fn, ins)
    # cast base memref -> llvm.ptr, then add byte offset via ptr-to-int round-trip
    base_ptr = ins(UnrealizedConversionCastOp.get([base], [LLVMPointerType()])).results[0]
    if flat is None:
        return base_ptr
    return ins(llvm.GEPOp(base_ptr, [GEP_USE_SSA_VAL], elem_type, ssa_indices=[flat], inbounds=True)).result


def _offset_ptr_raw(base: SSAValue, indices: Sequence[SSAValue], rank: int, dim_size_fn, elem_size: int, ins) -> SSAValue:
    # compute &base[indices] using ptrtoint/inttoptr (for subview. produces type-agnostic ptr)
    flat = _flat_offset(indices, rank, dim_size_fn, ins)
    base_ptr = ins(UnrealizedConversionCastOp.get([base], [LLVMPointerType()])).results[0]
    if flat is None:
        return base_ptr
    byte_offset = ins(llvm.MulOp(flat, _iconst(ins, elem_size))).res
    ptr_int = ins(llvm.PtrToIntOp(base_ptr)).output
    target_int = ins(llvm.AddOp(ptr_int, byte_offset)).res
    return ins(llvm.IntToPtrOp(target_int)).output


def _dim_size_fn(shape: tuple[int, ...], dims: Sequence[int | SSAValue], ins: Callable) -> Callable[[int], SSAValue]:
    # resolve dimension i to its runtime size. a variable dimension is absent from
    # the memref type, so it comes from the value IRGenerator recorded for it.
    def dim_size(i: int) -> SSAValue:
        if shape[i] != DYNAMIC_INDEX:
            return _iconst(ins, shape[i])
        assert i < len(dims), f"no recorded size for dynamic dimension {i}"
        size = dims[i]
        return _iconst(ins, size) if isinstance(size, int) else size

    return dim_size


def _get_target_ptr(memref_val: SSAValue, memref_type: builtin.MemRefType, indices: list[SSAValue], rewriter: PatternRewriter, dims: DimSizes) -> SSAValue:
    # compute &memref_val[indices] using gep (enables llvm auto-vectorization for scalar load/store)
    shape = memref_type.get_shape()
    ins = rewriter.insert_op
    return _offset_ptr_gep(memref_val, indices, len(shape), _dim_size_fn(shape, dims.get(memref_val, ()), ins), memref_type.element_type, ins)


@dataclass
class ConvertLoadPattern(RewritePattern):
    # memref.load %buf[%i, %j] => ptr arithmetic + llvm.load
    dims: DimSizes

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        memref_type = op.memref.type
        assert isa(memref_type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        target_ptr = _get_target_ptr(op.memref, memref_type, list(op.indices), rewriter, self.dims)
        rewriter.replace_op(op, llvm.LoadOp(target_ptr, op.res.type))


@dataclass
class ConvertStorePattern(RewritePattern):
    # memref.store %val, %buf[%i, %j] => ptr arithmetic + llvm.store
    dims: DimSizes

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        memref_type = op.memref.type
        assert isa(memref_type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        target_ptr = _get_target_ptr(op.memref, memref_type, list(op.indices), rewriter, self.dims)
        rewriter.replace_op(op, llvm.StoreOp(op.value, target_ptr))


@dataclass
class ConvertSubviewPattern(RewritePattern):
    # memref.subview %buf[offsets] => ptr to the start of the slice
    #
    # subview carries both static offsets (baked into the op) and dynamic offsets (ssa values).
    # mlir encodes "this offset is dynamic" by setting the static value to dynamic_index (-1);
    # the actual ssa value then comes from the op.offsets list in order.
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        src_type = op.source.type
        assert isa(src_type, builtin.MemRefType)
        if not isa(src_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        src_shape = src_type.get_shape()
        assert all(d != DYNAMIC_INDEX for d in src_shape), "dynamic source dims in subview not supported"

        ins = rewriter.insert_op

        # merge static_offsets (constants) and dynamic offsets (ssa values) into one list
        all_offsets: list[SSAValue] = []
        dyn_iter = iter(op.offsets)
        for soff in op.static_offsets.iter_values():
            if soff == DYNAMIC_INDEX:
                all_offsets.append(next(dyn_iter))
            else:
                all_offsets.append(_iconst(ins, soff))

        assert isinstance(src_type.element_type, builtin.FixedBitwidthType)
        result_ptr = _offset_ptr_raw(op.source, all_offsets, len(src_shape), lambda i: _iconst(ins, src_shape[i]), src_type.element_type.size, ins)

        # wrap result as memreftype so downstream load/store patterns still see the right shape for stride computation
        rewriter.replace_op(op, UnrealizedConversionCastOp.get([result_ptr], [op.result.type]))


@dataclass(frozen=True)
class ExtendedConvertMemRefToPtr(ModulePass):
    name = "extended-convert-memref-to-ptr"

    dims: DimSizes

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([ConvertCastOp(), ConvertLoadPattern(self.dims), ConvertStorePattern(self.dims), ConvertSubviewPattern()])).rewrite_module(op)


#
# erase memreftype on all remaining values
# (runs after the patterns above consumed shape info)
#
# before:  %x : memref<4x8xf32>
# after:   %x : !llvm.ptr
#


@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    recursive: bool = True

    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()
