from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.backend.llvm.convert_op import _CAST_OP_NAMES
from xdsl.context import Context
from xdsl.dialects import builtin, llvm, memref
from xdsl.dialects.builtin import DYNAMIC_INDEX, IntegerAttr, MemRefType, UnrealizedConversionCastOp, i64
from xdsl.dialects.llvm import GEP_USE_SSA_VAL, GenericCastOp, LLVMPointerType
from xdsl.ir import BlockArgument, OpResult, SSAValue
from xdsl.irdl import irdl_op_definition
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.transforms.convert_memref_to_ptr import ConvertCastOp
from xdsl.utils.hints import isa


@irdl_op_definition
class FPTruncOp(GenericCastOp):
    name = "llvm.fptrunc"


# xdsl's llvm dialect has fpext but no fptrunc, so teach its llvmlite converter about ours
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
# example (convertloadstorepattern):
# ----------------------------------
#     memref.store %v, %buf[%i, %j] : memref<4x4xf32>
#     =>
#     %c1     = llvm.mlir.constant(1)   ; stride[1] = 1
#     %c4     = llvm.mlir.constant(4)   ; dim[1] = 4
#     %stride = llvm.mul %c1, %c4       ; stride[0] = dim[1] = 4
#     %off0   = llvm.mul %i, %stride    ; i * 4
#     %off1   = llvm.mul %j, %c1        ; j * 1
#     %flat   = llvm.add %off0, %off1   ; i*4 + j
#     %ptr    = llvm.getelementptr inbounds %buf[%flat] : f32
#     llvm.store %v, %ptr : f32


def _loop_upper_bound_as_i64(index: SSAValue) -> SSAValue | None:
    # for dynamic dims: walk index -> block_arg#0 -> find llvm.icmp in the loop header -> extract the bound
    # e.g. `icmp slt %iv, %n` => return %n as the dim size
    if isinstance(index, OpResult) and isinstance(index.op, UnrealizedConversionCastOp):
        inputs = list(index.op.operands)
        iv = inputs[0] if len(inputs) == 1 else index
    else:
        iv = index
    if not isinstance(iv, BlockArgument) or iv.index != 0:
        return None
    for op in iv.block.ops:
        if isinstance(op, llvm.ICmpOp):
            if op.lhs == iv:
                return op.rhs if op.rhs.type == i64 else None
            if op.rhs == iv:
                return op.lhs if op.lhs.type == i64 else None
    return None


def _iconst(ins, n: int) -> SSAValue:
    return ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result


def _base_and_offset(base: SSAValue, indices: Sequence[SSAValue], shape: tuple[int, ...], ins) -> tuple[SSAValue, SSAValue | None]:
    def dim_size(i: int) -> SSAValue:
        # static constant, or the dynamic loop bound the index is derived from
        if shape[i] != DYNAMIC_INDEX:
            return _iconst(ins, shape[i])
        ub = _loop_upper_bound_as_i64(indices[i])
        assert ub is not None
        return ub

    # row-major strides: stride[last]=1, stride[i]=stride[i+1]*dim[i+1]
    strides: list[SSAValue] = [_iconst(ins, 1)] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = ins(llvm.MulOp(strides[i + 1], dim_size(i + 1))).res

    # flat element offset = sum(index_i * stride_i)
    flat: SSAValue | None = None
    for idx, stride in zip(indices, strides):
        # peek through unrealized_cast(x:i64 -> index) to recover the original i64
        if isinstance(idx, OpResult) and isinstance(idx.op, UnrealizedConversionCastOp) and len(idx.op.operands) == 1 and idx.op.operands[0].type == i64:
            idx = idx.op.operands[0]
        term = ins(llvm.MulOp(idx, stride)).res
        flat = term if flat is None else ins(llvm.AddOp(flat, term)).res

    return ins(UnrealizedConversionCastOp.get([base], [LLVMPointerType()])).results[0], flat


class ConvertLoadStorePattern(RewritePattern):
    # memref.load/store %buf[%i, %j] => ptr arithmetic + llvm.load/store
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp | memref.StoreOp, rewriter: PatternRewriter, /):
        memref_type = op.memref.type
        assert isa(memref_type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        ins = rewriter.insert_op
        ptr, flat = _base_and_offset(op.memref, list(op.indices), memref_type.get_shape(), ins)
        if flat is not None:
            # gep with element type (rather than raw byte math) enables llvm auto-vectorization
            ptr = ins(llvm.GEPOp(ptr, [GEP_USE_SSA_VAL], memref_type.element_type, ssa_indices=[flat], inbounds=True)).result
        rewriter.replace_op(op, llvm.LoadOp(ptr, op.res.type) if isinstance(op, memref.LoadOp) else llvm.StoreOp(op.value, ptr))


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
        assert isinstance(src_type.element_type, builtin.FixedBitwidthType)

        ins = rewriter.insert_op

        # merge static_offsets (constants) and dynamic offsets (ssa values) into one list
        dyn_iter = iter(op.offsets)
        all_offsets = [next(dyn_iter) if soff == DYNAMIC_INDEX else _iconst(ins, soff) for soff in op.static_offsets.iter_values()]

        # ptrtoint/inttoptr rather than gep, so the result stays type-agnostic
        ptr, flat = _base_and_offset(op.source, all_offsets, src_shape, ins)
        if flat is not None:
            byte_offset = ins(llvm.MulOp(flat, _iconst(ins, src_type.element_type.size))).res
            ptr_int = ins(llvm.PtrToIntOp(ptr)).output
            ptr = ins(llvm.IntToPtrOp(ins(llvm.AddOp(ptr_int, byte_offset)).res)).output

        # wrap result as memreftype so downstream load/store patterns still see the right shape for stride computation
        rewriter.replace_op(op, UnrealizedConversionCastOp.get([ptr], [op.result.type]))


@dataclass(frozen=True)
class ExtendedConvertMemRefToPtr(ModulePass):
    name = "extended-convert-memref-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([ConvertCastOp(), ConvertLoadStorePattern(), ConvertSubviewPattern()])).rewrite_module(op)


#
# erase memreftype on all remaining values
# (runs after the patterns above consumed shape info)
#
# before:  %x : memref<4x8xf32>
# after:   %x : !llvm.ptr
#


class RewriteMemRefTypes(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()
