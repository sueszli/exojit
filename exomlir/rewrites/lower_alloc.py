from functools import reduce

from xdsl.context import Context
from xdsl.dialects import arith, llvm
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, UnrealizedConversionCastOp, i64
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern

from exomlir.dialects import exo


class ConvertAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
        assert isinstance(op.result.type, MemRefType)

        rewriter.replace_matched_op(
            (
                const_op := arith.ConstantOp(IntegerAttr(reduce(lambda x, y: x * y, op.result.type.get_shape()), i64)),
                alloca_op := llvm.AllocaOp(const_op.result, op.result.type.element_type),
                UnrealizedConversionCastOp.get(alloca_op.res, op.result.type),
            )
        )


class ConvertFreeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.FreeOp, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class EraseMemorySpace(TypeConversionPattern):
    """
    Replaces `ptr_dxdsl.ptr` with `llvm.ptr`.
    """

    @attr_type_rewrite_pattern
    def convert_type(self, typ: MemRefType) -> MemRefType:
        return MemRefType(
            element_type=typ.element_type,
            shape=typ.shape,
            layout=typ.layout,
        )


class LowerAllocPass(ModulePass):
    name = "lower-alloc"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertAllocOp(),
                    ConvertFreeOp(),
                    EraseMemorySpace(recursive=True),
                ]
            )
        ).rewrite_module(m)
