from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern


class ConvertFuncOp(RewritePattern):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.name.startswith(self.prefix):
            return

        op.sym_name = StringAttr(f"{self.prefix}_{op.sym_name.data}")


class AddPrefixPass(ModulePass):
    name = "add-prefix"

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertFuncOp(self.prefix),
                ]
            ),
        ).rewrite_module(m)
