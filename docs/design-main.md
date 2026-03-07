#### 7. `_transform` defines `_rewrite` as a lambda (L582)

```python
_rewrite = lambda patterns: PatternRewriteWalker(GreedyRewritePatternApplier(patterns)).rewrite_module(module)
```

This lambda is used exactly twice (L584-585). It saves one line of repetition but introduces a local name that shadows nothing and reads as unnecessarily terse. Either inline both calls or make it a proper local function with a docstring explaining why it exists.

**Proposed change:** Inline:

```python
PatternRewriteWalker(GreedyRewritePatternApplier([RewriteMemRefTypes()])).rewrite_module(module)
PatternRewriteWalker(GreedyRewritePatternApplier([ConvertVecIntrinsic()])).rewrite_module(module)
```

Or keep it but use `def`:

```python
def rewrite(patterns):
    PatternRewriteWalker(GreedyRewritePatternApplier(patterns)).rewrite_module(module)
```

#### 9. `_memref_load` / `_memref_store` duplicate the zero-index pattern (L195-196, L203-205)

Both create a `ConstantOp(0, i64)` and cast it for scalar access. This 3-line pattern appears twice.

**Proposed change:** Extract a `_zero_index(self) -> list[SSAValue]` helper that returns `[cast(const(0))]`.
