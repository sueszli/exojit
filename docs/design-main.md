# Design Review: `xdsl_exo/main.py`

**Lines:** 642
**Role:** Core of the compiler. Lowers Exo's LoopIR to xDSL's LLVM dialect, runs optimization/lowering passes, and provides the CLI entry point.

## Summary

The file has three logical sections:

1. **Free helper functions** (L37-89): `_is_mutated`, `_window_access`, `_coerce_arg`, `_to_index_list` — pure utilities used by the IR generator.
2. **`IRGenerator` class** (L92-555): A stateful tree-walk that pattern-matches LoopIR nodes and emits xDSL ops. Manages a builder, symbol table, and type table.
3. **Pipeline + CLI** (L558-642): `_context`, `_transform`, `compile_procs`, `main` — wires the generator into a pass pipeline and exposes a CLI.

## What's Good

- **Pattern matching everywhere.** `_type`, `_expr`, `_stmt` use `match`/`case` for exhaustive dispatch — clear, flat, and easy to extend.
- **`_tmp_state` context manager** (L126-140). Cleanly saves/restores builder + symbol/type tables for nested scopes (procedure bodies). Single point of control for scope management.
- **Thin methods.** Most `_stmt_*` and `_expr_*` methods are 5-15 lines. Easy to read in isolation.
- **Table-driven op selection** (L285-286). `float_ops` and `int_ops` dicts avoid long if/elif chains.
- **`_to_index_list`** (L85-89). Neatly wraps the split/cast/rejoin dance for SubviewOp operands.
- **`_coerce_arg`** (L62-82). The scalar-passed-by-ref and shape-mismatch logic is well-isolated from `_stmt_call`.

## Issues

### Severity: High

#### 1. `_shape` is a dual-purpose function (L172-192)

```python
def _shape(self, tensor: T.Tensor, *, emit: bool = False) -> list[int | SSAValue]:
```

When `emit=False` it returns `int | DYNAMIC_INDEX` (for type declarations). When `emit=True` it returns `int | SSAValue` (for runtime arithmetic). The caller gets a `list[int | SSAValue]` regardless, but the actual contents are type-incompatible. This makes it impossible to reason about the return type statically and forces callers to handle both shapes.

**Proposed change:** Split into two functions:

```python
def _shape_static(self, tensor: T.Tensor) -> list[int]:
    """For MemRefType declarations. Variable dims -> DYNAMIC_INDEX."""
    ...

def _shape_dynamic(self, tensor: T.Tensor) -> list[int | SSAValue]:
    """For runtime arithmetic. Variable dims -> live SSA values."""
    ...
```

#### 2. `_stmt_for` manually saves/restores `symbol_table` (L415-427)

```python
parent_syms = self.symbol_table
self.symbol_table = ScopedDict(self._syms)
...
self.symbol_table = parent_syms
```

This duplicates what `_tmp_state` already does, but without restoring `builder` or `type_table`, and without a `try/finally` guard. If any statement in the loop body throws, the symbol table leaks.

**Proposed change:** Use `_tmp_state` and just override the builder afterwards, or extend `_tmp_state` to accept an `inherit_syms` option.

#### 3. `assert` used for user-facing validation in `main()` (L623-627)

```python
assert src.is_file() and src.suffix == ".py"
...
assert isinstance(library, list)
assert all(isinstance(proc, Procedure) for proc in library)
```

These assertions are silenced by `python -O` and produce unhelpful tracebacks for users. The first one doesn't even tell you *which* check failed.

**Proposed change:** Use `parser.error()` for the file check and raise `ValueError` with a message for the others:

```python
if not src.is_file() or src.suffix != ".py":
    parser.error(f"expected a .py file, got: {src}")
```

### Severity: Medium

#### 4. `compile_procs` uses set-add-in-comprehension trick (L603)

```python
seen: set[str] = set()
unique_procs = [proc for proc in all_procs if not (proc.name in seen or seen.add(proc.name))]
```

`seen.add()` returns `None`, so `None or True` is `True` — the `not` flips it. This is a clever side-effect-in-comprehension trick that's notoriously hard to read.

**Proposed change:** Use a plain loop or `dict.fromkeys`:

```python
unique_procs = list({proc.name: proc for proc in all_procs}.values())
```

#### 5. `_expr_read` has 4 sequential if/return guards (L238-249)

```python
if not isinstance(operand.type, MemRefType):
    return operand
if isinstance(read.type, (T.Window, T.Tensor)):
    return operand
if operand.type == self._type(read.type):
    return operand
return self._memref_load(operand, idx)
```

The precedence between these guards is non-obvious. The third guard (`operand.type == self._type(read.type)`) catches scalar-memref reads where the types already match — but that sounds like it overlaps with guard 1. A comment explaining the ordering would help, or better: merge guards 1 and 3 since they both return `operand` when "no load is needed."

**Proposed change:** Add a single comment block explaining the decision tree, or restructure as:

```python
needs_load = (
    isinstance(operand.type, MemRefType)
    and not isinstance(read.type, (T.Window, T.Tensor))
    and operand.type != self._type(read.type)
)
return self._memref_load(operand, idx) if needs_load else operand
```

#### 6. `_build_alloc` uses `reduce(lambda)` (L445)

```python
total_elements = reduce(lambda x, y: x * y, shape)
```

`math.prod` exists since Python 3.8 and is clearer:

```python
total_elements = math.prod(shape)
```

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

### Severity: Low

#### 8. `_cmp_binop` is `@staticmethod` that takes `emit` callable (L264-274)

```python
@staticmethod
def _cmp_binop(lhs, rhs, op, emit):
```

This is the only method on `IRGenerator` that takes an `emit` parameter instead of using `self._emit`. It's static because it needs to work without `self`, but the pattern is inconsistent with every other method. If it were a module-level function (like `_is_mutated`, `_window_access`, etc.) the inconsistency would be explicit and intentional.

**Proposed change:** Move to module level as `_cmp_binop(lhs, rhs, op, emit)`.

#### 9. `_memref_load` / `_memref_store` duplicate the zero-index pattern (L195-196, L203-205)

Both create a `ConstantOp(0, i64)` and cast it for scalar access. This 3-line pattern appears twice.

**Proposed change:** Extract a `_zero_index(self) -> list[SSAValue]` helper that returns `[cast(const(0))]`.

#### 10. `_context()` is `@cache`d (L558)

`@cache` on a no-arg function is effectively a module-level singleton. This is fine, but the cache is never invalidated and holds a reference to the Context forever. If this matters (e.g., tests creating fresh contexts), `@cache` is the wrong tool — a module-level variable would be more explicit.

#### 11. `os.makedirs` + `Path.write_text` (L640-641)

```python
os.makedirs(dst.parent, exist_ok=True)
dst.write_text(str(module))
```

Since `dst` is already a `Path`, use `dst.parent.mkdir(parents=True, exist_ok=True)` to stay within the pathlib API.
