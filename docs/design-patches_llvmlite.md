# Design Review: `xdsl_exo/patches_llvmlite.py`

**Lines:** 93
**Role:** Translates xDSL's LLVM dialect module into llvmlite IR and JIT-compiles it to native code. Bridges the gap between xDSL ops and llvmlite's builder API.

## Summary

The file has four parts:

1. **Type aliases** (L12-14): `ValMap`, `BlockMap`, `PhiMap` — thin dict aliases for readability.
2. **`_convert_type`** (L17-22): Extends xDSL's `_xdsl_convert_type` to handle `IndexType` (→ `i64`).
3. **`_convert_op`** (L25-52): Match-dispatches xDSL ops to llvmlite builder calls. Handles the custom ops (`FNegOp`, `FCmpOp`, `SelectOp`, `BrOp`, `CondBrOp`) that xDSL's default converter doesn't know about.
4. **`_emit_func` + `jit_compile`** (L55-92): Walks the module, forward-declares all functions, emits bodies block-by-block, then runs llvmlite's MCJIT pipeline.

## What's Good

- **Very short file.** 93 lines for a complete JIT backend is impressive. The delegation to `_xdsl_convert_op` (L52) means only the custom ops need explicit handling.
- **Clean phi insertion** (L60). The dict comprehension builds all phi nodes for non-entry blocks in one line, and the val_map merge (L61) makes phis available to all subsequent op conversions.
- **Forward declaration loop** (L74-77). Ensures call sites resolve regardless of function order. Simple and correct.
- **`_convert_type` is minimal** (L17-22). Only overrides `IndexType`; everything else falls through to xDSL's converter.
- **`_emit_func`** (L55-66) is 12 lines. Flat, no nesting beyond the block loop.

## Issues

### Severity: Medium

#### 1. `FCmpOp` case is a single ~300-character line (L32-33)

```python
case FCmpOp():
    pred, is_ordered = {"oeq": ("==", True), "ogt": (">", True), "oge": (">=", True), "olt": ("<", True), "ole": ("<=", True), "one": ("!=", True), "ord": ("ord", True), "ueq": ("==", False), "ugt": (">", False), "uge": (">=", False), "ult": ("<", False), "ule": ("<=", False), "une": ("!=", False), "uno": ("uno", False)}[op.predicate.data]
    val_map[op.res] = (builder.fcmp_ordered if is_ordered else builder.fcmp_unordered)(pred, val_map[op.lhs], val_map[op.rhs])
```

This is hard to read and hard to diff. The dict literal is a lookup table that maps LLVM predicate strings to `(python_pred, is_ordered)` pairs.

**Proposed change:** Extract the table to a module-level constant:

```python
_FCMP_PREDICATES: dict[str, tuple[str, bool]] = {
    "oeq": ("==", True),
    "ogt": (">", True),
    "oge": (">=", True),
    "olt": ("<", True),
    "ole": ("<=", True),
    "one": ("!=", True),
    "ord": ("ord", True),
    "ueq": ("==", False),
    "ugt": (">", False),
    "uge": (">=", False),
    "ult": ("<", False),
    "ule": ("<=", False),
    "une": ("!=", False),
    "uno": ("uno", False),
}
```

Then:
```python
case FCmpOp():
    pred, is_ordered = _FCMP_PREDICATES[op.predicate.data]
    cmp_fn = builder.fcmp_ordered if is_ordered else builder.fcmp_unordered
    val_map[op.res] = cmp_fn(pred, val_map[op.lhs], val_map[op.rhs])
```

#### 2. Phi insertion logic is duplicated between `BrOp` and `CondBrOp` handlers (L36-49)

```python
case BrOp():
    cur = builder.block
    for a, v in zip(op.successor.args, op.operands):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur)
    builder.branch(block_map[op.successor])

case CondBrOp():
    cur = builder.block
    for a, v in zip(op.successors[0].args, op.then_arguments):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur)
    for a, v in zip(op.successors[1].args, op.else_arguments):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur)
    builder.cbranch(...)
```

The 3-line `for a, v in zip(...): if a in phi_map: phi_map[a].add_incoming(...)` pattern appears three times (once in BrOp, twice in CondBrOp).

**Proposed change:** Extract a helper:

```python
def _add_phis(phi_map: PhiMap, val_map: ValMap, block_args, operands, cur_block):
    for a, v in zip(block_args, operands):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur_block)
```

Then:
```python
case BrOp():
    _add_phis(phi_map, val_map, op.successor.args, op.operands, builder.block)
    builder.branch(block_map[op.successor])

case CondBrOp():
    _add_phis(phi_map, val_map, op.successors[0].args, op.then_arguments, builder.block)
    _add_phis(phi_map, val_map, op.successors[1].args, op.else_arguments, builder.block)
    builder.cbranch(...)
```

### Severity: Low

#### 3. `jit_compile` returns `ExecutionEngine` but callers may not use it (L69)

```python
def jit_compile(module: ModuleOp) -> llvm_binding.ExecutionEngine:
```

In `main.py:L632`, the return value is discarded:
```python
jit_compile(module)
return
```

The function both compiles *and* returns the engine. If the intent is to JIT and immediately discard (e.g., for testing compilation), the return value is dead. If the intent is for callers to call functions on the engine, the current `main()` doesn't do that.

This isn't a bug — returning the engine is useful for programmatic callers — but the CLI path discards it, which means the compiled code is never actually *called*.

**Proposed change:** Either:
- Add a `# TODO: invoke the compiled function` comment in `main.py`, or
- Have `jit_compile` optionally invoke a function, or
- Document that `jit_compile` is a library API, not a CLI action.

#### 4. `_convert_type` fallback to `_xdsl_convert_type` could mask errors (L21-22)

```python
case _:
    return _xdsl_convert_type(mlir_type)
```

If `_xdsl_convert_type` doesn't handle a type, it presumably raises. But the error message won't mention that this was called from `patches_llvmlite`, making debugging harder.

**Proposed change:** This is minor. A comment noting the delegation chain is sufficient.

#### 5. No error handling for `llvm_binding.parse_assembly` (L86)

```python
llvm_mod = llvm_binding.parse_assembly(str(llvm_module))
```

If the generated LLVM IR is malformed, `parse_assembly` raises a generic llvmlite error. The `str(llvm_module)` output is lost — there's no way to inspect what was generated.

**Proposed change:** Wrap in a try/except that prints the IR on failure:

```python
ir_text = str(llvm_module)
try:
    llvm_mod = llvm_binding.parse_assembly(ir_text)
except RuntimeError as e:
    raise RuntimeError(f"llvmlite failed to parse generated IR:\n{ir_text}") from e
```
