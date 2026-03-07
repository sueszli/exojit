## Issues

### Severity: High

#### 1. `_get_target_ptr` and `ConvertSubviewPattern` duplicate stride/offset/ptr arithmetic (L118-150 vs L176-218)

Both functions compute row-major strides and flat byte offsets, then do `ptr_to_int + byte_offset + int_to_ptr`. The logic is nearly identical:

**`_get_target_ptr` (L131-150):**
```python
strides = [iconst(1)] * len(shape)
for i in range(len(shape) - 2, -1, -1):
    strides[i] = ins(llvm.MulOp(strides[i + 1], dim_size(i + 1))).res
flat = None
for idx, stride in zip(indices, strides):
    term = ins(llvm.MulOp(_unwrap_i64(idx), stride)).res
    flat = term if flat is None else ins(llvm.AddOp(flat, term)).res
base_ptr = ins(UnrealizedConversionCastOp.get([memref_val], [LLVMPointerType()])).results[0]
byte_offset = ins(llvm.MulOp(flat, iconst(memref_type.element_type.size))).res
ptr_int = ins(llvm.PtrToIntOp(base_ptr)).output
target_int = ins(llvm.AddOp(ptr_int, byte_offset)).res
return ins(llvm.IntToPtrOp(target_int)).output
```

**`ConvertSubviewPattern` (L197-215):**
```python
strides = [iconst(1)] * len(src_shape)
for i in range(len(src_shape) - 2, -1, -1):
    strides[i] = ins(llvm.MulOp(strides[i + 1], iconst(src_shape[i + 1]))).res
flat = None
for offset, stride in zip(all_offsets, strides):
    term = ins(llvm.MulOp(_unwrap_i64(offset), stride)).res
    flat = term if flat is None else ins(llvm.AddOp(flat, term)).res
base_ptr = ins(UnrealizedConversionCastOp.get([op.source], [LLVMPointerType()])).results[0]
byte_offset = ins(llvm.MulOp(flat, iconst(src_type.element_type.size))).res
ptr_int = ins(llvm.PtrToIntOp(base_ptr)).output
target_int = ins(llvm.AddOp(ptr_int, byte_offset)).res
result_ptr = ins(llvm.IntToPtrOp(target_int)).output
```

**Proposed change:** Extract a shared helper:

```python
def _offset_ptr(
    base: SSAValue,
    indices: list[SSAValue],
    dim_sizes: list[SSAValue],
    elem_size: int,
    ins: Callable,
) -> SSAValue:
    """Compute base_ptr + sum(index_i * stride_i) * elem_size using row-major strides."""
    ...
```

Then both `_get_target_ptr` and `ConvertSubviewPattern` call it with their respective dim-size providers.

#### 2. `iconst` lambda duplicated (L122, L186)

```python
# in _get_target_ptr:
iconst = lambda n: ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result

# in ConvertSubviewPattern:
iconst = lambda n: ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result
```

Identical lambda. Part of the duplication in issue #1.

**Proposed change:** Make `iconst` a module-level helper or part of the extracted `_offset_ptr`.

### Severity: Medium

#### 3. `_loop_ub_as_i64` is a fragile heuristic (L100-115)

```python
def _loop_ub_as_i64(index: SSAValue) -> SSAValue | None:
```

This function reverse-engineers the loop upper bound by tracing: `index -> unrealized_cast -> block_arg -> find ICmpOp in header block -> extract the other operand`. It assumes:
- The index is a loop IV (block arg at position 0).
- The header block has exactly one `ICmpOp` that compares the IV against the upper bound.
- The upper bound is `i64`.

If the loop structure changes (e.g., a different comparison, or the IV isn't at position 0, or there are multiple comparisons), this silently returns `None` and the caller asserts. The function has no way to report *why* it failed.

**Proposed change:** At minimum, add a docstring explaining what IR shape it expects and when it returns `None`. Ideally, carry the upper bound through metadata or a side table rather than reverse-engineering it from the CFG.

#### 4. `_get_target_ptr` uses `dim_size` closure with `assert ub is not None` (L124-129)

```python
def dim_size(i: int) -> SSAValue:
    if shape[i] != DYNAMIC_INDEX:
        return iconst(shape[i])
    ub = _loop_ub_as_i64(indices[i])
    assert ub is not None
    return ub
```

The `assert` will fire with no context if `_loop_ub_as_i64` returns `None`. This is a user-invisible crash in the middle of a rewrite pass.

**Proposed change:** Raise a descriptive error:

```python
if ub is None:
    raise ValueError(f"cannot determine dynamic dim size for axis {i} of {memref_type}")
```

### Severity: Low

#### 5. `RewriteMemRefTypes.recursive = True` is never changed (L254)

```python
@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    recursive: bool = True
```

The `recursive` field is always `True` (set by the dataclass default). It's never assigned any other value. If `TypeConversionPattern` requires it, it should be a class constant. If it doesn't, it's dead code.

**Proposed change:** Check if `TypeConversionPattern` reads `self.recursive`. If yes, set it as a class variable: `recursive: ClassVar[bool] = True`. If no, remove it.

#### 6. `CondBrOp.assembly_format` string concatenation (L80)

```python
assembly_format = "$cond `,` $then_block (`(` $then_arguments^ `:` type($then_arguments) `)`)? `,`" " $else_block ..."
```

This is an implicit string concatenation split across the line. It works, but the space between `","` and `" $else_block"` is easy to miss. Use an explicit `\` continuation or parentheses:

```python
assembly_format = (
    "$cond `,` $then_block (`(` $then_arguments^ `:` type($then_arguments) `)`)? `,`"
    " $else_block (`(` $else_arguments^ `:` type($else_arguments) `)`)? attr-dict"
)
```

#### 7. `ConvertSubviewPattern` asserts static source dims (L183)

```python
assert all(d != DYNAMIC_INDEX for d in src_shape), "dynamic source dims in subview not supported"
```

This is a limitation, not a bug, but it's enforced via `assert` which can be disabled with `-O`. If this constraint matters for correctness, use a proper guard:

```python
if any(d == DYNAMIC_INDEX for d in src_shape):
    return  # skip: dynamic source dims not yet supported
```
