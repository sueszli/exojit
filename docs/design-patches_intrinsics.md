# Design Review: `xdsl_exo/patches_intrinsics.py`

**Lines:** 260
**Role:** Rewrite pattern that lowers `vec_*` and `mm256_*` intrinsic calls (`llvm.CallOp`) into concrete LLVM/vector dialect ops. Handles both plain and prefix-masked variants for f32x8 and f64x4.

## Summary

The file defines:

1. **Type aliases** (L44-48): `MaskResult`, `BuildResult`, `Builder`, `MaskFn`, `Handler` — describe the signature contracts for the builder/handler system.
2. **Mask generation** (L51-74): `_make_mask` and its specializations (`_mask_f32x8`, `_mask_f64x4`, `_mask_f64x4_ext`).
3. **Core builder functions** (L77-157): `_build_copy`, `_build_abs`, `_build_neg`, `_build_add`, `_build_mul`, `_build_fma`, etc. — each one takes pointers + vec_type and returns `(ops, result_value)`.
4. **Handler factories** (L160-188): `_plain_handler`, `_pfx_handler`, `_reduce_handler` — wrap builders into `Handler` callables that include the final store/masked-store.
5. **`mm256_*` standalone builders** (L191-210): Legacy Intel intrinsic handlers.
6. **`_make_intrinsics` table** (L213-246): Builds the complete dispatch dict at import time.
7. **`ConvertVecIntrinsic`** (L249-259): The actual `RewritePattern` — a thin lookup + dispatch.

## What's Good

- **Table-driven dispatch** (L213-246). The nested loop over `(name, builder, pfx_builder, f64_mask)` x `(suffix, vec_type, mask_fn)` generates all 52+ entries without repetition. Adding a new intrinsic is a one-line table entry.
- **Builder/handler separation.** Builders produce ops + result; handlers add the store strategy (plain vs. masked). Clean separation of concerns.
- **`_make_intrinsics` runs once at class load** (L250). The dict is a `ClassVar`, so lookup in `match_and_rewrite` is O(1) with no repeated construction.
- **Prefix mask logic is well-documented** (L10-41). The file header clearly explains the naming convention and the plain vs. prefix lowering with ASCII examples.
- **`ConvertVecIntrinsic.match_and_rewrite`** (L253-259) is 6 lines. The pattern is trivially auditable.

## Issues

### Severity: High

#### 1. Builder functions are heavily duplicated (L77-157)

`_build_copy`, `_build_add`, `_build_mul`, `_build_neg`, `_build_add_red`, `_build_fma`, `_build_fma_red` all follow the same pattern:

```python
def _build_X(dst, *srcs, vec_type):
    loads = [llvm.LoadOp(src, vec_type) for src in srcs]
    result = SomeOp(*[l.dereferenced_value for l in loads])
    return [*loads, result], result.res
```

The only thing that varies is:
- How many sources to load (1, 2, or 3)
- Which op to apply (`FAddOp`, `FMulOp`, `FNegOp`, `FMAOp`, etc.)
- Whether `dst` is also loaded (reduction variants)

**Proposed change:** A generic builder that takes a "core op" factory:

```python
def _build_generic(
    dst: SSAValue,
    *srcs: SSAValue,
    vec_type: VectorType,
    op_fn: Callable[[list[ir.Value]], Operation],
    load_dst: bool = False,
) -> BuildResult:
    loads = []
    if load_dst:
        loads.append(llvm.LoadOp(dst, vec_type))
    loads.extend(llvm.LoadOp(s, vec_type) for s in srcs)
    values = [l.dereferenced_value for l in loads]
    result_op = op_fn(values)
    return [*loads, result_op], result_op.res  # or .result depending on op
```

Then:
```python
_build_add = partial(_build_generic, op_fn=lambda vs: llvm.FAddOp(*vs))
_build_mul = partial(_build_generic, op_fn=lambda vs: llvm.FMulOp(*vs))
_build_add_red = partial(_build_generic, op_fn=lambda vs: llvm.FAddOp(*vs), load_dst=True)
```

This would eliminate ~60 lines of near-identical code.

#### 2. `Builder` type alias shadows xDSL's `Builder` (L46)

```python
Builder: TypeAlias = Callable[..., BuildResult]
```

`xdsl.builder.Builder` is used elsewhere in the project (main.py L18). If someone adds an import of `xdsl.builder.Builder` to this file, they'll get a confusing name collision.

**Proposed change:** Rename to `IntrinsicBuilder` or `BuilderFn`.

### Severity: Medium

#### 3. `_build_abs_pfx` has an implicit side-effect (L90-96)

```python
def _build_abs_pfx(dst, src, *, vec_type):
    load = llvm.LoadOp(src, vec_type)
    fabs = FAbsOp(load.dereferenced_value, vec_type)
    return [load, fabs, llvm.StoreOp(load.dereferenced_value, dst)], fabs.result
```

The returned ops list includes a `StoreOp` that copies `src` into `dst` for *all* lanes — this is a setup step so the subsequent `MaskedStoreOp` (from `_pfx_handler`) only overwrites the active prefix lanes. But this side-effect is buried inside the builder's return value and is easy to miss. The comment on L91-93 helps, but the pattern is unique to this one builder.

**Proposed change:** Move the "pre-copy for tail lanes" logic into `_pfx_handler` itself, parameterized by a flag. That way the side-effect is visible at the handler level where the masked store lives:

```python
def _pfx_handler(builder, vec_type, mask_fn, *, pre_copy_src: bool = False):
    def handle(args):
        lane_count, dst, *srcs = args
        mask_ops, mask = mask_fn(lane_count)
        if pre_copy_src:
            # preserve tail lanes: dst[:] = src[:] before masked overwrite
            pre_ops = [llvm.LoadOp(srcs[0], vec_type), llvm.StoreOp(..., dst)]
        else:
            pre_ops = []
        core_ops, result = builder(dst, *srcs, vec_type=vec_type)
        return (*mask_ops, *pre_ops, *core_ops, MaskedStoreOp(result, dst, mask))
    return handle
```

Then `_build_abs_pfx` can be deleted — `_build_abs` is sufficient.

#### 4. `mm256_*` functions are structurally different from `vec_*` (L191-210)

The `mm256_*` functions return `tuple[Operation, ...]` directly (not `BuildResult`), and are wrapped with lambdas in the table (L241-244):

```python
entries["mm256_storeu_ps"] = lambda args: _build_mm256_storeu_ps(*args)
entries["mm256_fmadd_ps"] = lambda args: _build_mm256_fmadd_ps(*args)
```

Meanwhile `vec_*` entries use the `_plain_handler`/`_pfx_handler` factories. This means two different patterns coexist in the same dispatch table.

**Proposed change:** Rewrite `mm256_*` as `Builder` functions + `_plain_handler`, or at minimum add a comment in `_make_intrinsics` explaining the two conventions.

### Severity: Low

#### 5. `_mask_f32x8` and `_mask_f64x4` are one-liners (L65-70)

```python
def _mask_f32x8(lane_count):
    return _make_mask(lane_count, 8)

def _mask_f64x4(lane_count):
    return _make_mask(lane_count, 4)
```

These exist only to be passed as `MaskFn` callbacks. They could be replaced with `partial`:

```python
_mask_f32x8 = partial(_make_mask, n_lanes=8)
_mask_f64x4 = partial(_make_mask, n_lanes=4)
```

But the current form is fine for readability — this is a nit.

#### 6. `_build_broadcast` and `_build_zero` don't load from `dst` (L148-157)

`_build_broadcast` ignores its `dst` argument entirely (the handler's `_plain_handler` uses it for the store). `_build_zero` takes `dst` as its only positional argument. The signature `(dst, *, vec_type)` is misleading — `dst` is unused inside the builder.

**Proposed change:** Document that `dst` is consumed by the handler, not the builder. Or make it explicit by not passing it:

```python
_build_zero = lambda *, vec_type: ([...], zero.result)
```

But this would break the uniform `Builder` signature. A comment is sufficient.
