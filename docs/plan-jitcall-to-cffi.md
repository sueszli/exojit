# plan: drop `exojit/jitcall.c`, call through `cffi`, share the seam with xDSL

Status: **ready to implement.** xDSL's JIT uses stdlib `ctypes` (§1), but a
straight ctypes port costs 14.5x on the call boundary (§4). `cffi` reaches
**1.12x -- parity** with the C extension, and is the only pure-Python option
that preserves every safety check `jitcall.c` performs. §9 compares all five
options. Since exojit's author is also an xDSL author, the recommendation is to
land an FFI seam upstream (§1.1) and have exojit consume it rather than fork.

This document has been through three adversarial reviews; §10 records what they
broke.

## 0. summary

Every call to a jitted kernel crosses from Python to machine code. Something has
to translate a numpy array into a raw address, and check it is safe to use,
each time:

```
Python array ──► translator ──► machine code
                      ^
              this is exojit/jitcall.c
```

That translator costs **142 ns** today. Small kernels feel it; large ones do not.
The goal is to delete the C file without paying for it.

| option | vs today | |
| --- | --- | --- |
| `jitcall.c` today | 1.00x | the C file we want to delete |
| **cffi + `bind()`** | **1.12x** | **recommended** |
| ctypes + `bind()` | 2.00x | no new dependency, loses read-only support |
| cffi, every call | 6.3x | no API change needed |
| ctypes, every call | 14.5x | what xDSL does today |

`bind()` translates once and then calls many times, instead of translating on
every call. That is where the speed comes from, and it is why the recommended
option lands within 16 ns of hand-written C.

## 1. what xDSL's JIT actually uses

xDSL grew a JIT subsystem over the last two weeks. At `xdslproject/xdsl@419203f`
(`jit: extract the LLVM backend (#6397)`, HEAD at time of writing) it lives in
`xdsl/jit/` and is 500 lines:

| file | role |
| --- | --- |
| `xdsl/jit/context.py` | `JITContext` (the `@ctx.jit(...)` decorator) + `JITBackend` ABC |
| `xdsl/jit/function.py` | `RawJITFunc` (a bound ctypes callable), `WrappedJITFunc`, `wrap_jit_func` |
| `xdsl/jit/c_type_context.py` | `CTypeContext`: registry mapping xDSL type attributes to ctypes types |
| `xdsl/jit/py_type_context.py` | `PyTypeContext` / `TypeMap`: Python value to ctypes value marshalling |
| `xdsl/jit/llvm/backend.py` | `LLVMJITBackend`: lowers, then llvmlite MCJIT, then binds the symbol |
| `xdsl/jit/llvm/c_type_context.py` | `llvm.ptr -> c_void_p`, `llvm.void -> None`, `to_c_func_type` |

History, all extracted out of the filecheck prototype
`tests/filecheck/projects/pyjit/two_plus_two.py`:
`8b775ca` (`backend: (llvm) add ctypes conversion registry (#6183)`, 2026-08-13)
-> `ce55c84` (`JIT: move the ctypes type converter out of the pyjit prototype (#6357)`, 2026-08-14)
-> `e844274` (`jit: extract JIT callables (#6367)`, 2026-08-21)
-> `0838adc` (`jit: extract the JIT context and backend interface (#6391)`, 2026-08-24)
-> `419203f` (2026-08-27).

**The answer to "what technology": `ctypes`, from the Python standard library.**
Nothing else. The native calling boundary is `ctypes.CFUNCTYPE`, constructed from
IR types rather than hand-written:

```python
# xdsl/jit/c_type_context.py
def to_c_func_type(self, inputs, output):
    return ctypes.CFUNCTYPE(self.to_ctype(output), *(self.to_ctype(arg) for arg in inputs))

# xdsl/jit/llvm/backend.py:52,56
func_ptr = engine.get_function_address(symbol)
c_types_fn = c_func_type(func_ptr)
```

Two things xDSL deliberately does *not* use: no C extension, and no `cffi`.
Compilation is still llvmlite MCJIT (`llvmlite~=0.47.0`, declared under the
optional `llvm` extra) -- the same approach exojit already takes. The only piece
exojit does differently is the call boundary.

To use the registry you need **both** halves; exo `size`/`index` arguments lower
to `i64`, which is `builtin.IntegerType` and lives in the builtin registry:

```python
ctx = CTypeContext()
register_builtin_ctypes(ctx)   # xdsl/jit/c_type_context.py -- i64, f32, f64, ...
register_llvm_ctypes(ctx)      # xdsl/jit/llvm/c_type_context.py -- ptr, void
```

Omitting `register_builtin_ctypes` raises `JITException: No ctypes mapping for
type: i64` on every dynamically-shaped kernel.

## 1.1 xDSL's abstraction is already FFI-agnostic

`xdsl/jit/function.py` types the calling boundary as **structural Protocols**,
not as ctypes classes:

```python
class CFunc(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class CFuncType(Protocol):
    @overload
    def __call__(self, address: int, /) -> CFunc: ...
```

A cffi function pointer -- `ffi.cast("void(*)(void*, void*)", addr)` -- satisfies
`CFunc` as it stands, and a thin factory around `ffi.cast` satisfies `CFuncType`.
The design is 90% of the way to being FFI-agnostic already. Three places break
the promise those Protocols make:

1. `wrap_jit_func` (`xdsl/jit/function.py`) compares
   `raw_func.c_func_type != expected_c_func_type` -- ctypes object equality.
2. `CTypeContext.to_c_func_type` (`xdsl/jit/c_type_context.py`) hardcodes
   `ctypes.CFUNCTYPE`.
3. `FuncTypeMap.c_func_type` (`xdsl/jit/py_type_context.py`) hardcodes it again.

**Proposed upstream change: an `FFIBackend` seam beneath `JITBackend`**, with a
`ctypes` implementation as the default (no new core dependencies -- xDSL core
currently depends only on `immutabledict`, `ordered-set`, `typing-extensions`)
and a `cffi` implementation under a new optional extra, alongside the existing
`llvm` extra. Each implementation owns type-attribute-to-FFI-type mapping,
prototype construction, and prototype comparison. This is defensible on xDSL's
own merits -- it makes `RawJITFunc` honest about what its Protocols already
claim -- and is independent of exojit.

## 2. what exojit does today

`exojit/jitcall.c` (382 lines, built as the `exojit.jitcall` extension module by
`setup.py`) implements a single type, `JitFunc`, constructed as
`JitFunc(address, engine, arg_kinds)`:

- holds a strong ref to the llvmlite `ExecutionEngine` so the code pages outlive
  the callable;
- stores a precomputed `arg_kinds` byte per argument -- `ARG_INT`, `ARG_PTR_RO`,
  `ARG_PTR_RW` -- produced by `_jit_arg_kinds()` (`exojit/main.py:875`), which
  runs a LoopIR write analysis to decide which tensor args are written;
- implements `tp_vectorcall`, marshalling each Python argument into an
  `intptr_t`: ints via `PyLong_AsSsize_t`; pointers via `PyLong_AsVoidPtr` for
  raw addresses, or `PyObject_GetBuffer` for buffer objects with
  `PyBUF_C_CONTIGUOUS | (kind == ARG_PTR_RW ? PyBUF_WRITABLE : 0)` --
  writability is requested *only* for written args (`jitcall.c:42`);
- **holds every `Py_buffer` export across the native call**, releasing them only
  after the dispatch switch returns. This pins each operand for the duration of
  the kernel;
- **holds the GIL** across the call -- there is no `Py_BEGIN_ALLOW_THREADS`;
- checks arity exactly (`"JitFunc expected %zd arguments, got %zd"`) and rejects
  keyword arguments (`"JitFunc does not accept keyword arguments"`);
- dispatches through a `case 0` .. `case 64` switch of correctly-typed function
  pointer casts, avoiding UB from calling through a mismatched prototype.

It does **not** check dtype: `PyBUF_C_CONTIGUOUS` does not request
`PyBUF_FORMAT`, so an `f64` or `object` array into an `f32` kernel is accepted
today. That is a pre-existing hole and matters for pricing `ndpointer` (§4).

Callers:

- `exojit/main.py:1083` constructs it; `_jit_wrap` (`:1028`) layers a cffi-based
  Python-list marshaller on top for `jit(proc)`, and decides writability with
  `arg_kinds[i] == 2` -- a **second positional consumer** of the encoding;
- `jit(proc, raw=True)` returns
  `lambda *args, **kwargs: raw_fn(*_resolve_jit_args(arg_names, args, kwargs))`,
  which resolves kwargs to positions by exo arg name and has **no arity check of
  its own** -- `JitFunc` is the only arity guard on the raw path;
- **28 `jit(..., raw=True)` sites across 24 files in `benchmarks/kernels/*.py`**
  (all `*_exo.py` / `*_neon.py`), every one passing numpy arrays via
  `benchmarks/run.py` (e.g. `fn_exo(C_exo, A, B)` at `run.py:130`) inside
  `timeit`. The other 48 kernel files (`_numpy`, `_numba`, `_torch`, `_jax`)
  never touch exojit;
- `examples/microgpt/train.py:395` has exactly one `@jit(..., raw=True)`;
- `tests/e2e/test_jitcall.py` asserts `type(fn._raw).__name__ == "JitFunc"`
  twice (`:81`, `:87`), that non-C-contiguous buffers raise `TypeError` matching
  `"C-contiguous buffer"` (`:96`, the only `raw=True` test), and that kwargs
  raise `TypeError` matching `"keyword"`. `test_jit_exposes_raw_entrypoint`
  (`:84`) exercises both numpy arrays and `arr.ctypes.data` ints but does so
  through `jit(copy4)._raw`, i.e. the **non**-raw wrapper.

## 3. what makes the swap attractive

- exojit's LLVM function signature is already exactly what xDSL's converter
  expects. `to_mlir(proc)` emits an `llvm.FuncOp` whose inputs follow the proc's
  **declared parameter order**; since exo declares `size` params first,
  `copy_n(N: size, dst, src)` yields
  `!llvm.func<void (i64, !llvm.ptr, !llvm.ptr)>`. A type registry maps that
  straight to a prototype with no bespoke code -- `CFUNCTYPE(None, c_int64,
  c_void_p, c_void_p)` for ctypes, `"void(*)(int64_t, void*, void*)"` for cffi.
- That makes the ABI half of `_jit_arg_kinds` (int vs pointer) redundant -- it is
  currently re-derived from LoopIR when the LLVM dialect already states it.
- No compiler toolchain needed to install exojit; the wheel becomes pure-Python.
- The README already states the project is being ported to xDSL. Sharing xDSL's
  JIT boundary is the same direction of travel.

**The entry function is not the only `llvm.FuncOp`, and often not the first.**
`to_mlir` also emits `malloc` / `free` declarations; a proc that calls a subproc
emits the callee first; a `par()` proc emits `__omp_outlined_0` first and a
*variadic* `__kmpc_fork_call` (which xDSL's `to_c_func_type` rejects outright). Any
implementation must select by `sym_name.data == proc.name()`, as
`xdsl/jit/llvm/backend.py:105` does via `SymbolTable.lookup_symbol`, and assert
`len(func_type.inputs) == len(arg_kinds)`.

## 4. the cost, measured

Measured on this machine (darwin/arm64 M2 Pro, py3.14.5, numpy 2.4.2,
llvmlite **0.46.0** per `uv.lock`, cffi 2.0.0), against the real 3-argument `add`
kernel at n=256, `min(timeit.repeat(number=20000, repeat=9))`. Every row below is
a **faithful wrapper class** performing the full check set -- arity, contiguity,
writability -- not an inline expression; the first draft of this plan measured
inline expressions and understated the cost by 35-95%.

| call boundary | ns/call | vs baseline |
| --- | --- | --- |
| `jitcall.c` `JitFunc`, numpy arrays | **142** | 1.0x |
| **cffi, `bind()` steady state** | **158** | **1.12x** |
| ctypes, `bind()` steady state | 283 | 2.0x |
| cffi, per-call marshalling (lean) | 913 | 6.3x |
| cffi, per-call marshalling (generic loop) | 1140 | 8.0x |
| ctypes, per-call marshalling | 2053 | 14.5x |
| ctypes, ndarray fast path (`flags` + `__array_interface__`) | 3179 | 22.4x |
| `numpy.ctypeslib.ndpointer` argtypes | 3236 | 22.8x |

Primitives, per array argument:

| operation | ns | what it checks |
| --- | --- | --- |
| `ffi.from_buffer(arr, require_writable=...)` | **146** | **contiguity + writability + read-only + pins the export** |
| `memoryview(arr)` | 100 | contiguity + readonly flag only, no address |
| `arr.flags.c_contiguous` | 40 | contiguity only |
| `ctypes.addressof(ctypes.c_char.from_buffer(arr))` | 229 | address only, **fails on read-only** |
| `arr.ctypes.data` | 645 | address only |
| `hasattr(arr, "ctypes")` | 627 | nothing |
| `arr.__array_interface__["data"][0]` | 690 | address only |

`ffi.from_buffer` is the whole reason cffi wins. It returns the pointer *and*
performs both checks *and* holds the buffer export for the lifetime of the
returned cdata, in a single call cheaper than ctypes' address extraction alone.
The ctypes path needs `memoryview` (100) **plus** `c_char.from_buffer` (229) to
get less safety.

Two counter-intuitive ctypes results, retained because they explain why the
naive designs are the slow ones: `c_char.from_buffer` is 2.8x cheaper than
`arr.ctypes.data`, because numpy rebuilds its `_ctypes` helper on every attribute
access -- merely *probing* `hasattr(obj, "ctypes")` costs as much as reading it;
and a numpy-specialised fast path is **slower** than the generic buffer-protocol
path, for the same reason.

### the read-only problem, and why it decides the FFI choice

`ctypes.c_char.from_buffer` requires a writable buffer **unconditionally**. It
raises `TypeError: underlying buffer is not writable` on `bytes`, read-only
`mmap`, numpy scalars, and any `setflags(write=False)` array -- all of which
`jitcall.c` accepts today in `ARG_PTR_RO` slots. `from_buffer_copy` is not a
substitute: it returns a different address, so the kernel would read a copy that
is freed before it runs. **There is no pure-ctypes way to obtain a read-only
buffer's address.**

cffi has no such gap. Verified:

```
ffi.from_buffer(read_only_ndarray)              -> OK
ffi.from_buffer(b"abcd")                        -> OK
ffi.from_buffer(ro, require_writable=True)      -> ValueError: buffer source array is read-only
ffi.from_buffer(non_contiguous)                 -> ValueError: ndarray is not C-contiguous
```

So cffi is not merely faster; it is the only pure-Python option that reaches
feature parity with `jitcall.c` at all.

### end-to-end

Cost is per array argument, so it scales with arity: `adam` takes 10 arguments,
`matmul` / `matvec` / `weighted_sum` take 3. GFLOP/s retained on the `add` row,
for the two candidate designs:

| n | C ext | cffi `bind()` | cffi per-call | ctypes per-call |
| --- | --- | --- | --- | --- |
| 256 | 100% | ~99% | ~16% | **4.5%** |
| 4096 | 100% | ~99% | ~31% | **11.1%** |
| 65536 | 100% | ~100% | 87% | 69.5% |
| 1048576 | 100% | ~100% | ~99% | 95.0% |

(cffi columns scaled from the measured per-call deltas; the ctypes column was
measured end-to-end. The project's own `bench = min(timeit.repeat(number=1,
repeat=50))` carries +-10% at n >= 262144, so large-n deltas are not reliably
reviewable either way.)

`examples/microgpt` is **not** affected under any option: `train.py` makes
exactly one JIT call for the entire 1000-step run (the `seq(0, NUM_STEPS)` loop
is *inside* the jitted proc), so even the worst case is ~0.12%.

## 5. design

Delete `exojit/jitcall.c` and `setup.py`'s `ext_modules`. Add `exojit/jitcall.py`
exposing a `JitFunc` with the same construction signature and call semantics,
implemented over `cffi`, with the prototype derived from the IR.

```
llvm.FuncOp for SymbolTable.lookup_symbol(module, proc.name())
        |
        v
FFI-type registry (ctypes: CTypeContext / cffi: cdecl strings)   <- the §1.1 seam
        |
        v
ffi.cast("void(*)(int64_t, void*, void*)", address)              <- declared-order inputs
        |
        v
exojit.jitcall.JitFunc  -- arity gate, type gate, from_buffer, engine keepalive
        |
        +-- .bind(*args) -> zero-marshalling closure for hot loops
```

`cffi` is already a declared dependency of exojit (`pyproject.toml`) and is
already used by `_jit_wrap` for the Python-list marshalling path, so this adds
no new dependency. It is a wheel, not a C source file in this repo.

### per-argument marshalling

- `ARG_INT` -> `operator.index(obj)`. Not `type(obj) is int`: `np.int64` is not
  an `int` subclass, and routing it to the buffer branch would *succeed* and
  yield a pointer to the scalar.
- `int` in a pointer slot -> `ffi.cast("void*", obj)`, as today's raw-address
  path.
- otherwise -> `ffi.from_buffer(obj, require_writable=(kind == ARG_PTR_RW))`,
  which performs the contiguity and writability checks, handles read-only
  exporters, and pins the buffer. Translate its `ValueError` / `TypeError` into
  the message text the existing tests match (`"C-contiguous buffer"`).
- anything else -> `TypeError`. Reject by default.

### `bind()`

`bind(*args)` performs the marshalling once and returns a closure holding the
resulting cdata objects, which keep their buffer exports alive for the closure's
lifetime. Steady-state cost is 158 ns, parity with the C extension.

`bind()` is an **addition, not a replacement**. It fits callers that invoke the
same kernel against the same arrays repeatedly -- which is exactly all 28
`raw=True` benchmark sites, and matches exo's already-AOT-specialised model
(`partial_eval` per size). It does not fit the list-marshalling `jit(proc)` path,
which rebuilds its buffers every call anyway. Keep per-call `__call__` as the
default.

Because the returned cdata pins the exporter, a bound closure blocks resize and
close on its operands until released -- the same contract `jitcall.c` provides
per-call, extended to the closure's lifetime. Document it.

### what `cffi` still does not give you

Arity and keyword checks must be explicit in `__call__`. Measured against a bare
`CFUNCTYPE`, and the same hazards apply to a bare `ffi.cast` pointer:

| input | `jitcall.c` | bare FFI pointer |
| --- | --- | --- |
| too few args | `TypeError: expected 2, got 1` | `TypeError` |
| **too many args** | `TypeError: expected 2, got 5` | **accepted silently** (cdecl) |
| **keyword args** | `TypeError: ... does not accept keyword arguments` | accepted/ignored, message lacks `"keyword"` |
| **`None` / `str` / `2**64`** | `TypeError` / `OverflowError` | **accepted -> corruption** |

The "too many args accepted silently" row compounds §3's symbol-selection
hazard: a mis-selected 2-arg prototype called with 3 arguments *succeeds* with a
garbage third argument. Keep the explicit arity gate, and assert
`len(func_type.inputs) == len(arg_kinds)` at construction.

### GIL

`ffi.cast` function pointers release the GIL around the call, as `CFUNCTYPE`
does; `jitcall.c` holds it. This is safe here **only because `from_buffer` pins
every operand for the duration of the call** -- that pinning is what makes the
difference. An unpinned ctypes boundary with the GIL released was reproduced as
a SIGSEGV (a second thread calling `mmap.close()` mid-kernel: `BufferError`
under `jitcall.c`, `exit=139` unpinned). Releasing buys nothing measurable for
`par()` kernels -- OpenMP workers are native threads that never take the GIL --
but costs nothing either once operands are pinned.

## 6. steps

**Upstream first.** Land the §1.1 `FFIBackend` seam in xDSL: `ctypes`
implementation as default, `cffi` implementation under a new optional extra,
`wrap_jit_func`'s prototype comparison delegated to the backend. Then exojit
consumes `xdsl.jit` rather than forking it. If the seam stalls in review,
exojit can proceed by duplicating the small type-mapping surface (§8 R5) and
adopt the shared version later -- the two are not coupled in time.

Then, in exojit:

1. **Add `exojit/jitcall.py`** per §5, with `__call__` and `bind()`. Select the
   `llvm.FuncOp` by symbol name and assert its input count against `arg_kinds`.
2. **Switch `exojit/main.py`** to import from it and pass the IR-derived cdecl.
3. **Delete** `exojit/jitcall.c`, drop `ext_modules` from `setup.py`, and
   `rm -f exojit/*.so`. The built `exojit/jitcall.cpython-314-darwin.so` is on
   disk **and gitignored** (`.gitignore:25`, `*.so`), so `git status` hides it
   and `git clean -fd` will not remove it. Extension suffixes precede `.py` in
   CPython's `FileFinder`, so a stale `.so` silently shadows the new module: the
   full test and benchmark suite would keep exercising the old C extension, the
   regenerated CSVs would show no regression, and the PR would look clean with
   the new code dead. Add a `clean` target and a guard assertion
   (`assert exojit.jitcall.__file__.endswith(".py")`) to the test module.
4. **Strip `setup.py` from the Makefile**, which names it in four places (ruff
   check, ruff format, vulture, pyright); all three tools hard-fail on a missing
   path. `pyproject.toml` already carries `[build-system]` and
   `[tool.setuptools.packages.find]`, so deleting the file is safe. Drop the
   `find ... -name "*.c"` clang-format leg once no `.c` files remain.
5. **Extend `tests/e2e/test_jitcall.py`.** The two `type(...).__name__ ==
   "JitFunc"` assertions still hold. Add: over-arity (`raw(dst, src, extra)` ->
   `TypeError`); `None`, `str`, and `2**64` into a pointer slot; `bytes` into an
   `ARG_PTR_RW` slot; read-only exporters (`bytes`, read-only `mmap`,
   `setflags(write=False)`) into an `ARG_PTR_RO` slot -- these must now **pass**,
   where a ctypes design could not support them at all; `np.int64` into an
   `ARG_INT` slot; and a `bind()` round-trip including the resize-blocked
   contract.
6. **Switch the 28 `benchmarks/kernels/*.py` `raw=True` sites to `bind()`**,
   which is where the parity number comes from. This is a real calling-convention
   change to the benchmark harness and should be its own commit.
7. **Re-run** `make fmt && make tests && make benchmark` after a clean rebuild,
   and regenerate `benchmarks/results.csv` / `.pdf` separately. Do not regenerate
   `examples/microgpt/times/train.csv`: §4 shows the expected delta is ~0.12%,
   below that file's run-to-run noise.

**The xDSL version bump is not a prerequisite for exojit's side.** exojit pins
`e596f41c`, 278 commits behind HEAD, which predates `xdsl/jit` entirely
(`git ls-tree -d e596f41c xdsl/jit` is empty). The design needs only `cffi` and
an `llvm.FuncOp`, both of which the pinned revision already provides. Bump when
adopting the shared seam, not before.

## 7. if `bind()` is rejected

`bind()` changes the `raw=True` calling convention. If that is unacceptable,
the fallback is per-call cffi at 913 ns (6.3x) plus an address cache keyed on
`id(arr)` holding `(weakref.ref(arr), cdata)`. Measured on the ctypes
equivalent: 754 ns generic 2-arg, 978 ns on 3-arg add -- so a cffi version lands
near per-call cffi anyway, and is **strictly worse than `bind()`** while
carrying every hazard below. It is documented here to be dismissed, not adopted:

- `weakref.ref` is **unsupported** on `bytes`, `bytearray`, and numpy scalars --
  all of which `jitcall.c` accepts.
- In-place `shape`/`strides` assignment flips contiguity with **id and data
  pointer unchanged**, so a cached contiguity verdict goes stale silently.
- `setflags(write=False)` toggles writability at runtime with the weakref still
  alive, so a cached RW acceptance goes stale.
- `ndarray.resize()` can move the data pointer in place.
- The dict is unbounded without a weakref eviction callback, and its
  thread-safety is unspecified.

`bind()` has none of these: the cdata pins the exporter, so contiguity,
writability, and address cannot change underneath it. **`bind()` is the
principled version of this cache**, which is why §9 recommends it.

## 8. risks

- **R1 -- `bind()` changes the benchmark calling convention.** Step 6. The
  parity number depends on it; without it the migration costs 6.3x. Reviewers
  should confirm the comparison against numpy/torch/jax stays fair, since those
  do not hoist pointer derivation.
- **R2 -- silent acceptance of invalid arguments.** The §5 table. Every row a
  bare FFI pointer accepts and `jitcall.c` rejects is memory corruption, not an
  exception. Largest source of implementation risk under any option.
- **R3 -- bound cdata pins its operands.** A held closure blocks `resize()` and
  `mmap.close()` on its arrays until released. This is the correct contract but
  is new surface area; document and test it.
- **R4 -- upstream seam may stall.** exojit can ship against a duplicated
  type-mapping surface and adopt the shared one later; the two are decoupled in
  time. Do not block the migration on the xDSL PR.
- **R5 -- `xdsl.jit` is two weeks old and pre-1.0.** Even with the seam landed,
  expect churn. (An earlier draft justified caution with module churn in
  `xdsl/jit`; that was wrong -- `git show --stat -M` confirms every symbol was
  added once at its current path, zero renames. The pre-1.0 argument stands on
  its own.)
- **R6 -- renumbering `arg_kinds` breaks a second consumer.** `_jit_wrap`
  (`exojit/main.py:1075`) tests `arg_kinds[i] == 2` positionally. Dropping
  `ARG_INT` and renumbering to `0=RO, 1=RW` makes that never true: every
  writable tensor silently becomes read-only and `fn(dst, src)` leaves `dst`
  untouched with no exception. Make the kinds a named enum before renumbering.
  (`test_jit_syncs_nested_writable_sequences` would catch it.)

## 9. options

| # | option | ns/call | vs C ext | verdict |
| --- | --- | --- | --- | --- |
| 1 | **cffi + `bind()`, cffi per-call default** | **158** | **1.12x** | **recommended** |
| 2 | cffi per-call only | 913 | 6.3x | fallback if `bind()` is rejected |
| 3 | ctypes + `bind()` | 283 | 2.0x | matches xDSL as-is, no new dep, loses read-only support |
| 4 | ctypes per-call only | 2053 | 14.5x | what the first draft proposed; not viable |

Option 3 is the only one that needs no upstream change at all, and is the
conservative choice if the §1.1 seam is rejected -- but it cannot support
read-only buffer arguments in `ARG_PTR_RO` slots, which `jitcall.c` supports
today, so it is a feature regression as well as a 2x one.

A fifth option -- emitting the translation itself as generated LLVM IR -- was
considered and dropped. It could in principle beat the C extension, since a
generated thunk knows each proc's exact arity at codegen time. But it was never
measured, it moves the CPython ABI dependency from a `.c` file into hand-written
IR, and installing a vectorcall slot is not possible from pure Python anyway.

**Recommendation: option 1.** It drops the C extension (the actual goal), reaches
parity, is the only pure-Python option at feature parity with `jitcall.c`, adds
no new dependency, and -- via the §1.1 seam -- makes exojit a consumer of xDSL's
JIT rather than a fork of it.

## 10. review log

Three adversarial reviews (factual, design-feasibility, perf-methodology) ran
against the first draft, which proposed a straight `ctypes` port. What they
broke, all corrected above:

- The headline regression was understated: the draft claimed 14x from an inline
  expression that performed **no safety checks**, against a C extension that
  does. Apples-to-apples is 20.9x for the drafted design.
- The draft recommended `arr.ctypes.data`. It is 2.8x slower than
  `c_char.from_buffer`, and merely probing `hasattr(obj, "ctypes")` costs as much
  again -- the draft's own duck-typing rule was its dominant cost.
- The draft called `CFUNCTYPE`'s GIL release "an improvement". It is a behaviour
  change from `jitcall.c` that buys nothing measurable for OpenMP kernels and
  opens a use-after-free window, reproduced as a SIGSEGV. It is safe under §5
  only because `from_buffer` pins operands.
- The draft's `from_buffer` branch was inverted, and the read-only case it
  prescribed a test for **cannot work at all in pure ctypes**. This finding is
  what motivated evaluating cffi, and is the reason the recommendation changed.
- The draft claimed ctypes raises `ctypes.ArgumentError` on arity mismatch.
  Too-many-args raises **nothing**, kwargs raise a message without `"keyword"`,
  and `ArgumentError` is not a `TypeError` subclass.
- The draft's R6 (xDSL module churn) was fabricated; zero renames occurred.
- The draft's microgpt impact claim was fabricated; the true figure is ~0.12%
  because microgpt makes one JIT call for the whole run.
- The stale gitignored `.so` shadowing the new module would have made the entire
  change untestable while appearing to pass.
- `to_mlir` emits several `llvm.FuncOp`s and the entry point is often not the
  first; argument order follows exo's declared order (scalars first), not the
  pointers-first order the draft asserted.
- `weakref` -- which the draft's cache depended on -- does not work on `bytes`,
  `bytearray`, or numpy scalars. §7 now dismisses that cache in favour of
  `bind()`.
- Deleting `setup.py` breaks four Makefile invocations.
- Assorted: wrong line numbers (taken from an uncommitted working copy), wrong
  commit subjects, `benchmarks/run.py` cited for `raw=True` sites that live in
  `benchmarks/kernels/*.py`, and llvmlite pinned at 0.46.0 not 0.47.

The cffi evaluation postdates all three reviews and has **not** been
adversarially reviewed. Its numbers come from faithful wrapper classes with the
full check set (the methodology the reviews forced), but the design in §5 has
not had the same scrutiny the ctypes design received.
