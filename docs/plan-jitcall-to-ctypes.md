# plan: drop `exojit/jitcall.c`, call through `ctypes` like xDSL does

Status: **decision required before implementation.** Sections 1-3 are settled.
Section 4 says the migration costs ~14x on the call boundary; section 9 lays out
the four ways forward. This document has been through three adversarial reviews;
section 10 records what they broke.

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
  `!llvm.func<void (i64, !llvm.ptr, !llvm.ptr)>` and
  `to_c_func_type` gives `CFUNCTYPE(None, c_int64, c_void_p, c_void_p)` with no
  bespoke code.
- That makes the ABI half of `_jit_arg_kinds` (int vs pointer) redundant -- it is
  currently re-derived from LoopIR when the LLVM dialect already states it.
- No compiler toolchain needed to install exojit; the wheel becomes pure-Python.
- The README already states the project is being ported to xDSL. Sharing xDSL's
  JIT boundary is the same direction of travel.

**The entry function is not the only `llvm.FuncOp`, and often not the first.**
`to_mlir` also emits `malloc` / `free` declarations; a proc that calls a subproc
emits the callee first; a `par()` proc emits `__omp_outlined_0` first and a
*variadic* `__kmpc_fork_call` (which `to_c_func_type` rejects outright). Any
implementation must select by `sym_name.data == proc.name()`, as
`xdsl/jit/llvm/backend.py:105` does via `SymbolTable.lookup_symbol`, and assert
`len(func_type.inputs) == len(arg_kinds)`.

## 4. the cost, measured

Measured on this machine (darwin/arm64 M2 Pro, py3.14.5, numpy 2.4.2,
llvmlite **0.46.0** per `uv.lock`), against the real 3-argument
`add` kernel at n=256, `min(timeit.repeat(number=20000, repeat=9))`, statement
form (no lambda closure):

| call boundary | ns/call | vs baseline |
| --- | --- | --- |
| `jitcall.c` `JitFunc`, numpy arrays | **137** | 1.0x |
| pure ctypes, `PYFUNCTYPE` + `memoryview` checks + `c_char.from_buffer` | **1981** | **14.5x** |
| pure ctypes, `CFUNCTYPE`, same otherwise | 2027 | 14.8x |
| pure ctypes, ndarray fast path (`flags` + `__array_interface__`) | 3179 | 23.2x |
| pure ctypes, `memoryview` checks + `arr.ctypes.data` | 3197 | 23.3x |
| pure ctypes, `hasattr(obj, "ctypes")` probe + `arr.ctypes.data` | 3648 | 26.6x |
| `numpy.ctypeslib.ndpointer` argtypes | 3236 | 23.6x |

Component costs, per array argument:

| operation | ns |
| --- | --- |
| `memoryview(arr)` | 109 |
| `arr.flags.c_contiguous` | 40 |
| `ctypes.addressof(ctypes.c_char.from_buffer(arr))` | 227 |
| `arr.ctypes.data` | 645 |
| `hasattr(arr, "ctypes")` | 627 |
| `arr.__array_interface__["data"][0]` | 690 |
| `arr.ctypes.data_as(c_void_p)` | 1183 |

Two counter-intuitive results drove the design in §5. First,
`ctypes.c_char.from_buffer` is **2.8x cheaper** than `arr.ctypes.data`, because
numpy rebuilds its `_ctypes` helper object on every attribute access -- merely
*probing* for `.ctypes` costs as much as reading it. Second, a numpy-specialised
fast path is **slower** than the generic buffer-protocol path, for the same
reason. The naive designs are the slow ones.

Int-address path, for reference: 72 ns (C ext) vs 174 ns (`PYFUNCTYPE`, no
`argtypes`, pre-built `c_void_p` instances) to 240 ns (`CFUNCTYPE` with
`argtypes`) = **2.4x-3.3x**.

### end-to-end

Cost is ~700 ns *per array argument*, not a flat per-call figure. GFLOP/s
retained on the `add` row:

| n | C ext ns | ctypes ns | GFLOP/s kept |
| --- | --- | --- | --- |
| 256 | 144 | 3197 | **4.5%** |
| 4096 | 391 | 3511 | **11.1%** |
| 65536 | 6534 | 9395 | 69.5% |
| 262144 | 28975 | 27484 | ~100% (inside noise) |
| 1048576 | 72006 | 75808 | 95.0% |

`adam` takes 10 arguments and pays **+8.5 us**; `matmul`, `matvec`,
`weighted_sum` take 3. So three benchmark columns move materially, not one, and
the smallest two lose ~90% of their GFLOP/s. Note also that the project's own
`bench = min(timeit.repeat(number=1, repeat=50))` carries +-10% at n >= 262144,
so a large-n delta is not reliably reviewable.

`examples/microgpt` is **not** affected: `train.py` makes exactly one JIT call
for the entire 1000-step run (the `seq(0, NUM_STEPS)` loop is *inside* the
jitted proc), so the added cost is ~59 arrays x 700 ns ~= 41 us once, against
0.035 ms/step x 1000 = **0.12%**.

**No pure-ctypes arrangement avoids this.** The best safe design measured is
14.5x, and §7's cache recovers to ~6.8x, not to parity.

## 5. design

Delete `exojit/jitcall.c` and `setup.py`'s `ext_modules`. Add `exojit/jitcall.py`
exposing a `JitFunc` with the same construction signature and call semantics,
implemented over `ctypes` and over xDSL's registry where it fits.

```
llvm.FuncOp for SymbolTable.lookup_symbol(module, proc.name())
        |
        v
to_c_func_type(CTypeContext(builtin + llvm registries), func_op.function_type)
        |
        v
ctypes.PYFUNCTYPE(None, c_int64, c_void_p, c_void_p)   # declared-order inputs
        |
        v
exojit.jitcall.JitFunc  -- arity gate, type gate, buffer pinning, engine keepalive
```

`JitFunc.__init__(address, engine, arg_kinds, c_func_type)` keeps its existing
shape plus the IR-derived prototype, so `exojit/main.py` and
`tests/e2e/test_jitcall.py` need minimal edits.

**`ctypes` provides none of the safety `jitcall.c` provides. Every check must be
written explicitly in `__call__`, before any value reaches the function pointer.**
Measured against a real `CFUNCTYPE(None, c_void_p, c_void_p)`:

| input | `jitcall.c` | bare ctypes |
| --- | --- | --- |
| too few args | `TypeError: JitFunc expected 2, got 1` | `TypeError: ... takes at least 2 arguments` |
| **too many args** | `TypeError: JitFunc expected 2, got 5` | **accepted silently** (cdecl) |
| **keyword args** | `TypeError: ... does not accept keyword arguments` | accepted, ignored, message lacks `"keyword"` |
| **`None`** | `TypeError: ... got NoneType` | **accepted -> null deref** |
| **`str`** | `TypeError: ... got str` | **accepted -> pointer into interned str** |
| **`bytes` into RW slot** | `TypeError: ... writable C-contiguous buffer` | **accepted -> writes into immutable** |
| **`2**64`** | `OverflowError` | **accepted** |
| wrong ctypes type | -- | `ctypes.ArgumentError`, **not** a `TypeError` subclass |

The "too many args accepted silently" row compounds §3's symbol-selection
hazard: a mis-selected 2-arg prototype called with 3 arguments *succeeds* with a
garbage third argument.

Per-argument marshalling, in the order measured fastest in §4:

- `ARG_INT` -> `operator.index(obj)` (not `type(obj) is int`: `np.int64` is not
  an `int` subclass, and routing it to the buffer branch would succeed and yield
  a pointer to the scalar);
- `int` in a pointer slot -> passed through as a raw address, as today;
- otherwise -> `mv = memoryview(obj)`; reject if not `mv.c_contiguous`; reject if
  `mv.readonly` and the kind is `ARG_PTR_RW`; take the address via
  `ctypes.addressof(ctypes.c_char.from_buffer(obj))`;
- anything else -> `TypeError`. Reject by default; never let an unvalidated
  value reach `c_void_p`.

**`from_buffer` requires a writable buffer unconditionally.** It raises
`TypeError: underlying buffer is not writable` on `bytes`, read-only `mmap`,
numpy scalars, and any `setflags(write=False)` array -- all of which
`jitcall.c` accepts today in `ARG_PTR_RO` slots. `from_buffer_copy` is not a
substitute: it returns a different address, so the kernel would read a copy that
is freed before it runs. **There is no pure-ctypes way to obtain a read-only
buffer's address.** The fallback is `numpy.frombuffer(obj).ctypes.data` (numpy
is already a hard dependency), or an explicit decision to drop read-only
non-numpy exporters from the supported set. This must be settled before
implementation, not during.

**Pin every operand for the duration of the call.** `jitcall.c` holds each
`Py_buffer` export across the native call; a raw address taken and dropped does
not. Keep the `memoryview` and the `c_char` instance alive in a local list until
the call returns. Demonstrated failure without it: a second thread calling
`mmap.close()` mid-kernel gets `BufferError: cannot close exported pointers
exist` under `jitcall.c`, and **SIGSEGV (exit 139)** under an unpinned ctypes
boundary. A `bytearray` resized immediately after address extraction moved to a
different address.

**Use `PYFUNCTYPE`, not `CFUNCTYPE`.** `CFUNCTYPE` releases the GIL; `PYFUNCTYPE`
holds it, matching `jitcall.c` exactly. Releasing buys nothing measurable --
on a real `__kmpc_fork_call` add at n=1<<20, `CFUNCTYPE` 73.3 us vs `PYFUNCTYPE`
77.8 us, inside the spread, because OpenMP workers are native threads that never
take the GIL -- while costing 21-25% on the boundary and widening the
use-after-free window above. `PYFUNCTYPE` is the semantics-preserving choice.

Error messages must stay compatible with the existing tests: `TypeError` with
`"C-contiguous buffer"` and `"keyword"` in the message.

## 6. steps

1. **Add `exojit/jitcall.py`** per §5, alongside the existing extension. Select
   the `llvm.FuncOp` by symbol name and assert its input count against
   `arg_kinds`.
2. **Switch `exojit/main.py`** to import from it and to pass the IR-derived
   `c_func_type`.
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
   `find ... -name "*.c"` clang-format leg only once no `.c` files remain.
5. **Extend `tests/e2e/test_jitcall.py`.** The two `type(...).__name__ ==
   "JitFunc"` assertions still hold. Add: over-arity (`raw(dst, src, extra)` ->
   `TypeError`); `None`, `str`, and `2**64` into a pointer slot; `bytes` into an
   `ARG_PTR_RW` slot; a read-only buffer (`bytes`, read-only `mmap`) into an
   `ARG_PTR_RO` slot; and a `np.int64` into an `ARG_INT` slot.
6. **Re-run** `make fmt && make tests && make benchmark` after a clean rebuild,
   and regenerate `benchmarks/results.csv` / `.pdf` in a separate commit so the
   perf delta is reviewable on its own. Do not regenerate
   `examples/microgpt/times/train.csv`: §4 shows the expected delta there is
   0.12%, below that file's run-to-run noise.

**The xDSL bump is not a prerequisite and is not part of this PR.** exojit pins
`e596f41c`, 278 commits behind HEAD, which predates `xdsl/jit` entirely
(`git ls-tree -d e596f41c xdsl/jit` is empty). But the design needs only stdlib
`ctypes` and an `llvm.FuncOp`, both of which the pinned revision already
provides. Per R5 below, duplicate the 5-line `to_c_func_type` plus the type
registry rather than depending on `xdsl.jit`, and schedule the bump separately.

## 7. perf recovery (deferred, not in this PR)

The cheapest recovery is to avoid re-extracting addresses for arrays passed
repeatedly: a cache keyed on `id(arr)` holding `(weakref.ref(arr), address)`,
validated by checking the weakref is alive. Measured: **754 ns** generic 2-arg,
978 ns on the 3-arg add -- a **6.8x** residual regression, not parity. The
~400 ns figure is reachable only with a hand-unrolled per-arity class *and*
`PYFUNCTYPE`; the unsafe floor with no checks at all is 352 ns, and the
zero-marshalling floor (raw ints) is 190 ns.

Hazards, which are why this is not in this PR:

- `weakref.ref` is **unsupported** on `bytes`, `bytearray`, and numpy scalars --
  all of which `jitcall.c` accepts. The cache can only cover
  `type(obj) is np.ndarray`.
- In-place `shape`/`strides` assignment flips contiguity with **id and data
  pointer unchanged**, so any cached contiguity verdict goes stale silently.
- `setflags(write=False)` toggles writability at runtime with the weakref still
  alive, so a cached RW acceptance goes stale.
- `ndarray.resize()` can move the data pointer in place.
- The dict is unbounded without a weakref eviction callback, and its
  thread-safety is unspecified.

So: cache the address only, re-validate `flags.c_contiguous` and
`flags.writeable` on every hit (40 ns each), and register an eviction callback.
The `id()`-reuse hazard is the one part that is already sound -- if the weakref
is alive and keyed by `id(arr)`, it must be `arr`. A wrong answer anywhere else
here is a silent wild pointer write, not an exception.

## 8. risks

- **R1 -- benchmark regression.** Quantified in §4: 14.5x on the boundary, ~90%
  of GFLOP/s lost at n=256 and n=4096. Not mitigated by this PR; see §9.
- **R2 -- use-after-free from unpinned operands.** §5 mandates pinning and
  `PYFUNCTYPE`. Getting either wrong reintroduces a demonstrated SIGSEGV. Under
  a free-threaded 3.14t build the window is permanently open, not per-call.
- **R3 -- silent acceptance of invalid arguments.** The §5 table. Every row that
  `jitcall.c` rejects and bare ctypes accepts is memory corruption, not an
  exception. This is the single largest source of implementation risk.
- **R4 -- read-only non-numpy buffers have no pure-ctypes address path.** §5.
  Needs an explicit decision, and tests for `bytes` and read-only `mmap`.
- **R5 -- `xdsl.jit` is two weeks old and pre-1.0.** Mitigated by duplicating
  `to_c_func_type` (5 lines, `xdsl/jit/llvm/c_type_context.py:13-17`) plus
  `CTypeContext` / `register_builtin_ctypes` (63 lines) rather than importing
  them, which also removes the xDSL bump from the critical path. (An earlier
  draft justified this with module churn in `xdsl/jit`; that was wrong --
  `git show --stat -M` confirms every symbol was added once at its current path,
  zero renames. The pre-1.0 argument stands on its own.)
- **R6 -- renumbering `arg_kinds` breaks a second consumer.** `_jit_wrap`
  (`exojit/main.py:1075`) tests `arg_kinds[i] == 2` positionally. Dropping
  `ARG_INT` and renumbering to `0=RO, 1=RW` makes that never true: every
  writable tensor silently becomes read-only and `fn(dst, src)` leaves `dst`
  untouched with no exception. Make the kinds a named enum before renumbering.
  (`test_jit_syncs_nested_writable_sequences` would catch it.)

## 9. the decision this forces

§4 means "drop `jitcall.c` for ctypes" and "keep the current per-call overhead"
are not simultaneously satisfiable. Four ways forward:

- **(a) Ship the migration, accept the regression.** Simplest, matches xDSL
  exactly, pure-Python wheel. Costs ~90% of GFLOP/s at n=256 and n=4096 on
  every exo and neon benchmark row, which is the comparison against
  numpy/numba/torch that the project exists to make.
- **(b) Ship the migration plus §7's cache together.** Residual 6.8x rather than
  14.5x, at the cost of landing the R2/R4 lifetime hazards and a stale-validation
  cache unreviewed alongside a large refactor.
- **(c) Adopt the ctypes type derivation only, keep `jitcall.c` as the calling
  boundary.** Removes the duplicated ABI classification, gains the IR-derived
  prototype and xDSL compatibility at the type level, keeps 137 ns. Does not drop
  the C file, so it does not do what was asked.
- **(d) Migrate, and move the marshalling out of the hot path.** The boundary
  cost is per-call; the benchmarks call in a `timeit` loop with the same arrays
  every iteration. A `bind(*arrays)` API returning a zero-marshalling closure
  (addresses resolved and operands pinned once) would make the steady-state cost
  the 190 ns raw-int floor. This changes the `raw=True` calling convention and
  needs the benchmark harness updated, but it is the only option that both drops
  `jitcall.c` and keeps the numbers.

**Recommendation: prototype (d) for one kernel and measure before committing.**
It is the only option on the list that satisfies the actual goal -- drop the C
extension, gain xDSL compatibility -- without paying a 14.5x boundary tax, and
§4's component table suggests the 190 ns floor is reachable. If (d) does not
measure out, (a) is the honest fallback and the small-n benchmark regression
should be stated in the README rather than hidden.

## 10. review log

Three adversarial reviews (factual, design-feasibility, perf-methodology) ran
against the first draft of this document. What they broke, all now corrected
above:

- The headline regression was understated: the draft claimed 14x from an inline
  expression that performed **no safety checks**, against a C extension that
  does. Apples-to-apples is 20.9x for the drafted design, and 14.5x for the
  redesigned one in §5.
- The draft recommended `arr.ctypes.data`. It is 2.8x slower than
  `c_char.from_buffer`, and merely probing `hasattr(obj, "ctypes")` costs as much
  again -- the draft's own duck-typing rule was its dominant cost.
- The draft called `CFUNCTYPE`'s GIL release "an improvement". It is a behaviour
  change from `jitcall.c`, buys nothing measurable for OpenMP kernels, costs
  21-25%, and opens a use-after-free window that was reproduced as a SIGSEGV.
- The draft's `from_buffer` branch was inverted, and the read-only case it
  prescribed a test for cannot work at all in pure ctypes.
- The draft claimed ctypes raises `ctypes.ArgumentError` on arity mismatch.
  Too-many-args raises **nothing**, kwargs raise a message without `"keyword"`,
  and `ArgumentError` is not a `TypeError` subclass.
- The draft's R6 (xDSL module churn) was fabricated; zero renames occurred.
- The draft's microgpt impact claim was fabricated; the true figure is 0.12%
  because microgpt makes one JIT call for the whole run.
- The stale gitignored `.so` shadowing the new module would have made the entire
  change untestable while appearing to pass.
- `to_mlir` emits several `llvm.FuncOp`s and the entry point is often not the
  first; argument order follows exo's declared order (scalars first), not the
  pointers-first order the draft asserted.
- `weakref` -- which §7's cache depends on -- does not work on `bytes`,
  `bytearray`, or numpy scalars.
- Deleting `setup.py` breaks four Makefile invocations.
- Assorted: wrong line numbers (taken from an uncommitted working copy), wrong
  commit subjects, `benchmarks/run.py` cited for `raw=True` sites that live in
  `benchmarks/kernels/*.py`, and llvmlite pinned at 0.46.0 not 0.47.
