# what still needs to be ported to xDSL

exojit's lowering already runs on xDSL, and the LLVM conversion + call boundary
now come from `xdsl.backend.llvm.convert` and `xdsl.jit`. what is left below is
the work that cannot move upstream yet, with the reason it is blocked.

measurements were taken on arm64 darwin with saxpy f32 at `N = 2^20`, median of
9 runs of 100 calls.

## 1. `llvm.fptrunc` is missing from xDSL's LLVM dialect

xDSL has `FPExtOp`, `SIToFPOp` and `BitcastOp` but no `fptrunc`, so we define
`FPTruncOp` in `exojit/patches_xdsl_llvm.py` and register it into
`xdsl.backend.llvm.convert_op._CAST_OP_NAMES` -- a private dict -- to teach the
llvmlite converter about it. that reach into private state is the only reason
`convert_module` works on procs that call `expf` on `f64`.

port `FPTruncOp` upstream next to `FPExtOp` and add it to `_CAST_OP_NAMES`, then
delete the patch.

## 2. `xdsl.jit.llvm.backend.llvm_jit` is not usable for a performance compiler

`llvm_jit` builds its target machine with a bare
`Target.from_default_triple().create_target_machine()` and runs no module pass
pipeline at all. on our IR that is a 3.5x regression:

| pipeline | saxpy |
| --- | --- |
| `llvm_jit` as shipped (no O3) | 398.8 us |
| same IR + our O3 module passes | 120.4 us |
| same IR + O3 + host-cpu target machine | 130.0 us |

the gap is entirely the missing pass pipeline; the host-cpu/features target
machine is worth nothing measurable here, though it still matters on x86 where
feature detection gates AVX-512. the engine itself is fine -- handing `llvm_jit`
already-optimised IR gets back to parity (140.2 us).

so exojit keeps `_target_machine()` and `_to_llvmlite_moduleref()` and builds
its own MCJIT engine. to drop them, `llvm_jit` needs optional `target_machine`
and module-pass-pipeline parameters, or needs to accept a pre-built `ModuleRef`.

## 3. no assembly emission

`--asm` needs `TargetMachine.emit_assembly`. xDSL's `LLVMTarget.emit` only
prints LLVM IR, so `to_asm()` stays here until there is an asm target upstream.

## 4. OpenMP lowering stops at the dialect boundary

xDSL has an `omp` dialect and `convert_scf_to_openmp`, but nothing lowers
`omp.parallel` / `omp.wsloop` to `__kmpc_*` calls without shelling out to
`mlir-opt`. `_stmt_for_par` therefore hand-emits the outlined function,
`__kmpc_fork_call`, `__kmpc_for_static_init_8` and `__kmpc_for_static_fini`.

an omp-to-llvm pass upstream would delete ~100 lines of `main.py` plus the
`_load_libomp` dance around it.

### blocker: `par()` with a dynamic loop bound is broken

```python
@proc
def saxpy_par(N: size, y: f32[N], x: f32[N], a: f32[1]):
    for i in par(0, N):
        y[i] = a[0] * x[i] + y[i]

jit(saxpy_par)
# NotImplementedError: Conversion not implemented for op:
# builtin.unrealized_conversion_cast
```

`_stmt_for_par` casts every shared capture to `!llvm.ptr` with an
`unrealized_conversion_cast`. for memref captures that is an identity cast once
`RewriteMemRefTypes` has run, and reconcile-unrealized-casts folds it away. a
scalar capture is a genuine `i64 -> ptr` conversion that nothing folds:

```
UserWarning: Unable to remove cast ... i64 to !llvm.ptr
because it is not unifiable with its uses
```

it survives to the llvmlite backend and fails there. this predates the move to
`convert_module` -- it is not a regression. every benchmark kernel calls
`partial_eval(N=n)` first, which constant-folds the size away and leaves only
pointer captures, which is why the suite is green.

fix by boxing scalar captures through an alloca the way `lo`/`hi` already are,
or by emitting `llvm.inttoptr` instead of an unrealized cast.

## 5. externs without a runtime symbol

`_expr_extern` special-cases `select`, `expf` and the libm unary intrinsics into
LLVM ops. every other Exo extern -- `relu`, `sigmoid`, `fmaxf`, and any
user-defined `Extern` -- falls through to a plain `llvm.call`, which
`module.verify()` then rejects:

```
VerifyException: '@relu' could not be found in symbol table
```

so those procs simply do not compile. that is the behaviour on main and it is
deliberately unchanged here.

declaring the callee is not enough to fix it. Exo's C backend emits `relu` and
`sigmoid` as source-level helpers via `Extern.globl()`, so no such symbol exists
in the JIT process; MCJIT resolves the missing name to address 0 without any
diagnostic and the call jumps to NULL. supporting these needs the extern body
lowered into the module (or registered with the engine), plus a check that every
declared-but-undefined symbol actually resolves before handing back a callable.

note also that a name-keyed declaration cache is not sufficient on its own: the
same extern can be called at two precisions in one module, which needs one
declaration per signature.

## 6. dynamic-dimension memref lowering

`ExtendedConvertMemRefToPtr` recovers a dynamic dimension by walking an index
back to a block argument and reading the loop header's `llvm.icmp` bound
(`_loop_upper_bound_as_i64`). that is a workaround for not carrying dynamic
sizes in the IR at all.

xDSL's `ConvertMemRefToPtr` does handle dynamic strides, but it needs them
present as SSA values. fix our IR first -- a strided layout or explicit size
operands -- and xDSL's pass may replace ours outright.

## 7. `vec_*` / `neon_*` intrinsics

`exojit/patches_xdsl_intrinsics.py` (329 lines) is entirely ours: masked prefix
stores, fmadd reductions and NEON builders on top of `llvm.MaskedStoreOp`,
`ShuffleVectorOp` and `InsertElementOp`. a vector-dialect lowering upstream
could absorb it, but nothing in xDSL covers it today.

## 8. argument marshalling

`xdsl.jit.py_type_context` maps scalars only (`float` <-> `double`). it has no
concept of buffers, writable outputs synced back after the call, nested Python
sequences, or argument sizes that depend on other arguments. `_jit_tensor_converter`,
`_jit_eval_shape_expr` and the shape-env/keepalive/syncback machinery stay here
until that grows a buffer-shaped type map.

we do use `xdsl.jit`'s C type registry for the entry point signature, so the ABI
mapping itself is no longer duplicated.

the `bind()` entry point that pre-marshalled arguments once and returned a
zero-argument callable was dropped -- calls now go straight through `call()`.
nothing in the repo used it, and it was also the accidental owner of the MCJIT
engine reference. if that caching comes back it should hold the engine
explicitly, the way `xdsl.jit.llvm.backend.LLVMRawJITFunc` does.

## 9. compilation cache

xDSL's jit has no caching. `_ir_cache_dir` / `_disk_cache` key generated IR on a
hash of the compiler sources so edits invalidate the cache automatically.

## 10. `JITContext` is frontend-bound

`xdsl.jit.context.JITContext` wires `PyASTContext` to a backend, so it only
accepts Python functions parsed by xDSL's own frontend. exojit's frontend is Exo
LoopIR, so we talk to the backend layer directly and skip `JITContext`.
generalising it over frontends would let exojit use `@ctx.jit(...)` as-is.
