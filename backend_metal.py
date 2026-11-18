import math
from collections.abc import Callable
from importlib import import_module
from typing import Any

from cffi import FFI
from exo.API import Procedure
from exo.core.LoopIR import LoopIR, T, get_writes_of_stmts
from exo.core.prelude import Sym
from xdsl.backend.llvm.convert import convert_module
from xdsl.builder import Builder
from xdsl.dialects import llvm
from xdsl.dialects.builtin import MemRefType, ModuleOp, i32, i64
from xdsl.ir import Block, Region
from xdsl.rewriter import InsertPoint

from exojit import IRGenerator, JITRuntime, to_mlir

# ===----------------------------------------------------------------------=== #
# metal backend
# ===----------------------------------------------------------------------=== #


def thread_grid(ir: LoopIR.proc) -> tuple[int, Sym, Sym, int, int]:
    # the only supported shape: static f32 tensors, `for i in par(0, M): for j in par(0, N): <seq body>` writing only out[i, j]
    # exo does not check that par loops are race-free outside its c backend, so disjoint writes are checked here
    for arg in ir.args:
        assert isinstance(arg.type, T.Tensor) and not arg.type.is_window and isinstance(arg.type.basetype(), T.F32) and all(isinstance(dim, LoopIR.Const) for dim in arg.type.shape()), f"metal backend: argument {arg.name} must be a static f32 tensor, got {arg.type}"
        assert all(isinstance(dim, LoopIR.Const) and isinstance(dim.val, int) and dim.val > 0 for dim in arg.type.shape()), f"metal backend: argument {arg.name} must have positive integer extents, got {arg.type}"
    written = {sym for sym, _ in get_writes_of_stmts(ir.body)}
    outs = [i for i, arg in enumerate(ir.args) if arg.name in written]
    assert len(outs) == 1, f"metal backend: expected exactly one written argument, got {len(outs)}"
    out = ir.args[outs[0]]
    out_type = out.type
    assert isinstance(out_type, T.Tensor), f"metal backend: output {out.name} must be a tensor, got {out_type}"
    shape = [dim.val for dim in out_type.shape() if isinstance(dim, LoopIR.Const) and isinstance(dim.val, int)]
    assert len(shape) == 2, f"metal backend: output {out.name} must be a 2d matrix, got shape {shape}"
    assert shape[0] * shape[1] < 2**32, f"metal backend: {shape[0]}x{shape[1]} threads exceed the uint32 thread id"
    rows = ir.body[0] if len(ir.body) == 1 else None
    assert isinstance(rows, LoopIR.For) and isinstance(rows.loop_mode, LoopIR.Par) and len(rows.body) == 1, "metal backend: body must be exactly `for i in par(0, M): for j in par(0, N): ...`"
    cols = rows.body[0]
    assert isinstance(cols, LoopIR.For) and isinstance(cols.loop_mode, LoopIR.Par), "metal backend: body must be exactly `for i in par(0, M): for j in par(0, N): ...`"
    for loop, extent in zip((rows, cols), shape):
        assert isinstance(loop.lo, LoopIR.Const) and loop.lo.val == 0 and isinstance(loop.hi, LoopIR.Const) and loop.hi.val == extent, f"metal backend: par loop {loop.iter} must span 0..{extent} to match {out.name}"

    def is_cell(idx: list[LoopIR.expr]) -> bool:
        return len(idx) == 2 and all(isinstance(e, LoopIR.Read) and e.name == sym and not e.idx for e, sym in zip(idx, (rows.iter, cols.iter)))

    def check_expr(expr: LoopIR.expr) -> None:
        match expr:
            case LoopIR.Read():
                assert expr.name != out.name or is_cell(expr.idx), f"metal backend: {out.name} may only be read at [{rows.iter}, {cols.iter}]"
                for idx in expr.idx:
                    check_expr(idx)
            case LoopIR.USub():
                check_expr(expr.arg)
            case LoopIR.BinOp():
                check_expr(expr.lhs)
                check_expr(expr.rhs)
            case LoopIR.Const():
                pass
            case _:
                assert False, f"metal backend: unsupported expression {type(expr).__name__}"

    def check_stmt(stmt: LoopIR.stmt) -> None:
        match stmt:
            case LoopIR.Assign() | LoopIR.Reduce():
                assert stmt.name == out.name and is_cell(stmt.idx), f"metal backend: each thread may only write {out.name}[{rows.iter}, {cols.iter}]"
                check_expr(stmt.rhs)
            case LoopIR.For() if isinstance(stmt.loop_mode, LoopIR.Seq):
                check_expr(stmt.lo)
                check_expr(stmt.hi)
                for child in stmt.body:
                    check_stmt(child)
            case LoopIR.For():
                assert False, "metal backend: par loops nested inside the thread grid are not supported"
            case LoopIR.Pass():
                pass
            case _:
                assert False, f"metal backend: unsupported statement {type(stmt).__name__}"

    for stmt in cols.body:
        check_stmt(stmt)
    return outs[0], rows.iter, cols.iter, shape[0], shape[1]


class MetalIRGenerator(IRGenerator):
    # reuses the cpu lowering, but binds the two par loops of the grid to a trailing i32 gpu thread id
    def __init__(self, row: Sym, col: Sym, cols: int):
        super().__init__()
        self.row, self.col, self.cols = row, col, cols

    def _stmt_for(self, for_stmt: LoopIR.For) -> None:
        if for_stmt.iter not in (self.row, self.col):
            return super()._stmt_for(for_stmt)
        for stmt in for_stmt.body:  # one thread per iteration, the iterator is already bound to the thread id
            self._stmt(stmt)

    def generate(self, procs: list[LoopIR.proc]) -> ModuleOp:
        # the kernel has no noalias arg attrs (llvm_to_air only parses bare `ptr %x` arguments) and there is no gpu heap, so no malloc/free
        assert len(procs) == 1, "metal backend: a thread grid maps exactly one kernel procedure"
        proc = procs[0]
        input_types = [self._arg_type(arg, proc.body) for arg in proc.args] + [i32]
        region = Region(block := Block(arg_types=input_types))
        self.builder = Builder(insertion_point=InsertPoint.at_end(block))
        self.symbol_table = {arg.name: val for arg, val in zip(proc.args, block.args)}
        thread_id = self._emit(llvm.ZExtOp(block.args[-1], i64))  # linear grid: row = id // cols, col = id % cols
        self.symbol_table[self.row] = self._emit(llvm.UDivOp(thread_id, self._int_const(self.cols)))
        self.symbol_table[self.col] = self._emit(llvm.URemOp(thread_id, self._int_const(self.cols)))
        for stmt in proc.body:
            self._stmt(stmt)
        self.builder.insert(llvm.ReturnOp())
        fn_type = llvm.LLVMFunctionType([llvm.LLVMPointerType() if isinstance(t, MemRefType) else t for t in input_types], llvm.LLVMVoidType())
        self.module.body.block.add_op(llvm.FuncOp(proc.name, fn_type, linkage=llvm.LinkageAttr("external"), body=region))
        return self.module


def compile(proc: Procedure, raw: bool = False) -> Callable[..., None]:
    # exo -> llvm dialect (cpu lowering + grid) -> llvm ir text (xdsl) -> air -> metallib (llvm_to_air), wrapped by the cpu jit's argument conversion
    assert not raw, "raw=True is not supported by the metal backend"
    ir = proc._loopir_proc
    out, row, col, rows, cols = thread_grid(ir)  # reject before compiling or importing optional deps

    def compile_kernel(proc: Procedure, ffi: FFI) -> Callable[..., None]:
        metal, air = import_module("Metal"), import_module("llvm_to_air")  # optional deps, only loaded on the metal path
        llvm_module = convert_module(to_mlir(proc, MetalIRGenerator(row, col, cols)), fallback_target_triple="air64-apple-macosx")  # to_air replaces the triple and datalayout
        kernel_args = llvm_module.get_global(proc.name()).args
        overrides = {proc.name(): {kernel_args[out].name: "out", kernel_args[-1].name: "id"}}  # llvm_to_air derives read_write access and thread_position_in_grid from these names
        device, pso = air.create_compute_pipeline(air.compile_to_metallib(air.to_air(str(llvm_module), kernel_overrides=overrides)), proc.name())
        sizes = [4 * math.prod(dim.val for dim in arg.type.shape()) for arg in ir.args]
        threads = rows * cols

        def run(*ptrs: int) -> None:
            # buffer index i is kernel argument i, the trailing thread id is not a buffer
            buffers = [device.newBufferWithBytes_length_options_(ffi.buffer(ffi.cast("char *", ptr), size)[:], size, metal.MTLResourceStorageModeShared) for ptr, size in zip(ptrs, sizes, strict=True)]
            assert all(buffer is not None for buffer in buffers), "metal backend: buffer allocation failed"

            def encode(encoder: Any) -> None:
                for i, buffer in enumerate(buffers):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, i)

            air.execute_kernel(device, pso, metal.MTLSize(threads, 1, 1), metal.MTLSize(min(threads, pso.maxTotalThreadsPerThreadgroup()), 1, 1), encode)
            ffi.memmove(ffi.cast("char *", ptrs[out]), buffers[out].contents().as_buffer(sizes[out]), sizes[out])

        return run

    return JITRuntime.compile(proc, compiler=compile_kernel)
