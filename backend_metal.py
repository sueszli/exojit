import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TypeGuard

from cffi import FFI
from exo import DRAM
from exo.API import Procedure
from exo.core.LoopIR import LoopIR, T
from exo.core.prelude import Sym
from xdsl.backend.llvm.convert import convert_module
from xdsl.builder import Builder
from xdsl.dialects import llvm
from xdsl.dialects.builtin import MemRefType, ModuleOp, i32, i64
from xdsl.ir import Block, Region
from xdsl.rewriter import InsertPoint

from exojit import IRGenerator, JITRuntime, to_mlir

# ===----------------------------------------------------------------------=== #
# matmul validation
# ===----------------------------------------------------------------------=== #


GRID = "metal backend: body must be `for i in par(0, M): for j in par(0, N): ...` over out[M, N]"
INIT = "metal backend: each thread must first set out[i, j] = 0.0 (accumulation is unsupported)"
REDUCE = "metal backend: after the zero, thread body must be exactly `for k in seq(0, K): out[i, j] += A[i, k] * B[k, j]`"


def matrix_dims(arg: LoopIR.fnarg) -> list[int]:
    t, msg = arg.type, f"metal backend: {arg.name} must be a static f32[M, N] @ DRAM matrix with positive extents, got {arg.type}"
    assert isinstance(t, T.Tensor) and not t.is_window and isinstance(t.basetype(), T.F32) and arg.mem is DRAM, msg
    dims = [d.val for d in t.shape() if isinstance(d, LoopIR.Const) and type(d.val) is int and d.val > 0]
    assert len(dims) == len(t.shape()) == 2, msg
    return dims


def is_loop(s: LoopIR.stmt, mode: type, extent: int) -> TypeGuard[LoopIR.For]:
    return isinstance(s, LoopIR.For) and isinstance(s.loop_mode, mode) and isinstance(s.lo, LoopIR.Const) and s.lo.val == 0 and isinstance(s.hi, LoopIR.Const) and s.hi.val == extent


def is_access(e: LoopIR.expr | LoopIR.stmt, name: Sym, *syms: Sym) -> bool:
    # name[syms...], every index a bare loop variable
    idx = getattr(e, "idx", None)
    return getattr(e, "name", None) == name and idx is not None and len(idx) == len(syms) and all(isinstance(x, LoopIR.Read) and x.name == s and not x.idx for x, s in zip(idx, syms))


def check_matmul(ir: LoopIR.proc) -> tuple[int, int, int, Sym, Sym]:
    # exo skips race checks outside c backend, so accept exactly one shape of program
    assert len(ir.args) == 3, f"metal backend: kernel must be matmul(out, A, B), got {len(ir.args)} arguments"
    (m, n), (am, k), (bk, bn) = (matrix_dims(arg) for arg in ir.args)
    out, a, b = (arg.name for arg in ir.args)
    assert (am, bk, bn) == (m, k, n), f"metal backend: shapes must be out[M, N], A[M, K], B[K, N], got out[{m}, {n}], A[{am}, {k}], B[{bk}, {bn}]"
    assert m * n < 2**32, f"metal backend: {m}x{n} threads exceed the uint32 thread id"

    assert len(ir.body) == 1, GRID
    rows = ir.body[0]
    assert is_loop(rows, LoopIR.Par, m) and len(rows.body) == 1, GRID
    cols = rows.body[0]
    assert is_loop(cols, LoopIR.Par, n), GRID
    i, j, cell = rows.iter, cols.iter, cols.body

    assert cell, INIT
    init = cell[0]
    assert isinstance(init, LoopIR.Assign) and is_access(init, out, i, j), INIT
    assert isinstance(init.rhs, LoopIR.Const) and init.rhs.val == 0, INIT

    assert len(cell) == 2, REDUCE
    red = cell[1]
    assert is_loop(red, LoopIR.Seq, k) and len(red.body) == 1, REDUCE
    acc = red.body[0]
    assert isinstance(acc, LoopIR.Reduce) and is_access(acc, out, i, j), REDUCE
    rhs, kk = acc.rhs, red.iter
    assert isinstance(rhs, LoopIR.BinOp) and rhs.op == "*", REDUCE
    assert isinstance(rhs.lhs, LoopIR.Read) and is_access(rhs.lhs, a, i, kk), REDUCE
    assert isinstance(rhs.rhs, LoopIR.Read) and is_access(rhs.rhs, b, kk, j), REDUCE
    return m, n, k, i, j


# ===----------------------------------------------------------------------=== #
# exo -> llvm ir -> air
# ===----------------------------------------------------------------------=== #


class MetalIRGenerator(IRGenerator):
    # par grid -> trailing i32 thread id
    def __init__(self, row: Sym, col: Sym, cols: int):
        super().__init__()
        self.row, self.col, self.cols = row, col, cols

    def _stmt_for(self, for_stmt: LoopIR.For) -> None:
        if for_stmt.iter not in (self.row, self.col):
            return super()._stmt_for(for_stmt)
        for stmt in for_stmt.body:
            self._stmt(stmt)

    def generate(self, procs: list[LoopIR.proc]) -> ModuleOp:
        # bare ptr args, no malloc/free decls
        assert len(procs) == 1, "metal backend: expected exactly one kernel procedure"
        proc = procs[0]
        input_types = [self._arg_type(arg, proc.body) for arg in proc.args] + [i32]
        region = Region(block := Block(arg_types=input_types))
        self.builder = Builder(insertion_point=InsertPoint.at_end(block))
        self.symbol_table = {arg.name: val for arg, val in zip(proc.args, block.args)}
        thread_id = self._emit(llvm.ZExtOp(block.args[-1], i64))
        self.symbol_table[self.row] = self._emit(llvm.UDivOp(thread_id, self._int_const(self.cols)))
        self.symbol_table[self.col] = self._emit(llvm.URemOp(thread_id, self._int_const(self.cols)))
        for stmt in proc.body:
            self._stmt(stmt)
        self.builder.insert(llvm.ReturnOp())
        fn_type = llvm.LLVMFunctionType([llvm.LLVMPointerType() if isinstance(t, MemRefType) else t for t in input_types], llvm.LLVMVoidType())
        self.module.body.block.add_op(llvm.FuncOp(proc.name, fn_type, linkage=llvm.LinkageAttr("external"), body=region))
        return self.module


# tested with air 2.7 / macos 15 / metal 3.2
AIR_HEADER = 'target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"\ntarget triple = "air64_v27-apple-macosx15.0.0"\n'
AIR_OPCODES = {"zext", "udiv", "urem", "mul", "add", "icmp", "phi", "br", "getelementptr", "load", "store", "fmul", "fadd", "ret"}
DEVICE_PTR = "float addrspace(1)*"


def air_metadata(name: str) -> str:
    buffers = [f'!{{i32 {i}, !"air.buffer", !"air.location_index", i32 {i}, i32 1, !"{access}", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"{arg}"}}' for i, (arg, access) in enumerate((("out", "air.read_write"), ("A", "air.read"), ("B", "air.read")))]
    thread = '!{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}'
    sig = f"void ({DEVICE_PTR}, {DEVICE_PTR}, {DEVICE_PTR}, i32)*"
    nodes = [*buffers, thread, "!{}", f'!{{{sig} @"{name}", !4, !{{!0, !1, !2, !3}}}}', "!{i32 2, i32 7, i32 0}", '!{!"Metal", i32 3, i32 2, i32 0}']
    named = ["!air.kernel = !{!5}", "!air.version = !{!6}", "!air.language_version = !{!7}"]
    return "\n".join(named + [f"!{i} = {node}" for i, node in enumerate(nodes)])


def to_air(llvm_ir: str, name: str) -> str:
    # adapter for MetalIRGenerator output only: one kernel, three f32 buffers, i32 thread id
    assert llvm_ir.count("define ") == 1 and "declare " not in llvm_ir, "metal backend: expected a single kernel definition"
    sig = re.search(rf'define void @"?{re.escape(name)}"?\((?:ptr (%"?[\w.]+"?)), (?:ptr (%"?[\w.]+"?)), (?:ptr (%"?[\w.]+"?)), i32 (%"?[\w.]+"?)\)\s*\{{\n(.*?)\n\}}', llvm_ir, re.DOTALL)
    assert sig, f"metal backend: unexpected kernel signature in\n{llvm_ir}"
    *ptrs, tid, body = sig.groups()
    for line in body.splitlines():
        if line.strip() and not line.strip().endswith(":"):
            op = line.split("=", 1)[1].split()[0] if "=" in line else line.split()[0]
            assert op in AIR_OPCODES, f"metal backend: unexpected llvm instruction {line.strip()!r}"
    # no allocas or other pointer spaces, so every ptr is a device buffer
    body = re.sub(r"\bptr\b", DEVICE_PTR, body)
    params = ", ".join([f'{DEVICE_PTR} nocapture noundef "air-buffer-no-alias" {p}' for p in ptrs] + [f"i32 {tid}"])
    return f'{AIR_HEADER}\ndefine void @"{name}"({params}) {{\n{body}\n}}\n\n{air_metadata(name)}\n'


def kernel_to_air(proc: Procedure) -> str:
    _, cols, _, row, col = check_matmul(proc._loopir_proc)
    llvm_module = convert_module(to_mlir(proc, MetalIRGenerator(row, col, cols)), fallback_target_triple="air64-apple-macosx")
    return to_air(str(llvm_module), proc.name())


# ===----------------------------------------------------------------------=== #
# metal runtime
# ===----------------------------------------------------------------------=== #


XCODE = "/Applications/Xcode.app/Contents/Developer"


def has_metal(env: dict[str, str]) -> bool:
    return subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], check=False, capture_output=True, env=env).returncode == 0


def metal_toolchain_env() -> dict[str, str]:
    # selected toolchain, else full xcode. clts lack metal. explicit DEVELOPER_DIR must work
    env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": "15.0"}
    if has_metal(env):
        return env
    developer_dir = os.environ.get("DEVELOPER_DIR")
    assert not developer_dir, f"metal backend: no usable metal compiler in DEVELOPER_DIR={developer_dir}"
    fallback = env | {"DEVELOPER_DIR": XCODE}
    assert has_metal(fallback), f"metal backend: no metal compiler in the selected toolchain or {XCODE}. Install the Metal Toolchain or set DEVELOPER_DIR"
    return fallback


def create_pipeline(air: str, name: str):
    device = import_module("Metal").MTLCreateSystemDefaultDevice()
    assert device, "metal backend: no metal device"
    env = metal_toolchain_env()
    with tempfile.TemporaryDirectory() as d:
        src, obj, lib = (os.path.join(d, f"kernel.{ext}") for ext in ("ll", "air", "metallib"))
        Path(src).write_text(air)
        for cmd in (["xcrun", "-sdk", "macosx", "metal", "-x", "ir", "-c", src, "-o", obj], ["xcrun", "-sdk", "macosx", "metallib", obj, "-o", lib]):
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
            assert result.returncode == 0, f"metal backend: {cmd[3]} failed:\n{result.stdout}\n{result.stderr}"
        library, error = device.newLibraryWithURL_error_(import_module("Foundation").NSURL.fileURLWithPath_(lib), None)
        assert library, f"metal backend: failed to load metallib: {error}"
        fn = library.newFunctionWithName_(name)
        assert fn, f"metal backend: function {name} not found in metallib"
        pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
        assert pso, f"metal backend: failed to create pipeline: {error}"
    return device, pso


def execute_kernel(device, pso, grid, group, buffers) -> None:
    metal = import_module("Metal")
    queue = device.newCommandQueue()
    assert queue, "metal backend: failed to create command queue"
    cmd = queue.commandBuffer()
    assert cmd, "metal backend: failed to create command buffer"
    encoder = cmd.computeCommandEncoder()
    assert encoder, "metal backend: failed to create compute encoder"
    encoder.setComputePipelineState_(pso)
    for i, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, i)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid, group)
    encoder.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == metal.MTLCommandBufferStatusCompleted, f"metal backend: command buffer failed with status {cmd.status()}: {cmd.error()}"


def compile(proc: Procedure, raw: bool = False) -> Callable[..., None]:
    if not __debug__:  # validation is assert-based
        raise RuntimeError("metal backend: python -O disables kernel validation")
    assert not raw, "raw=True is not supported by the metal backend"
    # reject before optional imports
    assert not proc.is_instr(), f"metal backend: {proc.name()} is an instr, not a kernel"
    m, n, k, _, _ = check_matmul(proc._loopir_proc)
    sizes = [4 * m * n, 4 * m * k, 4 * k * n]

    def build(proc: Procedure, ffi: FFI) -> Callable[..., None]:
        metal = import_module("Metal")
        device, pso = create_pipeline(kernel_to_air(proc), proc.name())
        grid = metal.MTLSize(m * n, 1, 1)
        group = metal.MTLSize(min(m * n, pso.maxTotalThreadsPerThreadgroup()), 1, 1)

        def run(*ptrs: int) -> None:
            buffers = [device.newBufferWithBytes_length_options_(ffi.buffer(ffi.cast("char *", ptr), size)[:], size, metal.MTLResourceStorageModeShared) for ptr, size in zip(ptrs, sizes, strict=True)]
            assert all(buffer is not None for buffer in buffers), "metal backend: buffer allocation failed"
            execute_kernel(device, pso, grid, group, buffers)
            ffi.memmove(ffi.cast("char *", ptrs[0]), buffers[0].contents().as_buffer(sizes[0]), sizes[0])

        return run

    return JITRuntime.compile(proc, compiler=build)
