import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from functools import partial
from importlib import import_module
from pathlib import Path
from string import Template
from typing import TypeGuard

import llvmlite.ir
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


def matrix_dims(arg: LoopIR.fnarg) -> list[int]:
    t = arg.type
    assert isinstance(t, T.Tensor) and not t.is_window and isinstance(t.basetype(), T.F32) and arg.mem is DRAM
    dims = [d.val for d in t.shape() if isinstance(d, LoopIR.Const) and type(d.val) is int and d.val > 0]
    assert len(dims) == len(t.shape()) == 2
    return dims


def is_loop(s: LoopIR.stmt, mode: type, extent: int) -> TypeGuard[LoopIR.For]:
    return isinstance(s, LoopIR.For) and isinstance(s.loop_mode, mode) and isinstance(s.lo, LoopIR.Const) and s.lo.val == 0 and isinstance(s.hi, LoopIR.Const) and s.hi.val == extent


def is_access(e: LoopIR.Read | LoopIR.Assign | LoopIR.Reduce, name: Sym, *syms: Sym) -> bool:
    # name[syms...], every index a bare loop variable
    return e.name == name and len(e.idx) == len(syms) and all(isinstance(x, LoopIR.Read) and x.name == s and not x.idx for x, s in zip(e.idx, syms))


def check_matmul(ir: LoopIR.proc) -> tuple[int, int, int, Sym, Sym]:
    # exo race-checks only in its c backend
    assert len(ir.args) == 3
    (m, n), (am, k), (bk, bn) = (matrix_dims(arg) for arg in ir.args)
    out, a, b = (arg.name for arg in ir.args)
    assert (am, bk, bn) == (m, k, n)
    assert m * n < 2**32

    assert len(ir.body) == 1
    rows = ir.body[0]
    assert is_loop(rows, LoopIR.Par, m) and len(rows.body) == 1
    cols = rows.body[0]
    assert is_loop(cols, LoopIR.Par, n)
    i, j, cell = rows.iter, cols.iter, cols.body

    assert cell
    init = cell[0]
    assert isinstance(init, LoopIR.Assign) and is_access(init, out, i, j)
    assert isinstance(init.rhs, LoopIR.Const) and init.rhs.val == 0

    assert len(cell) == 2
    red = cell[1]
    assert is_loop(red, LoopIR.Seq, k) and len(red.body) == 1
    acc = red.body[0]
    assert isinstance(acc, LoopIR.Reduce) and is_access(acc, out, i, j)
    rhs, kk = acc.rhs, red.iter
    assert isinstance(rhs, LoopIR.BinOp) and rhs.op == "*"
    assert isinstance(rhs.lhs, LoopIR.Read) and is_access(rhs.lhs, a, i, kk)
    assert isinstance(rhs.rhs, LoopIR.Read) and is_access(rhs.rhs, b, kk, j)
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
        assert len(procs) == 1
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


AIR_OPCODES = {"zext", "udiv", "urem", "mul", "add", "icmp", "phi", "br", "getelementptr", "load", "store", "fmul", "fadd", "ret void"}
DEVICE_PTR = "float addrspace(1)*"

# tested with air 2.7 / macos 15 / metal 3.2
AIR_MODULE = Template("""\
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

define void $fn(float addrspace(1)* nocapture noundef "air-buffer-no-alias" $out, float addrspace(1)* nocapture noundef "air-buffer-no-alias" $a, float addrspace(1)* nocapture noundef "air-buffer-no-alias" $b, i32 $tid) {
$body}

!air.kernel = !{!5}
!air.version = !{!6}
!air.language_version = !{!7}
!0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!1 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"A"}
!2 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"B"}
!3 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!4 = !{}
!5 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* $fn, !4, !{!0, !1, !2, !3}}
!6 = !{i32 2, i32 7, i32 0}
!7 = !{!"Metal", i32 3, i32 2, i32 0}
""")


def is_opaque_ptr(t: llvmlite.ir.Type) -> bool:
    return isinstance(t, llvmlite.ir.PointerType) and t.is_opaque and t == llvmlite.ir.PointerType()


def to_air(module: llvmlite.ir.Module, name: str) -> str:
    fns = module.functions
    assert len(fns) == 1 and fns[0].name == name and fns[0].blocks
    fn = fns[0]
    types = [arg.type for arg in fn.args]
    assert len(types) == 4 and all(is_opaque_ptr(t) for t in types[:3]) and types[3] == llvmlite.ir.IntType(32)
    for block in fn.blocks:
        for instr in block.instructions:
            assert instr.opname in AIR_OPCODES
    lines: list[str] = []
    fn.descr_body(lines)
    # air needs typed device pointers. no allocas, so every ptr is a buffer
    body = re.sub(r"\bptr\b", DEVICE_PTR, "".join(lines))
    out, a, b, tid = (arg.get_reference() for arg in fn.args)
    return AIR_MODULE.substitute(fn=fn.get_reference(), out=out, a=a, b=b, tid=tid, body=body)


def kernel_to_air(proc: Procedure) -> str:
    _, cols, _, row, col = check_matmul(proc._loopir_proc)
    llvm_module = convert_module(to_mlir(proc, MetalIRGenerator(row, col, cols)), fallback_target_triple="air64-apple-macosx")
    return to_air(llvm_module, proc.name())


# ===----------------------------------------------------------------------=== #
# metal runtime
# ===----------------------------------------------------------------------=== #


XCODE = "/Applications/Xcode.app/Contents/Developer"


def has_metal(env: dict[str, str]) -> bool:
    return subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], check=False, capture_output=True, env=env).returncode == 0


def metal_toolchain_env() -> dict[str, str]:
    # selected toolchain, else full xcode (clts may lack metal). explicit DEVELOPER_DIR must work
    env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": "15.0"}
    if has_metal(env):
        return env
    assert not os.environ.get("DEVELOPER_DIR")
    fallback = env | {"DEVELOPER_DIR": XCODE}
    usable = has_metal(fallback)
    assert usable
    return fallback


def run_tool(env: dict[str, str], tool: str, *args: str) -> None:
    result = subprocess.run(["xcrun", "-sdk", "macosx", tool, *args], check=False, capture_output=True, text=True, env=env)
    assert result.returncode == 0


def build_metallib(air: str, workdir: Path, env: dict[str, str]) -> Path:
    src, obj, lib = workdir / "kernel.ll", workdir / "kernel.air", workdir / "kernel.metallib"
    src.write_text(air)
    run_tool(env, "metal", "-x", "ir", "-c", str(src), "-o", str(obj))
    run_tool(env, "metallib", str(obj), "-o", str(lib))
    return lib


def load_pipeline(device, metallib: Path, name: str):
    url = import_module("Foundation").NSURL.fileURLWithPath_(str(metallib))
    library, _ = device.newLibraryWithURL_error_(url, None)
    assert library
    fn = library.newFunctionWithName_(name)
    assert fn
    pso, _ = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso
    return pso


def create_pipeline(air: str, name: str):
    device = import_module("Metal").MTLCreateSystemDefaultDevice()
    assert device
    env = metal_toolchain_env()
    with tempfile.TemporaryDirectory() as d:
        metallib = build_metallib(air, Path(d), env)
        pso = load_pipeline(device, metallib, name)
    return device, pso


def device_buffer(device, ffi: FFI, ptr: int, size: int):
    data = ffi.buffer(ffi.cast("char *", ptr), size)[:]
    buffer = device.newBufferWithBytes_length_options_(data, size, import_module("Metal").MTLResourceStorageModeShared)
    assert buffer is not None
    return buffer


def execute_kernel(device, pso, grid, group, buffers) -> None:
    metal = import_module("Metal")
    queue = device.newCommandQueue()
    assert queue
    cmd = queue.commandBuffer()
    assert cmd
    encoder = cmd.computeCommandEncoder()
    assert encoder
    encoder.setComputePipelineState_(pso)
    for i, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, i)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid, group)
    encoder.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    assert cmd.status() == metal.MTLCommandBufferStatusCompleted


def build_kernel(proc: Procedure, ffi: FFI, *, air: str, sizes: tuple[int, int, int], threads: int) -> Callable[..., None]:
    metal = import_module("Metal")
    device, pso = create_pipeline(air, proc.name())
    grid = metal.MTLSize(threads, 1, 1)
    group = metal.MTLSize(min(threads, pso.maxTotalThreadsPerThreadgroup()), 1, 1)

    def run(*ptrs: int) -> None:
        buffers = [device_buffer(device, ffi, ptr, size) for ptr, size in zip(ptrs, sizes, strict=True)]
        execute_kernel(device, pso, grid, group, buffers)
        out_size = sizes[0]
        ffi.memmove(ffi.cast("char *", ptrs[0]), buffers[0].contents().as_buffer(out_size), out_size)

    return run


def compile(proc: Procedure, raw: bool = False) -> Callable[..., None]:
    assert not raw
    # reject before optional imports
    assert not proc.is_instr()
    m, n, k, _, _ = check_matmul(proc._loopir_proc)
    sizes = (4 * m * n, 4 * m * k, 4 * k * n)
    air = kernel_to_air(proc)
    compiler = partial(build_kernel, air=air, sizes=sizes, threads=m * n)
    return JITRuntime.compile(proc, compiler=compiler)
