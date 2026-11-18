import ctypes
import ctypes.util
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from functools import cache, lru_cache
from glob import glob
from importlib import import_module
from pathlib import Path
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
        # the kernel has no noalias arg attrs (to_air only parses bare `ptr %x` arguments) and there is no gpu heap, so no malloc/free
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


def kernel_to_air(proc: Procedure) -> str:
    # exo -> llvm dialect (cpu lowering + grid) -> llvm ir text (xdsl) -> air text. pure translation, no metal toolchain or pyobjc needed
    out, row, col, _, cols = thread_grid(proc._loopir_proc)
    llvm_module = convert_module(to_mlir(proc, MetalIRGenerator(row, col, cols)), fallback_target_triple="air64-apple-macosx")  # to_air replaces the triple and datalayout
    kernel_args = llvm_module.get_global(proc.name()).args
    overrides = {proc.name(): {kernel_args[out].name: "out", kernel_args[-1].name: "id"}}  # to_air derives read_write access and thread_position_in_grid from these names
    return to_air(str(llvm_module), kernel_overrides=overrides)


def compile(proc: Procedure, raw: bool = False) -> Callable[..., None]:
    # exo -> air (kernel_to_air) -> metallib (xcrun) -> metal pipeline, wrapped by the cpu jit's argument conversion
    assert not raw, "raw=True is not supported by the metal backend"
    ir = proc._loopir_proc
    out, _, _, rows, cols = thread_grid(ir)  # reject before compiling or importing optional deps

    def compile_kernel(proc: Procedure, ffi: FFI) -> Callable[..., None]:
        metal = import_module("Metal")  # optional dep, only loaded on the metal path
        device, pso = create_compute_pipeline(compile_to_metallib(kernel_to_air(proc)), proc.name())
        sizes = [4 * math.prod(dim.val for dim in arg.type.shape()) for arg in ir.args]
        threads = rows * cols

        def run(*ptrs: int) -> None:
            # buffer index i is kernel argument i, the trailing thread id is not a buffer
            buffers = [device.newBufferWithBytes_length_options_(ffi.buffer(ffi.cast("char *", ptr), size)[:], size, metal.MTLResourceStorageModeShared) for ptr, size in zip(ptrs, sizes, strict=True)]
            assert all(buffer is not None for buffer in buffers), "metal backend: buffer allocation failed"

            def encode(encoder: Any) -> None:
                for i, buffer in enumerate(buffers):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, i)

            execute_kernel(device, pso, metal.MTLSize(threads, 1, 1), metal.MTLSize(min(threads, pso.maxTotalThreadsPerThreadgroup()), 1, 1), encode)
            ffi.memmove(ffi.cast("char *", ptrs[out]), buffers[out].contents().as_buffer(sizes[out]), sizes[out])

        return run

    return JITRuntime.compile(proc, compiler=compile_kernel)


# ===----------------------------------------------------------------------=== #
# toolchain probes
# ===----------------------------------------------------------------------=== #


# the air module header and footer must match the local metal toolchain, with fallbacks so translation also runs off macos


def get_mac_version() -> str:
    return platform.mac_ver()[0] or "14.0.0"


def metal_probe_cmd() -> list[str] | None:
    # metadata probes only run on macos with metal on PATH, through the same selected xcrun toolchain as compile_to_metallib
    if platform.system() != "Darwin" or not (metal := shutil.which("metal")):
        return None
    return ["xcrun", "-sdk", "macosx", "metal"] if shutil.which("xcrun") else [metal]


def get_metal_version() -> str:
    default_version = "Apple metal version 32023.830 (metalfe-32023.830.2)"
    if not (cmd := metal_probe_cmd()):
        return default_version
    result = subprocess.run(cmd + ["--version"], check=False, capture_output=True, text=True, env=metal_toolchain_env())
    lines = result.stdout.splitlines() if result.returncode == 0 and result.stdout else []
    return lines[0].strip() if lines and "Apple metal version" in lines[0] else default_version


@lru_cache(maxsize=1)
def get_target_datalayout() -> str:
    default = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
    if not (cmd := metal_probe_cmd()):
        return default
    with tempfile.TemporaryDirectory() as d:
        Path(src := f"{d}/t.metal").write_text("kernel void e(){}")
        subprocess.run(cmd + ["-S", "-emit-llvm", src, "-o", out := f"{d}/t.ll"], check=True, capture_output=True, env=metal_toolchain_env())
        if m := re.search(r'target datalayout = "([^"]+)"', Path(out).read_text()):
            return m.group(1)
    return default


@cache
def metal_toolchain_env() -> dict[str, str]:
    # subprocess env for xcrun, os.environ itself is never mutated. xcode-select often points at the command line tools,
    # which lack the metal compiler: an explicit DEVELOPER_DIR is respected as is, otherwise the first full /Applications/Xcode*.app with metal is used
    # the translator emits air 2.7 (see the !air.version metadata), newer compilers target air 2.8 unless the deployment target is macos 15
    env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": os.environ.get("MACOSX_DEPLOYMENT_TARGET") or "15.0"}

    def has_metal(**overrides: str) -> bool:
        return subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], check=False, capture_output=True, env=env | overrides).returncode == 0

    if developer_dir := os.environ.get("DEVELOPER_DIR"):
        assert has_metal(), f"metal backend: no usable metal compiler in DEVELOPER_DIR={developer_dir}"
        return env
    if has_metal():
        return env
    for app in sorted(glob("/Applications/Xcode*.app"), key=lambda app: (app != "/Applications/Xcode.app", app)):
        if has_metal(DEVELOPER_DIR=(developer_dir := f"{app}/Contents/Developer")):
            return env | {"DEVELOPER_DIR": developer_dir}
    raise AssertionError("metal backend: no full Xcode with the Metal Toolchain found in /Applications. Install one or set DEVELOPER_DIR")


# ===----------------------------------------------------------------------=== #
# llvm ir -> air lowering
# ===----------------------------------------------------------------------=== #


# `llvm ir` -> `air llvm` lowering: rewrite plain llvm kernels into the dialect the metal backend accepts
#
# air is textual llvm ir plus apple's gpu abi: addrspace-qualified pointers, constant buffers for scalars,
# air.* intrinsics and !air.* metadata that binds each argument to a buffer slot. none of this exists in the input,
# so it is reconstructed from argument names and per-function addrspace tracking with regex text rewrites.
#
# pipeline order matters:
# ----------------------
#     1. signature      - buffers become addrspace(1|3) pointers, scalars become addrspace(2) constant buffers
#     2. instructions   - per-line rewrites that consume the addrspace map produced by the signature
#     3. metadata       - declarations and !air.* nodes, emitted last because they need every kernel and intrinsic seen
#
# example (signature + entry, scalar %n is loaded once and its uses renamed to %val_n):
# ------------------------------------------------------------------------------------
#     define void @k(float* %out, i32 %n, i32 %id) {
#     entry:
#     =>
#     define void @k(float addrspace(1)* nocapture noundef "air-buffer-no-alias" %out, i32 addrspace(2)* nocapture noundef readonly align 4 dereferenceable(4) "air-buffer-no-alias" %n, i32 %id) #0 {
#     entry:
#       %val_n = load i32, i32 addrspace(2)* %n, align 4


KernelArg = tuple[str, str, bool]  # (air type, semantic name, is output)

# thread position args stay plain values in addrspace 0, the signature and the metadata must agree on these names
GRID_POSITION_NAMES = ("id", "gid", "global_id")
THREADGROUP_POSITION_NAMES = ("tid", "lid", "local_id")

LLVM_TO_AIR_TYPES = {"float": "f32", "double": "f64", "i32": "i32", "i16": "i16", "i8": "i8", "i64": "i64"}
AIR_TO_LLVM_TYPES = {air: llvm for llvm, air in LLVM_TO_AIR_TYPES.items()}


def get_type_info(type_str: str) -> tuple[str, int, int] | None:
    # maps llvm type to (name, size, align) tuple.
    # examples:
    #   "i32" -> ("int", 4, 4)
    #   "<4 x float>" -> ("float", 16, 16)
    t = type_str.strip()

    # vector types: <N x type>, aligned to their full size
    if t.startswith("<") and t.endswith(">") and (match := re.search(r"<\s*(\d+)\s+x\s+([\w\d\.]+)\s*>", t)):
        info = get_type_info(match.group(2))
        assert info is not None, f"unsupported vector element type: {t}"
        base_name, elem_size, _ = info
        total_size = elem_size * int(match.group(1))
        return (base_name, total_size, total_size)

    # substring match in order, so pointer spellings resolve to their element and bare opaque `ptr` defaults to float
    scalars = [("double", ("double", 8, 8)), ("float", ("float", 4, 4)), ("i64", ("long", 8, 8)), ("i32", ("int", 4, 4)), ("i16", ("short", 2, 2)), ("i8", ("char", 1, 1)), ("ptr", ("float", 4, 4))]
    return next((info for key, info in scalars if key in t), None)


class SignatureParser:
    @staticmethod
    def parse(lines: list[str], start_idx: int, kernel_overrides: dict[str, dict[str, str]] | None = None) -> tuple[str, list[KernelArg], str, int, dict[str, int], dict[str, str]]:
        # rewrites one `define void` header and returns the per-function state the body rewrite consumes
        pkg, idx = SignatureParser._read_signature_lines(lines, start_idx)

        sig_match = re.search(r"define\s+void\s+@\"?([\w\.]+)\"?\s*\((.*?)\).*?{", pkg, re.DOTALL)
        assert sig_match, f"Failed to parse function signature: {pkg}"
        func_name, raw_args = sig_match.group(1), sig_match.group(2)
        arg_map = kernel_overrides.get(func_name, {}) if kernel_overrides else {}

        var_addrspaces: dict[str, int] = {}
        scalar_loads: dict[str, str] = {}
        args_list, new_sig_parts = SignatureParser._process_arguments(raw_args, var_addrspaces, scalar_loads, arg_map)

        air_sig = f"define void @{func_name}({', '.join(new_sig_parts)}) #0 {{"
        return func_name, args_list, air_sig, idx + 1, var_addrspaces, scalar_loads

    @staticmethod
    def _read_signature_lines(lines: list[str], start_idx: int) -> tuple[str, int]:
        # handles multi-line function signatures
        pkg = lines[start_idx]
        i = start_idx
        while "{" not in pkg:
            i += 1
            pkg += " " + lines[i]
        return pkg, i

    @staticmethod
    def _process_arguments(raw_args: str, var_addrspaces: dict[str, int], scalar_loads: dict[str, str], arg_map: dict[str, str] | None = None) -> tuple[list[KernelArg], list[str]]:
        new_sig_parts = []
        args_list = []
        for arg_chunk in (x.strip() for x in raw_args.split(",")):
            if not arg_chunk:
                continue
            *type_parts, clean_name = arg_chunk.split()
            a_type = " ".join(type_parts)
            name_no_prefix = clean_name.replace("%", "").replace('"', "")

            # overrides give generated names like %0 a semantic role, the metadata generator keys off the semantic name
            semantic_name = (arg_map or {}).get(name_no_prefix, name_no_prefix)
            res_type, is_output, sig_part = SignatureParser._process_single_argument(a_type, clean_name, semantic_name, var_addrspaces, scalar_loads)
            new_sig_parts.append(sig_part)
            args_list.append((res_type, semantic_name, is_output))
        return args_list, new_sig_parts

    @staticmethod
    def _process_single_argument(a_type: str, clean_name: str, name_no_prefix: str, var_addrspaces: dict[str, int], scalar_loads: dict[str, str]) -> tuple[str, bool, str]:
        if "*" in a_type or a_type == "ptr":
            return SignatureParser._process_buffer_argument(a_type, clean_name, name_no_prefix, var_addrspaces)
        return SignatureParser._process_scalar_argument(a_type, clean_name, name_no_prefix, var_addrspaces, scalar_loads)

    @staticmethod
    def _process_buffer_argument(a_type: str, clean_name: str, name_no_prefix: str, var_addrspaces: dict[str, int]) -> tuple[str, bool, str]:
        # llvm ir has no memory or access qualifiers, so threadgroup memory and read_write outputs are inferred from names
        as_id = 3 if "shared" in clean_name else 1
        var_addrspaces[clean_name] = as_id
        is_output = any(x in clean_name.lower() for x in ["out", "result"]) or clean_name in ["%C", "%c"] or name_no_prefix in ["out", "result"]

        res_type = a_type
        if a_type == "ptr":  # default to float for opaque pointers
            res_type = f"float addrspace({as_id})*"
        elif "addrspace" not in a_type:
            res_type = a_type.replace("*", f" addrspace({as_id})*")

        sig_part = f'{res_type} nocapture noundef "air-buffer-no-alias" {clean_name}'
        return res_type, is_output, sig_part

    @staticmethod
    def _process_scalar_argument(a_type: str, clean_name: str, name_no_prefix: str, var_addrspaces: dict[str, int], scalar_loads: dict[str, str]) -> tuple[str, bool, str]:
        if name_no_prefix in GRID_POSITION_NAMES + THREADGROUP_POSITION_NAMES:
            var_addrspaces[clean_name] = 0
            return a_type, False, f"{a_type} {clean_name}"

        # metal binds scalar kernel args as constant buffers (addrspace 2), the body reads them through one load at entry
        base_type = a_type.strip()
        ptr_type = f"{base_type} addrspace(2)*"
        info = get_type_info(base_type)
        assert info is not None, f"unsupported scalar argument type: {base_type}"
        _, size, align = info

        var_addrspaces[clean_name] = 2
        sig_part = f'{ptr_type} nocapture noundef readonly align {align} dereferenceable({size}) "air-buffer-no-alias" {clean_name}'
        scalar_loads[clean_name] = f"%val_{name_no_prefix}"
        return ptr_type, False, sig_part


class IntrinsicHandler:
    @staticmethod
    def handle_type_casts(line: str, used_intrinsics: set[str]) -> str:
        # the metal backend rejects raw int<->float cast opcodes, they must become air.convert calls
        conversions = [("uitofp", "@air.convert.f.{dst}.u.{src}", "tail call fast"), ("sitofp", "@air.convert.f.{dst}.s.{src}", "tail call fast"), ("fptoui", "@air.convert.u.{dst}.f.{src}", "tail call"), ("fptosi", "@air.convert.s.{dst}.f.{src}", "tail call")]
        for opcode, intr_template, call_attrs in conversions:
            if not (match := re.search(rf"(%\S+)\s*=\s*{opcode}\s+(\S+)\s+(%\S+)\s+to\s+(\S+)", line)):
                continue
            res, src_t, src_v, dst_t = match.groups()
            intr = intr_template.format(src=LLVM_TO_AIR_TYPES.get(src_t, src_t), dst=LLVM_TO_AIR_TYPES.get(dst_t, dst_t))
            used_intrinsics.add(intr)
            return f"  {res} = {call_attrs} {dst_t} {intr}({src_t} {src_v})"
        return line

    @staticmethod
    def replace_intrinsics(line: str, used_intrinsics: set[str]) -> str:
        if "call" not in line:
            return line
        math_ops = ["exp", "log", "sin", "cos", "sqrt", "ceil", "floor", "fabs", "pow", "tanh", "fma", "trunc", "round"]
        replacements = {f"@llvm.{op}.f32": f"@air.{op}.f32" for op in math_ops} | {"@llvm.minnum.f32": "@air.fmin.f32", "@llvm.maxnum.f32": "@air.fmax.f32"}
        for old, new in replacements.items():
            if old in line:
                line = line.replace(old, new)
                used_intrinsics.add(new)
        return line


class MetadataGenerator:
    @staticmethod
    def emit(kernels: list[tuple[str, list[KernelArg]]], used_intrinsics: set[str], metal_version_str: str | None = None) -> list[str]:
        lines = ["\ndeclare void @air.wg.barrier(i32, i32) local_unnamed_addr #1"]

        # sorted so repeated translations of the same input are byte-identical
        for intr in sorted(used_intrinsics):
            lines.extend(MetadataGenerator._emit_intrinsic_decl(intr))

        # #0 tags kernel definitions, #1 the barrier declaration, #2 intrinsic declarations and barrier call sites
        lines.append('attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
        lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
        lines.append("attributes #2 = { convergent nounwind willreturn }")

        if kernels:
            lines.extend(MetadataGenerator._generate_kernel_metadata(kernels, metal_version_str))
        return lines

    @staticmethod
    def _emit_intrinsic_decl(intr: str) -> list[str]:
        if "air.convert" in intr:
            # @air.convert.f.f32.u.i32 -> [f, f32, u, i32], index 1 is the dest type and index 3 the src type
            parts = intr.replace("@air.convert.", "").split(".")
            ret_type = AIR_TO_LLVM_TYPES.get(parts[1], parts[1])
            arg_type = AIR_TO_LLVM_TYPES.get(parts[3], parts[3])
            return [f"declare {ret_type} {intr}({arg_type}) #2"]

        # binary ops are checked first because "fmax" contains "fma"
        arg_types = "(float, float)" if any(x in intr for x in ["pow", "fmin", "fmax"]) else "(float, float, float)" if "fma" in intr else "(float)"
        name = intr.replace("@", "")
        return [f"declare float @{name}{arg_types} #2"]

    @staticmethod
    def _generate_kernel_metadata(kernels: list[tuple[str, list[KernelArg]]], metal_version_str: str | None = None) -> list[str]:
        lines: list[str] = []

        # every line here is one numbered node, so the next node id is the line count
        def m(content: str) -> str:
            lines.append(f"!{len(lines)} = {content}")
            return f"!{len(lines) - 1}"

        kernel_nodes = []
        for func_name, args_list in kernels:
            arg_meta_refs = []
            # buffer slots: device(1) and constant(2) share one location counter, threadgroup(3) has its own
            loc_indices = {1: 0, 3: 0}
            for idx, (at, an, is_out) in enumerate(args_list):
                content, as_id = MetadataGenerator._get_air_metadata_content(at, an, is_out)
                if "{loc_index}" in content:
                    link_id = 1 if as_id in [1, 2] else 3
                    content = content.replace("{loc_index}", str(loc_indices[link_id]))
                    loc_indices[link_id] += 1
                if "air.arg_name" not in content:
                    content += f', !"air.arg_name", !"{an}"'
                arg_meta_refs.append(m(f"!{{i32 {idx}, {content}}}"))

            empty = m("!{}")
            sig_str = f"void ({', '.join(arg_type for arg_type, _, _ in args_list)})*"
            kernel_nodes.append(m(f"!{{{sig_str} @{func_name}, {empty}, !{{{', '.join(arg_meta_refs)}}}}}"))

        descriptors = ['!"air.compile.denorms_disable"', '!"air.compile.fast_math_enable"', '!"air.compile.framebuffer_fetch_enable"', f'!"{metal_version_str}"']
        desc_refs = [m(f"!{{{d}}}") for d in descriptors]
        version = m("!{i32 2, i32 7, i32 0}")
        metal_ver = m('!{!"Metal", i32 3, i32 2, i32 0}')
        src_file = m('!{!"input.ll"}')

        # named metadata goes before the numbered nodes it references
        top_meta = [f"!air.kernel = !{{{', '.join(kernel_nodes)}}}", f"!air.compile_options = !{{{', '.join(desc_refs[:3])}}}", f"!llvm.ident = !{{{desc_refs[3]}}}", f"!air.version = !{{{version}}}", f"!air.language_version = !{{{metal_ver}}}", f"!air.source_file_name = !{{{src_file}}}", ""]
        return top_meta + lines

    @staticmethod
    def _get_air_metadata_content(arg_type: str, arg_name: str, is_output: bool) -> tuple[str, int]:
        if arg_name in GRID_POSITION_NAMES:
            return ('!"air.thread_position_in_grid", !"air.arg_type_name", !"uint"', 0)
        if arg_name in THREADGROUP_POSITION_NAMES:
            return ('!"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint"', 0)
        if "addrspace" not in arg_type and "*" not in arg_type:
            return (f'!"air.arg_type_name", !"{arg_type.strip()}"', 0)

        base_t_str = re.sub(r"addrspace\(\d+\)", "", arg_type.replace("*", "")).strip()
        info = get_type_info(base_t_str)
        assert info is not None, f"unsupported buffer element type: {base_t_str}"
        base_name, size, align = info

        # TODO: hacky "magic string" dependency, fix at MLIR lowering phase
        as_id = 3 if "addrspace(3)" in arg_type else (2 if "addrspace(2)" in arg_type else 1)
        access_mode = '!"air.read"' if (as_id == 2 or not is_output) else '!"air.read_write"'

        # {loc_index} is filled in by the caller, which owns the per-kernel slot counters
        meta_parts = ['!"air.buffer"', '!"air.location_index", i32 {loc_index}, i32 1', access_mode, f'!"air.address_space", i32 {as_id}', f'!"air.arg_type_size", i32 {size}', f'!"air.arg_type_align_size", i32 {align}', f'!"air.arg_type_name", !"{base_name}"']
        if as_id == 2:
            meta_parts.insert(1, f'!"air.buffer_size", i32 {size}')
        return (", ".join(meta_parts), as_id)


# ===----------------------------------------------------------------------=== #
# llvm-to-air
# ===----------------------------------------------------------------------=== #


class AirTranslator:
    lines: list[str]
    output_lines: list[str]
    kernels: list[tuple[str, list[KernelArg]]]
    kernel_overrides: dict[str, dict[str, str]] | None
    used_intrinsics: set[str]
    # per-function state, replaced by SignatureParser.parse at every `define void`
    var_addrspaces: dict[str, int]
    scalar_loads: dict[str, str]

    def __init__(self, llvm_ir: str, kernel_overrides: dict[str, dict[str, str]] | None = None):
        self.lines = llvm_ir.splitlines()
        self.output_lines = []
        self.kernels = []
        self.kernel_overrides = kernel_overrides
        self.used_intrinsics = set()
        self.var_addrspaces = {}
        self.scalar_loads = {}

    def translate(self) -> str:
        self.output_lines.append(f'target datalayout = "{get_target_datalayout()}"')
        self.output_lines.append(f'target triple = "air64_v27-apple-macosx{get_mac_version()}"\n')

        # only kernels, comments and labels survive at module scope
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            stripped = line.strip()
            if stripped.startswith("define void"):
                i = self._process_function(i)
                continue
            if stripped.startswith(";") or stripped.endswith(":"):
                self.output_lines.append(line)
            i += 1

        self.output_lines.extend(MetadataGenerator.emit(self.kernels, self.used_intrinsics, get_metal_version()))
        return "\n".join(self.output_lines)

    def _process_function(self, start_idx: int) -> int:
        func_name, args_list, air_sig, idx, self.var_addrspaces, self.scalar_loads = SignatureParser.parse(self.lines, start_idx, self.kernel_overrides)
        self.kernels.append((func_name, args_list))
        self.output_lines.append(air_sig)

        idx = self._skip_comments_and_empty(idx)
        has_line = idx < len(self.lines)
        if has_line and self.lines[idx].strip() == "}":
            self.output_lines.append("}")
            return idx + 1

        # scalar loads must follow the entry label, a label cannot come after instructions in its block
        if has_line and self.lines[idx].strip().endswith(":"):
            self.output_lines.append(self.lines[idx])
            idx += 1
        self._insert_scalar_loads(args_list)
        return self._convert_body(idx)

    def _skip_comments_and_empty(self, idx: int) -> int:
        while idx < len(self.lines):
            stripped = self.lines[idx].strip()
            if stripped and not stripped.startswith(";"):
                break
            self.output_lines.append(self.lines[idx])
            idx += 1
        return idx

    def _insert_scalar_loads(self, args_list: list[KernelArg]) -> None:
        for param_name, loaded_var in self.scalar_loads.items():
            name_check = param_name.replace("%", "")
            base_type = next((arg.split("addrspace")[0].strip() for arg, name, _ in args_list if name == name_check), None)
            if base_type:
                self.output_lines.append(f"  {loaded_var} = load {base_type}, {base_type} addrspace(2)* {param_name}, align 4")

    def _convert_body(self, idx: int) -> int:
        while idx < len(self.lines):
            line = self.lines[idx]
            stripped = line.strip()
            if stripped == "}":
                self.output_lines.append("}")
                return idx + 1
            is_passthrough = stripped.startswith(";") or stripped.endswith(":")
            self.output_lines.append(line if is_passthrough else self._convert_instruction(line))
            idx += 1
        return idx

    def _convert_instruction(self, line: str) -> str:
        # any barrier call becomes a threadgroup barrier that fences threadgroup memory (mem_flags 2)
        if "call" in line and "barrier" in line:
            return "  tail call void @air.wg.barrier(i32 2, i32 1) #2"

        # pointer rewrites must precede propagation, which reads the addrspace(N) they just inserted
        line = IntrinsicHandler.handle_type_casts(line, self.used_intrinsics)
        line = IntrinsicHandler.replace_intrinsics(line, self.used_intrinsics)
        line = self._rewrite_pointers(line)
        line = self._rewrite_opaque_pointers(line)
        self._propagate_address_spaces(line)
        return self._apply_scalar_loads(line)

    def _rewrite_pointers(self, line: str) -> str:
        def replacer(m: re.Match[str]) -> str:
            type_part, var_part = m.group(1), m.group(2)
            as_id = self.var_addrspaces.get(var_part, 0)
            return f"{type_part} addrspace({as_id})* {var_part}" if as_id > 0 else m.group(0)

        return re.sub(r"([\w\s<>\.]+)\*\s+(%[\w\.\"]+)", replacer, line)

    def _rewrite_opaque_pointers(self, line: str) -> str:
        def replacer(m: re.Match[str]) -> str:
            var_part = m.group(1)
            as_id = self.var_addrspaces.get(var_part, 0)
            # assume float for opaque pointers
            return f"float addrspace({as_id})* {var_part}" if as_id > 0 else m.group(0)

        return re.sub(r"\bptr\s+(%[\w\.\"]+)", replacer, line)

    def _propagate_address_spaces(self, line: str) -> None:
        # pointers derived via gep/bitcast/select inherit their base's addrspace so later loads/stores get rewritten too
        if "=" not in line or not (lhs_match := re.search(r"(%[\w\.\"]+)\s*=", line)):
            return
        if not any(x in line for x in ["getelementptr", "bitcast", "select"]):
            return
        # addrspace(N)* | ptr addrspace(N)
        if as_match := re.search(r"addrspace\((\d+)\)", line):
            self.var_addrspaces[lhs_match.group(1)] = int(as_match.group(1))

    def _apply_scalar_loads(self, line: str) -> str:
        for param_name, loaded_var in self.scalar_loads.items():
            # use word boundaries to avoid partial replacements
            line = re.sub(re.escape(param_name) + r"\b", loaded_var, line)
        return line


def to_air(llvm_ir_text: str, kernel_overrides: dict[str, dict[str, str]] | None = None) -> str:
    # llvm ir text -> air llvm text
    #
    # kernel_overrides renames arguments per kernel ({kernel: {raw arg name: semantic name}}) for generators
    # like mlir or tvm that emit non-semantic names. the semantic name drives the abi:
    #
    #     define void @vector_add(float* %0, float* %1, float* %2, i32 %3) { ... }
    #     kernel_overrides = {"vector_add": {"2": "out", "3": "gid"}}
    #     =>  %2 is a read_write buffer, %3 gets air.thread_position_in_grid metadata
    assert llvm_ir_text
    return AirTranslator(llvm_ir_text, kernel_overrides).translate()


# ===----------------------------------------------------------------------=== #
# metal runtime
# ===----------------------------------------------------------------------=== #


def compile_to_metallib(air_llvm_ir: str) -> bytes:
    assert shutil.which("xcrun") is not None, "xcrun not found"
    env = metal_toolchain_env()
    for tool in ("metal", "metallib"):
        assert subprocess.run(["xcrun", "-sdk", "macosx", tool, "--version"], check=False, capture_output=True, env=env).returncode == 0, f"{tool} not found"
    assert air_llvm_ir
    with tempfile.NamedTemporaryFile(suffix=".ll") as f_ll, tempfile.NamedTemporaryFile(suffix=".air") as f_air, tempfile.NamedTemporaryFile(suffix=".metallib") as f_lib:
        f_ll.write(air_llvm_ir.encode("utf-8"))
        f_ll.flush()
        # argv lists, no shell: temp paths with spaces or shell metacharacters stay single arguments. the linker only runs after a successful compile
        steps = (("metal", ["xcrun", "-sdk", "macosx", "metal", "-x", "ir", "-c", f_ll.name, "-o", f_air.name]), ("metallib", ["xcrun", "-sdk", "macosx", "metallib", f_air.name, "-o", f_lib.name]))
        for tool, cmd in steps:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
            assert result.returncode == 0, f"{tool} compilation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        lib_data = Path(f_lib.name).read_bytes()
        assert len(lib_data) > 0, "generated metallib is empty"
        return lib_data


# pyobjc is imported on first runtime call, so translation and metallib builds work without it
@cache  # one pipeline per (metallib, kernel), repeated dispatches reuse it
def create_compute_pipeline(metallib_binary: bytes, kernel_name: str):
    device = import_module("Metal").MTLCreateSystemDefaultDevice()
    assert device, "metal not supported on this device"

    # newLibraryWithData_error_ takes a dispatch_data_t, built here through libdispatch via ctypes
    libdispatch = ctypes.CDLL(ctypes.util.find_library("dispatch") or "/usr/lib/system/libdispatch.dylib")
    # dispatch_data_create(const void *buffer, size_t size, dispatch_queue_t queue, dispatch_block_t destructor)
    libdispatch.dispatch_data_create.restype = ctypes.c_void_p
    libdispatch.dispatch_data_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]

    # a NULL destructor makes libdispatch copy the bytes, so c_data may be freed when this function returns
    data_len = len(metallib_binary)
    c_data = ctypes.create_string_buffer(metallib_binary, data_len)
    dispatch_data = libdispatch.dispatch_data_create(c_data, data_len, None, None)
    assert dispatch_data, "failed to create dispatch_data"

    # bridge c_void_p to object for PyObjC
    ns_dispatch_data = import_module("objc").objc_object(c_void_p=dispatch_data)
    library, error = device.newLibraryWithData_error_(ns_dispatch_data, None)
    assert library, f"error loading library: {error}"

    fn = library.newFunctionWithName_(kernel_name)
    assert fn, f"function '{kernel_name}' not found."
    pso, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    assert pso, f"error creating pipeline state: {error}"
    return device, pso


def execute_kernel(device, pso, grid_size, threadgroup_size, encode_args_fn: Callable[[Any], None]) -> float:
    metal = import_module("Metal")
    queue = device.newCommandQueue()
    assert queue, "failed to create command queue"
    cmd_buffer = queue.commandBuffer()
    assert cmd_buffer, "failed to create command buffer"
    encoder = cmd_buffer.computeCommandEncoder()
    assert encoder, "failed to create compute command encoder"

    # buffer bindings are kernel-specific, so the caller encodes them between pipeline selection and dispatch
    encoder.setComputePipelineState_(pso)
    encode_args_fn(encoder)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    encoder.endEncoding()

    # timing covers commit to completion, encoding above is excluded
    start_time = time.perf_counter()
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()
    end_time = time.perf_counter()

    status = cmd_buffer.status()
    assert status == metal.MTLCommandBufferStatusCompleted, f"command buffer failed with status {status} and error: {cmd_buffer.error()}"
    return end_time - start_time
