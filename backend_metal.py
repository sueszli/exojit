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
from functools import cache
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
# toolchain probes
# ===----------------------------------------------------------------------=== #


def has_tool(tool: str, env: dict[str, str]) -> bool:
    return subprocess.run(["xcrun", "-sdk", "macosx", tool, "--version"], check=False, capture_output=True, env=env).returncode == 0


@cache
def metal_toolchain_env() -> dict[str, str]:
    # clts lack metal -> DEVELOPER_DIR, then Xcode*.app
    # air 2.7 needs macos 15 target
    env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": os.environ.get("MACOSX_DEPLOYMENT_TARGET") or "15.0"}
    if has_tool("metal", env):
        return env
    assert not (developer_dir := os.environ.get("DEVELOPER_DIR")), f"metal backend: no usable metal compiler in DEVELOPER_DIR={developer_dir}"
    for app in sorted(glob("/Applications/Xcode*.app"), key=lambda app: (app != "/Applications/Xcode.app", app)):
        if has_tool("metal", candidate := env | {"DEVELOPER_DIR": f"{app}/Contents/Developer"}):
            return candidate
    raise AssertionError("metal backend: no full Xcode with the Metal Toolchain found in /Applications. Install one or set DEVELOPER_DIR")


def metal_probe_cmd() -> list[str] | None:
    # same xcrun toolchain as compile
    if platform.system() != "Darwin" or not (metal := shutil.which("metal")):
        return None
    return ["xcrun", "-sdk", "macosx", "metal"] if shutil.which("xcrun") else [metal]


def get_mac_version() -> str:
    return platform.mac_ver()[0] or "14.0.0"


def get_metal_version() -> str:
    default_version = "Apple metal version 32023.830 (metalfe-32023.830.2)"
    if not (cmd := metal_probe_cmd()):
        return default_version
    result = subprocess.run(cmd + ["--version"], check=False, capture_output=True, text=True, env=metal_toolchain_env())
    lines = result.stdout.splitlines() if result.returncode == 0 and result.stdout else []
    return lines[0].strip() if lines and "Apple metal version" in lines[0] else default_version


@cache
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


# ===----------------------------------------------------------------------=== #
# llvm ir -> air lowering
# ===----------------------------------------------------------------------=== #


KernelArg = tuple[str, str, bool]  # (air type, semantic name, is output)

# addrspace 0, shared with metadata
GRID_POSITION_NAMES = ("id", "gid", "global_id")
THREADGROUP_POSITION_NAMES = ("tid", "lid", "local_id")

LLVM_TO_AIR_TYPES = {"float": "f32", "double": "f64", "i32": "i32", "i16": "i16", "i8": "i8", "i64": "i64"}
AIR_TO_LLVM_TYPES = {air: llvm for llvm, air in LLVM_TO_AIR_TYPES.items()}


def get_type_info(type_str: str) -> tuple[str, int, int] | None:
    t = type_str.strip()
    if t.startswith("<") and t.endswith(">") and (match := re.search(r"<\s*(\d+)\s+x\s+([\w\d\.]+)\s*>", t)):
        info = get_type_info(match.group(2))
        assert info is not None, f"unsupported vector element type: {t}"
        total_size = info[1] * int(match.group(1))
        return (info[0], total_size, total_size)
    # substring match, order matters
    scalars = [("double", ("double", 8, 8)), ("float", ("float", 4, 4)), ("i64", ("long", 8, 8)), ("i32", ("int", 4, 4)), ("i16", ("short", 2, 2)), ("i8", ("char", 1, 1)), ("ptr", ("float", 4, 4))]
    return next((info for key, info in scalars if key in t), None)


class SignatureParser:
    @staticmethod
    def parse(lines: list[str], start_idx: int, kernel_overrides: dict[str, dict[str, str]] | None = None) -> tuple[str, list[KernelArg], str, int, dict[str, int], dict[str, str]]:
        idx, pkg = start_idx, lines[start_idx]
        while "{" not in pkg:
            idx += 1
            pkg += " " + lines[idx]
        sig_match = re.search(r"define\s+void\s+@\"?([\w\.]+)\"?\s*\((.*?)\).*?{", pkg, re.DOTALL)
        assert sig_match, f"Failed to parse function signature: {pkg}"
        func_name, raw_args = sig_match.group(1), sig_match.group(2)
        var_addrspaces: dict[str, int] = {}
        scalar_loads: dict[str, str] = {}
        args_list, new_sig_parts = SignatureParser._process_arguments(raw_args, var_addrspaces, scalar_loads, (kernel_overrides or {}).get(func_name, {}))
        return func_name, args_list, f"define void @{func_name}({', '.join(new_sig_parts)}) #0 {{", idx + 1, var_addrspaces, scalar_loads

    @staticmethod
    def _process_arguments(raw_args: str, var_addrspaces: dict[str, int], scalar_loads: dict[str, str], arg_map: dict[str, str] | None = None) -> tuple[list[KernelArg], list[str]]:
        new_sig_parts = []
        args_list = []
        for arg_chunk in filter(None, (x.strip() for x in raw_args.split(","))):
            *type_parts, llvm_name = arg_chunk.split()
            a_type = " ".join(type_parts)
            name_no_prefix = llvm_name.replace("%", "").replace('"', "")
            # metadata keys off semantic name
            semantic_name = (arg_map or {}).get(name_no_prefix, name_no_prefix)
            if "*" in a_type or a_type == "ptr":
                res_type, is_output, sig_part = SignatureParser._process_buffer_argument(a_type, llvm_name, semantic_name, var_addrspaces)
            else:
                res_type, is_output, sig_part = SignatureParser._process_scalar_argument(a_type, llvm_name, semantic_name, var_addrspaces, scalar_loads)
            new_sig_parts.append(sig_part)
            args_list.append((res_type, semantic_name, is_output))
        return args_list, new_sig_parts

    @staticmethod
    def _process_buffer_argument(a_type: str, llvm_name: str, semantic_name: str, var_addrspaces: dict[str, int]) -> tuple[str, bool, str]:
        # no qualifiers in llvm ir -> infer from names
        as_id = 3 if "shared" in llvm_name else 1
        var_addrspaces[llvm_name] = as_id
        is_output = any(x in llvm_name.lower() for x in ["out", "result"]) or llvm_name in ["%C", "%c"] or semantic_name in ["out", "result"]
        res_type = a_type
        if a_type == "ptr":
            res_type = f"float addrspace({as_id})*"
        elif "addrspace" not in a_type:
            res_type = a_type.replace("*", f" addrspace({as_id})*")
        return res_type, is_output, f'{res_type} nocapture noundef "air-buffer-no-alias" {llvm_name}'

    @staticmethod
    def _process_scalar_argument(a_type: str, llvm_name: str, semantic_name: str, var_addrspaces: dict[str, int], scalar_loads: dict[str, str]) -> tuple[str, bool, str]:
        if semantic_name in GRID_POSITION_NAMES + THREADGROUP_POSITION_NAMES:
            var_addrspaces[llvm_name] = 0
            return a_type, False, f"{a_type} {llvm_name}"
        base_type = a_type.strip()
        info = get_type_info(base_type)
        assert info is not None, f"unsupported scalar argument type: {base_type}"
        _, size, align = info
        var_addrspaces[llvm_name] = 2
        scalar_loads[llvm_name] = f"%val_{semantic_name}"
        ptr_type = f"{base_type} addrspace(2)*"
        return ptr_type, False, f'{ptr_type} nocapture noundef readonly align {align} dereferenceable({size}) "air-buffer-no-alias" {llvm_name}'


class IntrinsicHandler:
    @staticmethod
    def handle_type_casts(line: str, used_intrinsics: set[str]) -> str:
        # metal rejects raw int<->float casts
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
        for intr in sorted(used_intrinsics):
            lines.extend(MetadataGenerator._emit_intrinsic_decl(intr))
        # #0 kernels, #1 barrier decl, #2 intrinsics
        lines.append('attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }')
        lines.append("attributes #1 = { convergent mustprogress nounwind willreturn }")
        lines.append("attributes #2 = { convergent nounwind willreturn }")
        if kernels:
            lines.extend(MetadataGenerator._generate_kernel_metadata(kernels, metal_version_str))
        return lines

    @staticmethod
    def _emit_intrinsic_decl(intr: str) -> list[str]:
        if "air.convert" in intr:
            # [kind, dst, kind, src]
            parts = intr.replace("@air.convert.", "").split(".")
            return [f"declare {AIR_TO_LLVM_TYPES.get(parts[1], parts[1])} {intr}({AIR_TO_LLVM_TYPES.get(parts[3], parts[3])}) #2"]
        # "fmax" contains "fma"
        arg_types = "(float, float)" if any(x in intr for x in ["pow", "fmin", "fmax"]) else "(float, float, float)" if "fma" in intr else "(float)"
        return [f"declare float @{intr.replace('@', '')}{arg_types} #2"]

    @staticmethod
    def _generate_kernel_metadata(kernels: list[tuple[str, list[KernelArg]]], metal_version_str: str | None = None) -> list[str]:
        lines: list[str] = []

        def m(content: str) -> str:
            lines.append(f"!{len(lines)} = {content}")
            return f"!{len(lines) - 1}"

        kernel_nodes = []
        for func_name, args_list in kernels:
            arg_meta_refs = []
            # device + constant share slots
            loc_indices = {1: 0, 3: 0}
            for idx, (at, an, is_out) in enumerate(args_list):
                content, as_id = MetadataGenerator._get_air_metadata_content(at, an, is_out)
                if as_id:
                    link_id = 1 if as_id in [1, 2] else 3
                    content = content.replace("{loc_index}", str(loc_indices[link_id]))
                    loc_indices[link_id] += 1
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
        # TODO: magic strings, fix in mlir lowering
        as_id = 3 if "addrspace(3)" in arg_type else (2 if "addrspace(2)" in arg_type else 1)
        access_mode = '!"air.read"' if (as_id == 2 or not is_output) else '!"air.read_write"'
        meta_parts = ['!"air.buffer"', '!"air.location_index", i32 {loc_index}, i32 1', access_mode, f'!"air.address_space", i32 {as_id}', f'!"air.arg_type_size", i32 {size}', f'!"air.arg_type_align_size", i32 {align}', f'!"air.arg_type_name", !"{base_name}"']
        if as_id == 2:
            meta_parts.insert(1, f'!"air.buffer_size", i32 {size}')
        return (", ".join(meta_parts), as_id)


class AirTranslator:
    lines: list[str]
    output_lines: list[str]
    kernels: list[tuple[str, list[KernelArg]]]
    kernel_overrides: dict[str, dict[str, str]] | None
    used_intrinsics: set[str]
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
        i = 0
        while i < len(self.lines):
            stripped = self.lines[i].strip()
            if stripped.startswith("define void"):
                i = self._process_function(i)
                continue
            if stripped.startswith(";") or stripped.endswith(":"):
                self.output_lines.append(self.lines[i])
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
            if base_type := next((arg.split("addrspace")[0].strip() for arg, name, _ in args_list if name == name_check), None):
                self.output_lines.append(f"  {loaded_var} = load {base_type}, {base_type} addrspace(2)* {param_name}, align 4")

    def _convert_body(self, idx: int) -> int:
        while idx < len(self.lines):
            line = self.lines[idx]
            stripped = line.strip()
            if stripped == "}":
                self.output_lines.append("}")
                return idx + 1
            self.output_lines.append(line if stripped.startswith(";") or stripped.endswith(":") else self._convert_instruction(line))
            idx += 1
        return idx

    def _convert_instruction(self, line: str) -> str:
        # mem_flags 2 = threadgroup memory
        if "call" in line and "barrier" in line:
            return "  tail call void @air.wg.barrier(i32 2, i32 1) #2"
        # pointer rewrites before propagation
        line = IntrinsicHandler.handle_type_casts(line, self.used_intrinsics)
        line = IntrinsicHandler.replace_intrinsics(line, self.used_intrinsics)
        line = self._rewrite_pointers(line)
        line = self._rewrite_opaque_pointers(line)
        self._propagate_address_spaces(line)
        return self._apply_scalar_loads(line)

    def _rewrite_pointers(self, line: str) -> str:
        def replacer(m: re.Match[str]) -> str:
            as_id = self.var_addrspaces.get(m.group(2), 0)
            return f"{m.group(1)} addrspace({as_id})* {m.group(2)}" if as_id > 0 else m.group(0)

        return re.sub(r"([\w\s<>\.]+)\*\s+(%[\w\.\"]+)", replacer, line)

    def _rewrite_opaque_pointers(self, line: str) -> str:
        def replacer(m: re.Match[str]) -> str:
            as_id = self.var_addrspaces.get(m.group(1), 0)
            return f"float addrspace({as_id})* {m.group(1)}" if as_id > 0 else m.group(0)

        return re.sub(r"\bptr\s+(%[\w\.\"]+)", replacer, line)

    def _propagate_address_spaces(self, line: str) -> None:
        if not any(x in line for x in ["getelementptr", "bitcast", "select"]):
            return
        if (lhs_match := re.search(r"(%[\w\.\"]+)\s*=", line)) and (as_match := re.search(r"addrspace\((\d+)\)", line)):
            self.var_addrspaces[lhs_match.group(1)] = int(as_match.group(1))

    def _apply_scalar_loads(self, line: str) -> str:
        for param_name, loaded_var in self.scalar_loads.items():
            line = re.sub(re.escape(param_name) + r"\b", loaded_var, line)
        return line


def to_air(llvm_ir_text: str, kernel_overrides: dict[str, dict[str, str]] | None = None) -> str:
    # overrides: {kernel: {raw name: semantic name}}
    assert llvm_ir_text
    return AirTranslator(llvm_ir_text, kernel_overrides).translate()


# ===----------------------------------------------------------------------=== #
# metal runtime
# ===----------------------------------------------------------------------=== #


def compile_to_metallib(air_llvm_ir: str) -> bytes:
    assert shutil.which("xcrun") is not None, "xcrun not found"
    env = metal_toolchain_env()
    for tool in ("metal", "metallib"):
        assert has_tool(tool, env), f"{tool} not found"
    assert air_llvm_ir
    with tempfile.NamedTemporaryFile(suffix=".ll") as f_ll, tempfile.NamedTemporaryFile(suffix=".air") as f_air, tempfile.NamedTemporaryFile(suffix=".metallib") as f_lib:
        f_ll.write(air_llvm_ir.encode("utf-8"))
        f_ll.flush()
        steps = (("metal", ["xcrun", "-sdk", "macosx", "metal", "-x", "ir", "-c", f_ll.name, "-o", f_air.name]), ("metallib", ["xcrun", "-sdk", "macosx", "metallib", f_air.name, "-o", f_lib.name]))
        for tool, cmd in steps:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
            assert result.returncode == 0, f"{tool} compilation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        lib_data = Path(f_lib.name).read_bytes()
    assert len(lib_data) > 0, "generated metallib is empty"
    return lib_data


@cache
def create_compute_pipeline(metallib_binary: bytes, kernel_name: str):
    device = import_module("Metal").MTLCreateSystemDefaultDevice()
    assert device, "metal not supported on this device"
    libdispatch = ctypes.CDLL(ctypes.util.find_library("dispatch") or "/usr/lib/system/libdispatch.dylib")
    libdispatch.dispatch_data_create.restype = ctypes.c_void_p
    libdispatch.dispatch_data_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    # null destructor copies bytes
    c_data = ctypes.create_string_buffer(metallib_binary, len(metallib_binary))
    dispatch_data = libdispatch.dispatch_data_create(c_data, len(metallib_binary), None, None)
    assert dispatch_data, "failed to create dispatch_data"
    library, error = device.newLibraryWithData_error_(import_module("objc").objc_object(c_void_p=dispatch_data), None)
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
    encoder.setComputePipelineState_(pso)
    encode_args_fn(encoder)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
    encoder.endEncoding()
    start_time = time.perf_counter()
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()
    elapsed = time.perf_counter() - start_time
    status = cmd_buffer.status()
    assert status == metal.MTLCommandBufferStatusCompleted, f"command buffer failed with status {status} and error: {cmd_buffer.error()}"
    return elapsed


# ===----------------------------------------------------------------------=== #
# metal backend
# ===----------------------------------------------------------------------=== #


def thread_grid(ir: LoopIR.proc) -> tuple[int, Sym, Sym, int, int]:
    # exo skips race checks outside c backend
    for arg in ir.args:
        assert isinstance(arg.type, T.Tensor) and not arg.type.is_window and isinstance(arg.type.basetype(), T.F32) and all(isinstance(dim, LoopIR.Const) for dim in arg.type.shape()), f"metal backend: argument {arg.name} must be a static f32 tensor, got {arg.type}"
        assert all(isinstance(dim, LoopIR.Const) and isinstance(dim.val, int) and dim.val > 0 for dim in arg.type.shape()), f"metal backend: argument {arg.name} must have positive integer extents, got {arg.type}"
    written = {sym for sym, _ in get_writes_of_stmts(ir.body)}
    outs = [i for i, arg in enumerate(ir.args) if arg.name in written]
    assert len(outs) == 1, f"metal backend: expected exactly one written argument, got {len(outs)}"
    out = ir.args[outs[0]]
    assert isinstance(out.type, T.Tensor)  # narrowing only
    shape = [dim.val for dim in out.type.shape() if isinstance(dim, LoopIR.Const) and isinstance(dim.val, int)]
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
        # bare ptr args for to_air, no gpu heap
        assert len(procs) == 1, "metal backend: a thread grid maps exactly one kernel procedure"
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


def kernel_to_air(proc: Procedure) -> str:
    out, row, col, _, cols = thread_grid(proc._loopir_proc)
    llvm_module = convert_module(to_mlir(proc, MetalIRGenerator(row, col, cols)), fallback_target_triple="air64-apple-macosx")
    kernel_args = llvm_module.get_global(proc.name()).args
    overrides = {proc.name(): {kernel_args[out].name: "out", kernel_args[-1].name: "id"}}  # names drive air abi
    return to_air(str(llvm_module), kernel_overrides=overrides)


def _compile_kernel(proc: Procedure, ffi: FFI) -> Callable[..., None]:
    out, _, _, rows, cols = thread_grid(proc._loopir_proc)  # reject before optional imports
    metal = import_module("Metal")
    device, pso = create_compute_pipeline(compile_to_metallib(kernel_to_air(proc)), proc.name())
    sizes = [4 * math.prod(dim.val for dim in arg.type.shape()) for arg in proc._loopir_proc.args]
    grid = metal.MTLSize(rows * cols, 1, 1)
    group = metal.MTLSize(min(rows * cols, pso.maxTotalThreadsPerThreadgroup()), 1, 1)

    def run(*ptrs: int) -> None:
        buffers = [device.newBufferWithBytes_length_options_(ffi.buffer(ffi.cast("char *", ptr), size)[:], size, metal.MTLResourceStorageModeShared) for ptr, size in zip(ptrs, sizes, strict=True)]
        assert all(buffer is not None for buffer in buffers), "metal backend: buffer allocation failed"

        def encode(encoder: Any) -> None:
            for i, buffer in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buffer, 0, i)

        execute_kernel(device, pso, grid, group, encode)
        ffi.memmove(ffi.cast("char *", ptrs[out]), buffers[out].contents().as_buffer(sizes[out]), sizes[out])

    return run


def compile(proc: Procedure, raw: bool = False) -> Callable[..., None]:
    assert not raw, "raw=True is not supported by the metal backend"
    return JITRuntime.compile(proc, compiler=_compile_kernel)
