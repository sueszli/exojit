import ctypes
import platform

import pytest

# skip off macos only, before importing Metal
if platform.system() != "Darwin":
    pytest.skip("metal gpu tests need macOS", allow_module_level=True)

import Metal

from backend_metal import compile_to_metallib, create_compute_pipeline, execute_kernel, to_air


def llvm_to_metallib(llvm_ir: str) -> bytes:
    assert llvm_ir, "empty LLVM IR"
    air_llvm_text = to_air(llvm_ir, kernel_overrides=None)
    assert air_llvm_text, "failed to convert LLVM IR to AIR"
    return compile_to_metallib(air_llvm_text)


# custom runtime wrappers


def _run_kernel_common_1d(metallib_binary: bytes, input_data: list, kernel_name: str, c_type, width: int, height: int, threadgroup_w: int, threadgroup_h: int, use_shared_mem: bool) -> list:
    device, pso = create_compute_pipeline(metallib_binary, kernel_name)

    thread_count_items = len(input_data)
    raw_data_array = (c_type * thread_count_items)(*input_data)
    data_size_bytes = ctypes.sizeof(raw_data_array)

    buffer_in = device.newBufferWithBytes_length_options_(raw_data_array, data_size_bytes, Metal.MTLResourceStorageModeShared)
    assert buffer_in, "failed to create input buffer"

    buffer_out = device.newBufferWithLength_options_(data_size_bytes, Metal.MTLResourceStorageModeShared)
    assert buffer_out, "failed to create output buffer"

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buffer_in, 0, 0)
        encoder.setBuffer_offset_atIndex_(buffer_out, 0, 1)
        if use_shared_mem:
            encoder.setThreadgroupMemoryLength_atIndex_(data_size_bytes, 0)

    grid_size = Metal.MTLSize(width=width, height=height, depth=1)
    threadgroup_size = Metal.MTLSize(width=threadgroup_w, height=threadgroup_h, depth=1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buffer_out.contents()
    output_buffer = output_ptr.as_buffer(data_size_bytes)

    if c_type == ctypes.c_float:
        fmt = "f"
    elif c_type == ctypes.c_double:
        fmt = "d"
    elif c_type == ctypes.c_int32:
        fmt = "i"
    elif c_type == ctypes.c_uint32:
        fmt = "I"
    else:
        assert False, f"unsupported c_type: {c_type}"

    results_view = memoryview(output_buffer).cast(fmt)
    return list(results_view)


def run_kernel_1d_double(metallib_binary: bytes, input_data: list[float], kernel_name: str, threadgroup_size: int = 0) -> list[float]:
    n = len(input_data)
    tg_size = threadgroup_size if threadgroup_size > 0 else n
    return _run_kernel_common_1d(metallib_binary, input_data, kernel_name, ctypes.c_double, n, 1, tg_size, 1, True)


def run_kernel_1d_float(metallib_binary: bytes, input_data: list[float], kernel_name: str, threadgroup_size: int = 0) -> list[float]:
    n = len(input_data)
    tg_size = threadgroup_size if threadgroup_size > 0 else n
    return _run_kernel_common_1d(metallib_binary, input_data, kernel_name, ctypes.c_float, n, 1, tg_size, 1, True)


def run_kernel_1d_int(metallib_binary: bytes, input_data: list[int], kernel_name: str, threadgroup_size: int = 0) -> list[int]:
    n = len(input_data)
    tg_size = threadgroup_size if threadgroup_size > 0 else n
    return _run_kernel_common_1d(metallib_binary, input_data, kernel_name, ctypes.c_int32, n, 1, tg_size, 1, True)
