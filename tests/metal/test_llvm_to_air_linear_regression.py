import ctypes
import struct

import pytest

from backend_metal import create_compute_pipeline, execute_kernel

from .utils import Metal, llvm_to_metallib

LLVM_IR_LINEAR_FORWARD = """
define void @linear_forward(float* %x, float* %w, float* %b, float* %y, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  %x_ptr = getelementptr inbounds float, float* %x, i64 %idx
  %x_val = load float, float* %x_ptr
  
  %w_ptr = getelementptr inbounds float, float* %w, i64 0
  %w_val = load float, float* %w_ptr
  
  %b_ptr = getelementptr inbounds float, float* %b, i64 0
  %b_val = load float, float* %b_ptr
  
  %wx = fmul float %w_val, %x_val
  %y_val = fadd float %wx, %b_val
  
  %y_ptr = getelementptr inbounds float, float* %y, i64 %idx
  store float %y_val, float* %y_ptr
  
  ret void
}
"""


def run_linear_forward(binary, x, w, b):
    device, pso = create_compute_pipeline(binary, "linear_forward")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_x = create_buffer(x)
    buf_w = create_buffer(w)
    buf_b = create_buffer(b)
    buf_y = device.newBufferWithLength_options_(len(x) * 4, Metal.MTLResourceStorageModeShared)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_x, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_w, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_b, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_y, 0, 3)

    grid_size = Metal.MTLSize(len(x), 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_y.contents()
    output_buffer = output_ptr.as_buffer(len(x) * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


@pytest.fixture(scope="module")
def binary_linear_forward():
    return llvm_to_metallib(LLVM_IR_LINEAR_FORWARD)


def test_linear_forward_simple(binary_linear_forward):
    x = [0.0, 1.0, 2.0, 3.0]
    w = [2.0]
    b = [1.0]

    expected = [1.0, 3.0, 5.0, 7.0]

    result = run_linear_forward(binary_linear_forward, x, w, b)
    assert result == pytest.approx(expected, rel=1e-5)


def test_linear_forward_zero_bias(binary_linear_forward):
    x = [2.0, 4.0, 6.0, 8.0]
    w = [0.5]
    b = [0.0]

    expected = [1.0, 2.0, 3.0, 4.0]

    result = run_linear_forward(binary_linear_forward, x, w, b)
    assert result == pytest.approx(expected, rel=1e-5)


def test_linear_forward_negative_weight(binary_linear_forward):
    x = [1.0, 2.0, 3.0, 4.0]
    w = [-1.0]
    b = [5.0]

    expected = [4.0, 3.0, 2.0, 1.0]

    result = run_linear_forward(binary_linear_forward, x, w, b)
    assert result == pytest.approx(expected, rel=1e-5)


LLVM_IR_LINEAR_GRADIENT = """
define void @linear_gradient(float* %x, float* %y_pred, float* %y_true, float* %grad_w, float* %grad_b, i32 %n, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  %x_ptr = getelementptr inbounds float, float* %x, i64 %idx
  %x_val = load float, float* %x_ptr
  
  %pred_ptr = getelementptr inbounds float, float* %y_pred, i64 %idx
  %pred_val = load float, float* %pred_ptr
  
  %true_ptr = getelementptr inbounds float, float* %y_true, i64 %idx
  %true_val = load float, float* %true_ptr
  
  %error = fsub float %pred_val, %true_val
  
  %grad_w_val = fmul float %error, %x_val
  
  %grad_w_ptr = getelementptr inbounds float, float* %grad_w, i64 %idx
  store float %grad_w_val, float* %grad_w_ptr
  
  %grad_b_ptr = getelementptr inbounds float, float* %grad_b, i64 %idx
  store float %error, float* %grad_b_ptr
  
  ret void
}
"""


def run_linear_gradient(binary, x, y_pred, y_true, n):
    device, pso = create_compute_pipeline(binary, "linear_gradient")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_x = create_buffer(x)
    buf_pred = create_buffer(y_pred)
    buf_true = create_buffer(y_true)
    buf_grad_w = device.newBufferWithLength_options_(len(x) * 4, Metal.MTLResourceStorageModeShared)
    buf_grad_b = device.newBufferWithLength_options_(len(x) * 4, Metal.MTLResourceStorageModeShared)

    n_bytes = struct.pack("i", n)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_x, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_pred, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_true, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_grad_w, 0, 3)
        encoder.setBuffer_offset_atIndex_(buf_grad_b, 0, 4)
        encoder.setBytes_length_atIndex_(n_bytes, 4, 5)

    grid_size = Metal.MTLSize(len(x), 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    grad_w_ptr = buf_grad_w.contents()
    grad_w_buffer = grad_w_ptr.as_buffer(len(x) * 4)
    grad_w_view = memoryview(grad_w_buffer).cast("f")

    grad_b_ptr = buf_grad_b.contents()
    grad_b_buffer = grad_b_ptr.as_buffer(len(x) * 4)
    grad_b_view = memoryview(grad_b_buffer).cast("f")

    grad_w_mean = sum(grad_w_view) / len(x)
    grad_b_mean = sum(grad_b_view) / len(x)

    return grad_w_mean, grad_b_mean


@pytest.fixture(scope="module")
def binary_linear_gradient():
    return llvm_to_metallib(LLVM_IR_LINEAR_GRADIENT)


def test_linear_gradient_perfect_fit(binary_linear_gradient):
    x = [1.0, 2.0, 3.0, 4.0]
    y_pred = [3.0, 5.0, 7.0, 9.0]
    y_true = [3.0, 5.0, 7.0, 9.0]

    grad_w, grad_b = run_linear_gradient(binary_linear_gradient, x, y_pred, y_true, len(x))

    assert grad_w == pytest.approx(0.0, abs=1e-5)
    assert grad_b == pytest.approx(0.0, abs=1e-5)


def test_linear_gradient_simple_error(binary_linear_gradient):
    x = [1.0, 2.0, 3.0, 4.0]
    y_pred = [4.0, 6.0, 8.0, 10.0]
    y_true = [3.0, 5.0, 7.0, 9.0]

    grad_w, grad_b = run_linear_gradient(binary_linear_gradient, x, y_pred, y_true, len(x))

    assert grad_w == pytest.approx(2.5, rel=1e-5)
    assert grad_b == pytest.approx(1.0, rel=1e-5)


def test_linear_gradient_negative_error(binary_linear_gradient):
    x = [1.0, 2.0, 3.0, 4.0]
    y_pred = [2.0, 4.0, 6.0, 8.0]
    y_true = [3.0, 5.0, 7.0, 9.0]

    grad_w, grad_b = run_linear_gradient(binary_linear_gradient, x, y_pred, y_true, len(x))

    assert grad_w == pytest.approx(-2.5, rel=1e-5)
    assert grad_b == pytest.approx(-1.0, rel=1e-5)
