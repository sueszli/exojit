import ctypes
import math
import sys
from pathlib import Path

import Metal
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from llvm_to_air import create_compute_pipeline, execute_kernel
from utils import llvm_to_metallib

LLVM_IR_MLP_FORWARD = """
declare float @llvm.exp.f32(float)
declare float @llvm.maxnum.f32(float, float)

define void @mlp_forward(float* %x, float* %w1, float* %b1, float* %w2, float* %b2, float* %output, i32 %global_id) {
entry:
  %x_ptr = getelementptr inbounds float, float* %x, i64 0
  %x_val = load float, float* %x_ptr
  
  %w1_0_ptr = getelementptr inbounds float, float* %w1, i64 0
  %w1_0 = load float, float* %w1_0_ptr
  %b1_0_ptr = getelementptr inbounds float, float* %b1, i64 0
  %b1_0 = load float, float* %b1_0_ptr
  
  %z1_0 = fmul float %w1_0, %x_val
  %z1_0_bias = fadd float %z1_0, %b1_0
  %h1_0 = call float @llvm.maxnum.f32(float 0.0, float %z1_0_bias)
  
  %w1_1_ptr = getelementptr inbounds float, float* %w1, i64 1
  %w1_1 = load float, float* %w1_1_ptr
  %b1_1_ptr = getelementptr inbounds float, float* %b1, i64 1
  %b1_1 = load float, float* %b1_1_ptr
  
  %z1_1 = fmul float %w1_1, %x_val
  %z1_1_bias = fadd float %z1_1, %b1_1
  %h1_1 = call float @llvm.maxnum.f32(float 0.0, float %z1_1_bias)
  
  %w2_0_ptr = getelementptr inbounds float, float* %w2, i64 0
  %w2_0 = load float, float* %w2_0_ptr
  %w2_1_ptr = getelementptr inbounds float, float* %w2, i64 1
  %w2_1 = load float, float* %w2_1_ptr
  %b2_ptr = getelementptr inbounds float, float* %b2, i64 0
  %b2_val = load float, float* %b2_ptr
  
  %z2_0 = fmul float %w2_0, %h1_0
  %z2_1 = fmul float %w2_1, %h1_1
  %z2_sum = fadd float %z2_0, %z2_1
  %z2 = fadd float %z2_sum, %b2_val
  
  %neg_z2 = fneg float %z2
  %exp_neg_z2 = call float @llvm.exp.f32(float %neg_z2)
  %one_plus_exp = fadd float 1.0, %exp_neg_z2
  %y = fdiv float 1.0, %one_plus_exp
  
  %output_ptr = getelementptr inbounds float, float* %output, i64 0
  store float %y, float* %output_ptr
  
  ret void
}
"""


def run_mlp_forward(binary, x, w1, b1, w2, b2):
    device, pso = create_compute_pipeline(binary, "mlp_forward")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_x = create_buffer(x)
    buf_w1 = create_buffer(w1)
    buf_b1 = create_buffer(b1)
    buf_w2 = create_buffer(w2)
    buf_b2 = create_buffer(b2)
    buf_output = device.newBufferWithLength_options_(4, Metal.MTLResourceStorageModeShared)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_x, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_w1, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_b1, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_w2, 0, 3)
        encoder.setBuffer_offset_atIndex_(buf_b2, 0, 4)
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 5)

    grid_size = Metal.MTLSize(1, 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_output.contents()
    output_buffer = output_ptr.as_buffer(4)
    results_view = memoryview(output_buffer).cast("f")
    return results_view[0]


@pytest.fixture(scope="module")
def binary_mlp_forward():
    return llvm_to_metallib(LLVM_IR_MLP_FORWARD)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def relu(z):
    return max(0.0, z)


def mlp_forward_reference(x, w1, b1, w2, b2):
    h1_0 = relu(w1[0] * x + b1[0])
    h1_1 = relu(w1[1] * x + b1[1])

    z2 = w2[0] * h1_0 + w2[1] * h1_1 + b2[0]
    y = sigmoid(z2)

    return y


def test_mlp_forward_all_positive(binary_mlp_forward):
    x = [2.0]
    w1 = [1.0, 0.5]
    b1 = [0.0, 0.0]
    w2 = [1.0, 1.0]
    b2 = [0.0]

    expected = sigmoid(3.0)

    result = run_mlp_forward(binary_mlp_forward, x, w1, b1, w2, b2)
    assert result == pytest.approx(expected, rel=1e-5)


def test_mlp_forward_with_relu_cutoff(binary_mlp_forward):
    x = [1.0]
    w1 = [1.0, -2.0]
    b1 = [0.0, 0.0]
    w2 = [1.0, 1.0]
    b2 = [0.0]

    expected = sigmoid(1.0)

    result = run_mlp_forward(binary_mlp_forward, x, w1, b1, w2, b2)
    assert result == pytest.approx(expected, rel=1e-5)


def test_mlp_forward_with_bias(binary_mlp_forward):
    x = [0.5]
    w1 = [2.0, 1.0]
    b1 = [-0.5, 0.5]
    w2 = [0.5, -0.5]
    b2 = [1.0]

    expected = sigmoid(0.75)

    result = run_mlp_forward(binary_mlp_forward, x, w1, b1, w2, b2)
    assert result == pytest.approx(expected, rel=1e-5)


def test_mlp_forward_zero_input(binary_mlp_forward):
    x = [0.0]
    w1 = [1.0, 1.0]
    b1 = [1.0, -1.0]
    w2 = [1.0, 1.0]
    b2 = [0.0]

    expected = sigmoid(1.0)

    result = run_mlp_forward(binary_mlp_forward, x, w1, b1, w2, b2)
    assert result == pytest.approx(expected, rel=1e-5)


def test_mlp_forward_reference_comparison(binary_mlp_forward):
    x_val = 1.5
    x = [x_val]
    w1 = [0.8, -0.3]
    b1 = [0.2, 0.5]
    w2 = [1.2, -0.7]
    b2 = [0.1]

    expected = mlp_forward_reference(x_val, w1, b1, w2, b2)
    result = run_mlp_forward(binary_mlp_forward, x, w1, b1, w2, b2)

    assert result == pytest.approx(expected, rel=1e-5)


LLVM_IR_MLP_BATCH = """
declare float @llvm.exp.f32(float)
declare float @llvm.maxnum.f32(float, float)

define void @mlp_batch(float* %x, float* %w1, float* %b1, float* %w2, float* %b2, float* %output, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  %x_ptr = getelementptr inbounds float, float* %x, i64 %idx
  %x_val = load float, float* %x_ptr
  
  %w1_0_ptr = getelementptr inbounds float, float* %w1, i64 0
  %w1_0 = load float, float* %w1_0_ptr
  %b1_0_ptr = getelementptr inbounds float, float* %b1, i64 0
  %b1_0 = load float, float* %b1_0_ptr
  
  %z1_0 = fmul float %w1_0, %x_val
  %z1_0_bias = fadd float %z1_0, %b1_0
  %h1_0 = call float @llvm.maxnum.f32(float 0.0, float %z1_0_bias)
  
  %w1_1_ptr = getelementptr inbounds float, float* %w1, i64 1
  %w1_1 = load float, float* %w1_1_ptr
  %b1_1_ptr = getelementptr inbounds float, float* %b1, i64 1
  %b1_1 = load float, float* %b1_1_ptr
  
  %z1_1 = fmul float %w1_1, %x_val
  %z1_1_bias = fadd float %z1_1, %b1_1
  %h1_1 = call float @llvm.maxnum.f32(float 0.0, float %z1_1_bias)
  
  %w2_0_ptr = getelementptr inbounds float, float* %w2, i64 0
  %w2_0 = load float, float* %w2_0_ptr
  %w2_1_ptr = getelementptr inbounds float, float* %w2, i64 1
  %w2_1 = load float, float* %w2_1_ptr
  %b2_ptr = getelementptr inbounds float, float* %b2, i64 0
  %b2_val = load float, float* %b2_ptr
  
  %z2_0 = fmul float %w2_0, %h1_0
  %z2_1 = fmul float %w2_1, %h1_1
  %z2_sum = fadd float %z2_0, %z2_1
  %z2 = fadd float %z2_sum, %b2_val
  
  %neg_z2 = fneg float %z2
  %exp_neg_z2 = call float @llvm.exp.f32(float %neg_z2)
  %one_plus_exp = fadd float 1.0, %exp_neg_z2
  %y = fdiv float 1.0, %one_plus_exp
  
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %y, float* %output_ptr
  
  ret void
}
"""


def run_mlp_batch(binary, x_batch, w1, b1, w2, b2):
    """Run MLP forward pass on a batch of inputs"""
    device, pso = create_compute_pipeline(binary, "mlp_batch")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_x = create_buffer(x_batch)
    buf_w1 = create_buffer(w1)
    buf_b1 = create_buffer(b1)
    buf_w2 = create_buffer(w2)
    buf_b2 = create_buffer(b2)
    buf_output = device.newBufferWithLength_options_(len(x_batch) * 4, Metal.MTLResourceStorageModeShared)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_x, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_w1, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_b1, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_w2, 0, 3)
        encoder.setBuffer_offset_atIndex_(buf_b2, 0, 4)
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 5)

    grid_size = Metal.MTLSize(len(x_batch), 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_output.contents()
    output_buffer = output_ptr.as_buffer(len(x_batch) * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


@pytest.fixture(scope="module")
def binary_mlp_batch():
    return llvm_to_metallib(LLVM_IR_MLP_BATCH)


def test_mlp_batch_single_input(binary_mlp_batch):
    x_batch = [2.0]
    w1 = [1.0, 0.5]
    b1 = [0.0, 0.0]
    w2 = [1.0, 1.0]
    b2 = [0.0]

    expected = [sigmoid(3.0)]

    result = run_mlp_batch(binary_mlp_batch, x_batch, w1, b1, w2, b2)
    assert result == pytest.approx(expected, rel=1e-5)


def test_mlp_batch_multiple_inputs(binary_mlp_batch):
    x_batch = [0.0, 1.0, 2.0, 3.0]
    w1 = [1.0, -1.0]
    b1 = [0.5, 0.5]
    w2 = [1.0, 1.0]
    b2 = [0.0]

    expected = [mlp_forward_reference(x, w1, b1, w2, b2) for x in x_batch]

    result = run_mlp_batch(binary_mlp_batch, x_batch, w1, b1, w2, b2)
    assert result == pytest.approx(expected, rel=1e-5)


def test_mlp_batch_xor_pattern(binary_mlp_batch):
    x_batch = [-1.0, 1.0]
    w1 = [2.0, -2.0]
    b1 = [0.0, 0.0]
    w2 = [1.0, 1.0]
    b2 = [0.0]

    expected_val = sigmoid(2.0)
    expected = [expected_val, expected_val]

    result = run_mlp_batch(binary_mlp_batch, x_batch, w1, b1, w2, b2)
    assert result == pytest.approx(expected, rel=1e-5)
