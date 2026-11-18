import ctypes
import sys
from pathlib import Path

import Metal
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from llvm_to_air import create_compute_pipeline, execute_kernel
from utils import llvm_to_metallib

# params[i] -= learning_rate * gradients[i]
LLVM_IR_SGD_BASIC = """
define void @sgd_basic(float* %params, float* %gradients, float %learning_rate, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  %param_ptr = getelementptr inbounds float, float* %params, i64 %idx
  %param_val = load float, float* %param_ptr
  
  %grad_ptr = getelementptr inbounds float, float* %gradients, i64 %idx
  %grad_val = load float, float* %grad_ptr
  
  %update = fmul float %learning_rate, %grad_val
  
  %new_param = fsub float %param_val, %update
  
  store float %new_param, float* %param_ptr
  
  ret void
}
"""


def run_sgd_basic(binary, params, gradients, learning_rate):
    device, pso = create_compute_pipeline(binary, "sgd_basic")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_params = create_buffer(params)
    buf_grads = create_buffer(gradients)

    import struct

    lr_bytes = struct.pack("f", learning_rate)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_params, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_grads, 0, 1)
        encoder.setBytes_length_atIndex_(lr_bytes, 4, 2)

    grid_size = Metal.MTLSize(len(params), 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    # read back updated parameters
    output_ptr = buf_params.contents()
    output_buffer = output_ptr.as_buffer(len(params) * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


@pytest.fixture(scope="module")
def binary_sgd_basic():
    return llvm_to_metallib(LLVM_IR_SGD_BASIC)


def test_sgd_basic_simple(binary_sgd_basic):
    # params = [1.0, 2.0, 3.0, 4.0]
    # gradients = [0.1, 0.2, 0.3, 0.4]
    # lr = 0.1
    # expected: params[i] -= 0.1 * gradients[i]
    # = [1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03, 4.0 - 0.04]
    # = [0.99, 1.98, 2.97, 3.96]

    params = [1.0, 2.0, 3.0, 4.0]
    gradients = [0.1, 0.2, 0.3, 0.4]
    learning_rate = 0.1

    expected = [0.99, 1.98, 2.97, 3.96]

    result = run_sgd_basic(binary_sgd_basic, params, gradients, learning_rate)
    assert result == pytest.approx(expected, rel=1e-5)


def test_sgd_basic_zero_gradients(binary_sgd_basic):
    params = [1.0, 2.0, 3.0, 4.0]
    gradients = [0.0, 0.0, 0.0, 0.0]
    learning_rate = 0.1

    expected = [1.0, 2.0, 3.0, 4.0]  # No change

    result = run_sgd_basic(binary_sgd_basic, params, gradients, learning_rate)
    assert result == pytest.approx(expected, rel=1e-5)


def test_sgd_basic_negative_gradients(binary_sgd_basic):
    params = [1.0, 2.0, 3.0, 4.0]
    gradients = [-0.1, -0.2, -0.3, -0.4]
    learning_rate = 0.1

    # params[i] -= lr * grad[i]
    # = params[i] -= 0.1 * (-grad[i])
    # = params[i] + 0.1 * abs(grad[i])
    expected = [1.01, 2.02, 3.03, 4.04]

    result = run_sgd_basic(binary_sgd_basic, params, gradients, learning_rate)
    assert result == pytest.approx(expected, rel=1e-5)


# sgd with momentum: velocity = momentum * velocity + grad; param -= lr * velocity
LLVM_IR_SGD_MOMENTUM = """
define void @sgd_momentum(float* %params, float* %gradients, float* %velocity, float %learning_rate, float %momentum, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  %param_ptr = getelementptr inbounds float, float* %params, i64 %idx
  %param_val = load float, float* %param_ptr
  
  %grad_ptr = getelementptr inbounds float, float* %gradients, i64 %idx
  %grad_val = load float, float* %grad_ptr
  
  %vel_ptr = getelementptr inbounds float, float* %velocity, i64 %idx
  %vel_val = load float, float* %vel_ptr
  
  %momentum_term = fmul float %momentum, %vel_val
  %new_velocity = fadd float %momentum_term, %grad_val
  
  store float %new_velocity, float* %vel_ptr
  
  %update = fmul float %learning_rate, %new_velocity
  
  %new_param = fsub float %param_val, %update
  
  store float %new_param, float* %param_ptr
  
  ret void
}
"""


def run_sgd_momentum(binary, params, gradients, velocity, learning_rate, momentum):
    device, pso = create_compute_pipeline(binary, "sgd_momentum")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_params = create_buffer(params)
    buf_grads = create_buffer(gradients)
    buf_velocity = create_buffer(velocity)

    import struct

    lr_bytes = struct.pack("f", learning_rate)
    momentum_bytes = struct.pack("f", momentum)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_params, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_grads, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_velocity, 0, 2)
        encoder.setBytes_length_atIndex_(lr_bytes, 4, 3)
        encoder.setBytes_length_atIndex_(momentum_bytes, 4, 4)

    grid_size = Metal.MTLSize(len(params), 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    params_ptr = buf_params.contents()
    params_buffer = params_ptr.as_buffer(len(params) * 4)
    params_view = memoryview(params_buffer).cast("f")

    velocity_ptr = buf_velocity.contents()
    velocity_buffer = velocity_ptr.as_buffer(len(velocity) * 4)
    velocity_view = memoryview(velocity_buffer).cast("f")

    return list(params_view), list(velocity_view)


@pytest.fixture(scope="module")
def binary_sgd_momentum():
    return llvm_to_metallib(LLVM_IR_SGD_MOMENTUM)


def test_sgd_momentum_simple(binary_sgd_momentum):
    params = [1.0, 2.0, 3.0, 4.0]
    gradients = [0.1, 0.2, 0.3, 0.4]
    velocity = [0.0, 0.0, 0.0, 0.0]
    learning_rate = 0.1
    momentum = 0.9

    # expected velocity: 0.9 * 0.0 + grad = grad
    expected_velocity = [0.1, 0.2, 0.3, 0.4]
    # expected params: param - lr * velocity
    expected_params = [0.99, 1.98, 2.97, 3.96]

    result_params, result_velocity = run_sgd_momentum(binary_sgd_momentum, params, gradients, velocity, learning_rate, momentum)

    assert result_params == pytest.approx(expected_params, rel=1e-5)
    assert result_velocity == pytest.approx(expected_velocity, rel=1e-5)


def test_sgd_momentum_with_existing_velocity(binary_sgd_momentum):
    params = [1.0, 2.0, 3.0, 4.0]
    gradients = [0.1, 0.2, 0.3, 0.4]
    velocity = [0.05, 0.1, 0.15, 0.2]
    learning_rate = 0.1
    momentum = 0.9

    expected_velocity = [0.145, 0.29, 0.435, 0.58]
    expected_params = [0.9855, 1.971, 2.9565, 3.942]

    result_params, result_velocity = run_sgd_momentum(binary_sgd_momentum, params, gradients, velocity, learning_rate, momentum)

    assert result_params == pytest.approx(expected_params, rel=1e-5)
    assert result_velocity == pytest.approx(expected_velocity, rel=1e-5)


# sgd with weigh decay: params[i] -= lr * (gradients[i] + weight_decay * params[i])
LLVM_IR_SGD_WEIGHT_DECAY = """
define void @sgd_weight_decay(float* %params, float* %gradients, float %learning_rate, float %weight_decay, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  ; Load current parameter value
  %param_ptr = getelementptr inbounds float, float* %params, i64 %idx
  %param_val = load float, float* %param_ptr
  
  ; Load gradient
  %grad_ptr = getelementptr inbounds float, float* %gradients, i64 %idx
  %grad_val = load float, float* %grad_ptr
  
  ; Compute weight decay term: weight_decay * param
  %decay_term = fmul float %weight_decay, %param_val
  
  ; Add weight decay to gradient: grad + weight_decay * param
  %effective_grad = fadd float %grad_val, %decay_term
  
  ; Compute update: lr * effective_grad
  %update = fmul float %learning_rate, %effective_grad
  
  ; Apply update: param -= lr * (grad + weight_decay * param)
  %new_param = fsub float %param_val, %update
  
  ; Store updated parameter
  store float %new_param, float* %param_ptr
  
  ret void
}
"""


def run_sgd_weight_decay(binary, params, gradients, learning_rate, weight_decay):
    device, pso = create_compute_pipeline(binary, "sgd_weight_decay")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_params = create_buffer(params)
    buf_grads = create_buffer(gradients)

    import struct

    lr_bytes = struct.pack("f", learning_rate)
    wd_bytes = struct.pack("f", weight_decay)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_params, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_grads, 0, 1)
        encoder.setBytes_length_atIndex_(lr_bytes, 4, 2)
        encoder.setBytes_length_atIndex_(wd_bytes, 4, 3)

    grid_size = Metal.MTLSize(len(params), 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_params.contents()
    output_buffer = output_ptr.as_buffer(len(params) * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


@pytest.fixture(scope="module")
def binary_sgd_weight_decay():
    return llvm_to_metallib(LLVM_IR_SGD_WEIGHT_DECAY)


def test_sgd_weight_decay_simple(binary_sgd_weight_decay):
    params = [1.0, 2.0, 3.0, 4.0]
    gradients = [0.1, 0.2, 0.3, 0.4]
    learning_rate = 0.1
    weight_decay = 0.01
    expected = [0.989, 1.978, 2.967, 3.956]

    result = run_sgd_weight_decay(binary_sgd_weight_decay, params, gradients, learning_rate, weight_decay)
    assert result == pytest.approx(expected, rel=1e-5)


def test_sgd_weight_decay_zero_decay(binary_sgd_weight_decay):
    params = [1.0, 2.0, 3.0, 4.0]
    gradients = [0.1, 0.2, 0.3, 0.4]
    learning_rate = 0.1
    weight_decay = 0.0

    expected = [0.99, 1.98, 2.97, 3.96]

    result = run_sgd_weight_decay(binary_sgd_weight_decay, params, gradients, learning_rate, weight_decay)
    assert result == pytest.approx(expected, rel=1e-5)
