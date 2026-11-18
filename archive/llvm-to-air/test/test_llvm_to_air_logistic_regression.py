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

# logistic regression forward pass: y = sigmoid(w * x + b)
# sigmoid(z) = 1 / (1 + exp(-z))
LLVM_IR_LOGISTIC_FORWARD = """
declare float @llvm.exp.f32(float)

define void @logistic_forward(float* %x, float* %w, float* %b, float* %y, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  %x_ptr = getelementptr inbounds float, float* %x, i64 %idx
  %x_val = load float, float* %x_ptr
  
  %w_ptr = getelementptr inbounds float, float* %w, i64 0
  %w_val = load float, float* %w_ptr
  
  %b_ptr = getelementptr inbounds float, float* %b, i64 0
  %b_val = load float, float* %b_ptr
  
  %wx = fmul float %w_val, %x_val
  %z = fadd float %wx, %b_val
  
  %neg_z = fneg float %z
  %exp_neg_z = call float @llvm.exp.f32(float %neg_z)
  %one_plus_exp = fadd float 1.0, %exp_neg_z
  %sigmoid = fdiv float 1.0, %one_plus_exp
  
  %y_ptr = getelementptr inbounds float, float* %y, i64 %idx
  store float %sigmoid, float* %y_ptr
  
  ret void
}
"""


def run_logistic_forward(binary, x, w, b):
    device, pso = create_compute_pipeline(binary, "logistic_forward")

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
def binary_logistic_forward():
    return llvm_to_metallib(LLVM_IR_LOGISTIC_FORWARD)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def test_logistic_forward_zero_input(binary_logistic_forward):
    x = [0.0, 0.0, 0.0, 0.0]
    w = [1.0]
    b = [0.0]

    # sigmoid(0) = 0.5
    expected = [0.5, 0.5, 0.5, 0.5]

    result = run_logistic_forward(binary_logistic_forward, x, w, b)
    assert result == pytest.approx(expected, rel=1e-5)


def test_logistic_forward_positive_values(binary_logistic_forward):
    x = [1.0, 2.0, 3.0, 4.0]
    w = [1.0]
    b = [0.0]

    expected = [sigmoid(1.0), sigmoid(2.0), sigmoid(3.0), sigmoid(4.0)]

    result = run_logistic_forward(binary_logistic_forward, x, w, b)
    assert result == pytest.approx(expected, rel=1e-5)


def test_logistic_forward_negative_values(binary_logistic_forward):
    x = [-1.0, -2.0, -3.0, -4.0]
    w = [1.0]
    b = [0.0]

    expected = [sigmoid(-1.0), sigmoid(-2.0), sigmoid(-3.0), sigmoid(-4.0)]

    result = run_logistic_forward(binary_logistic_forward, x, w, b)
    assert result == pytest.approx(expected, rel=1e-5)


def test_logistic_forward_with_bias(binary_logistic_forward):
    x = [1.0, 2.0, 3.0, 4.0]
    w = [2.0]
    b = [-3.0]

    # z = 2*x - 3 = [-1, 1, 3, 5]
    expected = [sigmoid(-1.0), sigmoid(1.0), sigmoid(3.0), sigmoid(5.0)]

    result = run_logistic_forward(binary_logistic_forward, x, w, b)
    assert result == pytest.approx(expected, rel=1e-5)


LLVM_IR_BCE_LOSS = """
declare float @llvm.log.f32(float)

define void @bce_loss(float* %y_pred, float* %y_true, float* %output, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  
  %pred_ptr = getelementptr inbounds float, float* %y_pred, i64 %idx
  %p = load float, float* %pred_ptr
  
  %true_ptr = getelementptr inbounds float, float* %y_true, i64 %idx
  %y = load float, float* %true_ptr
  
  %log_p = call float @llvm.log.f32(float %p)
  
  %one_minus_p = fsub float 1.0, %p
  
  %log_one_minus_p = call float @llvm.log.f32(float %one_minus_p)
  
  %one_minus_y = fsub float 1.0, %y
  
  %term1 = fmul float %y, %log_p
  
  %term2 = fmul float %one_minus_y, %log_one_minus_p
  
  %sum = fadd float %term1, %term2
  %bce = fneg float %sum
  
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %bce, float* %output_ptr
  
  ret void
}
"""


def run_bce_loss(binary, y_pred, y_true):
    device, pso = create_compute_pipeline(binary, "bce_loss")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_pred = create_buffer(y_pred)
    buf_true = create_buffer(y_true)
    buf_output = device.newBufferWithLength_options_(len(y_pred) * 4, Metal.MTLResourceStorageModeShared)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_pred, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_true, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 2)

    grid_size = Metal.MTLSize(len(y_pred), 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_output.contents()
    output_buffer = output_ptr.as_buffer(len(y_pred) * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


@pytest.fixture(scope="module")
def binary_bce_loss():
    return llvm_to_metallib(LLVM_IR_BCE_LOSS)


def bce(y_true, y_pred):
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


def test_bce_loss_perfect_prediction_class_1(binary_bce_loss):
    y_pred = [0.9999, 0.9999, 0.9999, 0.9999]
    y_true = [1.0, 1.0, 1.0, 1.0]

    result = run_bce_loss(binary_bce_loss, y_pred, y_true)

    for loss in result:
        assert loss == pytest.approx(0.0, abs=1e-3)


def test_bce_loss_perfect_prediction_class_0(binary_bce_loss):
    y_pred = [0.0001, 0.0001, 0.0001, 0.0001]
    y_true = [0.0, 0.0, 0.0, 0.0]

    result = run_bce_loss(binary_bce_loss, y_pred, y_true)

    for loss in result:
        assert loss == pytest.approx(0.0, abs=1e-3)


def test_bce_loss_uncertain_prediction(binary_bce_loss):
    y_pred = [0.5, 0.5, 0.5, 0.5]
    y_true = [1.0, 0.0, 1.0, 0.0]

    expected_loss = -math.log(0.5)

    result = run_bce_loss(binary_bce_loss, y_pred, y_true)

    for loss in result:
        assert loss == pytest.approx(expected_loss, rel=1e-5)


def test_bce_loss_mixed_predictions(binary_bce_loss):
    y_pred = [0.9, 0.1, 0.7, 0.3]
    y_true = [1.0, 0.0, 1.0, 0.0]

    expected = [bce(1.0, 0.9), bce(0.0, 0.1), bce(1.0, 0.7), bce(0.0, 0.3)]

    result = run_bce_loss(binary_bce_loss, y_pred, y_true)
    assert result == pytest.approx(expected, rel=1e-5)


LLVM_IR_LOGISTIC_GRADIENT = """
define void @logistic_gradient(float* %x, float* %y_pred, float* %y_true, float* %grad_w, float* %grad_b, i32 %global_id) {
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


def run_logistic_gradient(binary, x, y_pred, y_true):
    device, pso = create_compute_pipeline(binary, "logistic_gradient")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_x = create_buffer(x)
    buf_pred = create_buffer(y_pred)
    buf_true = create_buffer(y_true)
    buf_grad_w = device.newBufferWithLength_options_(len(x) * 4, Metal.MTLResourceStorageModeShared)
    buf_grad_b = device.newBufferWithLength_options_(len(x) * 4, Metal.MTLResourceStorageModeShared)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_x, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_pred, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_true, 0, 2)
        encoder.setBuffer_offset_atIndex_(buf_grad_w, 0, 3)
        encoder.setBuffer_offset_atIndex_(buf_grad_b, 0, 4)

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
def binary_logistic_gradient():
    return llvm_to_metallib(LLVM_IR_LOGISTIC_GRADIENT)


def test_logistic_gradient_perfect_predictions(binary_logistic_gradient):
    x = [1.0, 2.0, 3.0, 4.0]
    y_pred = [0.9, 0.1, 0.9, 0.1]
    y_true = [0.9, 0.1, 0.9, 0.1]

    grad_w, grad_b = run_logistic_gradient(binary_logistic_gradient, x, y_pred, y_true)

    assert grad_w == pytest.approx(0.0, abs=1e-5)
    assert grad_b == pytest.approx(0.0, abs=1e-5)


def test_logistic_gradient_uniform_error(binary_logistic_gradient):
    x = [1.0, 2.0, 3.0, 4.0]
    y_pred = [0.6, 0.6, 0.6, 0.6]
    y_true = [0.5, 0.5, 0.5, 0.5]

    grad_w, grad_b = run_logistic_gradient(binary_logistic_gradient, x, y_pred, y_true)

    assert grad_w == pytest.approx(0.25, rel=1e-5)
    assert grad_b == pytest.approx(0.1, rel=1e-5)
