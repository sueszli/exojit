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

# cross-entropy loss: loss = -sum(y_true * log(y_pred))
LLVM_IR_CROSS_ENTROPY = """
declare float @llvm.log.f32(float)

define void @cross_entropy(float* %y_pred, float* %y_true, float* %output, i32 %global_id) {
entry:
  ; y_pred: predicted probabilities (4 values)
  ; y_true: true labels (4 values, one-hot encoded)
  ; output: single loss value (computed by all threads, but only thread 0 writes)
  
  %idx = zext i32 %global_id to i64
  
  ; load all predicted probabilities
  %pred_ptr_0 = getelementptr inbounds float, float* %y_pred, i64 0
  %pred_0 = load float, float* %pred_ptr_0
  
  %pred_ptr_1 = getelementptr inbounds float, float* %y_pred, i64 1
  %pred_1 = load float, float* %pred_ptr_1
  
  %pred_ptr_2 = getelementptr inbounds float, float* %y_pred, i64 2
  %pred_2 = load float, float* %pred_ptr_2
  
  %pred_ptr_3 = getelementptr inbounds float, float* %y_pred, i64 3
  %pred_3 = load float, float* %pred_ptr_3

  ; load all true labels
  %true_ptr_0 = getelementptr inbounds float, float* %y_true, i64 0
  %true_0 = load float, float* %true_ptr_0
  
  %true_ptr_1 = getelementptr inbounds float, float* %y_true, i64 1
  %true_1 = load float, float* %true_ptr_1
  
  %true_ptr_2 = getelementptr inbounds float, float* %y_true, i64 2
  %true_2 = load float, float* %true_ptr_2
  
  %true_ptr_3 = getelementptr inbounds float, float* %y_true, i64 3
  %true_3 = load float, float* %true_ptr_3
  
  ; compute -y_true * log(y_pred) for each sample
  ; sample 0
  %log_pred_0 = call float @llvm.log.f32(float %pred_0)
  %contrib_0 = fmul float %true_0, %log_pred_0
  
  ; sample 1
  %log_pred_1 = call float @llvm.log.f32(float %pred_1)
  %contrib_1 = fmul float %true_1, %log_pred_1
  
  ; sample 2
  %log_pred_2 = call float @llvm.log.f32(float %pred_2)
  %contrib_2 = fmul float %true_2, %log_pred_2
  
  ; sample 3
  %log_pred_3 = call float @llvm.log.f32(float %pred_3)
  %contrib_3 = fmul float %true_3, %log_pred_3
  
  ; sum all contributions
  %sum_01 = fadd float %contrib_0, %contrib_1
  %sum_23 = fadd float %contrib_2, %contrib_3
  %sum_all = fadd float %sum_01, %sum_23
  
  ; negate to get loss
  %loss = fneg float %sum_all
  
  ; store result (all threads write the same value)
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %loss, float* %output_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_cross_entropy():
    return llvm_to_metallib(LLVM_IR_CROSS_ENTROPY)


def run_loss_function(binary, y_pred, y_true, kernel_name):
    device, pso = create_compute_pipeline(binary, kernel_name)

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


def test_cross_entropy_perfect_prediction(binary_cross_entropy):
    y_pred = [0.9999, 0.0001, 0.0001, 0.0001]
    y_true = [1.0, 0.0, 0.0, 0.0]

    result = run_loss_function(binary_cross_entropy, y_pred, y_true, "cross_entropy")
    assert result[0] == pytest.approx(0.0, abs=1e-3)


def test_cross_entropy_uniform_distribution(binary_cross_entropy):
    y_pred = [0.25, 0.25, 0.25, 0.25]
    y_true = [1.0, 0.0, 0.0, 0.0]

    expected_loss = -math.log(0.25)

    result = run_loss_function(binary_cross_entropy, y_pred, y_true, "cross_entropy")
    assert result[0] == pytest.approx(expected_loss, rel=1e-4)


def test_cross_entropy_multi_class(binary_cross_entropy):
    y_pred = [0.7, 0.2, 0.05, 0.05]
    y_true = [0.0, 1.0, 0.0, 0.0]

    expected_loss = -math.log(0.2)

    result = run_loss_function(binary_cross_entropy, y_pred, y_true, "cross_entropy")
    assert result[0] == pytest.approx(expected_loss, rel=1e-4)


# mean squared error loss: loss = mean((y_pred - y_true)^2)
LLVM_IR_MSE = """
define void @mse_loss(float* %y_pred, float* %y_true, float* %output, i32 %global_id) {
entry:
  ; y_pred: predicted values (4 values)
  ; y_true: true values (4 values)
  ; output: single loss value
  
  %idx = zext i32 %global_id to i64
  
  ; load all predictions
  %pred_ptr_0 = getelementptr inbounds float, float* %y_pred, i64 0
  %pred_0 = load float, float* %pred_ptr_0
  
  %pred_ptr_1 = getelementptr inbounds float, float* %y_pred, i64 1
  %pred_1 = load float, float* %pred_ptr_1
  
  %pred_ptr_2 = getelementptr inbounds float, float* %y_pred, i64 2
  %pred_2 = load float, float* %pred_ptr_2
  
  %pred_ptr_3 = getelementptr inbounds float, float* %y_pred, i64 3
  %pred_3 = load float, float* %pred_ptr_3
  
  ; load all true values
  %true_ptr_0 = getelementptr inbounds float, float* %y_true, i64 0
  %true_0 = load float, float* %true_ptr_0
  
  %true_ptr_1 = getelementptr inbounds float, float* %y_true, i64 1
  %true_1 = load float, float* %true_ptr_1
  
  %true_ptr_2 = getelementptr inbounds float, float* %y_true, i64 2
  %true_2 = load float, float* %true_ptr_2
  
  %true_ptr_3 = getelementptr inbounds float, float* %y_true, i64 3
  %true_3 = load float, float* %true_ptr_3
  
  ; compute squared errors
  %diff_0 = fsub float %pred_0, %true_0
  %sq_0 = fmul float %diff_0, %diff_0
  
  %diff_1 = fsub float %pred_1, %true_1
  %sq_1 = fmul float %diff_1, %diff_1
  
  %diff_2 = fsub float %pred_2, %true_2
  %sq_2 = fmul float %diff_2, %diff_2
  
  %diff_3 = fsub float %pred_3, %true_3
  %sq_3 = fmul float %diff_3, %diff_3
  
  ; sum squared errors
  %sum_01 = fadd float %sq_0, %sq_1
  %sum_23 = fadd float %sq_2, %sq_3
  %sum_all = fadd float %sum_01, %sum_23
  
  ; compute mean
  %mse = fdiv float %sum_all, 4.0
  
  ; store result
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %mse, float* %output_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_mse():
    return llvm_to_metallib(LLVM_IR_MSE)


def test_mse_perfect_prediction(binary_mse):
    y_pred = [1.0, 2.0, 3.0, 4.0]
    y_true = [1.0, 2.0, 3.0, 4.0]

    result = run_loss_function(binary_mse, y_pred, y_true, "mse_loss")
    assert result[0] == pytest.approx(0.0, abs=1e-5)


def test_mse_uniform_error(binary_mse):
    y_pred = [2.0, 3.0, 4.0, 5.0]
    y_true = [1.0, 2.0, 3.0, 4.0]

    expected_mse = 1.0
    expected_mse = 1.0

    result = run_loss_function(binary_mse, y_pred, y_true, "mse_loss")
    assert result[0] == pytest.approx(expected_mse, rel=1e-4)


def test_mse_varied_errors(binary_mse):
    y_pred = [1.0, 3.0, 5.0, 7.0]
    y_true = [2.0, 3.0, 4.0, 5.0]
    expected_mse = 1.5

    result = run_loss_function(binary_mse, y_pred, y_true, "mse_loss")
    assert result[0] == pytest.approx(expected_mse, rel=1e-4)
