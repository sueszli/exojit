import math

import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

# relu: out[i] = max(0.0, in[i])
LLVM_IR_RELU = """
define void @relu_kernel(float* %a, float* %b, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %a_ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %a_ptr
  %cmp = fcmp ogt float %val, 0.0
  %res = select i1 %cmp, float %val, float 0.0
  %b_ptr = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %b_ptr
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_relu():
    return llvm_to_metallib(LLVM_IR_RELU)


def test_relu(binary_relu):
    input_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    expected = [0.0, 0.0, 0.0, 1.0, 2.0]
    result = run_kernel_1d_float(binary_relu, input_data, "relu_kernel")
    assert result == pytest.approx(expected)


LLVM_IR_RELU_MAX = """
declare float @llvm.maxnum.f32(float, float)

define void @relu_max_kernel(float* %a, float* %b, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %a_ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %a_ptr
  %res = call float @llvm.maxnum.f32(float %val, float 0.0)
  %b_ptr = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %b_ptr
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_relu_max():
    return llvm_to_metallib(LLVM_IR_RELU_MAX)


def test_relu_max(binary_relu_max):
    input_data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    expected = [0.0, 0.0, 0.0, 1.0, 2.0]
    result = run_kernel_1d_float(binary_relu_max, input_data, "relu_max_kernel")
    assert result == pytest.approx(expected)


# out[i] = 1.0 / (1.0 + exp(-in[i]))
LLVM_IR_SIGMOID = """
declare float @llvm.exp.f32(float)

define void @sigmoid_kernel(float* %a, float* %b, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %a_ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %a_ptr
  
  ; -val
  %neg_val = fneg float %val
  
  ; exp(-val)
  %exp_neg_val = call float @llvm.exp.f32(float %neg_val)
  
  ; 1.0 + exp(-val)
  %denom = fadd float 1.0, %exp_neg_val
  
  ; 1.0 / denom
  %res = fdiv float 1.0, %denom
  
  %b_ptr = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %b_ptr
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_sigmoid():
    return llvm_to_metallib(LLVM_IR_SIGMOID)


def test_sigmoid(binary_sigmoid):
    input_data = [-1.0, 0.0, 1.0, 2.0]
    expected = [1.0 / (1.0 + math.exp(-x)) for x in input_data]
    result = run_kernel_1d_float(binary_sigmoid, input_data, "sigmoid_kernel")
    assert result == pytest.approx(expected)


LLVM_IR_TANH = """
declare float @llvm.tanh.f32(float)

define void @tanh_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  %res = call float @llvm.tanh.f32(float %val)
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_tanh():
    return llvm_to_metallib(LLVM_IR_TANH)


def test_tanh(binary_tanh):
    input_data = [-1.0, 0.0, 1.0, 2.0]
    expected = [math.tanh(x) for x in input_data]
    result = run_kernel_1d_float(binary_tanh, input_data, "tanh_kernel")
    assert result == pytest.approx(expected)
