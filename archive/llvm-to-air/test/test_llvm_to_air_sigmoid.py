import math

import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

# Sigmoid: out[i] = 1.0 / (1.0 + exp(-in[i]))
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
