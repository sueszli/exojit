import math

import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

#
# softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for all j)
#

LLVM_IR_SOFTMAX = """
declare float @llvm.exp.f32(float)
declare void @barrier()

define void @softmax(float* %input, float* %output, i32 %global_id, i32 %local_id, float* %shared_exp) {
entry:
  ; compute exp(input[i]) and store in shared memory
  %idx = zext i32 %global_id to i64
  %input_ptr = getelementptr inbounds float, float* %input, i64 %idx
  %val = load float, float* %input_ptr
  
  %exp_val = call float @llvm.exp.f32(float %val)
  
  %local_idx = zext i32 %local_id to i64
  %shared_ptr = getelementptr inbounds float, float* %shared_exp, i64 %local_idx
  store float %exp_val, float* %shared_ptr
  
  ; barrier to ensure all threads have computed exp
  call void @barrier()
  
  ; compute sum of all exp values (simple loop for small arrays)
  ; assume array size is 4
  %shared_ptr_0 = getelementptr inbounds float, float* %shared_exp, i64 0
  %exp_0 = load float, float* %shared_ptr_0
  
  %shared_ptr_1 = getelementptr inbounds float, float* %shared_exp, i64 1
  %exp_1 = load float, float* %shared_ptr_1
  
  %shared_ptr_2 = getelementptr inbounds float, float* %shared_exp, i64 2
  %exp_2 = load float, float* %shared_ptr_2
  
  %shared_ptr_3 = getelementptr inbounds float, float* %shared_exp, i64 3
  %exp_3 = load float, float* %shared_ptr_3
  
  %sum_01 = fadd float %exp_0, %exp_1
  %sum_012 = fadd float %sum_01, %exp_2
  %sum_total = fadd float %sum_012, %exp_3
  
  ; compute softmax = exp(x[i]) / sum
  %softmax_val = fdiv float %exp_val, %sum_total
  
  ; store result
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %softmax_val, float* %output_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_softmax():
    return llvm_to_metallib(LLVM_IR_SOFTMAX)


def test_softmax_simple(binary_softmax):
    # Input: [0, 1, 2, 3]
    # exp: [1, e, e^2, e^3]
    # sum: 1 + e + e^2 + e^3
    # softmax: [1/sum, e/sum, e^2/sum, e^3/sum]

    input_data = [0.0, 1.0, 2.0, 3.0]
    exp_vals = [math.exp(x) for x in input_data]
    sum_exp = sum(exp_vals)
    expected = [e / sum_exp for e in exp_vals]

    result = run_kernel_1d_float(binary_softmax, input_data, "softmax")
    assert result == pytest.approx(expected, rel=1e-5)


def test_softmax_uniform(binary_softmax):
    input_data = [1.0, 1.0, 1.0, 1.0]
    expected = [0.25, 0.25, 0.25, 0.25]

    result = run_kernel_1d_float(binary_softmax, input_data, "softmax")
    assert result == pytest.approx(expected, rel=1e-5)


def test_softmax_negative(binary_softmax):
    input_data = [-1.0, 0.0, 1.0, 2.0]

    exp_vals = [math.exp(x) for x in input_data]
    sum_exp = sum(exp_vals)
    expected = [e / sum_exp for e in exp_vals]

    result = run_kernel_1d_float(binary_softmax, input_data, "softmax")
    assert result == pytest.approx(expected, rel=1e-5)
