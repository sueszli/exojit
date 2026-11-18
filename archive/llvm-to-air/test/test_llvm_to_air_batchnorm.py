import math

import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

# batch Normalization: out[i] = gamma * (x[i] - mean) / sqrt(var + eps) + beta
# simplified: batch size = 4, single feature

LLVM_IR_BATCHNORM = """
declare float @llvm.sqrt.f32(float)

define void @batchnorm(float* %input, float* %output, i32 %global_id) {
entry:
  ; input: 4 values
  ; compute mean and variance, then normalize
  ; for simplicity: gamma=1, beta=0, eps=1e-5
  
  %idx = zext i32 %global_id to i64
  
  ; load all 4 values to compute mean
  %ptr_0 = getelementptr inbounds float, float* %input, i64 0
  %val_0 = load float, float* %ptr_0
  
  %ptr_1 = getelementptr inbounds float, float* %input, i64 1
  %val_1 = load float, float* %ptr_1
  
  %ptr_2 = getelementptr inbounds float, float* %input, i64 2
  %val_2 = load float, float* %ptr_2
  
  %ptr_3 = getelementptr inbounds float, float* %input, i64 3
  %val_3 = load float, float* %ptr_3
  
  ; compute mean = (sum of values) / 4
  %sum_01 = fadd float %val_0, %val_1
  %sum_23 = fadd float %val_2, %val_3
  %sum_all = fadd float %sum_01, %sum_23
  %mean = fdiv float %sum_all, 4.0
  
  ; compute variance = mean((x - mean)^2)
  %diff_0 = fsub float %val_0, %mean
  %diff_1 = fsub float %val_1, %mean
  %diff_2 = fsub float %val_2, %mean
  %diff_3 = fsub float %val_3, %mean
  
  %sq_0 = fmul float %diff_0, %diff_0
  %sq_1 = fmul float %diff_1, %diff_1
  %sq_2 = fmul float %diff_2, %diff_2
  %sq_3 = fmul float %diff_3, %diff_3
  
  %sq_sum_01 = fadd float %sq_0, %sq_1
  %sq_sum_23 = fadd float %sq_2, %sq_3
  %sq_sum_all = fadd float %sq_sum_01, %sq_sum_23
  %variance = fdiv float %sq_sum_all, 4.0
  
  ; add epsilon for numerical stability (compute 1e-5 via division)
  %eps = fdiv float 1.0, 100000.0
  %var_eps = fadd float %variance, %eps
  
  ; compute standard deviation
  %std = call float @llvm.sqrt.f32(float %var_eps)
  
  ; load current value and normalize
  %input_ptr = getelementptr inbounds float, float* %input, i64 %idx
  %input_val = load float, float* %input_ptr
  
  %centered = fsub float %input_val, %mean
  %normalized = fdiv float %centered, %std
  
  ; store result (gamma=1, beta=0)
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %normalized, float* %output_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_batchnorm():
    return llvm_to_metallib(LLVM_IR_BATCHNORM)


def test_batchnorm_basic(binary_batchnorm):
    # input: [1, 2, 3, 4]
    # mean = 2.5
    # variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
    #          = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
    # std = sqrt(1.25 + 1e-5) ≈ 1.118
    # normalized: [(1-2.5)/1.118, (2-2.5)/1.118, (3-2.5)/1.118, (4-2.5)/1.118]
    #           ≈ [-1.342, -0.447, 0.447, 1.342]

    input_data = [1.0, 2.0, 3.0, 4.0]

    mean = 2.5
    variance = 1.25
    std = math.sqrt(variance + 1e-5)
    expected = [(x - mean) / std for x in input_data]

    result = run_kernel_1d_float(binary_batchnorm, input_data, "batchnorm")
    assert result == pytest.approx(expected, rel=1e-4)


def test_batchnorm_zero_mean(binary_batchnorm):
    # input: [-2, -1, 1, 2]
    # mean = 0
    # variance = (4 + 1 + 1 + 4) / 4 = 2.5
    # std = sqrt(2.5 + 1e-5) ≈ 1.581

    input_data = [-2.0, -1.0, 1.0, 2.0]

    mean = 0.0
    variance = 2.5
    std = math.sqrt(variance + 1e-5)
    expected = [(x - mean) / std for x in input_data]

    result = run_kernel_1d_float(binary_batchnorm, input_data, "batchnorm")
    assert result == pytest.approx(expected, rel=1e-4)


def test_batchnorm_uniform(binary_batchnorm):
    # input: [5, 5, 5, 5]
    # mean = 5
    # variance = 0
    # std = sqrt(0 + 1e-5) ≈ 0.00316
    # normalized: all should be ~0

    input_data = [5.0, 5.0, 5.0, 5.0]
    expected = [0.0, 0.0, 0.0, 0.0]

    result = run_kernel_1d_float(binary_batchnorm, input_data, "batchnorm")
    assert result == pytest.approx(expected, abs=1e-2)
