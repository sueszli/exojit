import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

LLVM_IR_MIN_MAX = """
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)

define void @minmax_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  
  ; clamp to [0.0, 10.0]
  %val_max = call float @llvm.maxnum.f32(float %val, float 0.0)
  %val_clamped = call float @llvm.minnum.f32(float %val_max, float 10.0)
  
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %val_clamped, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_minmax():
    return llvm_to_metallib(LLVM_IR_MIN_MAX)


def test_minmax_clamp(binary_minmax):
    """Test min/max intrinsics for clamping."""
    input_data = [-5.0, 0.0, 5.0, 10.0, 15.0]
    # Clamp to [0, 10]
    expected = [0.0, 0.0, 5.0, 10.0, 10.0]
    result = run_kernel_1d_float(binary_minmax, input_data, "minmax_kernel")
    assert result == pytest.approx(expected)


# fma (fused multiply-add)
LLVM_IR_FMA = """
declare float @llvm.fma.f32(float, float, float)

define void @fma_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  
  ; result = val * 2.0 + 3.0
  %result = call float @llvm.fma.f32(float %val, float 2.0, float 3.0)
  
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %result, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_fma():
    return llvm_to_metallib(LLVM_IR_FMA)


def test_fma(binary_fma):
    """Test fused multiply-add intrinsic."""
    input_data = [0.0, 1.0, 2.0, 3.0]
    # result = val * 2.0 + 3.0
    expected = [3.0, 5.0, 7.0, 9.0]
    result = run_kernel_1d_float(binary_fma, input_data, "fma_kernel")
    assert result == pytest.approx(expected)


LLVM_IR_ABS = """
declare float @llvm.fabs.f32(float)

define void @abs_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  
  %result = call float @llvm.fabs.f32(float %val)
  
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %result, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_abs():
    return llvm_to_metallib(LLVM_IR_ABS)


def test_abs(binary_abs):
    """Test absolute value intrinsic."""
    input_data = [-5.0, -0.0, 0.0, 5.0]
    expected = [5.0, 0.0, 0.0, 5.0]
    result = run_kernel_1d_float(binary_abs, input_data, "abs_kernel")
    assert result == pytest.approx(expected)


LLVM_IR_TRUNC = """
declare float @llvm.trunc.f32(float)

define void @trunc_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  
  %result = call float @llvm.trunc.f32(float %val)
  
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %result, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_trunc():
    return llvm_to_metallib(LLVM_IR_TRUNC)


def test_trunc(binary_trunc):
    """Test truncate intrinsic."""
    input_data = [1.9, 2.1, -1.9, -2.1]
    expected = [1.0, 2.0, -1.0, -2.0]
    result = run_kernel_1d_float(binary_trunc, input_data, "trunc_kernel")
    assert result == pytest.approx(expected)


LLVM_IR_ROUND = """
declare float @llvm.round.f32(float)

define void @round_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  
  %result = call float @llvm.round.f32(float %val)
  
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %result, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_round():
    return llvm_to_metallib(LLVM_IR_ROUND)


def test_round(binary_round):
    """Test round intrinsic."""
    input_data = [1.4, 1.5, 1.6, -1.5]
    expected = [1.0, 2.0, 2.0, -2.0]
    result = run_kernel_1d_float(binary_round, input_data, "round_kernel")
    assert result == pytest.approx(expected)
