import math

import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

#
# out[i] = in[i] * 2.0 + 1.0
#

LLVM_IR_ARITHMETIC = """
define void @test_kernel(float* %in_ptr, float* %out_ptr, i32 %global_id, i32 %local_id, float* %shared_ptr) 
{
entry:
  %idx = zext i32 %global_id to i64
  %ptr_in = getelementptr float, float* %in_ptr, i64 %idx
  %val = load float, float* %ptr_in
  %val_mul = fmul float %val, 2.0
  %val_res = fadd float %val_mul, 1.0
  %ptr_out = getelementptr float, float* %out_ptr, i64 %idx
  store float %val_res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_arithmetic():
    return llvm_to_metallib(LLVM_IR_ARITHMETIC)


def test_arithmetic(binary_arithmetic):
    input_data = [0.0, 1.0, 2.0, -5.0]
    # expected: x * 2 + 1
    expected = [1.0, 3.0, 5.0, -9.0]
    result = run_kernel_1d_float(binary_arithmetic, input_data, "test_kernel")
    assert result == pytest.approx(expected)


#
# out[i] = in[i] + 1.0
#

LLVM_IR_ADD = """
define void @add_kernel(float* %a, float* %b, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %a_ptr = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %a_ptr
  %sum = fadd float %val, 1.0
  %b_ptr = getelementptr inbounds float, float* %b, i64 %idx
  store float %sum, float* %b_ptr
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_add():
    return llvm_to_metallib(LLVM_IR_ADD)


def test_add(binary_add):
    input_data = [i * 1.0 for i in range(32)]
    expected_output = [x + 1.0 for x in input_data]
    result = run_kernel_1d_float(binary_add, input_data, "add_kernel")
    assert result == expected_output


#
# out[i] = in[i] + i
#

LLVM_IR_ADD_INDEX = """
define void @add_index(float* %a, float* %b, i32 %id) {
  %1 = getelementptr inbounds float, float* %a, i32 %id
  %val = load float, float* %1, align 4
  %idx_float = sitofp i32 %id to float
  %res = fadd float %val, %idx_float
  %2 = getelementptr inbounds float, float* %b, i32 %id
  store float %res, float* %2, align 4
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_add_index():
    return llvm_to_metallib(LLVM_IR_ADD_INDEX)


def test_large_grid_execution(binary_add_index):
    size = 1024
    input_data = [1.0] * size
    output = run_kernel_1d_float(binary_add_index, input_data, "add_index", threadgroup_size=32)
    for i in range(size):
        expected = 1.0 + float(i)
        assert output[i] == expected, f"Mismatch at index {i}: expected {expected}, got {output[i]}"


#
# out[i] = in[i] - 1.0
#

LLVM_IR_SUB = """
define void @sub_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  %res = fsub float %val, 1.0
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_sub():
    return llvm_to_metallib(LLVM_IR_SUB)


def test_sub(binary_sub):
    input_data = [2.0, 3.0]
    expected = [1.0, 2.0]
    result = run_kernel_1d_float(binary_sub, input_data, "sub_kernel")
    assert result == pytest.approx(expected)


#
# out[i] = exp(in[i]) / 2.0
#

LLVM_IR_TENSOR_OPS = """
declare float @llvm.exp.f32(float)

define void @tensor_ops(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  
  ; exp(val)
  %val_exp = call float @llvm.exp.f32(float %val)
  
  ; / 2.0
  %res = fdiv float %val_exp, 2.0
  
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_tensor_ops():
    return llvm_to_metallib(LLVM_IR_TENSOR_OPS)


def test_div_exp(binary_tensor_ops):
    input_data = [0.0, 1.0, -1.0]
    expected = [math.exp(x) / 2.0 for x in input_data]
    result = run_kernel_1d_float(binary_tensor_ops, input_data, "tensor_ops")
    assert result == pytest.approx(expected)


LLVM_IR_FREM = """
define void @frem_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  %res = frem float %val, 2.0
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_frem():
    return llvm_to_metallib(LLVM_IR_FREM)


def test_frem(binary_frem):
    input_data = [2.5, 4.0, 5.1, 0.5]
    expected = [0.5, 0.0, 1.1, 0.5]
    result = run_kernel_1d_float(binary_frem, input_data, "frem_kernel")
    assert result == pytest.approx(expected, abs=1e-5)


LLVM_IR_RSQRT = """
declare float @llvm.sqrt.f32(float)

define void @rsqrt_kernel(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr inbounds float, float* %a, i64 %idx
  %val = load float, float* %ptr_in
  %sqrt_val = call float @llvm.sqrt.f32(float %val)
  %res = fdiv float 1.0, %sqrt_val
  %ptr_out = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_rsqrt():
    return llvm_to_metallib(LLVM_IR_RSQRT)


def test_rsqrt(binary_rsqrt):
    input_data = [1.0, 4.0, 16.0]
    expected = [1.0, 0.5, 0.25]
    result = run_kernel_1d_float(binary_rsqrt, input_data, "rsqrt_kernel")
    assert result == pytest.approx(expected, abs=1e-5)
