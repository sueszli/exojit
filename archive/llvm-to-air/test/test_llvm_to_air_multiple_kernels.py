import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

LLVM_IR_MULTIPLE = """
define void @kernel_add(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_a = getelementptr inbounds float, float* %a, i64 %idx
  %val_a = load float, float* %ptr_a
  %res = fadd float %val_a, 1.0
  %ptr_b = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_b
  ret void
}

define void @kernel_sub(float* %a, float* %b, i32 %id) {
  %idx = zext i32 %id to i64
  %ptr_a = getelementptr inbounds float, float* %a, i64 %idx
  %val_a = load float, float* %ptr_a
  %res = fsub float %val_a, 1.0
  %ptr_b = getelementptr inbounds float, float* %b, i64 %idx
  store float %res, float* %ptr_b
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_multiple():
    return llvm_to_metallib(LLVM_IR_MULTIPLE)


def test_multiple_kernels(binary_multiple):
    input_data = [1.0, 2.0, 3.0]
    res_add = run_kernel_1d_float(binary_multiple, input_data, "kernel_add")
    assert res_add == pytest.approx([2.0, 3.0, 4.0])
    res_sub = run_kernel_1d_float(binary_multiple, input_data, "kernel_sub")
    assert res_sub == pytest.approx([0.0, 1.0, 2.0])
