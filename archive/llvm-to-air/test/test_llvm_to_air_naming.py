import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

LLVM_IR = """
define void @test_kernel(float* %in_ptr, float* %out_ptr, i32 %global_id, i32 %local_id, float* %my_group_ptr) 
{
entry:
  %idx = zext i32 %global_id to i64
  %ptr_in = getelementptr float, float* %in_ptr, i64 %idx
  %val = load float, float* %ptr_in
  
  ; write to the "renamed" shared pointer to trigger mismatch
  %ptr_shared = getelementptr float, float* %my_group_ptr, i64 0
  store float %val, float* %ptr_shared
  
  %ptr_out = getelementptr float, float* %out_ptr, i64 %idx
  store float %val, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_fragility():
    return llvm_to_metallib(LLVM_IR)


def test_naming_fragility(binary_fragility):
    input_data = [42.0]
    expected = [42.0]
    result = run_kernel_1d_float(binary_fragility, input_data, "test_kernel")
    assert result == pytest.approx(expected)
