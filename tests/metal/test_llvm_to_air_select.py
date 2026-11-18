import pytest

from .utils import llvm_to_metallib, run_kernel_1d_float

# out[i] = i even ? in[i] : in[i+1]


LLVM_IR_SELECT_PTR_SINGLE = """
define void @select_ptr_single(float* %in, float* %out, i32 %id) {
  %idx = zext i32 %id to i64
  
  %idx_next = add i64 %idx, 1
  
  %ptr_curr = getelementptr inbounds float, float* %in, i64 %idx
  %ptr_next = getelementptr inbounds float, float* %in, i64 %idx_next
  
  ; toggle based on id is even/odd
  %rem = urem i32 %id, 2
  %cond = icmp eq i32 %rem, 0
  
  ; if even: take current. if odd: take next.
  %ptr_sel = select i1 %cond, float* %ptr_curr, float* %ptr_next
  
  %val = load float, float* %ptr_sel
  
  %ptr_out = getelementptr inbounds float, float* %out, i64 %idx
  store float %val, float* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_select_ptr_single():
    return llvm_to_metallib(LLVM_IR_SELECT_PTR_SINGLE)


def test_select_ptr_single(binary_select_ptr_single):
    input_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    expected = [0.0, 2.0, 2.0, 4.0]
    result = run_kernel_1d_float(binary_select_ptr_single, input_data, "select_ptr_single")
    assert result[:4] == pytest.approx(expected)
