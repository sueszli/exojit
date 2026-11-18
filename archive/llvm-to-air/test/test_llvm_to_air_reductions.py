import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

LLVM_IR_UITOFP_TEST = """
define void @uitofp_test(float* %input, float* %output, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  
  ; Convert thread ID to float
  %id_float = uitofp i32 %id to float
  
  ; Load input value
  %input_ptr = getelementptr inbounds float, float* %input, i64 %idx
  %input_val = load float, float* %input_ptr
  
  ; Add thread ID as float to input
  %result = fadd float %input_val, %id_float
  
  ; Store result
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %result, float* %output_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_uitofp_test():
    return llvm_to_metallib(LLVM_IR_UITOFP_TEST)


def test_uitofp_conversion(binary_uitofp_test):
    """Test that uitofp correctly converts unsigned int to float."""
    input_data = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected = [10.0, 21.0, 32.0, 43.0, 54.0]  # input[i] + i
    result = run_kernel_1d_float(binary_uitofp_test, input_data, "uitofp_test")
    assert result == pytest.approx(expected)


LLVM_IR_SITOFP_TEST = """
define void @sitofp_test(float* %input, float* %output, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  
  ; Create a signed value: id - 2
  %id_offset = sub i32 %id, 2
  
  ; Convert to float
  %offset_float = sitofp i32 %id_offset to float
  
  ; Load input value
  %input_ptr = getelementptr inbounds float, float* %input, i64 %idx
  %input_val = load float, float* %input_ptr
  
  ; Add offset to input
  %result = fadd float %input_val, %offset_float
  
  ; Store result
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %result, float* %output_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_sitofp_test():
    return llvm_to_metallib(LLVM_IR_SITOFP_TEST)


def test_sitofp_conversion(binary_sitofp_test):
    """Test that sitofp correctly converts signed int to float."""
    # input[i] + float(i - 2) should give us input[i] + (i - 2)
    input_data = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected = [8.0, 19.0, 30.0, 41.0, 52.0]  # input[i] + (i - 2)
    result = run_kernel_1d_float(binary_sitofp_test, input_data, "sitofp_test")
    assert result == pytest.approx(expected)


LLVM_IR_FPTOUI_TEST = """
define void @fptoui_test(float* %input, float* %output, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  
  ; Load input value
  %input_ptr = getelementptr inbounds float, float* %input, i64 %idx
  %input_val = load float, float* %input_ptr
  
  ; Convert to unsigned int (truncates decimal)
  %int_val = fptoui float %input_val to i32
  
  ; Convert back to float
  %result = uitofp i32 %int_val to float
  
  ; Store result
  %output_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %result, float* %output_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_fptoui_test():
    return llvm_to_metallib(LLVM_IR_FPTOUI_TEST)


def test_fptoui_conversion(binary_fptoui_test):
    input_data = [1.9, 2.1, 3.5, 4.0, 5.8]
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = run_kernel_1d_float(binary_fptoui_test, input_data, "fptoui_test")
    assert result == pytest.approx(expected)
