import pytest

from utils import llvm_to_metallib, run_kernel_1d_int

LLVM_IR_LOOP = """
define void @loop_kernel(i32* %in, i32* %out, i32 %id) {
entry:
  %idx = zext i32 %id to i64
  %ptr_in = getelementptr i32, i32* %in, i64 %idx
  %n = load i32, i32* %ptr_in
  %ptr_out = getelementptr i32, i32* %out, i64 %idx
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %sum_next = add i32 %sum, %i
  %i_next = add i32 %i, 1
  %cond = icmp slt i32 %i_next, %n
  br i1 %cond, label %loop, label %exit

exit:
  store i32 %sum_next, i32* %ptr_out
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_loop():
    return llvm_to_metallib(LLVM_IR_LOOP)


def test_loop_sum(binary_loop):
    input_data = [5, 10, 0, 1]
    expected = [10, 45, 0, 0]
    result = run_kernel_1d_int(binary_loop, input_data, "loop_kernel")
    assert result == expected
