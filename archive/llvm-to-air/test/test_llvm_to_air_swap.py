import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

#
# var swap
#

# kernel void test_kernel(
#    device const float* in_ptr [[buffer(0)]],         // pointer to input in global memory
#    device float* out_ptr [[buffer(1)]],              // pointer to output in global memory
#    uint global_id [[thread_position_in_grid]],       // thread across the entire execution grid
#    uint local_id [[thread_position_in_threadgroup]], // thread within the threadgroup
#    threadgroup float* shared_ptr [[threadgroup(0)]]  // pointer to shared memory within the threadgroup
# ) {
#     float val_in = in_ptr[global_id];
#     shared_ptr[local_id] = val_in;
#
#     threadgroup_barrier(mem_flags::mem_threadgroup); // wait until all threads have stored their values
#
#     // if (id % 2 == 0) { neighbor = id + 1 } else { neighbor = id - 1 }
#     uint neighbor_id = local_id ^ 1;
#
#     // swap
#     float val_neighbor = shared_ptr[neighbor_id];
#     out_ptr[global_id] = val_neighbor;
# }

LLVM_IR = """
declare void @"barrier"()

define void @"test_kernel"(float* %"in_ptr", float* %"out_ptr", i32 %"global_id", i32 %"local_id", float* %"shared_ptr")
{
entry:
  %"idx_global" = zext i32 %"global_id" to i64
  %"ptr_in" = getelementptr float, float* %"in_ptr", i64 %"idx_global"
  %"val_in" = load float, float* %"ptr_in"
  %"idx_local" = zext i32 %"local_id" to i64
  %"ptr_shared" = getelementptr float, float* %"shared_ptr", i64 %"idx_local"
  store float %"val_in", float* %"ptr_shared"
  call void @"barrier"()
  %"neighbor_id" = xor i32 %"local_id", 1
  %"idx_neighbor" = zext i32 %"neighbor_id" to i64
  %"ptr_shared_neighbor" = getelementptr float, float* %"shared_ptr", i64 %"idx_neighbor"
  %"val_neighbor" = load float, float* %"ptr_shared_neighbor"
  %"ptr_out" = getelementptr float, float* %"out_ptr", i64 %"idx_global"
  store float %"val_neighbor", float* %"ptr_out"
  ret void
}
"""


@pytest.fixture(scope="module")
def binary():
    return llvm_to_metallib(LLVM_IR)


def test_basic_swap(binary):
    input_data = [10.0, 20.0, 30.0, 40.0]
    expected = [20.0, 10.0, 40.0, 30.0]
    result = run_kernel_1d_float(binary, input_data, "test_kernel")
    assert result == pytest.approx(expected)


def test_larger_array(binary):
    input_data = [float(i) for i in range(8)]
    # 0<->1, 2<->3, 4<->5, 6<->7
    # 0->1, 1->0, 2->3, 3->2, etc.
    expected = [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]
    result = run_kernel_1d_float(binary, input_data, "test_kernel")
    assert result == pytest.approx(expected)


def test_negative_values(binary):
    input_data = [-1.0, -2.0, 5.5, 6.5]
    expected = [-2.0, -1.0, 6.5, 5.5]
    result = run_kernel_1d_float(binary, input_data, "test_kernel")
    assert result == pytest.approx(expected)


def test_minimal_pair(binary):
    input_data = [1.0, 2.0]
    expected = [2.0, 1.0]
    result = run_kernel_1d_float(binary, input_data, "test_kernel")
    assert result == pytest.approx(expected)


def test_decimals(binary):
    input_data = [0.1, 0.2, 0.3, 0.4]
    expected = [0.2, 0.1, 0.4, 0.3]
    result = run_kernel_1d_float(binary, input_data, "test_kernel")
    assert result == pytest.approx(expected)
