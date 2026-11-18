import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

#
# out[i] = in[i]
#

LLVM_IR_VEC_COPY = """
define void @vec_copy(<4 x float>* %a, <4 x float>* %b, i32 %id) {
  %1 = getelementptr inbounds <4 x float>, <4 x float>* %a, i32 %id
  %val = load <4 x float>, <4 x float>* %1, align 16
  %2 = getelementptr inbounds <4 x float>, <4 x float>* %b, i32 %id
  store <4 x float> %val, <4 x float>* %2, align 16
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_vec_copy():
    return llvm_to_metallib(LLVM_IR_VEC_COPY)


def test_vector_float4(binary_vec_copy):
    num_vectors = 64
    floats_per_vec = 4
    total_floats = num_vectors * floats_per_vec
    input_data = [float(i) for i in range(total_floats)]
    output = run_kernel_1d_float(binary_vec_copy, input_data, "vec_copy", threadgroup_size=32)
    assert output == input_data
