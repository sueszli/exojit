import pytest

from utils import llvm_to_metallib, run_kernel_1d_float

# max pooling 2D: out[y,x] = max(input[2*y:2*y+2, 2*x:2*x+2])
# input: 4x4, Pool size: 2x2, Stride: 2, Output: 2x2

LLVM_IR_MAXPOOL2D = """
declare float @llvm.maxnum.f32(float, float)

define void @maxpool2d(float* %input, float* %output, i32 %global_id) {
entry:
  ; input: 4x4 = 16 elements (row-major)
  ; output: 2x2 = 4 elements (row-major)
  ; pool size: 2x2, stride: 2
  ;
  ; global_id maps to output position:
  ;   id=0 -> out[0,0], id=1 -> out[0,1]
  ;   id=2 -> out[1,0], id=3 -> out[1,1]
  
  %idx = zext i32 %global_id to i64
  
  ; calculate output row and column
  ; out_row = id / 2, out_col = id % 2
  %out_row_32 = udiv i32 %global_id, 2
  %out_col_32 = urem i32 %global_id, 2
  %out_row = zext i32 %out_row_32 to i64
  %out_col = zext i32 %out_col_32 to i64
  
  ; calculate input starting position (stride = 2)
  %in_row_start = mul i64 %out_row, 2
  %in_col_start = mul i64 %out_col, 2
  
  ; load 2x2 pool window
  ; position [0, 0]
  %in_row_0 = add i64 %in_row_start, 0
  %in_col_0 = add i64 %in_col_start, 0
  %in_idx_00 = mul i64 %in_row_0, 4  ; input_width = 4
  %in_idx_00_x = add i64 %in_idx_00, %in_col_0
  %in_ptr_00 = getelementptr inbounds float, float* %input, i64 %in_idx_00_x
  %val_00 = load float, float* %in_ptr_00
  
  ; position [0, 1]
  %in_row_1 = add i64 %in_row_start, 0
  %in_col_1 = add i64 %in_col_start, 1
  %in_idx_01 = mul i64 %in_row_1, 4
  %in_idx_01_x = add i64 %in_idx_01, %in_col_1
  %in_ptr_01 = getelementptr inbounds float, float* %input, i64 %in_idx_01_x
  %val_01 = load float, float* %in_ptr_01
  
  ; position [1, 0]
  %in_row_2 = add i64 %in_row_start, 1
  %in_col_2 = add i64 %in_col_start, 0
  %in_idx_10 = mul i64 %in_row_2, 4
  %in_idx_10_x = add i64 %in_idx_10, %in_col_2
  %in_ptr_10 = getelementptr inbounds float, float* %input, i64 %in_idx_10_x
  %val_10 = load float, float* %in_ptr_10
  
  ; position [1, 1]
  %in_row_3 = add i64 %in_row_start, 1
  %in_col_3 = add i64 %in_col_start, 1
  %in_idx_11 = mul i64 %in_row_3, 4
  %in_idx_11_x = add i64 %in_idx_11, %in_col_3
  %in_ptr_11 = getelementptr inbounds float, float* %input, i64 %in_idx_11_x
  %val_11 = load float, float* %in_ptr_11
  
  ; compute max of 2x2 window
  %max_01 = call float @llvm.maxnum.f32(float %val_00, float %val_01)
  %max_23 = call float @llvm.maxnum.f32(float %val_10, float %val_11)
  %max_final = call float @llvm.maxnum.f32(float %max_01, float %max_23)
  
  ; store result
  %out_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %max_final, float* %out_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_maxpool2d():
    return llvm_to_metallib(LLVM_IR_MAXPOOL2D)


def test_maxpool2d_basic(binary_maxpool2d):
    # input: 4x4 matrix
    # [ 1,  2,  3,  4]
    # [ 5,  6,  7,  8]
    # [ 9, 10, 11, 12]
    # [13, 14, 15, 16]
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]

    # expected output: 2x2 matrix (2x2 pooling, stride 2)
    # out[0,0] = max(1,2,5,6) = 6
    # out[0,1] = max(3,4,7,8) = 8
    # out[1,0] = max(9,10,13,14) = 14
    # out[1,1] = max(11,12,15,16) = 16
    expected = [6.0, 8.0, 14.0, 16.0]

    result = run_kernel_1d_float(binary_maxpool2d, input_data, "maxpool2d")
    assert result[:4] == pytest.approx(expected)


def test_maxpool2d_negative(binary_maxpool2d):
    # input: 4x4 matrix with negative values
    input_data = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0]

    # expected: max of each 2x2 region (least negative)
    expected = [-1.0, -3.0, -9.0, -11.0]

    result = run_kernel_1d_float(binary_maxpool2d, input_data, "maxpool2d")
    assert result[:4] == pytest.approx(expected)


# average pooling 2D: out[y,x] = avg(input[2*y:2*y+2, 2*x:2*x+2])
LLVM_IR_AVGPOOL2D = """
define void @avgpool2d(float* %input, float* %output, i32 %global_id) {
entry:
  ; input: 4x4 = 16 elements (row-major)
  ; output: 2x2 = 4 elements (row-major)
  ; pool size: 2x2, stride: 2
  
  %idx = zext i32 %global_id to i64
  
  ; calculate output row and column
  %out_row_32 = udiv i32 %global_id, 2
  %out_col_32 = urem i32 %global_id, 2
  %out_row = zext i32 %out_row_32 to i64
  %out_col = zext i32 %out_col_32 to i64
  
  ; calculate input starting position (stride = 2)
  %in_row_start = mul i64 %out_row, 2
  %in_col_start = mul i64 %out_col, 2
  
  ; load 2x2 pool window
  ; position [0, 0]
  %in_row_0 = add i64 %in_row_start, 0
  %in_col_0 = add i64 %in_col_start, 0
  %in_idx_00 = mul i64 %in_row_0, 4
  %in_idx_00_x = add i64 %in_idx_00, %in_col_0
  %in_ptr_00 = getelementptr inbounds float, float* %input, i64 %in_idx_00_x
  %val_00 = load float, float* %in_ptr_00
  
  ; position [0, 1]
  %in_row_1 = add i64 %in_row_start, 0
  %in_col_1 = add i64 %in_col_start, 1
  %in_idx_01 = mul i64 %in_row_1, 4
  %in_idx_01_x = add i64 %in_idx_01, %in_col_1
  %in_ptr_01 = getelementptr inbounds float, float* %input, i64 %in_idx_01_x
  %val_01 = load float, float* %in_ptr_01
  
  ; position [1, 0]
  %in_row_2 = add i64 %in_row_start, 1
  %in_col_2 = add i64 %in_col_start, 0
  %in_idx_10 = mul i64 %in_row_2, 4
  %in_idx_10_x = add i64 %in_idx_10, %in_col_2
  %in_ptr_10 = getelementptr inbounds float, float* %input, i64 %in_idx_10_x
  %val_10 = load float, float* %in_ptr_10
  
  ; position [1, 1]
  %in_row_3 = add i64 %in_row_start, 1
  %in_col_3 = add i64 %in_col_start, 1
  %in_idx_11 = mul i64 %in_row_3, 4
  %in_idx_11_x = add i64 %in_idx_11, %in_col_3
  %in_ptr_11 = getelementptr inbounds float, float* %input, i64 %in_idx_11_x
  %val_11 = load float, float* %in_ptr_11
  
  ; compute average of 2x2 window
  %sum_01 = fadd float %val_00, %val_01
  %sum_23 = fadd float %val_10, %val_11
  %sum_all = fadd float %sum_01, %sum_23
  %avg = fdiv float %sum_all, 4.0
  
  ; store result
  %out_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %avg, float* %out_ptr
  
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_avgpool2d():
    return llvm_to_metallib(LLVM_IR_AVGPOOL2D)


def test_avgpool2d_basic(binary_avgpool2d):
    # input: 4x4 matrix
    input_data = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]

    # expected output: 2x2 matrix
    # out[0,0] = avg(1,2,5,6) = 3.5
    # out[0,1] = avg(3,4,7,8) = 5.5
    # out[1,0] = avg(9,10,13,14) = 11.5
    # out[1,1] = avg(11,12,15,16) = 13.5
    expected = [3.5, 5.5, 11.5, 13.5]

    result = run_kernel_1d_float(binary_avgpool2d, input_data, "avgpool2d")
    assert result[:4] == pytest.approx(expected)


def test_avgpool2d_uniform(binary_avgpool2d):
    # input: all 5.0
    input_data = [5.0] * 16

    # expected: all 5.0
    expected = [5.0, 5.0, 5.0, 5.0]

    result = run_kernel_1d_float(binary_avgpool2d, input_data, "avgpool2d")
    assert result[:4] == pytest.approx(expected)


LLVM_IR_MINPOOL2D = """
declare float @llvm.minnum.f32(float, float)

define void @minpool2d(float* %input, float* %output, i32 %global_id) {
entry:
  %idx = zext i32 %global_id to i64
  %out_row_32 = udiv i32 %global_id, 2
  %out_col_32 = urem i32 %global_id, 2
  %out_row = zext i32 %out_row_32 to i64
  %out_col = zext i32 %out_col_32 to i64
  %in_row_start = mul i64 %out_row, 2
  %in_col_start = mul i64 %out_col, 2
  %in_row_0 = add i64 %in_row_start, 0
  %in_col_0 = add i64 %in_col_start, 0
  %in_idx_00 = mul i64 %in_row_0, 4
  %in_idx_00_x = add i64 %in_idx_00, %in_col_0
  %in_ptr_00 = getelementptr inbounds float, float* %input, i64 %in_idx_00_x
  %val_00 = load float, float* %in_ptr_00
  %in_row_1 = add i64 %in_row_start, 0
  %in_col_1 = add i64 %in_col_start, 1
  %in_idx_01 = mul i64 %in_row_1, 4
  %in_idx_01_x = add i64 %in_idx_01, %in_col_1
  %in_ptr_01 = getelementptr inbounds float, float* %input, i64 %in_idx_01_x
  %val_01 = load float, float* %in_ptr_01
  %in_row_2 = add i64 %in_row_start, 1
  %in_col_2 = add i64 %in_col_start, 0
  %in_idx_10 = mul i64 %in_row_2, 4
  %in_idx_10_x = add i64 %in_idx_10, %in_col_2
  %in_ptr_10 = getelementptr inbounds float, float* %input, i64 %in_idx_10_x
  %val_10 = load float, float* %in_ptr_10
  %in_row_3 = add i64 %in_row_start, 1
  %in_col_3 = add i64 %in_col_start, 1
  %in_idx_11 = mul i64 %in_row_3, 4
  %in_idx_11_x = add i64 %in_idx_11, %in_col_3
  %in_ptr_11 = getelementptr inbounds float, float* %input, i64 %in_idx_11_x
  %val_11 = load float, float* %in_ptr_11
  %min_01 = call float @llvm.minnum.f32(float %val_00, float %val_01)
  %min_23 = call float @llvm.minnum.f32(float %val_10, float %val_11)
  %min_final = call float @llvm.minnum.f32(float %min_01, float %min_23)
  %out_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %min_final, float* %out_ptr
  ret void
}
"""


@pytest.fixture(scope="module")
def binary_minpool2d():
    return llvm_to_metallib(LLVM_IR_MINPOOL2D)


def test_minpool2d(binary_minpool2d):
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    expected = [1.0, 3.0, 9.0, 11.0]
    result = run_kernel_1d_float(binary_minpool2d, input_data, "minpool2d")
    assert result[:4] == pytest.approx(expected)
