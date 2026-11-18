import ctypes

import pytest

from backend_metal import create_compute_pipeline, execute_kernel

from .utils import Metal, llvm_to_metallib

# 2D Convolution: out[y,x] = sum over kernel of input[y+ky, x+kx] * kernel[ky, kx]
# simplified version: 3x3 input, 2x2 kernel, no padding, stride=1
# output will be 2x2
LLVM_IR_CONV2D = """
define void @conv2d(float* %input, float* %kernel, float* %output, i32 %global_id) {
entry:
  ; input: 3x3 = 9 elements (row-major)
  ; kernel: 2x2 = 4 elements (row-major)
  ; output: 2x2 = 4 elements (row-major)
  ; 
  ; global_id maps to output position:
  ;   id=0 -> out[0,0], id=1 -> out[0,1]
  ;   id=2 -> out[1,0], id=3 -> out[1,1]
  
  %idx = zext i32 %global_id to i64
  
  ; Calculate output row and column
  ; out_row = id / 2, out_col = id % 2
  %out_row_32 = udiv i32 %global_id, 2
  %out_col_32 = urem i32 %global_id, 2
  %out_row = zext i32 %out_row_32 to i64
  %out_col = zext i32 %out_col_32 to i64
  
  ; convolution computation: sum over 2x2 kernel
  ; result = 0
  %sum_init = fadd float 0.0, 0.0
  
  ; kernel position [0, 0]
  %in_y_0 = add i64 %out_row, 0
  %in_x_0 = add i64 %out_col, 0
  %in_idx_00 = mul i64 %in_y_0, 3  ; input_width = 3
  %in_idx_00_x = add i64 %in_idx_00, %in_x_0
  %in_ptr_00 = getelementptr inbounds float, float* %input, i64 %in_idx_00_x
  %in_val_00 = load float, float* %in_ptr_00
  
  %k_ptr_00 = getelementptr inbounds float, float* %kernel, i64 0
  %k_val_00 = load float, float* %k_ptr_00
  
  %prod_00 = fmul float %in_val_00, %k_val_00
  %sum_00 = fadd float %sum_init, %prod_00
  
  ; kernel position [0, 1]
  %in_y_1 = add i64 %out_row, 0
  %in_x_1 = add i64 %out_col, 1
  %in_idx_01 = mul i64 %in_y_1, 3
  %in_idx_01_x = add i64 %in_idx_01, %in_x_1
  %in_ptr_01 = getelementptr inbounds float, float* %input, i64 %in_idx_01_x
  %in_val_01 = load float, float* %in_ptr_01
  
  %k_ptr_01 = getelementptr inbounds float, float* %kernel, i64 1
  %k_val_01 = load float, float* %k_ptr_01
  
  %prod_01 = fmul float %in_val_01, %k_val_01
  %sum_01 = fadd float %sum_00, %prod_01
  
  ; kernel position [1, 0]
  %in_y_2 = add i64 %out_row, 1
  %in_x_2 = add i64 %out_col, 0
  %in_idx_10 = mul i64 %in_y_2, 3
  %in_idx_10_x = add i64 %in_idx_10, %in_x_2
  %in_ptr_10 = getelementptr inbounds float, float* %input, i64 %in_idx_10_x
  %in_val_10 = load float, float* %in_ptr_10
  
  %k_ptr_10 = getelementptr inbounds float, float* %kernel, i64 2
  %k_val_10 = load float, float* %k_ptr_10
  
  %prod_10 = fmul float %in_val_10, %k_val_10
  %sum_10 = fadd float %sum_01, %prod_10
  
  ; kernel position [1, 1]
  %in_y_3 = add i64 %out_row, 1
  %in_x_3 = add i64 %out_col, 1
  %in_idx_11 = mul i64 %in_y_3, 3
  %in_idx_11_x = add i64 %in_idx_11, %in_x_3
  %in_ptr_11 = getelementptr inbounds float, float* %input, i64 %in_idx_11_x
  %in_val_11 = load float, float* %in_ptr_11
  
  %k_ptr_11 = getelementptr inbounds float, float* %kernel, i64 3
  %k_val_11 = load float, float* %k_ptr_11
  
  %prod_11 = fmul float %in_val_11, %k_val_11
  %sum_final = fadd float %sum_10, %prod_11
  
  ; store result
  %out_ptr = getelementptr inbounds float, float* %output, i64 %idx
  store float %sum_final, float* %out_ptr
  
  ret void
}
"""


def run_conv2d(binary, input_img, kernel, output_size):
    device, pso = create_compute_pipeline(binary, "conv2d")

    def create_buffer(data):
        raw_array = (ctypes.c_float * len(data))(*data)
        return device.newBufferWithBytes_length_options_(raw_array, ctypes.sizeof(raw_array), Metal.MTLResourceStorageModeShared)

    buf_input = create_buffer(input_img)
    buf_kernel = create_buffer(kernel)
    buf_output = device.newBufferWithLength_options_(output_size * 4, Metal.MTLResourceStorageModeShared)

    def encode_args(encoder):
        encoder.setBuffer_offset_atIndex_(buf_input, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_kernel, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_output, 0, 2)

    grid_size = Metal.MTLSize(output_size, 1, 1)
    threadgroup_size = Metal.MTLSize(1, 1, 1)

    execute_kernel(device, pso, grid_size, threadgroup_size, encode_args)

    output_ptr = buf_output.contents()
    output_buffer = output_ptr.as_buffer(output_size * 4)
    results_view = memoryview(output_buffer).cast("f")
    return list(results_view)


@pytest.fixture(scope="module")
def binary_conv2d():
    return llvm_to_metallib(LLVM_IR_CONV2D)


def test_conv2d_identity(binary_conv2d):
    # input: 3x3 matrix
    # [1, 2, 3]
    # [4, 5, 6]
    # [7, 8, 9]
    input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    # kernel: 2x2 matrix (simple averaging kernel)
    # [0.25, 0.25]
    # [0.25, 0.25]
    kernel_data = [0.25, 0.25, 0.25, 0.25]

    # expected output: 2x2 matrix
    # out[0,0] = 0.25*(1+2+4+5) = 3.0
    # out[0,1] = 0.25*(2+3+5+6) = 4.0
    # out[1,0] = 0.25*(4+5+7+8) = 6.0
    # out[1,1] = 0.25*(5+6+8+9) = 7.0
    expected = [3.0, 4.0, 6.0, 7.0]

    result = run_conv2d(binary_conv2d, input_data, kernel_data, 4)
    assert result == pytest.approx(expected)


def test_conv2d_edge_detection(binary_conv2d):
    # input: 3x3 matrix
    # [1, 1, 1]
    # [1, 1, 1]
    # [1, 1, 1]
    input_data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # kernel: 2x2 Sobel-like
    # [1, -1]
    # [1, -1]
    kernel_data = [1.0, -1.0, 1.0, -1.0]

    # expected: all zeros (no edges in uniform image)
    expected = [0.0, 0.0, 0.0, 0.0]

    result = run_conv2d(binary_conv2d, input_data, kernel_data, 4)
    assert result == pytest.approx(expected)
