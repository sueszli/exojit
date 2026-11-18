import ctypes

import Metal

from llvm_to_air import compile_to_metallib, create_compute_pipeline, execute_kernel, to_air

# one thread per output element: C[row, col] = A[row, 0] * B[0, col] + A[row, 1] * B[1, col]
LLVM_IR = """
define void @matmul(float* %A, float* %B, float* %C, i32 %id) {
entry:
  %i = zext i32 %id to i64
  %row = udiv i64 %i, 2
  %col = urem i64 %i, 2
  %a0_idx = mul i64 %row, 2
  %a1_idx = add i64 %a0_idx, 1
  %b1_idx = add i64 %col, 2
  %a0_ptr = getelementptr float, float* %A, i64 %a0_idx
  %a0 = load float, float* %a0_ptr
  %a1_ptr = getelementptr float, float* %A, i64 %a1_idx
  %a1 = load float, float* %a1_ptr
  %b0_ptr = getelementptr float, float* %B, i64 %col
  %b0 = load float, float* %b0_ptr
  %b1_ptr = getelementptr float, float* %B, i64 %b1_idx
  %b1 = load float, float* %b1_ptr
  %p0 = fmul float %a0, %b0
  %p1 = fmul float %a1, %b1
  %sum = fadd float %p0, %p1
  %c_ptr = getelementptr float, float* %C, i64 %i
  store float %sum, float* %c_ptr
  ret void
}
"""

air = to_air(LLVM_IR)
metallib = compile_to_metallib(air)
device, pso = create_compute_pipeline(metallib, "matmul")

# row-major 2x2 inputs, kept alive until the gpu work is done
a = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
b = (ctypes.c_float * 4)(5.0, 6.0, 7.0, 8.0)
size = ctypes.sizeof(a)

buf_a = device.newBufferWithBytes_length_options_(a, size, Metal.MTLResourceStorageModeShared)
buf_b = device.newBufferWithBytes_length_options_(b, size, Metal.MTLResourceStorageModeShared)
buf_c = device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
assert buf_a and buf_b and buf_c


# slots follow the kernel argument order
def bind(encoder):
    encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_c, 0, 2)


execute_kernel(device, pso, Metal.MTLSize(4, 1, 1), Metal.MTLSize(4, 1, 1), bind)

# copy out of shared memory while buf_c is still alive
result = list(memoryview(buf_c.contents().as_buffer(size)).cast("f"))
assert result == [19.0, 22.0, 43.0, 50.0], result
print(result)
