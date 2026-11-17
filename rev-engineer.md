eapple silicon architecture:

- ARMv9-A, 64-bit. based on RISC but more instruction level parallelism.
- UMA unified memory = unified memory pool. CPU, GPU, ANE all share same address space.
- ANE apple neural engine = neural processing unit. asic just for specific tensor ops. purely inference focused and optimized for low precision / quantized models. can't be programed and targeted directly. best you can do is to recompile torch/tf models with `coremltools`.
- MPS metal performance shaders = not a compiler. library of single hand-tuned BLAS/LAPACK library operations.
- MPSGraph = directed acyclic graph of ops. minimizes memory roundtrips and dispatch latency by fusing multiple kernels together in a single metal kernel pass.
- MSL metal shading language = `.metal` shader files. similar to `.cuda` files. cpp-14 with added keywords. used to write graphics "shaders" or computation "kernels".
- https://www.youtube.com/watch?v=H6ZpMMDvB1M
- https://www.youtube.com/watch?v=6gQxhsZsawc
- https://developer.apple.com/metal/cpp/

targeting apple silicon gpus:

- pipeline: MSL `.metal` → AIR (LLVM based) `.metallib` → machine code`
- there is no MLIR→MSL lowering. but there are tools to convert shaders to MSL.
- the mojo-lang team wrote an LLVM→AIR bridge by reverse-engineering the metadata format
- https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295

inspect kernel output:

```bash
cat > /tmp/experiment.metal << 'EOF'
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a [[buffer(0)]],
                device float* b [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    b[id] = a[id] + 1.0f;
}
EOF
xcrun -sdk macosx metal -c /tmp/experiment.metal -o /tmp/experiment.air
xcrun -sdk macosx metal-objdump -d /tmp/experiment.air
```

outlook:

- https://github.com/JuliaGPU/GPUCompiler.jl/blob/d762273b0755012217124f57e875322fe6b8774f/src/metal.jl#L63
- https://github.com/JuliaGPU/Metal.jl/blob/f7b082942ecae649a2f993724e94de41b667a7d2/src/compiler/compilation.jl#L167-L241
- https://github.com/JuliaGPU/Metal.jl/blob/f7b082942ecae649a2f993724e94de41b667a7d2/bin/metallib-as#L8
- https://github.com/JuliaBinaryWrappers/LLVMDowngrader_jll.jl
- https://github.com/JuliaLLVM/llvm-downgrade
