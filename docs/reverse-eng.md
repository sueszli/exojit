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
