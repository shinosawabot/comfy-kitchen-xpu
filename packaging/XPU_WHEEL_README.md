# Comfy Kitchen XPU

Pure-Python Comfy Kitchen integration for Intel XPU. The wheel packages the
XPU, Triton, and eager backends and uses the target-specific
`omni_xpu_kernel` companion wheel for native Intel GPU kernels.

CUDA source is retained in the source repository for synchronization with
upstream Comfy Kitchen, but the CUDA backend and native CUDA artifacts are not
included in this wheel.

See the
[Comfy Kitchen XPU repository](https://github.com/xiangyuT/comfy-kitchen-xpu)
for installation and companion-wheel build instructions. This work is based on
and remains grateful to
[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen).

### Portable XPU format coverage

NVFP4/MXFP8 quantization and dequantization, W4A8 weight
quantization/dequantization/linear, and AWQ W4A16 GEMV use the existing eager
implementations when no higher-priority backend can handle the call. They are
not duplicated in the native XPU registry. Default backend priority and any
eligible Triton route remain unchanged.

The eager NVFP4/MXFP8 scaled-matrix operations first try the existing Torch
scaled-mm route. When Torch explicitly reports the operation unsupported
(including the current CPU/XPU swizzled-scale limitations), an eager reference
decodes operands to FP32, accumulates in FP32 and casts the result on the input
device. OOM and unrelated input/runtime errors propagate. For NVFP4, explicit
`alpha` replaces the product of global tensor scales; bias is added last.
Reference implementations can use substantially more temporary memory than a
native packed GEMM. They are general eager computations, not native XPU kernels.

`use_backend("eager")` selects these eager implementations. The existing
`use_backend("xpu")` context is a preference and can fall through for operations
that XPU does not register; requesting `registry.get_implementation(...,
backend="xpu")` directly for these reference operations raises
`BackendNotImplementedError`. Public API availability must not be inferred
solely from the native XPU registry.

`flash_attention_decode` accepts BF16 XPU BTHD tensors through a Torch SDPA
reference, including GQA and per-batch KV lengths. Query availability with the
explicit XPU device; the no-argument query retains the native CUDA/HIP meaning.
This does not change ordinary ComfyUI attention routing. Neighborhood attention
continues to have an eager XPU fallback.

Without the native Sol sidecar, the Sol reference supports `token_aug=0`;
nonzero token augmentation and the chunked producer API raise
`NotImplementedError`. Packed INT8 attention production and consumption are
not implemented on XPU and also raise explicitly. They are not replaced by
floating attention because their packed representation and early-release
contracts differ. Native Sol capability queries remain false on DG2.
