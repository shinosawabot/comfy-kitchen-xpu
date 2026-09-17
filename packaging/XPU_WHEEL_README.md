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

The XPU backend also exposes reference implementations for NVFP4 and MXFP8
quantization, dequantization and scaled matrix multiplication, W4A8 weight
quantization/dequantization/linear, and AWQ W4A16 GEMV. These routes use PyTorch
operations on the input XPU; registration does not mean native low-bit matrix
acceleration. The backend's `_REFERENCE_CAPABILITIES` distinguishes these
entries from Omni native capability flags. Scaled NVFP4/MXFP8 multiplication
decodes operands to FP32, accumulates in FP32 and casts the result. For NVFP4,
explicit `alpha` replaces the product of global tensor scales; bias is added
last. Reference implementations can use substantially more temporary memory
than a native packed GEMM.

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
