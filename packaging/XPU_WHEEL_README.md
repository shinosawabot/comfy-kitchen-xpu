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

Eager NVFP4/MXFP8 scaled-matrix operations are pure reference computations:
FP32 dequantization, matrix multiplication and final cast. Kitchen's separate
`torch` backend exposes only the optimized Torch scaled-mm calls and is ordered
after existing native/Triton backends and before eager. Its constraints check
the actual input device, API/kernel presence, CUDA 12.8+/SM10+ block-format
support, matching dtypes, matrix dimensions, scale storage and bias layout.
Kitchen's swizzled formats are not admitted to current CPU/XPU/ROCm Torch
recipes. Selection occurs in the registry. The `torch` and eager implementations do not
catch execution errors or retry a failed selected implementation.

`use_backend("eager")` explicitly requests reference computation. Backend
contexts remain preferences with normal registry fall-through; direct
`registry.get_implementation(..., backend="torch", kwargs=...)` rejects an
ineligible native call. Custom priority lists must include `torch` to select
that optimization. Reference operations are not duplicated in XPU's registry.
NVFP4 `alpha` replaces the global-scale product and bias is added last. Reference
paths can allocate full FP32 intermediates; no native throughput is claimed.

DG2 W4A4 GEMM is rejected by the native XPU call constraint, so the registry
selects eager. The tensor layout adapter transiently restores preconverted
unsigned U4 storage to signed Kitchen format and its original scale dtype
before dispatching to a nonnative backend; persistent storage is unchanged.
Native preconverted GEMM remains available where its constraints permit it.

Native `flash_attention_decode` remains unavailable on XPU and rejects XPU
calls. Ordinary Torch SDPA is not advertised as flash decode and existing
ComfyUI attention routing is unchanged. Neighborhood attention retains its
eager implementation.

Without the native Sol sidecar, the Sol reference supports `token_aug=0`;
nonzero token augmentation and the chunked producer API raise
`NotImplementedError`. Packed INT8 attention production and consumption are
not implemented on XPU and also raise explicitly. They are not replaced by
floating attention because their packed representation and early-release
contracts differ. Native Sol capability queries remain false on DG2.
