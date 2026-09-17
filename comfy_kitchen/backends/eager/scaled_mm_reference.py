"""Same-device FP32 decode/matmul references for unsupported scaled-mm formats.

These are eager computations, not hardware backend registrations. Kitchen
selects the eager backend before these functions run; they contain no backend
selection or exception fallback.
"""

import torch


def _require_same_device(*values):
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors or any(value.device != tensors[0].device for value in tensors[1:]):
        raise ValueError("scaled-mm reference operands must be on the same device")


def _validate_mm(a, b, bias, out_dtype, block_size, packed):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("scaled_mm requires matrices [M,K] and [N,K] with matching K")
    if a.shape[1] * (2 if packed else 1) % block_size:
        raise ValueError(f"scaled_mm logical K must be divisible by {block_size}")
    if bias is not None and (bias.ndim != 1 or bias.numel() != b.shape[0]):
        raise ValueError("scaled_mm bias must have shape [N]")
    if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("scaled_mm output dtype must be float32, float16 or bfloat16")


def scaled_mm_nvfp4_reference(
    a,
    b,
    tensor_scale_a,
    tensor_scale_b,
    block_scale_a,
    block_scale_b,
    bias=None,
    out_dtype=None,
    alpha=None,
):
    """Decode block scales, accumulate FP32, apply alpha once, then bias.

    Explicit alpha replaces tensor_scale_a * tensor_scale_b, matching the CUDA
    public boundary. Decode with unit global scale to avoid applying it twice.
    """
    _require_same_device(
        a, b, tensor_scale_a, tensor_scale_b, block_scale_a, block_scale_b, bias, alpha
    )
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    _validate_mm(a, b, bias, out_dtype, 16, True)
    if tensor_scale_a.numel() != 1 or tensor_scale_b.numel() != 1:
        raise ValueError("NVFP4 tensor scales must be scalar")
    if alpha is not None and alpha.numel() != 1:
        raise ValueError("NVFP4 alpha must be scalar")
    from .quantization import dequantize_nvfp4

    one = torch.ones((), device=a.device, dtype=torch.float32)
    decoded_a = dequantize_nvfp4(a, one, block_scale_a, torch.float32)
    decoded_b = dequantize_nvfp4(b, one, block_scale_b, torch.float32)
    result = decoded_a @ decoded_b.t()
    factor = tensor_scale_a * tensor_scale_b if alpha is None else alpha
    result = result * factor.float()
    if bias is not None:
        result = result + bias.float()
    return result.to(out_dtype)


def scaled_mm_mxfp8_reference(a, b, block_scale_a, block_scale_b, bias=None, out_dtype=None):
    """Portable block-scaled A @ B.T with FP32 accumulation and final cast."""
    _require_same_device(a, b, block_scale_a, block_scale_b, bias)
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    _validate_mm(a, b, bias, out_dtype, 32, False)
    from .quantization import dequantize_mxfp8

    decoded_a = dequantize_mxfp8(a, block_scale_a, torch.float32)
    decoded_b = dequantize_mxfp8(b, block_scale_b, torch.float32)
    result = decoded_a @ decoded_b.t()
    if bias is not None:
        result = result + bias.float()
    return result.to(out_dtype)
