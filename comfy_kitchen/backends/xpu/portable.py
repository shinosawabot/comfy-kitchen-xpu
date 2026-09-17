"""Correctness-first XPU routes using existing portable PyTorch operations.

These entries are reference implementations, not native low-bit matrix kernels.
They retain packed Kitchen formats and keep all computation on the input XPU.
"""

from __future__ import annotations

from functools import wraps

import torch

from comfy_kitchen.backends.eager import quantization as _quantization
from comfy_kitchen.backends.eager import w4a8_int8 as _w4a8
from comfy_kitchen.backends.eager.awq import gemv_awq_w4a16 as _awq

__all__ = [
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "scaled_mm_nvfp4",
    "quantize_mxfp8",
    "dequantize_mxfp8",
    "scaled_mm_mxfp8",
    "quantize_w4a8_int8_weight",
    "dequantize_w4a8_int8_weight",
    "w4a8_int8_linear",
    "gemv_awq_w4a16",
]


def _require_same_xpu(*values):
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors or tensors[0].device.type != "xpu":
        raise ValueError("portable XPU operators require XPU tensors")
    if any(value.device != tensors[0].device for value in tensors[1:]):
        raise ValueError("portable XPU operands must be on the same device")


def _portable(function):
    @wraps(function)
    def call(*args, **kwargs):
        _require_same_xpu(*args, *kwargs.values())
        return function(*args, **kwargs)

    return call


quantize_nvfp4 = _portable(_quantization.quantize_nvfp4)
dequantize_nvfp4 = _portable(_quantization.dequantize_nvfp4)
quantize_mxfp8 = _portable(_quantization.quantize_mxfp8)
dequantize_mxfp8 = _portable(_quantization.dequantize_mxfp8)
quantize_w4a8_int8_weight = _portable(_w4a8.quantize_w4a8_int8_weight)
dequantize_w4a8_int8_weight = _portable(_w4a8.dequantize_w4a8_int8_weight)
w4a8_int8_linear = _portable(_w4a8.w4a8_int8_linear)
gemv_awq_w4a16 = _portable(_awq)


def _validate_mm(a, b, bias, out_dtype, block_size, packed):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("scaled_mm requires matrices [M,K] and [N,K] with matching K")
    if a.shape[1] * (2 if packed else 1) % block_size:
        raise ValueError(f"scaled_mm logical K must be divisible by {block_size}")
    if bias is not None and (bias.ndim != 1 or bias.numel() != b.shape[0]):
        raise ValueError("scaled_mm bias must have shape [N]")
    if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("scaled_mm output dtype must be float32, float16 or bfloat16")


def scaled_mm_nvfp4(
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
    _require_same_xpu(
        a, b, tensor_scale_a, tensor_scale_b, block_scale_a, block_scale_b, bias, alpha
    )
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    _validate_mm(a, b, bias, out_dtype, 16, True)
    if tensor_scale_a.numel() != 1 or tensor_scale_b.numel() != 1:
        raise ValueError("NVFP4 tensor scales must be scalar")
    if alpha is not None and alpha.numel() != 1:
        raise ValueError("NVFP4 alpha must be scalar")
    one = torch.ones((), device=a.device, dtype=torch.float32)
    decoded_a = dequantize_nvfp4(a, one, block_scale_a, torch.float32)
    decoded_b = dequantize_nvfp4(b, one, block_scale_b, torch.float32)
    result = decoded_a @ decoded_b.t()
    factor = tensor_scale_a * tensor_scale_b if alpha is None else alpha
    result = result * factor.float()
    if bias is not None:
        result = result + bias.float()
    return result.to(out_dtype)


def scaled_mm_mxfp8(a, b, block_scale_a, block_scale_b, bias=None, out_dtype=None):
    """Portable block-scaled A @ B.T with FP32 accumulation and final cast."""
    _require_same_xpu(a, b, block_scale_a, block_scale_b, bias)
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    _validate_mm(a, b, bias, out_dtype, 32, False)
    decoded_a = dequantize_mxfp8(a, block_scale_a, torch.float32)
    decoded_b = dequantize_mxfp8(b, block_scale_b, torch.float32)
    result = decoded_a @ decoded_b.t()
    if bias is not None:
        result = result + bias.float()
    return result.to(out_dtype)
