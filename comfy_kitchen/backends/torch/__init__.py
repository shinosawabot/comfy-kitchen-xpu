"""Torch optimized block-scaled GEMM, selected by Kitchen's registry.

Only NVFP4/MXFP8 are exposed here. Their Kitchen swizzled scale layouts match
CUDA cuBLASLt (CUDA >=12.8, SM >=10); current Torch CPU/XPU and ROCm recipes
cannot consume that layout. No reference computation or exception fallback.
Contract sources: PyTorch cf30153 ScaledBlas.cpp and cuBLAS 12.8 release notes.
"""

import sys
from dataclasses import replace
from functools import partial

import torch
from comfy_kitchen.constraints import ValidationResult
from comfy_kitchen.registry import registry
from comfy_kitchen.scaled_mm_v2 import (
    ScalingType,
    SwizzleType,
    _scaled_mm_v2_torch,
    has_scaled_mm_v2,
)

__all__ = ["scaled_mm_nvfp4", "scaled_mm_mxfp8"]


def _api_available():
    if has_scaled_mm_v2():
        return callable(getattr(torch.nn.functional, "scaled_mm", None))
    return callable(getattr(torch, "_scaled_mm", None))


def _cuda_format_device(device):
    if device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return False
    version = getattr(torch.version, "cuda", None)
    if version is None or tuple(int(v) for v in version.split(".")[:2]) < (12, 8):
        return False
    op = "aten::_scaled_mm_v2" if has_scaled_mm_v2() else "aten::_scaled_mm"
    if not torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA"):
        return False
    return torch.cuda.get_device_capability(device) >= (10, 0)


def _round_up(n, block):
    return ((n + block - 1) // block) * block


def _format_call_rule(kwargs, *, packed):
    def fail(why):
        return ValidationResult.fail("__torch_scaled_mm__", why)

    a, b = kwargs.get("a"), kwargs.get("b")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return fail("matrix operands are required")
    if not _api_available() or not _cuda_format_device(a.device):
        return fail("Torch CUDA block-scaled GEMM API/device contract is unavailable")
    if packed and not hasattr(torch, "float4_e2m1fn_x2"):
        return fail("Torch packed FP4 dtype is unavailable")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        return fail("expected [M,K] and [N,K] matrices")
    block = 16 if packed else 32
    k = a.shape[1] * (2 if packed else 1)
    if (
        min(a.shape[0], b.shape[0], k) == 0
        or k % block
        or b.shape[0] % 16
        or (packed and not has_scaled_mm_v2() and k % 32)
    ):
        return fail("nonempty matrices require logical K block alignment and N%16==0")
    for value in kwargs.values():
        if isinstance(value, torch.Tensor) and (
            value.device != a.device or value.layout != torch.strided
        ):
            return fail("all operands must be strided tensors on the input device")
    for key, rows in (("block_scale_a", a.shape[0]), ("block_scale_b", b.shape[0])):
        scales = kwargs.get(key)
        if (
            scales is None
            or not scales.is_contiguous()
            or scales.numel() != _round_up(rows, 128) * _round_up(k // block, 4)
        ):
            return fail("scale storage must be contiguous Kitchen SWIZZLE_32_4_4")
    if packed:
        for key in ("tensor_scale_a", "tensor_scale_b"):
            if kwargs.get(key) is None or kwargs[key].numel() != 1:
                return fail("global scales must be scalar")
        alpha = kwargs.get("alpha")
        if alpha is not None and alpha.numel() != 1:
            return fail("alpha must be scalar")
    bias = kwargs.get("bias")
    out_dtype = kwargs.get("out_dtype") or torch.bfloat16
    if bias is not None and (
        bias.ndim != 1
        or bias.numel() != b.shape[0]
        or not bias.is_contiguous()
        or out_dtype not in (torch.float16, torch.bfloat16)
        or bias.dtype != out_dtype
    ):
        return fail("native bias must be a contiguous [N] vector matching FP16/BF16 output")
    return ValidationResult.ok()


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
    if alpha is not None:
        alpha = alpha.to(dtype=torch.float32).reshape(1)
    return _scaled_mm_v2_torch(
        a.view(torch.float4_e2m1fn_x2),
        b.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[block_scale_a.view(-1), tensor_scale_a if alpha is None else alpha],
        scale_b=[
            block_scale_b.view(-1),
            tensor_scale_b if alpha is None else torch.ones_like(alpha),
        ],
        bias=bias,
        out_dtype=out_dtype or torch.bfloat16,
        scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    )


def scaled_mm_mxfp8(a, b, block_scale_a, block_scale_b, bias=None, out_dtype=None):
    return _scaled_mm_v2_torch(
        a,
        b.t(),
        scale_a=block_scale_a,
        scale_b=block_scale_b,
        bias=bias,
        out_dtype=out_dtype or torch.bfloat16,
        scale_recipe_a=ScalingType.BlockWise1x32,
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
    )


def _build_constraints():
    from comfy_kitchen.backends.eager import _build_constraints as eager_constraints

    source = eager_constraints()
    out = {}
    for name in __all__:
        if name in source:
            out[name] = replace(
                source[name],
                default_devices=frozenset({"cuda"}),
                call_rules=(partial(_format_call_rule, packed=name.endswith("nvfp4")),),
            )
    return out


if _api_available():
    registry.register("torch", sys.modules[__name__], _build_constraints())
else:
    registry.mark_unavailable("torch", "Torch scaled-mm API is unavailable")
