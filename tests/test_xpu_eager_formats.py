"""A770 public format contracts through eager, without duplicate XPU registration."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as reference
from comfy_kitchen.backends.eager import scaled_mm_reference as reference_mm
from comfy_kitchen.backends.eager import w4a8_int8 as w4_reference

pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not torch.xpu.is_available() or not ck.list_backends().get("xpu", {}).get("available"),
        reason="XPU backend required",
    ),
]


def _cpu(x):
    return None if x is None else x.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hi_first", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_nvfp4_qdq_packed_cpu_oracle(dtype, hi_first, pad):
    torch.manual_seed(117)
    x = torch.randn((5, 37) if pad else (5, 64), dtype=dtype)
    x[0] = 0
    scale = torch.tensor([0.25])
    with ck.use_backend("eager"):
        q, s = ck.quantize_nvfp4(x.to("xpu"), scale.to("xpu"), pad_16x=pad, hi_first=hi_first)
        out = ck.dequantize_nvfp4(q, scale.to("xpu"), s, output_type=dtype, hi_first=hi_first)
    expected_q, expected_s = reference.quantize_nvfp4(x, scale, pad_16x=pad, hi_first=hi_first)
    torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        s.view(torch.uint8).cpu(), expected_s.view(torch.uint8), rtol=0, atol=0
    )
    expected = reference.dequantize_nvfp4(expected_q, scale, expected_s, dtype, hi_first)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert out.device.type == "xpu"
    assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pad", [False, True])
def test_mxfp8_qdq_packed_cpu_oracle(dtype, pad):
    torch.manual_seed(118)
    x = torch.randn((5, 37) if pad else (5, 64), dtype=dtype)
    x[0] = 0
    with ck.use_backend("eager"):
        q, s = ck.quantize_mxfp8(x.to("xpu"), pad_32x=pad)
        out = ck.dequantize_mxfp8(q, s, output_type=dtype)
    eq, es = reference.quantize_mxfp8(x, pad)
    torch.testing.assert_close(q.view(torch.uint8).cpu(), eq.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(s.view(torch.uint8).cpu(), es.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), reference.dequantize_mxfp8(eq, es, dtype), rtol=0, atol=0)
    assert out.device.type == "xpu"


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_scaled_mm_asymmetric_linear_semantics(format, dtype, with_bias):
    torch.manual_seed(119)
    a, b = torch.randn(7, 64), torch.randn(11, 64)
    bias = torch.randn(11) if with_bias else None
    sa, sb = torch.tensor([0.125]), torch.tensor([0.5])
    with ck.use_backend("eager"):
        if format == "nvfp4":
            qa, ba = ck.quantize_nvfp4(a.to("xpu"), sa.to("xpu"))
            qb, bb = ck.quantize_nvfp4(b.to("xpu"), sb.to("xpu"))
            out = ck.scaled_mm_nvfp4(
                qa,
                qb,
                sa.to("xpu"),
                sb.to("xpu"),
                ba,
                bb,
                bias=None if bias is None else bias.to("xpu"),
                out_dtype=dtype,
            )
            da = reference.dequantize_nvfp4(qa.cpu(), sa, ba.cpu(), torch.float32)
            db = reference.dequantize_nvfp4(qb.cpu(), sb, bb.cpu(), torch.float32)
        else:
            qa, ba = ck.quantize_mxfp8(a.to("xpu"))
            qb, bb = ck.quantize_mxfp8(b.to("xpu"))
            out = ck.scaled_mm_mxfp8(
                qa, qb, ba, bb, bias=None if bias is None else bias.to("xpu"), out_dtype=dtype
            )
            da = reference.dequantize_mxfp8(qa.cpu(), ba.cpu(), torch.float32)
            db = reference.dequantize_mxfp8(qb.cpu(), bb.cpu(), torch.float32)
    expected = da @ db.T
    if bias is not None:
        expected += bias
    torch.testing.assert_close(
        out.cpu(),
        expected.to(dtype),
        rtol=1e-5 if dtype == torch.float32 else 0.002,
        atol=1e-5 if dtype == torch.float32 else 0.015625,
    )
    assert out.shape == (7, 11) and out.dtype == dtype and out.device.type == "xpu"


def test_nvfp4_explicit_alpha_replaces_global_scale_product():
    # All FP4 nibbles = +1, all block scales = 1: exact dot product K=32.
    from comfy_kitchen.float_utils import to_blocked

    q = torch.full((3, 16), 0x22, dtype=torch.uint8, device="xpu")
    blocks = to_blocked(torch.ones(3, 2, device="xpu").to(torch.float8_e4m3fn), flatten=False)
    sa = torch.tensor([2.0], device="xpu")
    sb = torch.tensor([3.0], device="xpu")
    alpha = torch.tensor([0.25], device="xpu")
    bias = torch.arange(3, device="xpu").float()
    with ck.use_backend("eager"):
        actual = ck.scaled_mm_nvfp4(
            q, q, sa, sb, blocks, blocks, bias=bias, alpha=alpha, out_dtype=torch.float32
        )
    torch.testing.assert_close(actual, torch.full((3, 3), 8.0, device="xpu") + bias, rtol=0, atol=0)


@pytest.mark.parametrize("group", [16, 32])
@pytest.mark.parametrize("correction", [False, True])
def test_w4a8_packed_and_linear_cpu_reference(group, correction):
    torch.manual_seed(120)
    w = torch.randn(8, 256)
    x = torch.randn(2, 3, 256)
    bias = torch.randn(8)
    kwargs = {
        "group_size": group,
        "convrot_groupsize": 256,
        "scale_dtype": torch.float32,
        "codebook": False,
        "symmetric": not correction,
    }
    with ck.use_backend("eager"):
        packed = ck.quantize_w4a8_int8_weight(w.to("xpu"), **kwargs)
        operands = (*packed[:3], packed[4], packed[3])
        decoded = ck.dequantize_w4a8_int8_weight(
            *operands, group_size=group, output_dtype=torch.float32
        )
        actual = ck.w4a8_int8_linear(
            x.to("xpu"), *operands, bias=bias.to("xpu"), group_size=group, out_dtype=torch.float32
        )
    cpu_packed = tuple(_cpu(v) for v in operands)
    expected_decoded = w4_reference.dequantize_w4a8_int8_weight(
        *cpu_packed, group_size=group, output_dtype=torch.float32
    )
    expected = w4_reference.w4a8_int8_linear(
        x, *cpu_packed, bias=bias, group_size=group, out_dtype=torch.float32
    )
    torch.testing.assert_close(decoded.cpu(), expected_decoded, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    assert packed[0].dtype == torch.int8 and packed[0].shape == (8, 128)


@pytest.mark.parametrize("shape", [(64,), (1, 64), (2, 3, 64)])
def test_awq_independent_uint4_oracle(shape):
    torch.manual_seed(121)
    x = torch.randn(shape)
    codes = torch.arange(5 * 64).reshape(5, 64) % 16
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.int8)
    scales = torch.rand(2, 5) + 0.1
    zeros = torch.randn(2, 5)
    bias = torch.randn(5)
    weight = (
        (codes.float().reshape(5, 2, 32) - 8) * scales.T.unsqueeze(-1) + zeros.T.unsqueeze(-1)
    ).reshape(5, 64)
    with ck.use_backend("eager"):
        out = ck.gemv_awq_w4a16(
            x.to("xpu"),
            packed.to("xpu"),
            scales.to("xpu"),
            zeros.to("xpu"),
            bias.to("xpu"),
            group_size=32,
        )
    torch.testing.assert_close(out.cpu(), x @ weight.T + bias, rtol=1e-5, atol=1e-5)


def test_reference_device_guard_and_shape_error():
    q = torch.zeros(2, 32, dtype=torch.float8_e4m3fn, device="xpu")
    scales = torch.zeros(128, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="same device"):
        reference_mm.scaled_mm_mxfp8_reference(q, q, scales, scales)
    with pytest.raises(ValueError, match="matching K"):
        reference_mm.scaled_mm_mxfp8_reference(q, q[:, :16], q, q)


@pytest.mark.parametrize("causal", [(False, False, False), (True, False, True)])
def test_na3d_existing_xpu_eager_matches_independent_window(causal):
    from .test_na import _ref_na3d

    torch.manual_seed(122)
    q, k, v = [torch.randn(1, 2, 3, 4, 1, 16, device="xpu") for _ in range(3)]
    with ck.use_backend("eager"):
        actual = ck.na3d(q, k, v, [3, 3, 3], list(causal))
    expected = _ref_na3d(q.cpu(), k.cpu(), v.cpu(), (3, 3, 3), causal, None)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)


def test_na2d_existing_xpu_eager_single_frame():
    torch.manual_seed(123)
    q, k, v = [torch.randn(1, 3, 4, 1, 16, device="xpu") for _ in range(3)]
    with ck.use_backend("eager"):
        actual = ck.na2d(q, k, v, [3, 3])
        expected = ck.na3d(q[:, None], k[:, None], v[:, None], [1, 3, 3])[:, 0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["TensorCoreNVFP4Layout", "TensorCoreMXFP8Layout"])
@pytest.mark.parametrize("operation", ["linear", "mm", "addmm"])
def test_quantized_tensor_public_layout(layout, operation):
    from comfy_kitchen.tensor import QuantizedTensor

    torch.manual_seed(124)
    with ck.use_backend("eager"):
        a = QuantizedTensor.from_float(torch.randn(5, 64, device="xpu"), layout)
        b = QuantizedTensor.from_float(torch.randn(32, 64, device="xpu"), layout)
        bias = torch.randn(32, device="xpu")
        if operation == "linear":
            actual = torch.nn.functional.linear(a, b, bias)
        elif operation == "mm":
            actual = torch.mm(a, b.t())
        else:
            actual = torch.addmm(bias, a, b.t())
        da = a.dequantize().float()
        db = b.dequantize().float()
    expected = da @ db.T
    if operation != "mm":
        expected += bias
    torch.testing.assert_close(actual.float(), expected, rtol=1e-4, atol=1e-4)
    assert actual.shape == (5, 32)


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_portable_compile_and_nondefault_stream(format):
    from comfy_kitchen.float_utils import to_blocked

    # Compile only the public already-quantized GEMM, including portable decode.
    if format == "nvfp4":
        q = torch.full((3, 16), 0x22, dtype=torch.uint8, device="xpu")
        scales = to_blocked(torch.ones(3, 2, device="xpu").to(torch.float8_e4m3fn), flatten=False)
        scalar = torch.ones(1, device="xpu")

        def fn(a, b):
            return ck.scaled_mm_nvfp4(a, b, scalar, scalar, scales, scales, out_dtype=torch.float32)
    else:
        q = torch.ones(3, 32, device="xpu").to(torch.float8_e4m3fn)
        scales = to_blocked(
            torch.full((3, 1), 127, dtype=torch.uint8, device="xpu"), flatten=False
        ).view(torch.float8_e8m0fnu)

        def fn(a, b):
            return ck.scaled_mm_mxfp8(a, b, scales, scales, out_dtype=torch.float32)

    torch.xpu.synchronize()
    stream = torch.xpu.Stream()
    with ck.use_backend("eager"), torch.xpu.stream(stream):
        out = torch.compile(fn, backend="eager", fullgraph=True)(q, q)
    stream.synchronize()
    torch.testing.assert_close(out, torch.full((3, 3), 32.0, device="xpu"), rtol=0, atol=0)


@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e4m3fn])
def test_w4a8_codebook_and_seeded_quantization(scale_dtype):
    torch.manual_seed(125)
    weight = torch.randn(8, 256, device="xpu")
    table = torch.linspace(-7, 7, 16, device="xpu")
    with ck.use_backend("eager"):
        first = ck.quantize_w4a8_int8_weight(
            weight, scale_dtype=scale_dtype, codebook_tensor=table, stochastic_rounding=13
        )
        second = ck.quantize_w4a8_int8_weight(
            weight, scale_dtype=scale_dtype, codebook_tensor=table, stochastic_rounding=13
        )
        for a, b in zip(first, second, strict=False):
            if a is not None:
                torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)
        decoded = ck.dequantize_w4a8_int8_weight(
            *first[:3], codebook=first[4], correction=first[3], output_dtype=torch.float32
        )
    expected = w4_reference.dequantize_w4a8_int8_weight(
        *[_cpu(x) for x in first[:3]],
        codebook=_cpu(first[4]),
        correction=_cpu(first[3]),
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(decoded.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("layout", ["TensorCoreNVFP4Layout", "TensorCoreMXFP8Layout"])
def test_quantized_tensor_unaligned_bias_contract(layout):
    from comfy_kitchen.tensor import QuantizedTensor

    torch.manual_seed(126)
    with ck.use_backend("eager"):
        a = QuantizedTensor.from_float(torch.randn(5, 65, device="xpu"), layout)
        b = QuantizedTensor.from_float(torch.randn(7, 65, device="xpu"), layout)
        bias = torch.randn(7, device="xpu")
        actual = torch.nn.functional.linear(a, b, bias)
    torch.testing.assert_close(
        actual, a.dequantize() @ b.dequantize().T + bias, rtol=1e-4, atol=1e-4
    )
