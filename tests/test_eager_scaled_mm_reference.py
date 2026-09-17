"""Host reference oracles and preservation of supported Torch scaled-mm calls."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as eager
from comfy_kitchen.backends.eager import scaled_mm_reference as reference
from comfy_kitchen.float_utils import to_blocked


def operands(format):
    if format == "nvfp4":
        q = torch.full((3, 16), 0x22, dtype=torch.uint8)
        scales = to_blocked(torch.ones(3, 2).to(torch.float8_e4m3fn), flatten=False)
        return (q, q, torch.tensor([2.0]), torch.tensor([3.0]), scales, scales)
    q = torch.ones(3, 32).to(torch.float8_e4m3fn)
    scales = to_blocked(torch.full((3, 1), 127, dtype=torch.uint8), flatten=False).view(
        torch.float8_e8m0fnu
    )
    return (q, q, scales, scales)


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_cpu_public_eager_fallback_known_codes(monkeypatch, format):
    # Simulate an absent Torch kernel; the numerical oracle is independent of decode.
    def missing(*args, **kwargs):
        raise NotImplementedError("no kernel for this backend")

    monkeypatch.setattr(eager, "scaled_mm_v2", missing)
    bias = torch.arange(3).float()
    kwargs = {"bias": bias, "out_dtype": torch.float32}
    if format == "nvfp4":
        kwargs["alpha"] = torch.tensor(0.25, dtype=torch.float16)
    with ck.use_backend("eager"):
        actual = getattr(ck, "scaled_mm_" + format)(*operands(format), **kwargs)
    expected = torch.full((3, 3), 8.0 if format == "nvfp4" else 32.0) + bias
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_supported_torch_route_is_retained(monkeypatch, format):
    seen = []
    sentinel = torch.ones(3, 3)

    def native(*args, **kwargs):
        seen.append((args, kwargs))
        return sentinel

    def forbidden(*args, **kwargs):
        raise AssertionError("supported native route replaced")

    monkeypatch.setattr(eager, "scaled_mm_v2", native)
    monkeypatch.setattr(eager, "scaled_mm_" + format + "_reference", forbidden)
    kwargs = {"out_dtype": torch.float32}
    if format == "nvfp4":
        kwargs["alpha"] = torch.tensor(0.25, dtype=torch.float16)
    actual = getattr(eager, "scaled_mm_" + format)(*operands(format), **kwargs)
    assert actual is sentinel and len(seen) == 1
    if format == "nvfp4":
        torch.testing.assert_close(
            seen[0][1]["scale_a"][1], kwargs["alpha"].float().reshape(1), rtol=0, atol=0
        )
        torch.testing.assert_close(seen[0][1]["scale_b"][1], torch.ones(1), rtol=0, atol=0)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("out of memory"),
        RuntimeError("invalid shape"),
        ValueError("invalid scale layout"),
    ],
)
@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_unrelated_native_error_is_not_hidden(monkeypatch, error, format):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(eager, "scaled_mm_v2", fail)
    with pytest.raises(type(error), match=str(error)):
        getattr(eager, "scaled_mm_" + format)(*operands(format), out_dtype=torch.float32)


def test_known_xpu_swizzle_rejection_is_narrow():
    error = ValueError("XPU does not support swizzle yet.")
    assert reference.scaled_mm_is_unsupported(error, "xpu")
    assert not reference.scaled_mm_is_unsupported(error, "cuda")
    assert not reference.scaled_mm_is_unsupported(ValueError("different error"), "xpu")


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_cpu_actual_eager_route_known_codes(format):
    with ck.use_backend("eager"):
        actual = getattr(ck, "scaled_mm_" + format)(*operands(format), out_dtype=torch.float32)
    torch.testing.assert_close(
        actual, torch.full((3, 3), 192.0 if format == "nvfp4" else 32.0), rtol=0, atol=0
    )


def test_known_cpu_swizzle_rejection_is_narrow():
    error = ValueError("CPU does not support swizzle.")
    assert reference.scaled_mm_is_unsupported(error, "cpu")
    assert not reference.scaled_mm_is_unsupported(error, "cuda")
