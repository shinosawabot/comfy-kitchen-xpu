"""Host reference oracles and preservation of supported Torch scaled-mm calls."""

import pytest
import torch

import comfy_kitchen as ck
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
def test_cpu_actual_eager_route_known_codes(format):
    with ck.use_backend("eager"):
        actual = getattr(ck, "scaled_mm_" + format)(*operands(format), out_dtype=torch.float32)
    torch.testing.assert_close(
        actual, torch.full((3, 3), 192.0 if format == "nvfp4" else 32.0), rtol=0, atol=0
    )


def test_eager_never_calls_torch_native(monkeypatch):
    from comfy_kitchen.backends import torch as native
    def forbidden(*args, **kwargs):
        raise AssertionError("eager attempted native dispatch")
    monkeypatch.setattr(native, "_scaled_mm_v2_torch", forbidden)
    with ck.use_backend("eager"):
        output=ck.scaled_mm_nvfp4(*operands("nvfp4"), alpha=torch.tensor([0.25]),
                                  bias=torch.arange(3).float(), out_dtype=torch.float32)
    torch.testing.assert_close(output, torch.full((3,3),8.)+torch.arange(3), rtol=0,atol=0)
