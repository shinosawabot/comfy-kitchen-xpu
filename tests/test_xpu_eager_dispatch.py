"""Public API availability is distinct from a native XPU registry entry."""

from dataclasses import replace

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.exceptions import BackendNotImplementedError
from comfy_kitchen.registry import registry

OPS = (
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
)
pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


def test_reference_ops_are_not_duplicate_xpu_entries():
    assert set(OPS) <= registry._capabilities["eager"]
    assert not set(OPS) & registry._capabilities["xpu"]
    for op in OPS:
        with pytest.raises(BackendNotImplementedError):
            registry.get_implementation(op, backend="xpu")


@pytest.mark.parametrize("enabled", [False, True])
def test_existing_triton_selection_preserved(monkeypatch, enabled):
    op = "w4a8_int8_linear"
    constraints = registry.get_constraints("triton", op)
    monkeypatch.setitem(
        registry._constraints, ("triton", op), replace(constraints, min_compute_capability=None)
    )
    monkeypatch.setattr(registry, "_priority", ["xpu", "triton", "eager"])
    if not enabled:
        registry.disable("triton")
    kwargs = {
        "x": torch.zeros(2, 256, device="xpu"),
        "qdata": torch.zeros(8, 128, device="xpu", dtype=torch.int8),
        "s_rel": torch.ones(8, 16, device="xpu"),
        "s_channel": torch.ones(8, device="xpu"),
        "out_dtype": torch.float32,
    }
    expected = "triton" if enabled else "eager"
    assert registry.get_capable_backend(op, kwargs) == expected
    # use_backend is a preferred backend context; unsupported ops still fall through.
    with ck.use_backend("xpu"):
        assert registry.get_capable_backend(op, kwargs) == expected
    with ck.use_backend("eager"):
        assert registry.get_capable_backend(op, kwargs) == "eager"


def _require_dg2(selected_device):
    from omni_xpu_kernel import device

    if device.info(selected_device.index).get("physical_build_target") != "dg2":
        pytest.skip("DG2-specific default eager route")


def test_actual_a770_default_calls_eager(monkeypatch):
    from comfy_kitchen.backends import eager

    selected_device = torch.device("xpu", torch.xpu.current_device())
    _require_dg2(selected_device)
    torch.manual_seed(127)
    packed = ck.quantize_w4a8_int8_weight(
        torch.randn(8, 256, device=selected_device), scale_dtype=torch.float32, codebook=False
    )
    kwargs = {
        "x": torch.randn(2, 256, device=selected_device),
        "qdata": packed[0],
        "s_rel": packed[1],
        "s_channel": packed[2],
        "correction": packed[3],
        "codebook": packed[4],
        "out_dtype": torch.float32,
    }
    calls = []
    original = eager.w4a8_int8_linear

    def observed(**args):
        calls.append(args["x"].device)
        return original(**args)

    monkeypatch.setattr(eager, "w4a8_int8_linear", observed)
    actual = ck.w4a8_int8_linear(**kwargs)
    assert calls == [kwargs["x"].device]
    expected = original(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
