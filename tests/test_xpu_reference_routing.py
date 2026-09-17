"""Reference fallback policy, not simulated BMG/PTL kernel execution."""

from dataclasses import replace

import pytest
import torch
from omni_xpu_kernel import device as omni_device

import comfy_kitchen as ck
from comfy_kitchen.backends import xpu
from comfy_kitchen.constraints import ExactDims, ParamConstraint
from comfy_kitchen.registry import registry

pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not torch.xpu.is_available() or not ck.list_backends().get("xpu", {}).get("available"),
        reason="XPU backend required",
    ),
]

OP = "w4a8_int8_linear"


@pytest.fixture
def operands():
    return {
        "x": torch.zeros(2, 256, device="xpu"),
        "qdata": torch.zeros(8, 128, device="xpu", dtype=torch.int8),
        "s_rel": torch.ones(8, 16, device="xpu"),
        "s_channel": torch.ones(8, device="xpu"),
        "group_size": 16,
        "convrot_groupsize": 256,
        "out_dtype": torch.float32,
    }


@pytest.fixture
def eligible_triton(monkeypatch):
    constraints = registry.get_constraints("triton", OP)
    assert constraints is not None, "test runtime must package the existing Triton W4A8 boundary"
    # Simulate an eligible Triton policy without claiming its CUDA capability
    # predicate passed on A770, changing device identity, or launching its kernel.
    monkeypatch.setitem(
        registry._constraints, ("triton", OP), replace(constraints, min_compute_capability=None)
    )
    monkeypatch.setattr(registry, "_priority", ["xpu", "triton", "eager"])
    monkeypatch.setattr(registry, "_disabled", registry._disabled - {"triton"})


@pytest.mark.parametrize("target", ["bmg", "ptl-h"])
def test_existing_eligible_triton_default_is_preserved(
    monkeypatch, operands, eligible_triton, target
):
    indices = []

    def info(index):
        indices.append(index)
        return {"physical_build_target": target}

    monkeypatch.setattr(omni_device, "info", info)
    # A different current device must not override the actual input ordinal.
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 99)
    with monkeypatch.context() as before:
        before.setitem(registry._capabilities, "xpu", registry._capabilities["xpu"] - {OP})
        old_selection = registry.get_capable_backend(OP, operands)
    assert old_selection == "triton"
    assert registry.get_capable_backend(OP, operands) == old_selection
    assert indices == [operands["x"].device.index]
    with ck.use_backend("xpu"):
        assert registry.get_capable_backend(OP, operands) == "xpu"


@pytest.mark.parametrize("state", ["disabled", "missing", "rejects_shape", "not_in_priority"])
def test_reference_remains_eligible_when_triton_cannot_handle_call(
    monkeypatch, operands, eligible_triton, state
):
    monkeypatch.setattr(omni_device, "info", lambda index: {"physical_build_target": "bmg"})
    if state == "disabled":
        registry.disable("triton")
    elif state == "missing":
        monkeypatch.delitem(registry._backends, "triton")
    elif state == "not_in_priority":
        monkeypatch.setattr(registry, "_priority", ["xpu", "eager"])
    else:
        constraints = registry.get_constraints("triton", OP)
        params = dict(constraints.params)
        params["x"] = ParamConstraint(shape_rules=(ExactDims(4),))
        monkeypatch.setitem(
            registry._constraints, ("triton", OP), replace(constraints, params=params)
        )
    assert registry.get_capable_backend(OP, operands) == "xpu"


def test_actual_dg2_default_executes_reference(monkeypatch, eligible_triton):
    assert omni_device.info(0)["physical_build_target"] == "dg2"
    torch.manual_seed(127)
    with ck.use_backend("xpu"):
        packed = ck.quantize_w4a8_int8_weight(
            torch.randn(8, 256, device="xpu"), scale_dtype=torch.float32, codebook=False
        )
    values = torch.randn(2, 256, device="xpu")
    kwargs = {
        "x": values,
        "qdata": packed[0],
        "s_rel": packed[1],
        "s_channel": packed[2],
        "correction": packed[3],
        "codebook": packed[4],
        "out_dtype": torch.float32,
    }
    calls = []
    original = xpu.w4a8_int8_linear

    def observed(**arguments):
        calls.append(arguments["x"].device)
        return original(**arguments)

    monkeypatch.setattr(xpu, OP, observed)
    actual = ck.w4a8_int8_linear(**kwargs)
    assert calls == [values.device]
    with ck.use_backend("eager"):
        expected = ck.w4a8_int8_linear(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
