"""Host policy checks; no BMG/PTL tensor allocation or kernel execution."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from . import test_xpu_eager_dispatch as dispatch_tests


@pytest.mark.parametrize("target", ["bmg", "ptl-h"])
def test_non_dg2_skips_only_dg2_case(monkeypatch, target):
    indices = []

    def info(index):
        indices.append(index)
        return {"physical_build_target": target}

    fake = ModuleType("omni_xpu_kernel")
    fake.device = SimpleNamespace(info=info)
    monkeypatch.setitem(sys.modules, "omni_xpu_kernel", fake)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 7)

    def no_allocation(*args, **kwargs):
        raise AssertionError("non-DG2 test allocated tensors")

    monkeypatch.setattr(torch, "randn", no_allocation)
    with pytest.raises(pytest.skip.Exception, match="DG2-specific"):
        dispatch_tests.test_actual_a770_default_calls_eager(monkeypatch)
    assert indices == [7]
    # Generic registry coverage executes without the DG2 guard or any device work.
    registry = dispatch_tests.registry
    monkeypatch.setattr(registry, "_capabilities", {"eager": set(dispatch_tests.OPS), "xpu": set()})
    monkeypatch.setattr(
        registry, "_backends", {"eager": SimpleNamespace(), "xpu": SimpleNamespace()}
    )
    monkeypatch.setattr(registry, "_disabled", set())
    dispatch_tests.test_reference_ops_are_not_duplicate_xpu_entries()
    assert indices == [7]


def test_dg2_guard_uses_explicit_selected_device(monkeypatch):
    indices = []

    def info(index):
        indices.append(index)
        return {"physical_build_target": "dg2"}

    fake = ModuleType("omni_xpu_kernel")
    fake.device = SimpleNamespace(info=info)
    monkeypatch.setitem(sys.modules, "omni_xpu_kernel", fake)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 99)
    dispatch_tests._require_dg2(torch.device("xpu", 3))
    assert indices == [3]
