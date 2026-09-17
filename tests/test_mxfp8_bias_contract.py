"""Logical Torch bias semantics precede packed MXFP8 column padding."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.mxfp8 import _handle_mxfp8_linear

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


def make_case(n):
    torch.manual_seed(141)
    k = 65 if n == 7 else 64
    with ck.use_backend("eager"):
        a = QuantizedTensor.from_float(torch.randn(3, k, device="xpu"), "TensorCoreMXFP8Layout")
        b = QuantizedTensor.from_float(torch.randn(n, k, device="xpu"), "TensorCoreMXFP8Layout")
        da = a.dequantize().cpu()
        db = b.dequantize().cpu()
    return a, b, da, db


def bias_case(kind, n):
    shapes = {
        "scalar": (),
        "singleton": (1,),
        "full_n": (n,),
        "row": (1, n),
        "full_matrix": (3, n),
        "column": (3, 1),
        "invalid_short": (2,),
        "invalid_rows": (4, n),
        "invalid_rank": (1, 3, n),
    }
    return torch.full(shapes[kind], 2.0, device="xpu")


@pytest.mark.parametrize("n", [7, 32])
@pytest.mark.parametrize("operation", ["linear", "linear_handler", "addmm"])
@pytest.mark.parametrize(
    "kind",
    [
        "scalar",
        "singleton",
        "full_n",
        "row",
        "full_matrix",
        "column",
        "invalid_short",
        "invalid_rows",
        "invalid_rank",
    ],
)
def test_logical_bias_matches_dequantized_torch(n, operation, kind):
    a, b, da, db = make_case(n)
    bias = bias_case(kind, n)

    def reference():
        return (
            torch.addmm(bias.cpu(), da, db.T)
            if operation == "addmm"
            else torch.nn.functional.linear(da, db, bias.cpu())
        )

    def call():
        with ck.use_backend("eager"):
            if operation == "addmm":
                return torch.addmm(bias, a, b.t())
            if operation == "linear_handler":
                return _handle_mxfp8_linear(a, (a, b, bias), {})
            return torch.nn.functional.linear(a, b, bias)

    try:
        expected = reference()
    except RuntimeError:
        with pytest.raises(RuntimeError):
            call()
    else:
        actual = call()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
        assert actual.shape == expected.shape


@pytest.mark.parametrize("n", [7, 32])
@pytest.mark.parametrize("kind", ["scalar", "singleton", "full_n", "row", "full_matrix"])
@pytest.mark.parametrize("alpha,beta", [(0.5, 2.0), (0.0, 0.0), (-1.0, 0.0)])
def test_addmm_alpha_beta_semantics(n, kind, alpha, beta):
    a, b, da, db = make_case(n)
    bias = bias_case(kind, n)
    if beta == 0:
        bias.fill_(float("nan"))
    expected = torch.addmm(bias.cpu(), da, db.T, alpha=alpha, beta=beta)
    with ck.use_backend("eager"):
        actual = torch.addmm(bias, a, b.t(), alpha=alpha, beta=beta)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


def test_only_full_logical_column_bias_reaches_packed_epilogue(monkeypatch):
    a, b, da, db = make_case(7)
    seen = []
    original = ck.scaled_mm_mxfp8

    def observe(*args, **kwargs):
        seen.append(kwargs["bias"].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(ck, "scaled_mm_mxfp8", observe)
    with ck.use_backend("eager"):
        torch.addmm(torch.full((7,), 2.0, device="xpu"), a, b.t())
    assert len(seen) == 1 and seen[0].shape == (32,)
    torch.testing.assert_close(seen[0][:7], torch.full((7,), 2.0, device="xpu"))
    torch.testing.assert_close(seen[0][7:], torch.zeros(25, device="xpu"))
    seen.clear()
    with ck.use_backend("eager"):
        actual = torch.addmm(torch.tensor([2.0], device="xpu"), a, b.t())
    assert not seen
    torch.testing.assert_close(actual.cpu(), da @ db.T + 2, rtol=1e-4, atol=1e-4)
