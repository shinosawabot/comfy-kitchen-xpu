"""Host-only native policy tests. Fake CUDA tensors do not execute GPU kernels."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from comfy_kitchen.backends import torch as native
from comfy_kitchen.exceptions import NoCapableBackendError
from comfy_kitchen.registry import registry


def case(packed=True, n=32):
    mode = FakeTensorMode()

    def tensor(shape, dtype):
        return FakeTensor(
            mode, torch.empty(shape, device="meta", dtype=dtype), torch.device("cuda", 2)
        )

    args = {
        "a": tensor((16, 32 if packed else 64), torch.uint8 if packed else torch.float8_e4m3fn),
        "b": tensor((n, 32 if packed else 64), torch.uint8 if packed else torch.float8_e4m3fn),
        "block_scale_a": tensor((128, 4), torch.float8_e4m3fn if packed else torch.float8_e8m0fnu),
        "block_scale_b": tensor((128, 4), torch.float8_e4m3fn if packed else torch.float8_e8m0fnu),
        "out_dtype": torch.bfloat16,
    }
    if packed:
        args.update(
            tensor_scale_a=tensor((1,), torch.float32), tensor_scale_b=tensor((1,), torch.float32)
        )
    return args


@pytest.fixture
def eligible(monkeypatch):
    monkeypatch.setattr(native, "_cuda_format_device", lambda device: True)
    monkeypatch.setattr(native, "_api_available", lambda: True)


@pytest.mark.parametrize("packed", [False, True])
def test_default_and_explicit_selection(eligible, packed):
    op = "scaled_mm_nvfp4" if packed else "scaled_mm_mxfp8"
    args = case(packed)
    assert registry.get_capable_backend(op, args) == "torch"
    with registry.use_backend("eager"):
        assert registry.get_capable_backend(op, args) == "eager"
    assert registry.get_implementation(op, backend="torch", kwargs=args) is getattr(native, op)


@pytest.mark.parametrize(
    "reason", ["api", "device", "n", "dtype", "scale_size", "scale_stride", "bias"]
)
def test_ineligible_native_selects_reference(monkeypatch, eligible, reason):
    args = case()
    op = "scaled_mm_nvfp4"
    if reason == "api":
        monkeypatch.setattr(native, "_api_available", lambda: False)
    elif reason == "device":
        monkeypatch.setattr(native, "_cuda_format_device", lambda device: False)
    elif reason == "n":
        args = case(n=7)
    elif reason == "dtype":
        args["a"] = args["a"].to(torch.bfloat16)
    elif reason == "scale_size":
        args["block_scale_a"] = FakeTensor(
            args["a"].fake_mode,
            torch.empty((1, 4), device="meta", dtype=torch.float8_e4m3fn),
            args["a"].device,
        )
    elif reason == "scale_stride":
        args["block_scale_a"] = FakeTensor(
            args["a"].fake_mode,
            torch.empty_strided((128, 4), (1, 128), device="meta", dtype=torch.float8_e4m3fn),
            args["a"].device,
        )
    else:
        args["bias"] = FakeTensor(
            args["a"].fake_mode,
            torch.empty((32,), device="meta", dtype=torch.float32),
            args["a"].device,
        )
    if reason == "dtype":
        with pytest.raises(NoCapableBackendError):
            registry.get_capable_backend(op, args)
    else:
        assert registry.get_capable_backend(op, args) == "eager"
    with pytest.raises(NoCapableBackendError):
        registry.get_implementation(op, backend="torch", kwargs=args)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("out of memory"),
        RuntimeError("kernel fault"),
        ValueError("XPU does not support swizzle yet."),
        NotImplementedError("unsupported"),
    ],
)
def test_selected_native_errors_propagate(monkeypatch, eligible, error):
    args = case()

    def fail(*a, **k):
        raise error

    monkeypatch.setattr(native, "_scaled_mm_v2_torch", fail)
    implementation = registry.get_implementation("scaled_mm_nvfp4", kwargs=args)
    with pytest.raises(type(error), match=str(error)):
        implementation(**args)


@pytest.mark.parametrize("packed", [False, True])
def test_selected_native_calls_optimized_torch_once(monkeypatch, eligible, packed):
    args = case(packed)
    calls = []
    sentinel = object()

    def run(*a, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(native, "_scaled_mm_v2_torch", run)
    op = "scaled_mm_nvfp4" if packed else "scaled_mm_mxfp8"
    assert registry.get_implementation(op, kwargs=args)(**args) is sentinel
    assert len(calls) == 1


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("cc,allowed", [((9, 0), False), ((10, 0), True), ((12, 0), True)])
def test_cuda_contract_uses_input_device_and_api(monkeypatch, v2, cc, allowed):
    seen = []
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(native, "has_scaled_mm_v2", lambda: v2)
    monkeypatch.setattr(
        torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        lambda name, key: seen.append((name, key)) or True,
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: seen.append(device.index) or cc
    )
    assert native._cuda_format_device(torch.device("cuda", 2)) is allowed
    assert seen == [("aten::_scaled_mm_v2" if v2 else "aten::_scaled_mm", "CUDA"), 2]


@pytest.mark.parametrize("family", ["cpu", "xpu"])
def test_swizzled_layout_not_admitted_on_unsupported_device(family):
    assert not native._cuda_format_device(torch.device(family))


def test_disabled_torch_backend_selects_eager(eligible):
    registry.disable("torch")
    assert registry.get_capable_backend("scaled_mm_nvfp4", case()) == "eager"


@pytest.mark.parametrize("build,kernel", [("12.7", True), ("12.8", False)])
def test_cuda_build_or_kernel_unavailable(monkeypatch, build, kernel):
    monkeypatch.setattr(torch.version, "cuda", build)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda *args: kernel)

    def forbidden(*args):
        raise AssertionError("hardware queried before API/build eligibility")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    assert not native._cuda_format_device(torch.device("cuda", 3))


def test_native_flash_availability_keeps_cuda_and_rejects_xpu(monkeypatch):
    from types import SimpleNamespace

    from comfy_kitchen import flash_attention

    monkeypatch.setattr(
        flash_attention,
        "_cuda_backend",
        SimpleNamespace(_EXT_AVAILABLE=True, _C=SimpleNamespace(flash_attention_decode=object())),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    assert flash_attention.is_available(torch.device("cuda", 3))
    assert not flash_attention.is_available(torch.device("xpu", 1))


@pytest.mark.parametrize("alpha_value", [None, 0.0, -0.25, 2.0])
def test_native_adapter_preserves_alpha_and_bias_contract(monkeypatch, alpha_value):
    # CPU tensors test adapter arguments only; no CUDA kernel is executed.
    captured = []
    result = object()

    def optimized(*args, **kwargs):
        captured.append(kwargs)
        return result

    monkeypatch.setattr(native, "_scaled_mm_v2_torch", optimized)
    data = torch.zeros((16, 32), dtype=torch.uint8)
    scales = torch.ones((128, 4)).to(torch.float8_e4m3fn)
    bias = torch.ones(16, dtype=torch.bfloat16)
    alpha = None if alpha_value is None else torch.tensor(alpha_value, dtype=torch.float16)
    assert (
        native.scaled_mm_nvfp4(
            data,
            data,
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            scales,
            scales,
            bias,
            torch.bfloat16,
            alpha,
        )
        is result
    )
    call = captured[0]
    expected = 6.0 if alpha_value is None else alpha_value
    torch.testing.assert_close(
        call["scale_a"][1] * call["scale_b"][1], torch.tensor([expected]), rtol=0, atol=0
    )
    assert call["bias"] is bias and call["out_dtype"] == torch.bfloat16
