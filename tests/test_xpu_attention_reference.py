"""Portable attention API coverage; no native DG2 attention claim."""

import pytest
import torch

import comfy_kitchen as ck

pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required"),
]


def test_native_flash_decode_is_unavailable_on_xpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("flash decode called ordinary SDPA")
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
    assert not ck.flash_attention_decode_is_available("xpu")
    q=torch.zeros(1,1,2,128,device="xpu",dtype=torch.bfloat16)
    assert not ck.flash_attention_decode_is_available(q.device)
    k=torch.zeros(1,8,2,128,device=q.device,dtype=q.dtype)
    with pytest.raises(NotImplementedError, match="Native flash_attention_decode"):
        ck.flash_attention_decode(q,k,k,torch.tensor([8],device=q.device,dtype=torch.int32))


@pytest.mark.parametrize("causal_controls", [False, True])
def test_sol_reference_matches_cpu_existing_semantics(causal_controls):
    from comfy_kitchen.backends.eager.sol_attn import sol_attn

    torch.manual_seed(131)
    q, k, v = [torch.randn(1, 129, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    kwargs = {
        "tau": 1.0,
        "scale": None,
        "sink_blocks": [0, 3],
        "sink_q": [0, 0],
        "topk_ratio": 0.0,
        "tail": True,
        "token_aug": 0,
    }
    if causal_controls:
        kwargs.update(
            key_bias=torch.linspace(-1, 1, 129),
            block_len=torch.tensor([64, 63, 1], dtype=torch.int32),
            coarse_gate=torch.ones_like(q) * 0.2,
        )
    expected = sol_attn(q, k, v, **kwargs)
    device_kwargs = {
        name: value.to("xpu") if isinstance(value, torch.Tensor) else value
        for name, value in kwargs.items()
    }
    with ck.use_backend("eager"):
        actual = ck.sol_attn(q.to("xpu"), k.to("xpu"), v.to("xpu"), **device_kwargs)
    # Tail block dead rows have unspecified output, compare only live queries.
    live = torch.ones(129, dtype=torch.bool)
    if causal_controls:
        live[127] = False
    torch.testing.assert_close(
        actual[:, live].float().cpu(), expected[:, live].float(), rtol=0.02, atol=0.02
    )


def test_unimplemented_native_attention_contracts_are_explicit():
    q = torch.zeros(1, 2, 3, 128, device="xpu", dtype=torch.bfloat16)
    assert not ck.int8_attention_is_available(q.device)
    for operation in (ck.prequantize_int8_attention, ck.int8_attention):
        with pytest.raises(NotImplementedError, match="packed INT8 attention"):
            operation(q, q, q)
    if not ck.sol_attn_is_available(q.device):
        bthd = q.transpose(1, 2)
        with pytest.raises(NotImplementedError, match="token_aug"):
            ck.sol_attn(bthd, bthd, bthd, token_aug=64)
        with pytest.raises(NotImplementedError, match="native XPU Sol sidecar"):
            ck.sol_attn_chunked(
                [], 3, 2, torch.ones(1, device="xpu"), (torch.ones(128, device="xpu"),) * 2
            )


def test_prequantized_xpu_consumer_rejects_packed_object_explicitly():
    packed = torch.zeros(1, 1, 1, 128, device="xpu", dtype=torch.int8)
    scales = torch.ones(1, device="xpu")
    item = ck.PrequantizedInt8Attention(
        packed, packed, packed, scales, scales, scales, 128, torch.bfloat16, 128**-0.5, 64, None
    )
    with pytest.raises(NotImplementedError, match="consumption is not implemented"):
        ck.int8_attention_from_prequantized(item)
