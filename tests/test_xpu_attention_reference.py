"""Portable attention API coverage; no native DG2 attention claim."""

import pytest
import torch

import comfy_kitchen as ck

pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required"),
]


@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("lengths", [(1, 17), (0, 32), (32, 32)])
def test_flash_decode_reference_gqa_and_lengths(groups, lengths):
    torch.manual_seed(130)
    q = torch.randn(2, 1, 2 * groups, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(2, 32, 2, 128, device="xpu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lens = torch.tensor(lengths, device="xpu", dtype=torch.int32)
    actual = ck.flash_attention_decode(q, k, v, lens)
    expected = []
    for b, length in enumerate(lengths):
        if length == 0:
            expected.append(torch.zeros_like(q[b]).float().cpu())
            continue
        query = q[b].float().cpu().transpose(0, 1)
        key = k[b, :length].float().cpu().transpose(0, 1).repeat_interleave(groups, dim=0)
        value = v[b, :length].float().cpu().transpose(0, 1).repeat_interleave(groups, dim=0)
        probabilities = torch.softmax(query @ key.transpose(-1, -2) * 128**-0.5, dim=-1)
        expected.append((probabilities @ value).transpose(0, 1))
    torch.testing.assert_close(actual.float().cpu(), torch.stack(expected), rtol=0.02, atol=0.01)
    assert actual.shape == q.shape and actual.dtype == q.dtype and actual.is_contiguous()
    assert ck.flash_attention_decode_is_available(q.device)


@pytest.mark.parametrize("bad", ["host_lengths", "length_shape", "dtype", "heads"])
def test_flash_decode_reference_rejects_invalid_contract(bad):
    q = torch.zeros(1, 1, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(1, 8, 2, 128, device="xpu", dtype=torch.bfloat16)
    v = k.clone()
    lens = torch.tensor([8], device="xpu", dtype=torch.int32)
    if bad == "host_lengths":
        lens = lens.cpu()
    elif bad == "length_shape":
        lens = lens[:, None]
    elif bad == "dtype":
        q = q.float()
    elif bad == "heads":
        q = q[:, :, :3]
    with pytest.raises((ValueError, TypeError)):
        ck.flash_attention_decode(q, k, v, lens)


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
