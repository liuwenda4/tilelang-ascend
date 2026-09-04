import pytest
import torch

from dispatch_quant_int8 import quantize_dispatch_payload
from moe_contract import dequantize_per_token, quantize_per_token


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires NPU")
def test_dispatch_quant_ub_matches_cpu_contract_on_hidden_tail():
    torch.npu.set_device(0)
    torch.manual_seed(7)
    x = torch.randn((4, 37), dtype=torch.bfloat16, device="npu")
    x[0].zero_()
    payload, scale = quantize_dispatch_payload(x)
    expected_payload, expected_scale = quantize_per_token(x.cpu())
    torch.testing.assert_close(payload.cpu(), expected_payload, atol=0, rtol=0)
    torch.testing.assert_close(scale.cpu(), expected_scale, atol=1e-5, rtol=1e-5)
    restored = dequantize_per_token(payload.cpu(), scale.cpu())
    assert float((restored - x.cpu().float()).abs().max()) < 0.1
