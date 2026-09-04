import pytest
import torch

from fixed_graph_runtime import GraphBucket, capture_or_fallback, validate_fixed_tensors
from moe_contract import DispatchCombineABI


def test_graph_bucket_is_exact():
    abi = DispatchCombineABI(4, 8, 2, 2, 2, 4)
    bucket = GraphBucket.from_abi(abi)
    assert bucket.matches(abi)
    assert not bucket.matches(DispatchCombineABI(4, 8, 2, 2, 2, 5))


def test_non_npu_inputs_are_rejected_before_capture():
    abi = DispatchCombineABI(4, 8, 2, 2, 2, 4)
    tensors = {
        "x": torch.zeros((4, 8), dtype=torch.bfloat16),
        "expert_ids": torch.zeros((4, 2), dtype=torch.int32),
        "active_mask": torch.zeros((8,), dtype=torch.uint8),
        "generation_id": torch.ones((1,), dtype=torch.int32),
        "iteration_id": torch.ones((1,), dtype=torch.int32),
    }
    with pytest.raises(ValueError, match="must already be on NPU"):
        validate_fixed_tensors(abi, **tensors)


def test_capture_helper_falls_back_without_npu_graph():
    calls = []
    telemetry = capture_or_fallback(
        lambda: calls.append("graph"),
        lambda: calls.append("eager"),
        input_tensors=[("x", torch.zeros(1))],
        replay_count=3,
    )
    if hasattr(torch, "npu") and hasattr(torch.npu, "NPUGraph"):
        pytest.skip("requires a runtime without NPUGraph")
    assert calls == ["eager"]
    assert telemetry.fallback
    assert telemetry.fallback_reason == "NPUGRAPH_UNAVAILABLE"
