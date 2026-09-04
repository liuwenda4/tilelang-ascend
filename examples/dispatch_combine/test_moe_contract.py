import pytest
import torch

from moe_contract import (
    CapacityOverflowError,
    ContractError,
    DispatchCombineABI,
    combine_reference,
    dequantize_per_token,
    dispatch_reference,
    expert_destination,
    quantize_per_token,
    validate_router_expert_ids,
)


def make_abi(tokens=4, hidden=5, topk=2, capacity=4):
    return DispatchCombineABI(
        local_tokens=tokens,
        hidden=hidden,
        topk=topk,
        num_ranks=2,
        num_local_experts=2,
        global_capacity=capacity,
    )


def test_expert_mapping_and_expert_major_offsets():
    abi = make_abi()
    assert [expert_destination(i, abi.num_local_experts, abi.num_ranks) for i in range(4)] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    x = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5)
    expert_ids = torch.tensor([[3, 0], [1, 3], [0, 1], [3, 0]], dtype=torch.int32)
    ref = dispatch_reference(x, expert_ids, source_rank=1, abi=abi)
    assert ref.actual_count == 8
    assert ref.global_prefix.tolist() == [3, 5, 5, 8]
    assert ref.ep_receive_count.tolist() == [3, 5]
    assert ref.expert_offsets.tolist() == [0, 3, 8]
    assert ref.expand_ids[:5].tolist() == [
        [1, 0, 1],
        [1, 2, 0],
        [1, 3, 1],
        [1, 1, 0],
        [1, 2, 1],
    ]
    assert ref.active_mask[:8].tolist() == [1] * 8
    assert ref.active_mask[8:].sum().item() == 0
    torch.testing.assert_close(ref.expert_output, ref.expand_x)


def test_destination_filter_and_combine_weighted_sum():
    abi = make_abi()
    x = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5)
    expert_ids = torch.tensor([[3, 0], [1, 3], [0, 1], [3, 0]], dtype=torch.int32)
    ref = dispatch_reference(x, expert_ids, source_rank=0, abi=abi, destination_rank=1)
    assert ref.actual_count == 3
    assert ref.ep_receive_count.tolist() == [0, 3]
    scales = torch.tensor([[0.5, 2.0], [1.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=torch.float32)
    out = combine_reference(ref.expand_x, ref.expand_ids, scales, ref.active_mask, abi)
    expected = torch.zeros_like(x)
    expected[0] = x[0] * 0.5
    expected[1] = x[1] * 3.0
    expected[3] = x[3] * 6.0
    torch.testing.assert_close(out, expected)


def test_zero_token_and_capacity_overflow():
    abi = make_abi(tokens=0, hidden=5, topk=2, capacity=1)
    ref = dispatch_reference(torch.empty((0, 5), dtype=torch.bfloat16), torch.empty((0, 2), dtype=torch.int32), 0, abi)
    assert ref.actual_count == 0
    assert ref.expert_offsets.tolist() == [0, 0, 0]

    abi = make_abi(tokens=4, hidden=5, topk=2, capacity=2)
    x = torch.ones((4, 5), dtype=torch.bfloat16)
    expert_ids = torch.zeros((4, 2), dtype=torch.int32)
    with pytest.raises(CapacityOverflowError):
        dispatch_reference(x, expert_ids, 0, abi)
    ref = dispatch_reference(x, expert_ids, 0, abi, drop_overflow=True)
    assert ref.actual_count == abi.max_capacity
    assert ref.overflow_count == 4


def test_invalid_expert_and_triplet_are_rejected():
    abi = make_abi()
    x = torch.zeros((4, 5), dtype=torch.bfloat16)
    with pytest.raises(ContractError):
        dispatch_reference(x, torch.full((4, 2), 4, dtype=torch.int32), 0, abi)

    ref = dispatch_reference(x, torch.zeros((4, 2), dtype=torch.int32), 0, abi)
    ref.expand_ids[0, 1] = abi.local_tokens
    with pytest.raises(ContractError):
        combine_reference(ref.expand_x, ref.expand_ids, torch.ones((4, 2)), ref.active_mask, abi)


def test_router_route_invariants_are_validated_before_npu_launch():
    expert_ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32)
    validate_router_expert_ids(expert_ids, 4, topk=2)
    with pytest.raises(ContractError, match="same expert"):
        validate_router_expert_ids(torch.tensor([[1, 1]], dtype=torch.int32), 4, topk=2)
    with pytest.raises(ContractError, match="out-of-range"):
        validate_router_expert_ids(torch.tensor([[0, 4]], dtype=torch.int32), 4, topk=2)


def test_per_token_int8_contract_zero_extrema_and_tail_hidden():
    x = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 2.0, 0.0], [127.0, -127.0, 1.0]], dtype=torch.bfloat16)
    payload, scale = quantize_per_token(x)
    assert payload.dtype == torch.int8
    assert payload.tolist()[0] == [0, 0, 0]
    assert scale.tolist()[0] == 1.0
    assert payload[1].tolist() == [-127, 127, 0]
    assert payload[2, 0].item() == 127
    assert payload[2, 1].item() == -127
    restored = dequantize_per_token(payload, scale)
    assert restored.shape == x.shape
    assert torch.linalg.vector_norm(restored - x.float(), dim=-1).max().item() < 1.0


def test_scale_and_route_order_mismatch_is_detectable():
    x = torch.tensor([[1.0, 2.0], [10.0, 20.0]], dtype=torch.bfloat16)
    payload, scale = quantize_per_token(x)
    correct = dequantize_per_token(payload, scale)
    swapped = dequantize_per_token(payload, scale.flip(0))
    assert not torch.allclose(correct, swapped)
    with pytest.raises(ContractError):
        dequantize_per_token(payload, scale[:, None])
