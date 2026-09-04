"""CPU contracts for fixed-capacity MoE dispatch/combine experiments.

The reference deliberately keeps the communication payload separate from the
control metadata.  It is used to validate the ABI before an NPU kernel is
allowed to consume a fixed-capacity buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


class ContractError(ValueError):
    """Base class for invalid MoE contract inputs."""


class CapacityOverflowError(ContractError):
    """Raised when a fixed-capacity buffer cannot hold all accepted routes."""


@dataclass(frozen=True)
class DispatchCombineABI:
    """Fixed tensor ABI shared by eager and Graph dispatch/combine paths."""

    local_tokens: int
    hidden: int
    topk: int
    num_ranks: int
    num_local_experts: int
    global_capacity: int

    def __post_init__(self) -> None:
        for name in (
            "local_tokens",
            "hidden",
            "topk",
            "num_ranks",
            "num_local_experts",
            "global_capacity",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        if self.topk == 0 and self.local_tokens != 0:
            raise ContractError("topk must be positive for non-empty input")
        if self.num_ranks == 0 or self.num_local_experts == 0:
            raise ContractError("num_ranks and num_local_experts must be positive")

    @property
    def num_global_experts(self) -> int:
        return self.num_ranks * self.num_local_experts

    @property
    def max_capacity(self) -> int:
        """Maximum route rows in fixed Dispatch/Combine buffers."""
        return self.global_capacity * self.topk

    @property
    def max_destination_capacity(self) -> int:
        """Maximum receive rows for expert-major source slots on one rank."""
        return self.num_ranks * self.local_tokens * self.num_local_experts

    @property
    def tensors(self) -> dict[str, dict[str, Any]]:
        return {
            "x": {"shape": [self.local_tokens, self.hidden], "dtype": "bfloat16", "layout": "row-major", "owner": "source-rank"},
            "expert_ids": {"shape": [self.local_tokens, self.topk], "dtype": "int32", "layout": "token-major", "owner": "source-rank"},
            "expand_x": {"shape": [self.max_capacity, self.hidden], "dtype": "bfloat16", "layout": "expert-major", "owner": "destination-rank"},
            "expert_output": {"shape": [self.max_capacity, self.hidden], "dtype": "bfloat16", "layout": "expert-major", "owner": "destination-rank"},
            "expand_ids": {"shape": [self.max_capacity, 3], "dtype": "int32", "layout": "[source_rank, token_id, topk_slot]", "owner": "destination-rank"},
            "global_prefix": {"shape": [self.num_global_experts], "dtype": "int32", "layout": "inclusive expert-major prefix; internal", "owner": "device"},
            "ep_receive_count": {"shape": [self.num_local_experts], "dtype": "int32", "layout": "per-local-expert count", "owner": "destination-rank"},
            "active_mask": {"shape": [self.max_capacity], "dtype": "uint8", "layout": "route-major", "owner": "destination-rank"},
            "actual_count": {"shape": [1], "dtype": "int32", "layout": "scalar tensor", "owner": "device"},
            "generation_id": {"shape": [1], "dtype": "int32", "layout": "scalar tensor", "owner": "device"},
            "iteration_id": {"shape": [1], "dtype": "int32", "layout": "scalar tensor", "owner": "device"},
            "int8_payload": {"shape": [self.max_capacity, self.hidden], "dtype": "int8", "layout": "expert-major", "owner": "SHMEM window"},
            "scale": {"shape": [self.max_capacity], "dtype": "float32", "layout": "route-major", "owner": "SHMEM window"},
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "abi": "fixed-capacity-dispatch-combine-v1",
            "local_tokens": self.local_tokens,
            "hidden": self.hidden,
            "topk": self.topk,
            "num_ranks": self.num_ranks,
            "num_local_experts": self.num_local_experts,
            "num_global_experts": self.num_global_experts,
            "global_capacity": self.global_capacity,
            "max_capacity": self.max_capacity,
            "max_destination_capacity": self.max_destination_capacity,
            "actual_count_semantics": "device tensor only; never used for kernel shape",
            "active_mask_semantics": "1 means route row is valid; 0 means row must be ignored",
            "generation_semantics": "positive monotonically increasing int32; stale generations are invalid",
            "iteration_semantics": "host-visible logical iteration carried as a fixed input tensor",
            "expert_mapping": "global_expert_id // num_local_experts -> destination_rank; modulo -> local_expert_id",
            "triplet": ["source_rank", "token_id", "topk_slot"],
            "quantization": quantization_contract(),
            "tensors": self.tensors,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quantization_contract() -> dict[str, Any]:
    """Return the only INT8 format supported by this first experiment."""
    return {
        "mode": "per-token-symmetric-int8",
        "input_dtype": ["bfloat16", "float16"],
        "payload_dtype": "int8",
        "scale_dtype": "float32",
        "scale_shape": "[route_count]",
        "scale_formula": "max(abs(x[row])) / 127, or 1.0 for an all-zero row",
        "rounding": "round-to-nearest-even",
        "saturation": [-127, 127],
        "zero_row": {"scale": 1.0, "payload": 0},
        "payload_layout": "[route, hidden] row-major, hidden tail padded to transfer alignment outside logical shape",
    }


def expert_destination(global_expert_id: int, num_local_experts: int, num_ranks: int) -> tuple[int, int]:
    if not isinstance(global_expert_id, int) or not 0 <= global_expert_id < num_local_experts * num_ranks:
        raise ContractError(f"global expert id {global_expert_id} is out of range")
    return global_expert_id // num_local_experts, global_expert_id % num_local_experts


def validate_router_expert_ids(
    expert_ids: torch.Tensor,
    num_global_experts: int,
    *,
    topk: int | None = None,
) -> None:
    """Validate the route invariants required by the expert-major NPU layout."""
    if expert_ids.ndim != 2 or expert_ids.dtype not in (torch.int32, torch.int64):
        raise ContractError("expert_ids must be a 2D int32 or int64 tensor")
    if num_global_experts <= 0:
        raise ContractError("num_global_experts must be positive")
    if topk is not None and expert_ids.shape[1] != topk:
        raise ContractError(f"expert_ids must have topk={topk}, got {expert_ids.shape[1]}")
    if expert_ids.numel() == 0:
        return
    if int(expert_ids.min()) < 0 or int(expert_ids.max()) >= num_global_experts:
        raise ContractError("expert_ids contains an out-of-range global expert")
    for token_id, row in enumerate(expert_ids.tolist()):
        if len(set(row)) != len(row):
            raise ContractError(f"token {token_id} routes to the same expert more than once")


@dataclass
class DispatchReference:
    expand_x: torch.Tensor
    expert_output: torch.Tensor
    expand_ids: torch.Tensor
    active_mask: torch.Tensor
    global_prefix: torch.Tensor
    ep_receive_count: torch.Tensor
    expert_offsets: torch.Tensor
    actual_count: int
    overflow_count: int


def dispatch_reference(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    source_rank: int,
    abi: DispatchCombineABI,
    *,
    destination_rank: int | None = None,
    drop_overflow: bool = False,
) -> DispatchReference:
    """Build expert-major fixed-capacity Dispatch output on CPU.

    Routes are sorted by global expert id and retain token-major order within
    each expert.  This matches the grouped-expert contract and makes the
    ``expand_ids`` mapping independently checkable.
    """
    if x.ndim != 2 or tuple(x.shape) != (abi.local_tokens, abi.hidden):
        raise ContractError(f"x must have shape {(abi.local_tokens, abi.hidden)}, got {tuple(x.shape)}")
    if expert_ids.ndim != 2 or tuple(expert_ids.shape) != (abi.local_tokens, abi.topk):
        raise ContractError(f"expert_ids must have shape {(abi.local_tokens, abi.topk)}, got {tuple(expert_ids.shape)}")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise ContractError("expert_ids must use int32 or int64")
    if not 0 <= source_rank < abi.num_ranks:
        raise ContractError("source_rank is out of range")
    if destination_rank is not None and not 0 <= destination_rank < abi.num_ranks:
        raise ContractError("destination_rank is out of range")

    grouped: list[list[tuple[int, int, int]]] = [[] for _ in range(abi.num_global_experts)]
    for token_id in range(abi.local_tokens):
        for topk_slot in range(abi.topk):
            global_expert_id = int(expert_ids[token_id, topk_slot])
            dst_rank, _ = expert_destination(global_expert_id, abi.num_local_experts, abi.num_ranks)
            if destination_rank is None or dst_rank == destination_rank:
                grouped[global_expert_id].append((source_rank, token_id, topk_slot))

    selected: list[tuple[int, int, int, int]] = []
    overflow_count = 0
    for global_expert_id, routes in enumerate(grouped):
        for route in routes:
            if len(selected) >= abi.max_capacity:
                overflow_count += 1
                continue
            selected.append((global_expert_id, *route))

    if overflow_count and not drop_overflow:
        raise CapacityOverflowError(
            f"{overflow_count} routes exceed fixed max_capacity={abi.max_capacity}"
        )

    expand_x = torch.zeros((abi.max_capacity, abi.hidden), dtype=x.dtype, device=x.device)
    expert_output = torch.zeros((abi.max_capacity, abi.hidden), dtype=x.dtype, device=x.device)
    expand_ids = torch.full((abi.max_capacity, 3), -1, dtype=torch.int32, device=x.device)
    active_mask = torch.zeros((abi.max_capacity,), dtype=torch.uint8, device=x.device)
    global_prefix = torch.zeros((abi.num_global_experts,), dtype=torch.int32, device=x.device)
    ep_receive_count = torch.zeros((abi.num_local_experts,), dtype=torch.int32, device=x.device)
    expert_offsets = torch.zeros((abi.num_local_experts + 1,), dtype=torch.int32, device=x.device)

    for row, (global_expert_id, source, token_id, topk_slot) in enumerate(selected):
        local_expert_id = global_expert_id % abi.num_local_experts
        expand_x[row].copy_(x[token_id])
        expert_output[row].copy_(x[token_id])
        expand_ids[row] = torch.tensor([source, token_id, topk_slot], dtype=torch.int32, device=x.device)
        active_mask[row] = 1
        ep_receive_count[local_expert_id] += 1
        global_prefix[global_expert_id] += 1

    global_prefix = torch.cumsum(global_prefix, dim=0)
    expert_offsets[1:] = torch.cumsum(ep_receive_count, dim=0)
    return DispatchReference(
        expand_x=expand_x,
        expert_output=expert_output,
        expand_ids=expand_ids,
        active_mask=active_mask,
        global_prefix=global_prefix,
        ep_receive_count=ep_receive_count,
        expert_offsets=expert_offsets,
        actual_count=len(selected),
        overflow_count=overflow_count,
    )


def combine_reference(
    expand_x: torch.Tensor,
    expand_ids: torch.Tensor,
    expert_scales: torch.Tensor,
    active_mask: torch.Tensor,
    abi: DispatchCombineABI,
) -> torch.Tensor:
    """Apply route weights using only fixed buffers and active_mask."""
    if tuple(expand_x.shape) != (abi.max_capacity, abi.hidden):
        raise ContractError("expand_x shape does not match fixed ABI")
    if tuple(expand_ids.shape) != (abi.max_capacity, 3):
        raise ContractError("expand_ids shape does not match fixed ABI")
    if tuple(active_mask.shape) != (abi.max_capacity,):
        raise ContractError("active_mask shape does not match fixed ABI")
    if tuple(expert_scales.shape) != (abi.local_tokens, abi.topk):
        raise ContractError("expert_scales shape does not match fixed ABI")

    output = torch.zeros((abi.local_tokens, abi.hidden), dtype=expand_x.dtype, device=expand_x.device)
    for row in range(abi.max_capacity):
        if int(active_mask[row]) == 0:
            continue
        source_rank, token_id, topk_slot = (int(v) for v in expand_ids[row])
        if source_rank < 0 or not 0 <= token_id < abi.local_tokens or not 0 <= topk_slot < abi.topk:
            raise ContractError(f"invalid active triplet at row {row}: {(source_rank, token_id, topk_slot)}")
        output[token_id] += expand_x[row] * expert_scales[token_id, topk_slot].to(expand_x.dtype)
    return output


def quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 rows with symmetric per-token INT8 scales."""
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ContractError("x must be a 2D BF16, FP16 or FP32 tensor")
    x_fp32 = x.float()
    absmax = x_fp32.abs().amax(dim=-1)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 127.0)
    payload = torch.round(x_fp32 / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload, scale


def dequantize_per_token(payload: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if payload.ndim != 2 or payload.dtype != torch.int8:
        raise ContractError("payload must be a 2D int8 tensor")
    if scale.ndim != 1 or scale.shape[0] != payload.shape[0] or scale.dtype != torch.float32:
        raise ContractError("scale must be a float32 vector aligned with payload rows")
    return (payload.float() * scale[:, None]).to(dtype)
