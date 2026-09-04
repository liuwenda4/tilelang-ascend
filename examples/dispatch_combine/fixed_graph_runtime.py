"""Host-side guards for fixed-capacity Dispatch/Combine Graph experiments.

This module intentionally does not initialize ACLSHMEM. Bootstrap, symmetric
window allocation, and host barriers belong to the caller and must happen
before :func:`capture_or_fallback` is entered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Callable, Iterable

import torch

from moe_contract import DispatchCombineABI


@dataclass(frozen=True)
class GraphBucket:
    """The dimensions that select one compiled fixed-shape graph binary."""

    local_tokens: int
    hidden: int
    topk: int
    max_capacity: int

    @classmethod
    def from_abi(cls, abi: DispatchCombineABI) -> "GraphBucket":
        return cls(abi.local_tokens, abi.hidden, abi.topk, abi.max_capacity)

    def matches(self, abi: DispatchCombineABI) -> bool:
        return self == GraphBucket.from_abi(abi)


@dataclass
class GraphTelemetry:
    capture_success: bool = False
    replay_success: bool = False
    fallback: bool = False
    fallback_reason: str | None = None
    replay_count: int = 0
    input_addresses: dict[str, int] = field(default_factory=dict)
    output_addresses: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "capture_success": self.capture_success,
            "replay_success": self.replay_success,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "replay_count": self.replay_count,
            "input_addresses": self.input_addresses,
            "output_addresses": self.output_addresses,
        }


def _tensor_addresses(tensors: Iterable[tuple[str, torch.Tensor]]) -> dict[str, int]:
    return {name: tensor.data_ptr() for name, tensor in tensors}


def validate_fixed_tensors(
    abi: DispatchCombineABI,
    *,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    active_mask: torch.Tensor,
    generation_id: torch.Tensor,
    iteration_id: torch.Tensor,
) -> None:
    """Reject a bucket mismatch before capture or launch."""
    expected = {
        "x": ((abi.local_tokens, abi.hidden), torch.bfloat16),
        "expert_ids": ((abi.local_tokens, abi.topk), torch.int32),
        "active_mask": ((abi.max_capacity,), torch.uint8),
        "generation_id": ((1,), torch.int32),
        "iteration_id": ((1,), torch.int32),
    }
    actual = {
        "x": x,
        "expert_ids": expert_ids,
        "active_mask": active_mask,
        "generation_id": generation_id,
        "iteration_id": iteration_id,
    }
    for name, (shape, dtype) in expected.items():
        tensor = actual[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"Graph bucket mismatch for {name}: expected shape={shape}, dtype={dtype}; "
                f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
        if not tensor.is_npu:
            raise ValueError(f"Graph tensor {name} must already be on NPU")


def capture_or_fallback(
    graph_body: Callable[[], None],
    eager_body: Callable[[], None],
    *,
    input_tensors: Iterable[tuple[str, torch.Tensor]],
    output_tensors: Iterable[tuple[str, torch.Tensor]] = (),
    replay_count: int = 1,
) -> GraphTelemetry:
    """Capture/replay a fixed graph, falling back before any replay launch.

    ``graph_body`` must only launch already-compiled fixed-shape kernels. The
    helper does not call barriers, read device counts, or allocate tensors.
    """
    if replay_count < 1:
        raise ValueError("replay_count must be positive")
    telemetry = GraphTelemetry(
        input_addresses=_tensor_addresses(input_tensors),
        output_addresses=_tensor_addresses(output_tensors),
    )
    if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
        telemetry.fallback = True
        telemetry.fallback_reason = "NPUGRAPH_UNAVAILABLE"
        eager_body()
        return telemetry

    try:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_body()
        telemetry.capture_success = True
    except Exception as exc:  # Graph support is platform/runtime dependent.
        telemetry.fallback = True
        telemetry.fallback_reason = f"CAPTURE_FAILED:{type(exc).__name__}:{exc}"
        eager_body()
        return telemetry

    try:
        for _ in range(replay_count):
            graph.replay()
        torch.npu.synchronize()
        telemetry.replay_success = True
        telemetry.replay_count = replay_count
    except Exception as exc:
        telemetry.fallback = True
        telemetry.fallback_reason = f"REPLAY_FAILED:{type(exc).__name__}:{exc}"
    return telemetry


def append_telemetry(path: str, telemetry: GraphTelemetry, **extra: object) -> None:
    record = telemetry.as_dict()
    record.update(extra)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
