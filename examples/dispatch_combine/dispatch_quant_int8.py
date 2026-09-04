"""Standalone UB per-token symmetric INT8 quantization for Dispatch payloads."""

from __future__ import annotations

import argparse
from typing import Callable

import tilelang
import tilelang.language as T
import torch

from attempt_log import AttemptLogger, tensor_digest
from moe_contract import ContractError, dequantize_per_token, quantize_per_token


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1, 2], pass_configs=PASS_CONFIGS)
def dispatch_quantize_int8_kernel(M: int, logical_hidden: int, physical_hidden: int, input_dtype: str = "bfloat16"):
    """Quantize one or two rows per AI core using only fixed physical shapes."""
    rows_per_core = 2
    core_count = (M + rows_per_core - 1) // rows_per_core
    cast_mode = "CAST_NONE"

    @T.prim_func
    def main(
        x: T.Tensor([M, logical_hidden], input_dtype),
        payload: T.Tensor([M, physical_hidden], "int8"),
        scale: T.Tensor([M], "float32"),
    ):
        with T.Kernel(core_count, is_npu=True) as (cid, vid):
            row = cid * rows_per_core + vid
            x_ub = T.alloc_ub([1, physical_hidden], input_dtype)
            x_fp32_ub = T.alloc_ub([1, physical_hidden], "float32")
            abs_ub = T.alloc_ub([1, physical_hidden], "float32")
            scale_ub = T.alloc_ub([1], "float32")
            x_fp16_ub = T.alloc_ub([1, physical_hidden], "float16")
            payload_ub = T.alloc_ub([1, physical_hidden], "int8")

            with T.Scope("V"):
                if row < M:
                    T.copy(x[row, 0:logical_hidden], x_ub, pad_value=0)
                    T.tile.cast(x_fp32_ub, x_ub, mode=cast_mode, count=physical_hidden)
                    T.tile.abs(abs_ub, x_fp32_ub)
                    T.reduce_max(abs_ub, scale_ub, dim=-1)
                    if scale_ub[0] == 0.0:
                        scale_ub[0] = 1.0
                    else:
                        scale_ub[0] = scale_ub[0] / 127.0
                    for j in T.Parallel(physical_hidden):
                        x_fp32_ub[0, j] = x_fp32_ub[0, j] / scale_ub[0]
                    T.tile.clamp(x_fp32_ub, x_fp32_ub, -127.0, 127.0, physical_hidden)
                    T.tile.round(x_fp32_ub, x_fp32_ub, physical_hidden)
                    T.tile.cast(x_fp16_ub, x_fp32_ub, mode=cast_mode, count=physical_hidden)
                    T.tile.cast(payload_ub, x_fp16_ub, mode=cast_mode, count=physical_hidden)
                    T.copy(payload_ub, payload[row, 0])
                    T.copy(scale_ub, scale[row : row + 1])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def dispatch_quantize_int8_fixed_kernel(M: int, logical_hidden: int, physical_hidden: int, input_dtype: str = "bfloat16"):
    """Quantize into caller-owned fixed buffers for Graph capture."""
    rows_per_core = 2
    core_count = (M + rows_per_core - 1) // rows_per_core
    cast_mode = "CAST_NONE"

    @T.prim_func
    def main(
        x: T.Tensor([M, logical_hidden], input_dtype),
        payload: T.Tensor([M, physical_hidden], "int8"),
        scale: T.Tensor([M], "float32"),
    ):
        with T.Kernel(core_count, is_npu=True) as (cid, vid):
            row = cid * rows_per_core + vid
            x_ub = T.alloc_ub([1, physical_hidden], input_dtype)
            x_fp32_ub = T.alloc_ub([1, physical_hidden], "float32")
            abs_ub = T.alloc_ub([1, physical_hidden], "float32")
            scale_ub = T.alloc_ub([1], "float32")
            x_fp16_ub = T.alloc_ub([1, physical_hidden], "float16")
            payload_ub = T.alloc_ub([1, physical_hidden], "int8")

            with T.Scope("V"):
                if row < M:
                    T.copy(x[row, 0:logical_hidden], x_ub, pad_value=0)
                    T.tile.cast(x_fp32_ub, x_ub, mode=cast_mode, count=physical_hidden)
                    T.tile.abs(abs_ub, x_fp32_ub)
                    T.reduce_max(abs_ub, scale_ub, dim=-1)
                    if scale_ub[0] == 0.0:
                        scale_ub[0] = 1.0
                    else:
                        scale_ub[0] = scale_ub[0] / 127.0
                    for j in T.Parallel(physical_hidden):
                        x_fp32_ub[0, j] = x_fp32_ub[0, j] / scale_ub[0]
                    T.tile.clamp(x_fp32_ub, x_fp32_ub, -127.0, 127.0, physical_hidden)
                    T.tile.round(x_fp32_ub, x_fp32_ub, physical_hidden)
                    T.tile.cast(x_fp16_ub, x_fp32_ub, mode=cast_mode, count=physical_hidden)
                    T.tile.cast(payload_ub, x_fp16_ub, mode=cast_mode, count=physical_hidden)
                    T.copy(payload_ub, payload[row, 0])
                    T.copy(scale_ub, scale[row : row + 1])

    return main


def _require_npu() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is required for dispatch_quantize_int8_kernel")


def quantize_dispatch_payload(
    x: torch.Tensor,
    logger: AttemptLogger | None = None,
    *,
    trim_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a logical [route, hidden] tensor and trim padded hidden output."""
    _require_npu()
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16):
        raise ContractError("dispatch quantization accepts a 2D BF16 or FP16 tensor")
    x = x.contiguous()
    logical_hidden = x.shape[1]
    physical_hidden = (logical_hidden + 31) // 32 * 32
    kernel = dispatch_quantize_int8_kernel(
        x.shape[0], logical_hidden, physical_hidden, "bfloat16" if x.dtype == torch.bfloat16 else "float16"
    )
    payload_physical, scale = kernel(x)
    payload = payload_physical[:, :logical_hidden] if trim_output else payload_physical
    if logger is not None:
        logger.event(
            "quant_finished",
            rows=x.shape[0],
            logical_hidden=logical_hidden,
            physical_hidden=physical_hidden,
            payload_digest=tensor_digest(payload),
            scale_digest=tensor_digest(scale),
        )
    return payload, scale


def run_golden(x: torch.Tensor) -> None:
    """Run the NPU kernel and compare against the frozen CPU contract."""
    logger = AttemptLogger.create()
    logger.event("quant_configuration", rows=x.shape[0], hidden=x.shape[1], dtype=str(x.dtype))
    payload, scale = quantize_dispatch_payload(x, logger)
    expected_payload, expected_scale = quantize_per_token(x.cpu())
    torch.testing.assert_close(payload.cpu(), expected_payload, atol=0, rtol=0)
    torch.testing.assert_close(scale.cpu(), expected_scale, atol=1e-5, rtol=1e-5)
    restored = dequantize_per_token(payload.cpu(), scale.cpu())
    logger.event(
        "quant_golden_pass",
        max_abs_error=float((restored - x.cpu().float()).abs().max()),
        relative_l2=float(torch.linalg.vector_norm(restored - x.cpu().float()) / torch.linalg.vector_norm(x.cpu().float()).clamp_min(1e-12)),
    )
    print("DISPATCH_QUANT_REFERENCE_PASS")
    print("DISPATCH_QUANT_NPU_PASS")


def main(custom_args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=37)
    args = parser.parse_args(custom_args)
    _require_npu()
    torch.npu.set_device(0)
    torch.manual_seed(0)
    x = torch.randn((args.rows, args.hidden), dtype=torch.bfloat16, device="npu")
    x[0].zero_()
    run_golden(x)


if __name__ == "__main__":
    main()
