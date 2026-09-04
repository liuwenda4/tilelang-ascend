"""Fixed-capacity SHMEM payload probe used before full MoE Graph capture.

The probe intentionally has no router or Expert MLP. It verifies the fixed
payload, scale, metadata and status ABI for a ring of ranks. The same fixed
buffers can be captured in an NPUGraph after SHMEM bootstrap has completed.
"""

from __future__ import annotations

import argparse
from multiprocessing import Barrier, Process
import os
from pathlib import Path
import sys
import time

import tilelang
import tilelang.language as T
import torch
import shmem as aclshmem_module

from attempt_log import AttemptLogger, tensor_digest
from dispatch_quant_int8 import quantize_dispatch_payload
from moe_contract import dequantize_per_token, quantize_per_token


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
DEBUG_DUMP = os.environ.get("MOE_DEBUG_DUMP", "0") == "1"


@tilelang.jit(pass_configs=PASS_CONFIGS)
def fixed_payload_probe_kernel(
    rows: int,
    hidden: int,
    rank: int,
    peer: int,
    payload_dtype: str,
    receive_mode: int,
    publish_with_signal_op: bool,
):
    """Send every fixed row, then wait for and optionally read peer rows."""
    @T.prim_func
    def main(
        payload: T.Tensor([rows, hidden], payload_dtype),
        scale: T.Tensor([rows], "float32"),
        active_mask: T.Tensor([rows], "int32"),
        generation_id: T.Tensor([1], "int32"),
        iteration_id: T.Tensor([1], "int32"),
        window_payload: T.Tensor([rows, hidden], payload_dtype),
        window_scale: T.Tensor([rows, 8], "float32"),
        window_status: T.Tensor([rows, 8], "int32"),
        received: T.Tensor([rows, hidden], payload_dtype),
        received_scale: T.Tensor([rows], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            payload_ub = T.alloc_ub([hidden], payload_dtype)
            scale_ub = T.alloc_ub([8], "float32")
            status_int_ub = T.alloc_ub([8], "int32")
            receive_status_int_ub = T.alloc_ub([8], "int32")
            generation_ub = T.alloc_ub([1], "int32")
            iteration_ub = T.alloc_ub([1], "int32")

            with T.Scope("V"):
                if vid == 0:
                    T.copy(generation_id, generation_ub)
                    T.copy(iteration_id, iteration_ub)
                    T.tile.fill(scale_ub, 0.0)
                    T.tile.fill(status_int_ub, 0)
                    status_int_ub[0] = generation_ub[0]
                    status_int_ub[2] = iteration_ub[0]

                    for row in T.serial(rows):
                        active = active_mask[row]
                        if active != 0:
                            T.copy(payload[row, 0], payload_ub)
                            T.shmem_ub_put_nbi(payload_ub, window_payload, hidden, peer, row * hidden)
                            T.shmem_mte_quiet()
                            scale_ub[0] = scale[row]
                            T.shmem_ub_put_nbi(scale_ub, window_scale, 8, peer, row * 8)
                            T.shmem_mte_quiet()
                        status_int_ub[1] = active
                        T.shmem_ub_put_nbi(status_int_ub, window_status, 8, peer, row * 8)
                        T.shmem_mte_quiet()
                        if publish_with_signal_op:
                            T.shmem_signal_op(window_status, row * 8, generation_ub[0], 0, peer)
                        if receive_mode > 0:
                            if DEBUG_DUMP:
                                T.printf("wait_start rank=%d row=%d\n", rank, row)
                            T.shmem_signal_wait_until(window_status, row * 8, 0, generation_ub[0])
                            if DEBUG_DUMP:
                                T.printf("wait_done rank=%d row=%d\n", rank, row)
                            if receive_mode != 3:
                                T.copy(window_status[row, 0], receive_status_int_ub)
                                T.set_flag("mte2", "v", 13)
                                T.wait_flag("mte2", "v", 13)
                                if DEBUG_DUMP:
                                    T.dump_tensor(receive_status_int_ub, 301, 8, (8,))
                            if receive_mode == 1 and receive_status_int_ub[1] != 0:
                                T.copy(window_payload[row, 0], payload_ub)
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                if DEBUG_DUMP:
                                    T.dump_tensor(payload_ub, 302, hidden, (hidden,))
                                T.copy(payload_ub, received[row, 0])
                                T.copy(window_scale[row, 0], scale_ub)
                                T.set_flag("mte2", "v", 1)
                                T.wait_flag("mte2", "v", 1)
                                T.copy(scale_ub[0:1], received_scale[row : row + 1])
                                T.set_flag("mte3", "s", 7)
                                T.wait_flag("mte3", "s", 7)
                                T.tile.datacachecleanandinvalid_experiment(received, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                                T.tile.datacachecleanandinvalid_experiment(received_scale, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                                T.shmem_mte_quiet()
                                if DEBUG_DUMP:
                                    T.dump_tensor(received, 303, hidden, (hidden,))
                                    T.dump_tensor(received_scale, 304, 1, (1,))
                            else:
                                for hidden_index in T.serial(hidden):
                                    received[row, hidden_index] = 0
                                received_scale[row] = 0.0

    return main


def _physical_hidden(hidden: int) -> int:
    return max(32, (hidden + 31) // 32 * 32)


def _make_payload(rank: int, rows: int, hidden: int, payload_dtype: str) -> torch.Tensor:
    if payload_dtype == "bfloat16":
        payload = torch.full((rows, hidden), float(rank + 1), dtype=torch.bfloat16, device="npu")
    else:
        payload = torch.full((rows, hidden), rank + 1, dtype=torch.int8, device="npu")
    return payload


def _worker(rank: int, args: argparse.Namespace, barrier: Barrier, root: str, attempt_id: str) -> None:
    logger = AttemptLogger(Path(root), attempt_id)
    logger.rank_event(
        rank,
        "probe_rank_started",
        mode=args.mode,
        payload_dtype=args.dtype,
        receive_mode=args.receive_mode,
        publish_with_signal_op=args.publish_with_signal_op,
    )
    tensors = []
    try:
        torch.npu.set_device(rank)
        hidden = _physical_hidden(args.hidden)
        payload_dtype = "bfloat16" if args.dtype == "bf16" else "int8"
        if payload_dtype == "bfloat16":
            payload = _make_payload(rank, args.rows, hidden, payload_dtype)
            scale = torch.full((args.rows,), 0.25 + rank, dtype=torch.float32, device="npu")
        else:
            source_cpu = torch.full((args.rows, args.hidden), float(rank + 1), dtype=torch.bfloat16)
            source = source_cpu.npu()
            payload, scale = quantize_dispatch_payload(source, logger, trim_output=False)
        active_mask = torch.ones((args.rows,), dtype=torch.int32, device="npu")
        generation_id = torch.ones((1,), dtype=torch.int32, device="npu")
        iteration_id = torch.zeros((1,), dtype=torch.int32, device="npu")
        received = torch.zeros_like(payload)
        received_scale = torch.zeros_like(scale)
        peer = (rank + 1) % args.world_size
        tensors.extend([payload, scale, active_mask, generation_id, iteration_id, received, received_scale])

        ret = aclshmem_module.set_conf_store_tls(False, "")
        logger.rank_event(rank, "probe_tls_configured", returncode=ret)
        if ret != 0:
            raise RuntimeError(f"set_conf_store_tls failed: {ret}")
        attributes = aclshmem_module.InitAttr()
        attributes.my_rank = rank
        attributes.n_ranks = args.world_size
        attributes.local_mem_size = args.local_mem_size
        attributes.ip_port = args.ip_port
        attributes.option_attr.data_op_engine_type = aclshmem_module.OpEngineType.MTE
        ret = aclshmem_module.aclshmem_init(attributes)
        logger.rank_event(rank, "probe_shmem_initialized", returncode=ret, world_size=args.world_size, peer=peer)
        if ret != 0:
            raise RuntimeError(f"aclshmem_init failed: {ret}")

        window_payload_tensor = aclshmem_module.aclshmem_create_tensor([args.rows, hidden], dtype=getattr(torch, payload_dtype), device_id=rank)
        window_scale_tensor = aclshmem_module.aclshmem_create_tensor([args.rows, 8], dtype=torch.float32, device_id=rank)
        window_status_tensor = aclshmem_module.aclshmem_create_tensor([args.rows, 8], dtype=torch.int32, device_id=rank)
        window_payload = window_payload_tensor.zero_()
        window_scale = window_scale_tensor.zero_()
        window_status = window_status_tensor.zero_()
        if args.preseed_status:
            window_status[:, 0].fill_(1)
        tensors.extend([window_payload, window_scale, window_status])
        logger.rank_event(rank, "probe_windows_allocated", rows=args.rows, hidden=hidden)
        torch.npu.synchronize()
        barrier.wait()

        if args.two_phase:
            send_kernel = fixed_payload_probe_kernel(
                args.rows,
                hidden,
                rank,
                peer,
                payload_dtype,
                0,
                args.publish_with_signal_op,
            )
            logger.rank_event(rank, "probe_send_kernel_compiled")
            send_kernel(
                payload,
                scale,
                active_mask,
                generation_id,
                iteration_id,
                window_payload,
                window_scale,
                window_status,
                received,
                received_scale,
            )
            torch.npu.synchronize()
            logger.rank_event(
                rank,
                "send_phase_finished",
                window_first=float(window_payload[0, 0].cpu()),
                window_scale_first=float(window_scale[0, 0].cpu()),
                window_status_words=window_status[0].cpu().tolist(),
            )
            barrier.wait()
            wait_kernel = fixed_payload_probe_kernel(
                args.rows,
                hidden,
                rank,
                peer,
                payload_dtype,
                2,
                args.publish_with_signal_op,
            )
            logger.rank_event(rank, "probe_wait_kernel_compiled")
            wait_kernel(
                payload,
                scale,
                active_mask,
                generation_id,
                iteration_id,
                window_payload,
                window_scale,
                window_status,
                received,
                received_scale,
            )
            torch.npu.synchronize()
            logger.rank_event(
                rank,
                "wait_only_finished",
                window_status_words=window_status[0].cpu().tolist(),
            )
            barrier.wait()
            return

        kernel = fixed_payload_probe_kernel(
            args.rows,
            hidden,
            rank,
            peer,
            payload_dtype,
            args.receive_mode,
            args.publish_with_signal_op,
        )
        logger.rank_event(rank, "probe_kernel_compiled", graph_requested=args.mode == "graph")
        logger.rank_event(rank, "host_barrier", phase="pre_launch")
        barrier.wait()

        def launch():
            return kernel(payload, scale, active_mask, generation_id, iteration_id, window_payload, window_scale, window_status, received, received_scale)

        if args.mode == "graph":
            logger.rank_event(rank, "graph_capture_barrier_wait")
            barrier.wait()
            graph = torch.npu.NPUGraph()
            logger.rank_event(rank, "graph_capture_started", generation=int(generation_id.cpu().item()))
            try:
                with torch.npu.graph(graph):
                    launch()
            except Exception as exc:
                logger.rank_event(rank, "graph_capture_failed", exception_type=type(exc).__name__, exception=str(exc))
                raise
            torch.npu.synchronize()
            logger.rank_event(rank, "graph_capture_finished")
            barrier.wait()
            for replay in range(args.replays):
                generation_id.fill_(replay + 2)
                iteration_id.fill_(replay + 1)
                graph.replay()
                torch.npu.synchronize()
                logger.rank_event(
                    rank,
                    "graph_replay_finished",
                    replay=replay + 1,
                    generation=replay + 2,
                    actual_count=int(active_mask.sum().cpu().item()),
                    output_digest=tensor_digest(received),
                    scale_digest=tensor_digest(received_scale),
                    stale_status_count=int((window_status[:, 0].cpu() != generation_id.cpu()).sum()),
                    timeout=False,
                )
        else:
            for replay in range(args.replays):
                generation_id.fill_(replay + 1)
                iteration_id.fill_(replay)
                logger.rank_event(rank, "eager_launch_started", replay=replay + 1, generation=replay + 1)
                result = launch()
                torch.npu.synchronize()
                logger.rank_event(
                    rank,
                    "eager_launch_finished",
                    replay=replay + 1,
                    generation=replay + 1,
                    actual_count=int(active_mask.sum().cpu().item()),
                    output_digest=tensor_digest(received),
                    scale_digest=tensor_digest(received_scale),
                    returned_first=float(result[0][0, 0].cpu()) if isinstance(result, tuple) else None,
                    stale_status_count=int((window_status[:, 0].cpu() != generation_id.cpu()).sum()),
                    timeout=False,
                )
                barrier.wait()

        if args.receive_mode == 0:
            logger.rank_event(
                rank,
                "send_only_finished",
                window_first=float(window_payload[0, 0].cpu()),
                window_scale_first=float(window_scale[0, 0].cpu()),
                window_status_words=window_status[0].cpu().tolist(),
            )
            barrier.wait()
            return

        if args.receive_mode in (2, 3):
            logger.rank_event(
                rank,
                "wait_only_finished" if args.receive_mode == 2 else "bare_wait_finished",
                window_status_words=window_status[0].cpu().tolist(),
            )
            barrier.wait()
            return

        expected_rank = (rank - 1) % args.world_size
        if args.dtype == "int8":
            expected_source = torch.full((args.rows, args.hidden), float(expected_rank + 1), dtype=torch.bfloat16)
            expected_payload_logical, expected_scale = quantize_per_token(expected_source)
            expected_payload = torch.zeros((args.rows, hidden), dtype=torch.int8)
            expected_payload[:, : args.hidden] = expected_payload_logical
        else:
            expected_payload = torch.full((args.rows, hidden), expected_rank + 1, dtype=torch.bfloat16)
            expected_scale = torch.full((args.rows,), 0.25 + expected_rank, dtype=torch.float32)
        golden_error = None
        logger.rank_event(
            rank,
            "probe_payload_observed",
            input_first=float(payload[0, 0].cpu()),
            window_first=float(window_payload[0, 0].cpu()),
            received_first=float(received[0, 0].cpu()),
            received_float_first=float(received.float()[0, 0].cpu()),
            input_scale=float(scale[0].cpu()),
            received_scale_first=float(received_scale[0].cpu()),
            received_scale_float_first=float(received_scale.float()[0].cpu()),
            window_status_words=window_status[0].cpu().tolist(),
        )
        try:
            torch.testing.assert_close(received.cpu(), expected_payload)
            torch.testing.assert_close(received_scale.cpu(), expected_scale)
            if args.dtype == "int8":
                restored = dequantize_per_token(received.cpu()[:, : args.hidden], received_scale.cpu())
                torch.testing.assert_close(restored, expected_source.float(), rtol=0.0, atol=1e-6)
                logger.rank_event(
                    rank,
                    "quant_roundtrip_error",
                    max_abs_error=float((restored - expected_source.float()).abs().max()),
                    relative_l2=float(torch.linalg.vector_norm(restored - expected_source.float()) / torch.linalg.vector_norm(expected_source.float()).clamp_min(1e-12)),
                )
        except Exception as exc:
            golden_error = exc
            logger.rank_event(
                rank,
                "probe_golden_fail",
                failure_class="SHMEM_PAYLOAD_READ_BLOCKED",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )
        else:
            logger.rank_event(rank, "probe_golden_pass", output_digest=tensor_digest(received))
            print(f"Rank {rank}: fixed payload golden passed")
        barrier.wait()
        if golden_error is not None:
            raise golden_error
    except Exception as exc:
        logger.rank_event(rank, "probe_exception", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        for tensor in tensors:
            del tensor
        try:
            aclshmem_module.aclshmem_finialize()
            logger.rank_event(rank, "probe_shmem_finalized")
        except Exception as exc:
            logger.rank_event(rank, "probe_finalize_exception", exception_type=type(exc).__name__, exception=str(exc))
        logger.rank_event(rank, "probe_rank_exited")


def main(custom_args=None) -> None:
    parser = argparse.ArgumentParser(description="Fixed BF16/INT8 SHMEM payload probe")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--dtype", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=37)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--ip-port", default="tcp://192.168.0.131:8666")
    parser.add_argument("--local-mem-size", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-base", default="artifacts/moe")
    parser.add_argument(
        "--receive-mode",
        choices=("full", "send-only", "wait-only", "bare-wait"),
        default="full",
        help="select the SHMEM receive phase used by the diagnostic probe",
    )
    parser.add_argument(
        "--signal-op",
        action="store_true",
        dest="publish_with_signal_op",
        help="publish generation with SHMEM SIGNAL_SET after the status put",
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        help="send from both ranks, synchronize on the host, then run wait-only kernels",
    )
    parser.add_argument(
        "--preseed-status",
        action="store_true",
        help="initialize local status generation to one before launching the probe",
    )
    args = parser.parse_args(custom_args)
    if args.rows < 1 or args.world_size < 2 or args.replays < 1:
        raise ValueError("rows, world-size and replays must be positive; world-size must be at least 2")
    if args.receive_mode != "full" and args.mode != "eager":
        raise ValueError("diagnostic receive modes are only supported in eager mode")
    if args.two_phase and args.mode != "eager":
        raise ValueError("--two-phase is only supported in eager mode")
    if args.two_phase and args.receive_mode != "full":
        raise ValueError("--two-phase requires --receive-mode full")
    args.receive_mode = {"send-only": 0, "full": 1, "wait-only": 2, "bare-wait": 3}[args.receive_mode]
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is required for the SHMEM payload probe")

    attempt = AttemptLogger.create(args.log_base)
    attempt.event("probe_configuration", argv=sys.argv, **vars(args))
    barrier = Barrier(args.world_size, timeout=args.timeout)
    processes = []
    for rank in range(args.world_size):
        process = Process(target=_worker, args=(rank, args, barrier, str(attempt.root), attempt.attempt_id))
        process.start()
        processes.append(process)
    deadline = time.monotonic() + args.timeout
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        for process in processes:
            if process.is_alive():
                process.join(0.1)
    for rank, process in enumerate(processes):
        if process.is_alive():
            attempt.event("probe_rank_timeout", rank=rank, pid=process.pid, timeout_seconds=args.timeout)
            process.terminate()
            process.join()
            attempt.event("probe_rank_terminated", rank=rank, pid=process.pid)
    exitcodes = {str(rank): process.exitcode for rank, process in enumerate(processes)}
    success = all(code == 0 for code in exitcodes.values())
    attempt.event("probe_rank_exit_codes", exitcodes=exitcodes)
    attempt.event("probe_finished", success=success)
    print("DISPATCH_COMBINE_PROBE_PASS" if success else "DISPATCH_COMBINE_PROBE_FAIL")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
