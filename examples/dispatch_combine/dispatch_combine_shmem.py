import argparse
import os
import tilelang
import tilelang.language as T
import torch
import shmem as aclshmem_module
import multiprocessing as mp
import random
import sys
from multiprocessing import Barrier
from pathlib import Path
import time
from attempt_log import AttemptLogger, tensor_digest
from dispatch_quant_int8 import dispatch_quantize_int8_fixed_kernel
from moe_contract import validate_router_expert_ids
tilelang.cache.clear_cache()
DEBUG_DUMP = os.environ.get("MOE_DEBUG_DUMP", "0") == "1"
G_IP_PORT = "tcp://192.168.0.131:8666"
g_ash_size = 1024 * 1024 * 1024
pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
#--- 1. Implement Dispatch Operator ---
@tilelang.jit(pass_configs=pass_configs)
def moe_dispatch_kernel(
    Bs,     # Total number of tokens
    H,      # Token length
    K,      # Number of MOE experts to send
    ep_world_size,  # Total number of ranks
    local_expert_num,   # Number of MOE experts per rank
    rank,   # Current rank ID
    ub_size,    # Single row size of the win data area
    aiv_num,    # v-core count
):
    total_expert_num = ep_world_size * local_expert_num     # Total MOE experts count
    assist_size = 3     # Triple size
    ub_align = 32   # UB requires 32-byte alignment
    ub_float_int32_align = 8
    status_per_core = (total_expert_num + aiv_num - 1) // aiv_num   # Number of states to be processed per v-core, rounded up to the nearest integer
    floor_status_per_core = max(status_per_core - 1, 1)
    # Calculate the number of tokens already received by the current MoE expert.
    @T.macro
    def cal_token_send_expert_cnt(
        dst_expert_id,
        cal_cnt,
        dst_expert_id_ub: T.Tensor([Bs * K], "int32"),
        sub_ub: T.Tensor([Bs * K], "int32"),
        expert_ids_ub: T.Tensor([Bs, 8], "int32"),
        tmp_fp_32: T.Tensor([Bs * K], "float"),
        tmp_out_fp_32: T.Tensor([Bs * K], "float"),
        work_local_ub: T.Tensor([total_expert_num * ub_float_int32_align], "float"),
    ):
        dst_expert_id_ub[0] = 0
        for token_index in T.serial(cal_cnt):
            if expert_ids_ub[token_index // K, token_index % K] == dst_expert_id:
                dst_expert_id_ub[0] = dst_expert_id_ub[0] + 1
    # Synchronization function between different pipelines
    @T.macro
    def sync_func(src, dst, event_id: "str"):
        T.set_flag(src, dst, event_id)
        T.wait_flag(src, dst, event_id)
    @T.prim_func
    def main_dispatch(
        x: T.Tensor([Bs, H], "bfloat16"),   # Tokens to be dispatched
        expert_ids: T.Tensor([Bs, K], "int32"),     # Target MoE expert index
        win_data: T.Tensor([total_expert_num * Bs, ub_size], "bfloat16"),   # Shared memory space for receiving tokens sent from other ranks
        win_status: T.Tensor([total_expert_num, ub_float_int32_align], "float"),    # Shared memory space for receiving status sent from other ranks
        win_credit: T.Tensor([total_expert_num, ub_float_int32_align], "int32"),
        generation_id: T.Tensor([1], "int32"),
        iteration_id: T.Tensor([1], "int32"),
        expand_x_out: T.Tensor([ep_world_size * Bs * local_expert_num, H], "bfloat16"),     # Dispatch output: tokens received by this card
        expand_ids: T.Tensor([ep_world_size * Bs * local_expert_num, assist_size], "int32"),    # Dispatch output: Triplet information indicating the source of the current token
        global_prefix: T.Tensor([total_expert_num], "int32"),      # Internal global-expert prefix
        expert_token_nums_out: T.Tensor([local_expert_num], "int64"),   # Number of tokens received by each MOE expert of the current rank
        active_mask: T.Tensor([ep_world_size * Bs * local_expert_num], "int32"),
        actual_count: T.Tensor([1], "int32"),
        workspace: T.Tensor([aiv_num, ub_float_int32_align], "int32"),  # Global buffer for storing the prefix sum of tokens received by ranks
    ):
        with T.Kernel(aiv_num // 2, is_npu=True) as (cid, vid):     # Enable kernel logic, with the first parameter being the number of AI Cores
            # Allocate ub space
            x_ub = T.alloc_ub([ub_size], "bfloat16")
            x_ub_cast32 = T.alloc_ub([ub_size], "int32")
            x_win_ub = T.alloc_ub([H], "bfloat16")  # Local win area token -> ub
            expert_ids_ub = T.alloc_ub([Bs, 8], "int32")
            dst_expert_id_ub = T.alloc_ub([Bs* K], "int32") # Used for cal_token_send_expert_cnt, filling in the target MOE expert index
            sub_ub = T.alloc_ub([Bs* K], "int32")
            tmp_fp_32 = T.alloc_ub([Bs* K], "float")
            tmp_out_fp_32 = T.alloc_ub([Bs* K], "float")
            work_local_ub = T.alloc_ub([Bs* K], "float")
            win_status_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "int32")    # Store the status to be sent to the win status area of other ranks
            win_status_fp_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "float")
            status_sum_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "float")
            status_sum_int_ub = T.alloc_ub([status_per_core * ub_float_int32_align], "int32")
            status_sum_out = T.alloc_ub([status_per_core * ub_float_int32_align], "float")
            gather_mask_out_ub = T.alloc_ub([status_per_core], "float")
            receive_count_max_ub = T.alloc_ub([status_per_core], "int32")
            sum_local_ub = T.alloc_ub([aiv_num, ub_float_int32_align], "int32")
            sum_continue_ub = T.alloc_ub([aiv_num], "int32")
            sum_continue_fp_ub = T.alloc_ub([aiv_num], "float")
            win_status_ub_single = T.alloc_ub([ub_float_int32_align], "float")
            status_sum_on_core_ub = T.alloc_ub([ub_float_int32_align], "float")
            status_sum_on_core_int_ub = T.alloc_ub([ub_float_int32_align], "int32")
            gather_sum_pattern_ub = T.alloc_ub([ub_float_int32_align], "uint32")
            recv_cnt_sum_out_ub = T.alloc_ub([ub_float_int32_align], "float")
            out_count_ub = T.alloc_ub([ub_float_int32_align], "int32")
            status_local_data = T.alloc_ub([assist_size * 2], "bfloat16")
            tmp_triple = T.alloc_ub([assist_size], "int32")
            token_repeat_num = T.alloc_ub([1], "int32")
            cur_vid = T.alloc_ub([1], "int32")
            sum_of_flag = T.alloc_ub([1], "float")
            count = T.alloc_ub([1], "int32")
            win_data_offset = T.alloc_ub([1], "int32")
            state_reset_ub = T.alloc_ub([status_per_core, ub_float_int32_align], "float")
            gather_tmp_ub = T.alloc_ub([1], "uint32")
            begin_idx_ub = T.alloc_ub([1], "int32")
            state_reset_floor_ub = T.alloc_ub([floor_status_per_core, ub_float_int32_align], "float")
            receive_count_floor_ub = T.alloc_ub([floor_status_per_core], "int32")
            generation_ub = T.alloc_ub([1], "int32")
            iteration_ub = T.alloc_ub([1], "int32")
            credit_ub = T.alloc_ub([ub_float_int32_align], "int32")
            cur_vid[0] = (vid + 2 * cid)
            cur_send_token_cnt = Bs * K
            with T.Scope("C"):
                T.sync_all()       # The number of C cores involved in the sync_all synchronization must match the number of V cores.
                T.sync_all()
                T.sync_all()
            with T.Scope("V"):
                if cur_vid[0] == 0:
                    for mask_index in T.serial(ep_world_size * Bs * local_expert_num):
                        active_mask[mask_index] = 0
                T.sync_all()
                # Send data distributed across cores
                send_token_num = cur_send_token_cnt // aiv_num
                remainder_token_num = cur_send_token_cnt % aiv_num
                start_send_token_id = send_token_num * cur_vid[0]
                start_send_token_id = T.if_then_else(cur_vid[0] < remainder_token_num, start_send_token_id + cur_vid[0], start_send_token_id + remainder_token_num)
                send_token_num = T.if_then_else(cur_vid[0] < remainder_token_num, send_token_num + 1, send_token_num)
                T.tile.fill(state_reset_ub, 0.0)
                T.tile.fill(state_reset_floor_ub, 0.0)
                for token_index in T.serial(Bs):
                    T.copy(expert_ids[token_index, 0:K], expert_ids_ub[token_index, 0:8])
                T.copy(generation_id, generation_ub)
                T.copy(iteration_id, iteration_ub)
                sync_func("mte2", "s", 0)
                T.tile.fill(credit_ub, 0)
                credit_ub[0] = generation_ub[0]
                aiv_expert_num = total_expert_num // aiv_num
                remainder_expert_num = total_expert_num % aiv_num
                start_expert_id = aiv_expert_num * cur_vid[0]
                start_expert_id = T.if_then_else(cur_vid[0] < remainder_expert_num, start_expert_id + cur_vid[0], start_expert_id + remainder_expert_num)
                aiv_expert_num = T.if_then_else(cur_vid[0] < remainder_expert_num, aiv_expert_num + 1, aiv_expert_num)
                if generation_ub[0] > 1:
                    for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                        T.shmem_signal_wait_until(
                            win_credit,
                            cur_expert_id * ub_float_int32_align,
                            0,
                            generation_ub[0] - 1,
                        )
                T.barrier_all()
                if DEBUG_DUMP and cur_vid[0] == 0:
                    T.dump_tensor(expert_ids_ub, 400, Bs * K, (Bs * K,))
                token_repeat_num[0] = 0
                # Send data:AlltoAllDispatch
                for cur_send_token_id in range(start_send_token_id, start_send_token_id + send_token_num):
                    expert_id = expert_ids_ub[cur_send_token_id // K, cur_send_token_id % K]
                    cal_token_send_expert_cnt(expert_id, cur_send_token_id, dst_expert_id_ub, sub_ub, expert_ids_ub, tmp_fp_32, tmp_out_fp_32, work_local_ub)
                    token_repeat_num[0] = dst_expert_id_ub[0]
                    dest_rank_id = expert_id // local_expert_num
                    dest_expert_id = expert_id % local_expert_num
                    sync_func("s", "mte2", 1)
                    T.copy(x[cur_send_token_id // K, 0], x_ub)
                    # Calculate triple
                    token_in_topkid = cur_send_token_id % K
                    sync_func("mte2", "v", 2)
                    T.reinterpretcast(x_ub_cast32, x_ub, "int32_t")
                    sync_func("v", "s", 3)
                    x_ub_cast32[(H + 16) // 2] = rank
                    x_ub_cast32[(H + 16) // 2 + 1] = cur_send_token_id // K
                    x_ub_cast32[(H + 16) // 2 + 2] = token_in_topkid
                    sync_func("s", "mte3", 4)
                    data_offset = rank * Bs * local_expert_num + dest_expert_id * Bs + token_repeat_num[0]
                    if dest_rank_id == rank:
                        T.copy(x_ub, win_data[data_offset, 0])
                    else:
                        T.shmem_ub_put_nbi(x_ub, win_data, ub_size, dest_rank_id, data_offset * ub_size)     # Dispatch tokens
                        T.shmem_mte_quiet()
                # Send status:SetStatus
                # Status distributed across cores
                total_send_token_num = Bs * K
                for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                    cal_token_send_expert_cnt(cur_expert_id, total_send_token_num, dst_expert_id_ub, sub_ub, expert_ids_ub, tmp_fp_32, tmp_out_fp_32, work_local_ub)
                    cnt_pos_index = (cur_expert_id - start_expert_id) * 8
                    win_status_ub[cnt_pos_index + 1] = dst_expert_id_ub[0]
                    win_status_ub[cnt_pos_index] = generation_ub[0]
                    win_status_ub[cnt_pos_index + 2] = iteration_ub[0]
                T.barrier_all()
                T.sync_all()    # Ensure that all cores have completed sending data previously.
                T.reinterpretcast(win_status_fp_ub, win_status_ub, "float")
                T.barrier_all()
                for cur_expert_id in range(start_expert_id, start_expert_id + aiv_expert_num):
                    dest_rank_id = cur_expert_id // local_expert_num    # Target rank
                    local_expert_id = cur_expert_id % local_expert_num  # Target MOE expert
                    index = (cur_expert_id - start_expert_id) * 8
                    T.copy(win_status_fp_ub[index:index+8], win_status_ub_single)
                    status_offset = local_expert_id * ep_world_size * 8 + rank * 8
                    if dest_rank_id == rank:
                        T.copy(win_status_ub_single, win_status[local_expert_id * ep_world_size + rank, 0])
                    else:
                        T.shmem_ub_put_nbi(win_status_ub_single, win_status, 8, dest_rank_id, status_offset)
                        T.shmem_mte_quiet()
                    if DEBUG_DUMP:
                        T.printf("dispatch_status_sent rank=%d core=%d expert=%d dest=%d\n", rank, cur_vid[0], cur_expert_id, dest_rank_id)
                    sync_func("mte3", "s", 5)
                # Loop waiting for status WaitDispatch
                start_status_index = start_expert_id
                status_num_per_core = aiv_expert_num
                sync_func("mte3", "s", 6)
                if DEBUG_DUMP:
                    T.printf("dispatch_wait_start rank=%d core=%d status=%d\n", rank, cur_vid[0], start_status_index)
                for status_index in T.serial(status_num_per_core):
                    T.shmem_signal_wait_until(
                        win_status,
                        (start_status_index + status_index) * ub_float_int32_align,
                        0,
                        generation_ub[0],
                    )
                if DEBUG_DUMP:
                    T.printf("dispatch_wait_done rank=%d core=%d\n", rank, cur_vid[0])
                T.copy(win_status[start_status_index, 0], status_sum_ub)
                sync_func("mte2", "v", 7)
                T.reinterpretcast(status_sum_int_ub, status_sum_ub, "int32_t")
                if DEBUG_DUMP:
                    T.dump_tensor(status_sum_int_ub, 401, status_per_core * ub_float_int32_align, (status_per_core * ub_float_int32_align,))
                sync_func("v", "mte3", 9)
                # Clear status area
                if status_num_per_core > 0 and status_num_per_core == status_per_core:
                    T.copy(state_reset_ub, win_status[start_status_index, 0])
                elif status_num_per_core > 0 and status_num_per_core == status_per_core - 1:
                    T.copy(state_reset_floor_ub, win_status[start_status_index, 0])
                sync_func("mte3", "s", 10)
                if DEBUG_DUMP:
                    T.printf("dispatch_status_staged rank=%d core=%d\n", rank, cur_vid[0])
                # SyncCntOnCore: Computes the total token count for the current core, used by GetCumSum to compute the prefix sum.
                status_sum_on_core_int_ub[0] = 0
                for status_index in T.serial(status_num_per_core):
                    status_sum_on_core_int_ub[0] = (
                        status_sum_on_core_int_ub[0]
                        + status_sum_int_ub[status_index * ub_float_int32_align + 1]
                    )
                sync_func("v", "mte3", 11)
                T.copy(status_sum_on_core_int_ub, workspace[cur_vid[0], 0])
                T.barrier_all()
                T.sync_all()
                # GetCumSum: Prefix sum of token counts; used to compute copy offsets for local data.
                T.copy(workspace, sum_local_ub)
                T.barrier_all()
                out_count_ub[0] = 0
                for core_index in T.serial(cur_vid[0]):
                    out_count_ub[0] = (
                        out_count_ub[0]
                        + sum_local_ub[core_index, 0]
                    )
                if DEBUG_DUMP:
                    T.printf("dispatch_prefix_ready rank=%d core=%d\n", rank, cur_vid[0])
                begin_idx_ub[0] = out_count_ub[0]
                # Local data copy
                for i in range(status_num_per_core):
                    begin_idx = begin_idx_ub[0]
                    count = status_sum_int_ub[i * 8 + 1]
                    receive_count_max_ub[i] = begin_idx + count
                    if status_num_per_core == status_per_core - 1:
                        receive_count_floor_ub[i] = begin_idx + count
                    win_data_offset[0] = (i + start_status_index) % ep_world_size * (Bs * local_expert_num) + (i + start_status_index) // ep_world_size * Bs
                    for j in range(count):
                        T.copy(win_data[win_data_offset[0] + j, 0:H], x_win_ub)
                        # Decompose triple
                        T.copy(win_data[win_data_offset[0] + j, H+16:H+22], status_local_data)
                        sync_func("mte2", "v", 12)
                        T.reinterpretcast(tmp_triple, status_local_data, "int32_t")
                        sync_func("v", "mte3", 13)
                        T.copy(tmp_triple, expand_ids[begin_idx + j, 0])
                        T.copy(x_win_ub, expand_x_out[begin_idx + j, 0])
                        active_mask[begin_idx + j] = 1
                        if DEBUG_DUMP and begin_idx == 0 and j == 0:
                            T.dump_tensor(x_win_ub, 402, H, (H,))
                            T.dump_tensor(tmp_triple, 403, assist_size, (assist_size,))
                    source_rank = (i + start_status_index) % ep_world_size
                    local_expert_id = (i + start_status_index) // ep_world_size
                    credit_offset = (rank * local_expert_num + local_expert_id) * ub_float_int32_align
                    if source_rank == rank:
                        T.copy(credit_ub, win_credit[rank * local_expert_num + local_expert_id, 0])
                        sync_func("mte3", "s", 14)
                    else:
                        T.shmem_ub_put_nbi(credit_ub, win_credit, ub_float_int32_align, source_rank, credit_offset)
                        T.shmem_mte_quiet()
                    begin_idx_ub[0] = begin_idx + count # Update prefix sum to obtain output ep_receive_count
                T.tile.datacachecleanandinvalid_experiment(active_mask, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                # Obtain ep_receive_count
                if status_num_per_core > 0 and status_num_per_core == status_per_core:
                    T.copy(receive_count_max_ub, global_prefix[start_status_index])
                elif status_num_per_core > 0 and status_num_per_core == status_per_core - 1:
                    T.copy(receive_count_floor_ub, global_prefix[start_status_index])
                if DEBUG_DUMP:
                    T.printf("dispatch_copy_done rank=%d core=%d\n", rank, cur_vid[0])
    return main_dispatch


@tilelang.jit(pass_configs=pass_configs)
def dispatch_metadata_kernel(max_capacity, total_expert_num, ep_world_size, local_expert_num):
    """Build fixed metadata from the synchronized expert-major prefix."""
    @T.prim_func
    def main_metadata(
        active_mask: T.Tensor([max_capacity], "int32"),
        global_prefix: T.Tensor([total_expert_num], "int32"),
        actual_count: T.Tensor([1], "int32"),
        expert_token_nums: T.Tensor([local_expert_num], "int64"),
        ep_receive_count: T.Tensor([local_expert_num], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                if vid == 0:
                    total_count = global_prefix[total_expert_num - 1]
                    actual_count[0] = total_count
                    for index in T.serial(max_capacity):
                        active_mask[index] = index < total_count
                    for local_expert_id in T.serial(local_expert_num):
                        end_index = (local_expert_id + 1) * ep_world_size - 1
                        start_index = local_expert_id * ep_world_size - 1
                        if local_expert_id == 0:
                            expert_token_nums[local_expert_id] = global_prefix[end_index]
                        else:
                            expert_token_nums[local_expert_id] = global_prefix[end_index] - global_prefix[start_index]
                        ep_receive_count[local_expert_id] = expert_token_nums[local_expert_id]
                T.tile.datacachecleanandinvalid_experiment(actual_count, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                T.tile.datacachecleanandinvalid_experiment(active_mask, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                T.tile.datacachecleanandinvalid_experiment(expert_token_nums, "SINGLE_CACHE_LINE", "CACHELINE_OUT")
                T.tile.datacachecleanandinvalid_experiment(ep_receive_count, "SINGLE_CACHE_LINE", "CACHELINE_OUT")

    return main_metadata


@tilelang.jit(pass_configs=pass_configs)
def moe_dispatch_int8_kernel(Bs, H, physical_H, K, ep_world_size, local_expert_num, rank):
    """Route fixed INT8 payload, scale, and triplet rows with native status."""
    total_expert_num = ep_world_size * local_expert_num
    max_capacity = ep_world_size * Bs * local_expert_num

    @T.prim_func
    def main_dispatch_int8(
        payload: T.Tensor([Bs, physical_H], "int8"),
        scale: T.Tensor([Bs], "float32"),
        expert_ids: T.Tensor([Bs, K], "int32"),
        generation_id: T.Tensor([1], "int32"),
        iteration_id: T.Tensor([1], "int32"),
        win_payload: T.Tensor([max_capacity, physical_H], "int8"),
        win_scale: T.Tensor([max_capacity, 8], "float32"),
        win_triplet: T.Tensor([max_capacity, 8], "int32"),
        win_status: T.Tensor([total_expert_num, 8], "int32"),
        win_credit: T.Tensor([total_expert_num, 8], "int32"),
        expand_payload: T.Tensor([max_capacity, physical_H], "int8"),
        expand_scale: T.Tensor([max_capacity], "float32"),
        expand_ids: T.Tensor([max_capacity, 3], "int32"),
        global_prefix: T.Tensor([total_expert_num], "int32"),
        expert_token_nums: T.Tensor([local_expert_num], "int64"),
        ep_receive_count: T.Tensor([local_expert_num], "int32"),
        active_mask: T.Tensor([max_capacity], "int32"),
        actual_count: T.Tensor([1], "int32"),
        send_counts: T.Tensor([total_expert_num], "int32"),
        received_counts: T.Tensor([total_expert_num], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            payload_ub = T.alloc_ub([physical_H], "int8")
            scale_ub = T.alloc_ub([8], "float32")
            triplet_ub = T.alloc_ub([8], "int32")
            status_ub = T.alloc_ub([8], "int32")
            receive_status_ub = T.alloc_ub([8], "int32")
            zero_status_ub = T.alloc_ub([8], "int32")
            credit_ub = T.alloc_ub([8], "int32")
            expert_ids_ub = T.alloc_ub([Bs, 8], "int32")
            generation_ub = T.alloc_ub([1], "int32")
            iteration_ub = T.alloc_ub([1], "int32")
            occurrence_ub = T.alloc_ub([1], "int32")
            route_count_ub = T.alloc_ub([1], "int32")
            output_row_ub = T.alloc_ub([1], "int32")
            expert_start_ub = T.alloc_ub([1], "int32")

            with T.Scope("V"):
                if vid == 0:
                    if DEBUG_DUMP:
                        T.printf("int8_dispatch_start rank=%d\n", rank)
                    T.copy(generation_id, generation_ub)
                    T.copy(iteration_id, iteration_ub)
                    T.tile.fill(scale_ub, 0.0)
                    T.tile.fill(triplet_ub, 0)
                    T.tile.fill(status_ub, 0)
                    T.tile.fill(zero_status_ub, 0)
                    T.tile.fill(credit_ub, 0)
                    for token_id in T.serial(Bs):
                        T.copy(expert_ids[token_id, 0:K], expert_ids_ub[token_id, 0:8])
                    T.set_flag("mte2", "v", 3)
                    T.wait_flag("mte2", "v", 3)
                    for row in T.serial(max_capacity):
                        active_mask[row] = 0

                    if generation_ub[0] > 1:
                        for expert_id in T.serial(total_expert_num):
                            T.shmem_signal_wait_until(win_credit, expert_id * 8, 0, generation_ub[0] - 1)

                    for token_id in T.serial(Bs):
                        for topk_slot in T.serial(K):
                            expert_id = expert_ids_ub[token_id, topk_slot]
                            destination_rank = expert_id // local_expert_num
                            local_expert_id = expert_id % local_expert_num
                            occurrence_ub[0] = 0
                            for previous_token in T.serial(Bs):
                                if previous_token < token_id:
                                    for previous_slot in T.serial(K):
                                        if expert_ids_ub[previous_token, previous_slot] == expert_id:
                                            occurrence_ub[0] = occurrence_ub[0] + 1
                            for previous_slot in T.serial(K):
                                if previous_slot < topk_slot and expert_ids_ub[token_id, previous_slot] == expert_id:
                                    occurrence_ub[0] = occurrence_ub[0] + 1
                            remote_slot = rank * Bs * local_expert_num + local_expert_id * Bs + occurrence_ub[0]
                            T.copy(payload[token_id, 0], payload_ub)
                            scale_ub[0] = scale[token_id]
                            triplet_ub[0] = rank
                            triplet_ub[1] = token_id
                            triplet_ub[2] = topk_slot
                            if destination_rank == rank:
                                T.copy(payload_ub, win_payload[remote_slot, 0])
                                T.copy(scale_ub, win_scale[remote_slot, 0])
                                T.copy(triplet_ub, win_triplet[remote_slot, 0])
                                T.set_flag("mte3", "s", 4)
                                T.wait_flag("mte3", "s", 4)
                            else:
                                T.shmem_ub_put_nbi(payload_ub, win_payload, physical_H, destination_rank, remote_slot * physical_H)
                                T.shmem_mte_quiet()
                                T.shmem_ub_put_nbi(scale_ub, win_scale, 8, destination_rank, remote_slot * 8)
                                T.shmem_mte_quiet()
                                T.shmem_ub_put_nbi(triplet_ub, win_triplet, 8, destination_rank, remote_slot * 8)
                                T.shmem_mte_quiet()

                    if DEBUG_DUMP:
                        T.printf("int8_payload_sent rank=%d\n", rank)

                    for expert_id in T.serial(total_expert_num):
                        route_count_ub[0] = 0
                        for token_id in T.serial(Bs):
                            for topk_slot in T.serial(K):
                                if expert_ids_ub[token_id, topk_slot] == expert_id:
                                    route_count_ub[0] = route_count_ub[0] + 1
                        destination_rank = expert_id // local_expert_num
                        local_expert_id = expert_id % local_expert_num
                        status_row = local_expert_id * ep_world_size + rank
                        status_ub[0] = generation_ub[0]
                        status_ub[1] = route_count_ub[0]
                        status_ub[2] = iteration_ub[0]
                        send_counts[expert_id] = route_count_ub[0]
                        if destination_rank == rank:
                            T.copy(status_ub, win_status[status_row, 0])
                            T.set_flag("mte3", "s", 5)
                            T.wait_flag("mte3", "s", 5)
                        else:
                            T.shmem_ub_put_nbi(status_ub, win_status, 8, destination_rank, status_row * 8)
                            T.shmem_mte_quiet()

                    if DEBUG_DUMP:
                        T.printf("int8_status_sent rank=%d\n", rank)

                    output_row_ub[0] = 0
                    for local_expert_id in T.serial(local_expert_num):
                        expert_start_ub[0] = output_row_ub[0]
                        for source_rank in T.serial(ep_world_size):
                            status_row = local_expert_id * ep_world_size + source_rank
                            if DEBUG_DUMP:
                                T.printf("int8_wait_start rank=%d status=%d\n", rank, status_row)
                            T.shmem_signal_wait_until(win_status, status_row * 8, 0, generation_ub[0])
                            if DEBUG_DUMP:
                                T.printf("int8_wait_done rank=%d status=%d\n", rank, status_row)
                            T.copy(win_status[status_row, 0], receive_status_ub)
                            T.set_flag("mte2", "v", 6)
                            T.wait_flag("mte2", "v", 6)
                            route_count_ub[0] = receive_status_ub[1]
                            received_counts[status_row] = route_count_ub[0]
                            for route_index in T.serial(Bs):
                                if route_index < route_count_ub[0]:
                                    source_slot = source_rank * Bs * local_expert_num + local_expert_id * Bs + route_index
                                    T.copy(win_payload[source_slot, 0], payload_ub)
                                    T.copy(win_scale[source_slot, 0], scale_ub)
                                    T.copy(win_triplet[source_slot, 0], triplet_ub)
                                    T.set_flag("mte2", "v", 7)
                                    T.wait_flag("mte2", "v", 7)
                                    T.set_flag("v", "mte3", 9)
                                    T.wait_flag("v", "mte3", 9)
                                    T.copy(payload_ub, expand_payload[output_row_ub[0], 0])
                                    T.copy(scale_ub[0:1], expand_scale[output_row_ub[0] : output_row_ub[0] + 1])
                                    T.copy(triplet_ub[0:3], expand_ids[output_row_ub[0], 0:3])
                                    T.set_flag("mte3", "mte2", 10)
                                    T.wait_flag("mte3", "mte2", 10)
                                    active_mask[output_row_ub[0]] = 1
                                    output_row_ub[0] = output_row_ub[0] + 1
                            global_prefix[status_row] = output_row_ub[0]
                            T.copy(zero_status_ub, win_status[status_row, 0])
                            credit_ub[0] = generation_ub[0]
                            if source_rank == rank:
                                T.copy(credit_ub, win_credit[rank * local_expert_num + local_expert_id, 0])
                                T.set_flag("mte3", "s", 11)
                                T.wait_flag("mte3", "s", 11)
                            else:
                                T.shmem_ub_put_nbi(
                                    credit_ub,
                                    win_credit,
                                    8,
                                    source_rank,
                                    (rank * local_expert_num + local_expert_id) * 8,
                                )
                                T.shmem_mte_quiet()
                        route_count_ub[0] = output_row_ub[0] - expert_start_ub[0]
                        expert_token_nums[local_expert_id] = route_count_ub[0]
                        ep_receive_count[local_expert_id] = route_count_ub[0]
                    actual_count[0] = output_row_ub[0]
                    T.set_flag("mte3", "s", 8)
                    T.wait_flag("mte3", "s", 8)
                    T.tile.datacachecleanandinvalid_experiment(expand_payload, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(expand_scale, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(expand_ids, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(global_prefix, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(expert_token_nums, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(ep_receive_count, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(active_mask, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(actual_count, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(send_counts, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    T.tile.datacachecleanandinvalid_experiment(received_counts, "SINGLE_CACHE_LINE", "CACHELINE_ALL")
                    if DEBUG_DUMP:
                        T.printf("int8_dispatch_done rank=%d count=%d\n", rank, output_row_ub[0])

    return main_dispatch_int8


@tilelang.jit(pass_configs=pass_configs)
def dispatch_dequantize_int8_kernel(max_capacity, H, physical_H):
    """Dequantize fixed expert-major INT8 rows before the BF16 expert stage."""
    @T.prim_func
    def main_dequant(
        payload: T.Tensor([max_capacity, physical_H], "int8"),
        scale: T.Tensor([max_capacity], "float32"),
        active_mask: T.Tensor([max_capacity], "int32"),
        output: T.Tensor([max_capacity, H], "bfloat16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            payload_ub = T.alloc_ub([physical_H], "int8")
            payload_fp16_ub = T.alloc_ub([physical_H], "float16")
            payload_fp32_ub = T.alloc_ub([physical_H], "float32")
            output_ub = T.alloc_ub([physical_H], "bfloat16")
            with T.Scope("V"):
                if vid == 0:
                    if DEBUG_DUMP:
                        T.printf("int8_dequant_start\n")
                    for row in T.serial(max_capacity):
                        if active_mask[row] != 0:
                            T.copy(payload[row, 0], payload_ub)
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.cast(payload_fp16_ub, payload_ub, mode="CAST_NONE", count=physical_H)
                            T.pipe_barrier("v")
                            T.tile.cast(payload_fp32_ub, payload_fp16_ub, mode="CAST_NONE", count=physical_H)
                            T.pipe_barrier("v")
                            T.tile.mul(payload_fp32_ub, payload_fp32_ub, scale[row])
                            T.pipe_barrier("v")
                            T.tile.cast(output_ub, payload_fp32_ub, mode="CAST_RINT", count=physical_H)
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            T.copy(output_ub[0:H], output[row, 0:H])
                            T.set_flag("mte3", "mte2", 2)
                            T.wait_flag("mte3", "mte2", 2)
                    if DEBUG_DUMP:
                        T.printf("int8_dequant_done\n")

    return main_dequant


# --- 2. Implement the Combine operator ---
@tilelang.jit(pass_configs=pass_configs)
def moe_combine_kernel(
    Bs,
    send_token_cnt,
    H,
    K,
    ep_world_size,
    local_expert_num,
    rank,
    aiv_num,
):
    assist_size = 3
    token_per_core = (send_token_cnt + aiv_num - 1) // aiv_num
    float_align_ub = 8
    @T.prim_func
    def main_combine(
        expand_x: T.Tensor([send_token_cnt, H], "bfloat16"),    # Data processed by the MOE expert to be returned by the current rank
        assist_info_combine: T.Tensor([send_token_cnt, assist_size], "int32"),      # Triple information of returned tokens
        active_mask: T.Tensor([send_token_cnt], "int32"),
        generation_id: T.Tensor([1], "int32"),
        iteration_id: T.Tensor([1], "int32"),
        ep_send_counts: T.Tensor([local_expert_num], "int32"),
        expert_scales: T.Tensor([Bs, K], "float"),  # The weight coefficients for the MOE experts.
        win_data: T.Tensor([Bs * K, H], "bfloat16"),
        win_status: T.Tensor([Bs * K, float_align_ub], "float"),
        combine_out: T.Tensor([Bs, H], "bfloat16")
    ):
        with T.Kernel(aiv_num // 2, is_npu=True) as (cid, vid):
            # Allocate ub
            x_ub = T.alloc_ub([H], "bfloat16")
            status_ub_int = T.alloc_ub([float_align_ub], "int32")
            state_ub = T.alloc_ub([K * float_align_ub], "float")
            work_local_ub = T.alloc_ub([K * float_align_ub], "float")
            state_sum_out = T.alloc_ub([K * float_align_ub], "float")
            status_ub = T.alloc_ub([float_align_ub], "float")
            state_reset = T.alloc_ub([K * float_align_ub], "float")
            win_data_ub_bfloat = T.alloc_ub([H], "bfloat16")
            win_data_ub_float = T.alloc_ub([H], "float")
            combine_out_ub_float = T.alloc_ub([H], "float")
            combine_out_ub_bfloat = T.alloc_ub([H], "bfloat16")
            expert_scales_ub = T.alloc_ub([K], "float")
            cur_vid = T.alloc_ub([1], "int32")
            generation_ub = T.alloc_ub([1], "int32")
            iteration_ub = T.alloc_ub([1], "int32")
            sum_of_flag = T.alloc_ub([1], "float")
            cur_vid[0] = vid + 2 * cid
            # Returned tokens across cores
            send_token_num = send_token_cnt // aiv_num
            remainder_send_token_num = send_token_cnt % aiv_num
            start_send_token_id = send_token_num * cur_vid[0]
            start_send_token_id = T.if_then_else(cur_vid[0] < remainder_send_token_num, start_send_token_id + cur_vid[0], start_send_token_id + remainder_send_token_num)
            send_token_num = T.if_then_else(cur_vid[0] < remainder_send_token_num, send_token_num + 1, send_token_num)
            with T.Scope("V"):
                T.copy(generation_id, generation_ub)
                T.copy(iteration_id, iteration_ub)
                T.tile.fill(status_ub_int, 0)
                status_ub_int[0] = generation_ub[0]
                status_ub_int[1] = 1
                status_ub_int[2] = iteration_ub[0]
                T.reinterpretcast(status_ub, status_ub_int, "float")
                T.tile.fill(state_reset, 0.0)
                T.barrier_all()
                for loop in range(send_token_num):
                    tk_index = start_send_token_id + ((loop + rank) % send_token_num)
                    if active_mask[tk_index] != 0:
                        to_rank_id = assist_info_combine[tk_index, 0]
                        token_id = assist_info_combine[tk_index, 1]
                        topk_id = assist_info_combine[tk_index, 2]
                        T.copy(expand_x[tk_index, 0], x_ub)
                        T.barrier_all()
                        win_gm = token_id * K + topk_id
                        if to_rank_id == rank:
                            T.copy(x_ub, win_data[win_gm, 0])
                            T.set_flag("mte3", "s", 15)
                            T.wait_flag("mte3", "s", 15)
                            T.copy(status_ub, win_status[win_gm, 0])
                        else:
                            T.shmem_ub_put_nbi(x_ub, win_data, H, to_rank_id, win_gm * H)   # Return data
                            T.shmem_mte_quiet()
                            T.shmem_ub_put_nbi(status_ub, win_status, float_align_ub, to_rank_id, win_gm * float_align_ub)  # Return status
                            T.shmem_mte_quiet()
                        if DEBUG_DUMP:
                            T.printf("combine send rank=%d token=%d topk=%d to=%d\\n", rank, token_id, topk_id, to_rank_id)
                T.barrier_all()
                # Local tokens are distributed across cores with Bs.
                token_num = Bs // aiv_num
                remainder_token_num = Bs % aiv_num
                start_token_id = token_num * cur_vid[0]
                start_token_id = T.if_then_else(cur_vid[0] < remainder_token_num, start_token_id + cur_vid[0], start_token_id + remainder_token_num)
                token_num = T.if_then_else(cur_vid[0] < remainder_token_num, token_num + 1, token_num)
                # Loop processing combine returned data, from win to ub to global output.
                for cur_idx in range(start_token_id, start_token_id + token_num):
                    T.tile.fill(combine_out_ub_float, 0.0)
                    T.copy(expert_scales[cur_idx, 0], expert_scales_ub)
                    state_gm = cur_idx * K
                    for index in range(K):
                        if DEBUG_DUMP:
                            T.printf("combine wait rank=%d token=%d topk=%d\\n", rank, cur_idx, index)
                        T.shmem_signal_wait_until(
                            win_status,
                            (state_gm + index) * float_align_ub,
                            0,
                            generation_ub[0],
                        )
                    T.copy(state_reset, win_status[state_gm, 0])
                    T.set_flag("mte3", "s", 14)
                    T.wait_flag("mte3", "s", 14)
                    T.barrier_all()
                    # Compute a weighted sum of MoE-processed tokens returned from other cards.
                    for index in range(K):
                        token_index_offset = cur_idx * K + index
                        T.copy(win_data[token_index_offset, 0], win_data_ub_bfloat)
                        T.barrier_all()
                        T.tile.cast(win_data_ub_float, win_data_ub_bfloat, "CAST_NONE", H)
                        T.pipe_barrier("v")
                        T.tile.mul(win_data_ub_float, win_data_ub_float, expert_scales_ub[index])
                        T.pipe_barrier("v")
                        T.tile.add(combine_out_ub_float, combine_out_ub_float, win_data_ub_float)
                    T.barrier_all()
                    T.tile.cast(combine_out_ub_bfloat, combine_out_ub_float, "CAST_RINT", H)
                    T.barrier_all()
                    T.copy(combine_out_ub_bfloat, combine_out[cur_idx, 0])
    return main_combine

def worker(rank, barrier, x, expert_ids, aiv_num, ep_world_size, local_expert_num, Bs, mode, replays, dispatch_dtype, log_root, attempt_id):
    logger = AttemptLogger(Path(log_root), attempt_id)
    logger.rank_event(rank, "rank_started", argv=sys.argv)
    try:
        print(f"Rank {rank}: Setting device")
        torch.npu.set_device(rank)
        logger.rank_event(rank, "device_selected", device=rank)
        x = x.npu()
        expert_ids = expert_ids.npu()
        byte_bf16 = torch.tensor([], dtype=torch.float16).element_size()
        token_byte = (H * byte_bf16 + 31) // 32 * 32
        quant_byte = 32
        threeinfo_byte = 3 * 4
        ub_byte = (token_byte + quant_byte + threeinfo_byte + 511) // 512 * 512
        ub_size = ub_byte // 2
        physical_H = max(32, (H + 31) // 32 * 32)
        logger.rank_event(rank, "buffer_geometry", hidden=H, physical_hidden=physical_H, ub_size=ub_size, token_bytes=token_byte)

        ret = aclshmem_module.set_conf_store_tls(False, "")
        logger.rank_event(rank, "shmem_tls_configured", returncode=ret)
        if ret != 0:
            raise ValueError("[ERROR] set_conf_store_tls failed")
        attributes = aclshmem_module.InitAttr()
        npu_num = num_processes
        attributes.my_rank = rank
        attributes.n_ranks = npu_num
        attributes.local_mem_size = g_ash_size
        attributes.ip_port = G_IP_PORT
        attributes.option_attr.data_op_engine_type = aclshmem_module.OpEngineType.MTE
        ret = aclshmem_module.aclshmem_init(attributes)
        logger.rank_event(rank, "shmem_initialized", returncode=ret, world_size=npu_num)
        if ret == 0:
            print(f"Rank {rank}: Initialization successful")
            torch.manual_seed(0)
            max_capacity = ep_world_size * Bs * local_expert_num
            if dispatch_dtype == "int8":
                tensorData_dispatch = aclshmem_module.aclshmem_create_tensor([max_capacity, physical_H], dtype=torch.int8, device_id=rank)
                tensorScale_dispatch = aclshmem_module.aclshmem_create_tensor([max_capacity, 8], dtype=torch.float32, device_id=rank)
                tensorTriplet_dispatch = aclshmem_module.aclshmem_create_tensor([max_capacity, 8], dtype=torch.int32, device_id=rank)
                tensorStatus_dispatch = aclshmem_module.aclshmem_create_tensor([ep_world_size * local_expert_num, 8], dtype=torch.int32, device_id=rank)
                tensorCredit_dispatch = aclshmem_module.aclshmem_create_tensor([ep_world_size * local_expert_num, 8], dtype=torch.int32, device_id=rank)
            else:
                tensorData_dispatch = aclshmem_module.aclshmem_create_tensor([max_capacity, ub_size], dtype=torch.bfloat16, device_id=rank)
                tensorScale_dispatch = None
                tensorTriplet_dispatch = None
                tensorStatus_dispatch = aclshmem_module.aclshmem_create_tensor([ep_world_size * local_expert_num, 8], dtype=torch.float, device_id=rank)
                tensorCredit_dispatch = aclshmem_module.aclshmem_create_tensor([ep_world_size * local_expert_num, 8], dtype=torch.int32, device_id=rank)
            tensor_combine = aclshmem_module.aclshmem_create_tensor([Bs * K, H], dtype=torch.bfloat16, device_id = rank)
            tensorStatus_combine = aclshmem_module.aclshmem_create_tensor([Bs * K, 8], dtype=torch.float, device_id=rank)
            generation_id = torch.ones(size=[1], dtype=torch.int32).npu()
            iteration_id = torch.zeros(size=[1], dtype=torch.int32).npu()
            expand_x = torch.zeros((max_capacity, H), dtype=torch.bfloat16, device="npu")
            expand_idx = torch.zeros((max_capacity, 3), dtype=torch.int32, device="npu")
            quant_payload = torch.zeros((Bs, physical_H), dtype=torch.int8, device="npu")
            quant_scale = torch.zeros((Bs,), dtype=torch.float32, device="npu")
            expand_payload = torch.zeros((max_capacity, physical_H), dtype=torch.int8, device="npu")
            expand_scale = torch.zeros((max_capacity,), dtype=torch.float32, device="npu")
            int8_send_counts = torch.empty((ep_world_size * local_expert_num,), dtype=torch.int32, device="npu")
            int8_received_counts = torch.empty_like(int8_send_counts)
            global_prefix = torch.empty((ep_world_size * local_expert_num,), dtype=torch.int32, device="npu")
            ep_receive_count = torch.empty((local_expert_num,), dtype=torch.int32, device="npu")
            expert_token_nums = torch.empty((local_expert_num,), dtype=torch.int64, device="npu")
            active_mask = torch.empty((max_capacity,), dtype=torch.int32, device="npu")
            actual_count = torch.empty((1,), dtype=torch.int32, device="npu")
            workspace = torch.empty((aiv_num, 8), dtype=torch.int32, device="npu")
            x_out = torch.empty((Bs, H), dtype=torch.bfloat16, device="npu")
            expert_output = torch.empty_like(expand_x)
            expert_scales = torch.empty(size=[Bs, K], dtype=torch.float32).uniform_(-1, 1).npu()
            logger.rank_event(rank, "windows_allocated", dispatch_rows=max_capacity, combine_rows=Bs * K, dispatch_dtype=dispatch_dtype)
            win_dispatch = tensorData_dispatch.fill_(0)
            win_status_dispatch = tensorStatus_dispatch.fill_(0)
            win_scale_dispatch = tensorScale_dispatch.fill_(0) if tensorScale_dispatch is not None else None
            win_triplet_dispatch = tensorTriplet_dispatch.fill_(0) if tensorTriplet_dispatch is not None else None
            win_credit_dispatch = tensorCredit_dispatch.fill_(0) if tensorCredit_dispatch is not None else None
            win_combine = tensor_combine.fill_(0)
            win_status_combine = tensorStatus_combine.fill_(0)
            torch.npu.synchronize()
            logger.rank_event(rank, "host_barrier", phase="pre_dispatch")
            barrier.wait()
            if dispatch_dtype == "int8":
                quant_kernel = dispatch_quantize_int8_fixed_kernel(Bs, H, physical_H)
                func_dispatch = moe_dispatch_int8_kernel(Bs, H, physical_H, K, ep_world_size, local_expert_num, rank)
                dequant_kernel = dispatch_dequantize_int8_kernel(max_capacity, H, physical_H)
                metadata_kernel = None
            else:
                quant_kernel = None
                dequant_kernel = None
                func_dispatch = moe_dispatch_kernel(Bs, H, K, ep_world_size, local_expert_num, rank, ub_size, aiv_num)
                metadata_kernel = dispatch_metadata_kernel(max_capacity, ep_world_size * local_expert_num, ep_world_size, local_expert_num)
            func_combine = moe_combine_kernel(Bs, max_capacity, H, K, ep_world_size, local_expert_num, rank, aiv_num)
            buffer_addresses = {
                "x": x.data_ptr(),
                "expert_ids": expert_ids.data_ptr(),
                "expand_x": expand_x.data_ptr(),
                "expand_ids": expand_idx.data_ptr(),
                "global_prefix": global_prefix.data_ptr(),
                "ep_receive_count": ep_receive_count.data_ptr(),
                "active_mask": active_mask.data_ptr(),
                "actual_count": actual_count.data_ptr(),
                "expert_output": expert_output.data_ptr(),
                "combine_out": x_out.data_ptr(),
            }
            if dispatch_dtype == "int8":
                buffer_addresses.update({
                    "quant_payload": quant_payload.data_ptr(),
                    "quant_scale": quant_scale.data_ptr(),
                    "expand_payload": expand_payload.data_ptr(),
                    "expand_scale": expand_scale.data_ptr(),
                    "int8_send_counts": int8_send_counts.data_ptr(),
                    "int8_received_counts": int8_received_counts.data_ptr(),
                    "int8_credit": win_credit_dispatch.data_ptr(),
                })
            logger.rank_event(
                rank,
                "kernels_compiled",
                mode=mode,
                dispatch_dtype=dispatch_dtype,
                buffer_addresses=buffer_addresses,
            )

            def launch_int8_dispatch():
                func_dispatch(
                    quant_payload,
                    quant_scale,
                    expert_ids,
                    generation_id,
                    iteration_id,
                    win_dispatch,
                    win_scale_dispatch,
                    win_triplet_dispatch,
                    win_status_dispatch,
                    win_credit_dispatch,
                    expand_payload,
                    expand_scale,
                    expand_idx,
                    global_prefix,
                    expert_token_nums,
                    ep_receive_count,
                    active_mask,
                    actual_count,
                    int8_send_counts,
                    int8_received_counts,
                )

            def launch_combine():
                expert_output.copy_(expand_x)
                func_combine(
                    expert_output,
                    expand_idx,
                    active_mask,
                    generation_id,
                    iteration_id,
                    ep_receive_count,
                    expert_scales,
                    win_combine,
                    win_status_combine,
                    x_out,
                )

            def launch_full_path():
                if dispatch_dtype == "int8":
                    quant_kernel(x, quant_payload, quant_scale)
                    launch_int8_dispatch()
                    dequant_kernel(expand_payload, expand_scale, active_mask, expand_x)
                else:
                    func_dispatch(
                        x, expert_ids, win_dispatch, win_status_dispatch, win_credit_dispatch, generation_id, iteration_id,
                        expand_x, expand_idx, global_prefix, expert_token_nums, active_mask, actual_count, workspace,
                    )
                    metadata_kernel(active_mask, global_prefix, actual_count, expert_token_nums, ep_receive_count)
                launch_combine()

            if mode == "graph":
                graph = torch.npu.NPUGraph()
                logger.rank_event(rank, "graph_capture_started", generation=1)
                with torch.npu.graph(graph):
                    launch_full_path()
                torch.npu.synchronize()
                logger.rank_event(
                    rank,
                    "graph_capture_finished",
                    credit_words=win_credit_dispatch.cpu().tolist() if dispatch_dtype == "int8" else None,
                )
                barrier.wait()
                for replay in range(replays):
                    generation_id.fill_(replay + 1)
                    iteration_id.fill_(replay)
                    graph.replay()
                    torch.npu.synchronize()
                    logger.rank_event(
                        rank,
                        "graph_replay_finished",
                        replay=replay + 1,
                        generation=replay + 1,
                        actual_count=int(actual_count.cpu().item()),
                        mask_digest=tensor_digest(active_mask),
                        output_digest=tensor_digest(x_out),
                        timeout=False,
                    )
            else:
                for replay in range(replays):
                    generation_id.fill_(replay + 1)
                    iteration_id.fill_(replay)
                    logger.rank_event(rank, "eager_launch_started", replay=replay + 1, generation=replay + 1)
                    if dispatch_dtype == "int8":
                        quant_kernel(x, quant_payload, quant_scale)
                        torch.npu.synchronize()
                        logger.rank_event(
                            rank,
                            "int8_quant_finished",
                            payload_digest=tensor_digest(quant_payload),
                            scale_digest=tensor_digest(quant_scale),
                        )
                        launch_int8_dispatch()
                        torch.npu.synchronize()
                        route_count = int(actual_count.cpu().item())
                        logger.rank_event(
                            rank,
                            "int8_route_finished",
                            actual_count=route_count,
                            status_words=win_status_dispatch.cpu().tolist(),
                            credit_words=win_credit_dispatch.cpu().tolist(),
                            window_payload_head=win_dispatch[: min(max_capacity, 8), 0].cpu().tolist(),
                            expand_payload_head=expand_payload[: min(max_capacity, 16), 0].cpu().tolist(),
                            expand_scale=expand_scale.cpu().tolist(),
                            active_mask=active_mask.cpu().tolist(),
                            expand_ids=expand_idx.cpu().tolist(),
                            send_counts=int8_send_counts.cpu().tolist(),
                            received_counts=int8_received_counts.cpu().tolist(),
                        )
                        dequant_kernel(expand_payload, expand_scale, active_mask, expand_x)
                        torch.npu.synchronize()
                        logger.rank_event(rank, "int8_dequant_finished", output_digest=tensor_digest(expand_x))
                        launch_combine()
                    else:
                        launch_full_path()
                    torch.npu.synchronize()
                    logger.rank_event(
                        rank,
                        "eager_launch_finished",
                        replay=replay + 1,
                        generation=replay + 1,
                        actual_count=int(actual_count.cpu().item()),
                        output_digest=tensor_digest(x_out),
                        timeout=False,
                    )
                    barrier.wait()

            dispatch_count = actual_count.item()
            logger.rank_event(
                rank,
                "dispatch_finished",
                actual_count=dispatch_count,
                count_digest=tensor_digest(global_prefix),
                mask_digest=tensor_digest(active_mask),
                expand_ids_head=expand_idx[: min(max_capacity, 16)].cpu().tolist(),
                expand_x_head=expand_x[: min(max_capacity, 16), 0].cpu().tolist(),
                global_prefix=global_prefix.cpu().tolist(),
                ep_receive_count=ep_receive_count.cpu().tolist(),
                expert_token_nums=expert_token_nums.cpu().tolist(),
                payload_digest=tensor_digest(expand_payload) if dispatch_dtype == "int8" else None,
                dispatch_quant_scale_digest=tensor_digest(expand_scale) if dispatch_dtype == "int8" else None,
            )
            if dispatch_dtype == "int8":
                ids_cpu = expand_idx[:dispatch_count].cpu()
                expected_values = torch.tensor(
                    [int(source_rank) * Bs + int(token_id) + 1 for source_rank, token_id, _ in ids_cpu],
                    dtype=torch.float32,
                )
                expected_payload = torch.zeros((dispatch_count, physical_H), dtype=torch.int8)
                expected_payload[:, :H] = 127
                expected_scale = expected_values / 127.0
                expected_dequant = expected_values[:, None].expand(dispatch_count, H).to(torch.bfloat16)
                torch.testing.assert_close(expand_payload[:dispatch_count].cpu(), expected_payload, rtol=0.0, atol=0.0)
                torch.testing.assert_close(expand_scale[:dispatch_count].cpu(), expected_scale, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(expand_x[:dispatch_count].cpu(), expected_dequant, rtol=0.0, atol=0.0)
                logger.rank_event(
                    rank,
                    "int8_route_binding_pass",
                    max_abs_error=(
                        float((expand_x[:dispatch_count].cpu().float() - expected_dequant.float()).abs().max())
                        if dispatch_count > 0
                        else 0.0
                    ),
                    payload_digest=tensor_digest(expand_payload[:dispatch_count]),
                    scale_digest=tensor_digest(expand_scale[:dispatch_count]),
                )
            logger.rank_event(
                rank,
                "expert_identity_finished",
                output_digest=tensor_digest(expert_output),
                output_head=expert_output[: min(max_capacity, 16), 0].cpu().tolist(),
            )
            logger.rank_event(
                rank,
                "combine_finished",
                output_digest=tensor_digest(x_out),
                window_head=win_combine[:, 0].cpu().tolist(),
                status_words=win_status_combine[:, 0].view(torch.int32).cpu().tolist(),
            )
            barrier.wait()
            logger.rank_event(rank, "host_barrier", phase="pre_free")
            aclshmem_module.aclshmem_free_tensor(tensorData_dispatch)
            aclshmem_module.aclshmem_free_tensor(tensorStatus_dispatch)
            if tensorScale_dispatch is not None:
                aclshmem_module.aclshmem_free_tensor(tensorScale_dispatch)
            if tensorTriplet_dispatch is not None:
                aclshmem_module.aclshmem_free_tensor(tensorTriplet_dispatch)
            if tensorCredit_dispatch is not None:
                aclshmem_module.aclshmem_free_tensor(tensorCredit_dispatch)
            aclshmem_module.aclshmem_free_tensor(tensor_combine)
            aclshmem_module.aclshmem_free_tensor(tensorStatus_combine)
            logger.rank_event(rank, "windows_freed")
        else:
            print(f"Rank {rank}: Initialization failed with code {ret}")
        aclshmem_module.aclshmem_finialize()
        logger.rank_event(rank, "shmem_finalized")
        print(f"Rank {rank}: Finalization")
        x_f32 = x.to(torch.float32)
        weight_sum_f32 = (x_f32.unsqueeze(1) * expert_scales.unsqueeze(-1)).sum(dim=1)
        dispatch_combine_golden = weight_sum_f32.to(torch.bfloat16)
        logger.rank_event(
            rank,
            "golden_observed",
            output_head=x_out[:, 0].cpu().tolist(),
            golden_head=dispatch_combine_golden[:, 0].cpu().tolist(),
            scales=expert_scales.cpu().tolist(),
        )
        torch.testing.assert_close(x_out, dispatch_combine_golden, rtol=1e-2, atol=1e-2)
        logger.rank_event(rank, "golden_pass")
        print("Kernel Output Match!")
    except Exception as exc:
        logger.rank_event(rank, "rank_exception", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        logger.rank_event(rank, "rank_exited")

# Construct input
def init_input(rank, Bs, H, K, ep_world_size, local_expert_num, route_profile="random"):
    start = rank * Bs + 1
    end = start + Bs
    x = torch.tensor([[i] for i in range(start, end)], dtype=torch.bfloat16)
    x = x.repeat(1, H)
    global_expert_num = ep_world_size * local_expert_num
    if route_profile in ("self", "remote") and K > local_expert_num:
        raise ValueError(f"route profile {route_profile} requires topk <= local experts")
    if route_profile == "empty-expert" and K >= global_expert_num:
        raise ValueError("empty-expert profile requires topk < global expert count")
    if route_profile == "random":
        seed = 1
        random.seed(seed)
        expert_ids_list = []
        for _ in range(Bs):
            full_range = list(range(global_expert_num))
            random.shuffle(full_range)
            expert_ids_list.append(full_range[:K])
    elif route_profile == "uniform":
        expert_ids_list = [[(token_id + slot) % global_expert_num for slot in range(K)] for token_id in range(Bs)]
    elif route_profile == "hotspot":
        expert_ids_list = [list(range(K)) for _ in range(Bs)]
    elif route_profile == "empty-expert":
        expert_ids_list = [list(range(1, K + 1)) for _ in range(Bs)]
    elif route_profile == "self":
        expert_base = rank * local_expert_num
        expert_ids_list = [[expert_base + slot for slot in range(K)] for _ in range(Bs)]
    elif route_profile == "remote":
        expert_base = ((rank + 1) % ep_world_size) * local_expert_num
        expert_ids_list = [[expert_base + slot for slot in range(K)] for _ in range(Bs)]
    else:
        raise ValueError(f"unknown route profile: {route_profile}")
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32)
    return x, expert_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fixed-capacity BF16 SHMEM Dispatch/Combine")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--dispatch-dtype", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--ep-world-size", type=int, default=16)
    parser.add_argument("--local-experts", type=int, default=3)
    parser.add_argument("--aiv-num", type=int, default=48)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--ip-port", default=G_IP_PORT)
    parser.add_argument("--log-base", default="artifacts/moe")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--route-profile",
        choices=("random", "uniform", "hotspot", "empty-expert", "self", "remote"),
        default="random",
    )
    args = parser.parse_args()
    Bs = args.tokens
    H = args.hidden
    K = args.topk
    ep_world_size = args.ep_world_size
    local_expert_num = args.local_experts
    aiv_num = args.aiv_num
    num_processes = args.num_processes or ep_world_size
    max_capacity = ep_world_size * Bs * local_expert_num
    if args.replays < 1:
        raise ValueError("--replays must be positive")
    if K > 8:
        raise ValueError("--topk must be at most 8 for the fixed metadata tile")
    if K > ep_world_size * local_expert_num:
        raise ValueError("--topk cannot exceed the number of global experts")
    if num_processes != ep_world_size:
        raise ValueError("--num-processes must match --ep-world-size for SHMEM rank identity")
    G_IP_PORT = args.ip_port
    attempt = AttemptLogger.create(args.log_base)
    attempt.event(
        "run_configuration",
        argv=sys.argv,
        Bs=Bs,
        H=H,
        K=K,
        ep_world_size=ep_world_size,
        local_expert_num=local_expert_num,
        aiv_num=aiv_num,
        num_processes=num_processes,
        max_capacity=max_capacity,
        capacity_policy="one route per token/expert; reject duplicate top-k experts and topk > global experts",
        route_profile=args.route_profile,
        mode=args.mode,
        dispatch_dtype=args.dispatch_dtype,
        replays=args.replays,
        ip_port=G_IP_PORT,
        timeout_seconds=args.timeout,
    )
    barrier = Barrier(num_processes, timeout=args.timeout)
    processes = []
    for rank in range(num_processes):
        x, expert_ids = init_input(rank, Bs, H, K, ep_world_size, local_expert_num, args.route_profile)
        validate_router_expert_ids(
            expert_ids,
            ep_world_size * local_expert_num,
            topk=K,
        )
        p = mp.Process(
            target=worker,
            args=(
                rank,
                barrier,
                x,
                expert_ids,
                aiv_num,
                ep_world_size,
                local_expert_num,
                Bs,
                args.mode,
                args.replays,
                args.dispatch_dtype,
                attempt.root,
                attempt.attempt_id,
            ),
        )
        p.start()
        processes.append(p)
    deadline = time.monotonic() + args.timeout
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        for process in processes:
            if process.is_alive():
                process.join(0.1)
    for rank, process in enumerate(processes):
        if process.is_alive():
            attempt.event("rank_timeout", rank=rank, pid=process.pid, timeout_seconds=args.timeout)
            process.terminate()
            process.join()
            attempt.event("rank_terminated", rank=rank, pid=process.pid)
    attempt.event("rank_exit_codes", exitcodes={str(rank): process.exitcode for rank, process in enumerate(processes)})
    attempt.event("run_finished", success=all(process.exitcode == 0 for process in processes))
    print("All processes completed")
