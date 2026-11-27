"""Two-CTA GC task handoff and the four task phase identifiers.

The 32-byte task lives in each CTA's SMEM. The scheduler warp in the leader
CTA publishes both copies with transaction-counted DSM stores; the two
epilogue warpgroups per CTA retire a slot through the leader empty barrier.
"""

from __future__ import annotations

from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Uint64

from ..common.math import warp_inclusive_sum
from ..comm.gc_barrier import wait_route_count_doorbell
from ..ptx.ld_st import (
    ld_volatile_u32,
    ld_volatile_u64,
    ld_acquire_gpu_u32,
    ld_acquire_sys_u64,
    atom_add_u32,
    st_async_cluster_u32x4,
    st_shared_u32x4,
)
from ..ptx.utils import (
    elect_one_sync,
    warp_first_set_bit,
    warp_nth_set_bit,
)


class GCBlockPhase(IntEnum):
    None_ = 0
    Linear1 = 1
    Linear2 = 2
    SharedLinear1 = 3
    SharedLinear2 = 4


TASK_WORDS = 8
TASK_BYTES = 32
SCHEDULE_STAGES = 2
TASK_EMPTY_CONSUMERS = 512


@cute.jit
def publish_task32(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                   stage: Int32, peer_cta: Int32,
                   block_phase: Uint32, local_expert: Uint32,
                   m_block: Uint32, n_cluster: Uint32,
                   pool_block: Uint32, valid_m: Uint32,
                   shape_n: Uint32, shape_k: Uint32) -> None:
    cute.arch.mbarrier_arrive_and_expect_tx(full_ptr + stage, TASK_BYTES, peer_cta)
    base = stage * TASK_WORDS
    st_async_cluster_u32x4(task_ptr, full_ptr + stage, peer_cta, base,
                  block_phase, local_expert, m_block, n_cluster)
    st_async_cluster_u32x4(task_ptr, full_ptr + stage, peer_cta, base + 4,
                  pool_block, valid_m, shape_n, shape_k)


@cute.jit
def release_task_info(empty_ptr: cute.Pointer, stage: Int32,
                      scheduler_cta: cutlass.Constexpr = 0,
                      local_tasks: cutlass.Constexpr = False) -> None:
    cute.arch.fence_acq_rel_cta()
    if cutlass.const_expr(local_tasks):
        cute.arch.mbarrier_arrive(empty_ptr + stage)
    else:
        cute.arch.mbarrier_arrive(empty_ptr + stage, peer_cta_rank_in_cluster=scheduler_cta)


@cute.jit
def claim_task(task_count_ptr: cute.Pointer, lane: Int32) -> Uint32:
    task_idx = Uint32(0)
    if elect_one_sync():
        task_idx = atom_add_u32(task_count_ptr, Uint32(1))
    return cute.arch.shuffle_sync(task_idx, 0)


@cute.jit
def wait_ready_tile(queue: cute.Pointer, ticket: Uint32, lane: Int32) -> Uint32:
    encoded = Uint32(0)
    if lane == Int32(0):
        while encoded == Uint32(0):
            encoded = ld_acquire_gpu_u32(queue + ticket)
    cute.arch.sync_warp()
    return cute.arch.shuffle_sync(encoded, 0) - Uint32(1)


@cute.jit
def wait_task_slot(empty_ptr: cute.Pointer, iteration: Uint32,
                   task_slots: cutlass.Constexpr = 2) -> None:
    """Wait for the next task slot before entering its claim loop."""
    stage = Int32(iteration % task_slots)
    parity = Int32((iteration // task_slots) % 2)
    cute.arch.mbarrier_wait(empty_ptr + stage, parity ^ 1)


@cute.jit
def publish_iteration(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                      iteration: Uint32,
                      lane: Int32, block_phase: Uint32,
                      local_expert: Uint32,
                      m_block: Uint32, n_cluster: Uint32,
                      pool_block: Uint32, valid_m: Uint32,
                      shape_n: Uint32, shape_k: Uint32,
                      local_tasks: cutlass.Constexpr = False,
                      task_slots: cutlass.Constexpr = 2) -> None:
    stage = Int32(iteration % task_slots)
    if cutlass.const_expr(local_tasks):
        if lane == Int32(0):
            dst = task_ptr + stage * TASK_WORDS
            st_shared_u32x4(dst, block_phase, local_expert, m_block, n_cluster)
            st_shared_u32x4(dst + 4, pool_block, valid_m, shape_n, shape_k)
            cute.arch.mbarrier_arrive(full_ptr + stage)
    else:
        if lane < 2:
            publish_task32(task_ptr, full_ptr, stage, lane,
                           block_phase, local_expert, m_block, n_cluster,
                           pool_block, valid_m, shape_n, shape_k)
    cute.arch.sync_warp()


@cute.jit
def publish_shared_phase(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                         empty_ptr: cute.Pointer, counter_ptr: cute.Pointer,
                         iteration: Uint32, lane: Int32,
                         cluster_idx: Uint32,
                         phase: Uint32, num_tokens: Uint32,
                         block_m: cutlass.Constexpr,
                         n_clusters: cutlass.Constexpr,
                         num_compute_sms: cutlass.Constexpr,
                         shape_n: cutlass.Constexpr,
                         shape_k: cutlass.Constexpr,
                         local_tasks: cutlass.Constexpr = False,
                         task_slots: cutlass.Constexpr = 2) -> Uint32:
    """Reference shared_mainloop: static one task/cluster, else global claims."""
    m_blocks = (num_tokens + Uint32(block_m - 1)) // Uint32(block_m)
    total = m_blocks * Uint32(n_clusters)
    if total <= Uint32(num_compute_sms // 2):
        wait_task_slot(empty_ptr, iteration, task_slots)
        if cluster_idx < Uint32(total):
            m_block = cluster_idx // Uint32(n_clusters)
            n_cluster = cluster_idx % Uint32(n_clusters)
            valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
            publish_iteration(task_ptr, full_ptr, iteration,
                              lane, phase, Uint32(0), m_block, n_cluster, m_block,
                              valid_m, Uint32(shape_n), Uint32(shape_k), local_tasks, task_slots)
            iteration += Uint32(1)
            # Match the reference's wait before the static path's failed claim.
            wait_task_slot(empty_ptr, iteration, task_slots)
    else:
        scheduling = True
        while scheduling:
            wait_task_slot(empty_ptr, iteration, task_slots)
            task_idx = claim_task(counter_ptr, lane)
            if task_idx >= Uint32(total):
                scheduling = False
            else:
                m_block = task_idx // Uint32(n_clusters)
                n_cluster = task_idx % Uint32(n_clusters)
                valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
                publish_iteration(task_ptr, full_ptr, iteration,
                                  lane, phase, Uint32(0), m_block, n_cluster, m_block,
                                  valid_m, Uint32(shape_n), Uint32(shape_k), local_tasks, task_slots)
                iteration += Uint32(1)
    return iteration


@cute.jit
def publish_one_decode_shared_w2(
        task: cute.Pointer, full: cute.Pointer, counter: cute.Pointer, ready: cute.Pointer,
        iteration: Uint32, lane: Int32, num_tokens: Uint32,
        l1_clusters: cutlass.Constexpr, l2_clusters: cutlass.Constexpr,
        hidden: cutlass.Constexpr, intermediate: cutlass.Constexpr,
        require_ready: cutlass.Constexpr, task_slots: cutlass.Constexpr = 2) -> tuple[Uint32, Uint32]:
    """Claim a shared output tile from the same workers that claim routed work."""
    ticket = Uint32(l2_clusters)
    if lane == Int32(0):
        if ld_volatile_u32(counter) < Uint32(l2_clusters):
            may_claim = True
            if cutlass.const_expr(require_ready):
                may_claim = ld_acquire_gpu_u32(ready) == Uint32(2 * l1_clusters)
            if may_claim:
                ticket = Uint32(atom_add_u32(counter, Uint32(1)))
    ticket = cute.arch.shuffle_sync(ticket, 0)
    valid = Uint32(ticket < Uint32(l2_clusters))
    if valid != Uint32(0):
        publish_iteration(
            task, full, iteration, lane, Uint32(GCBlockPhase.SharedLinear2),
            Uint32(0), Uint32(0), ticket, Uint32(0), num_tokens,
            Uint32(hidden), Uint32(intermediate), task_slots=task_slots)
        iteration += Uint32(1)
    return iteration, valid


@cute.jit
def physical_expert(slot: Uint32, reverse: Uint32, count: cutlass.Constexpr) -> Uint32:
    """Both workers visit the ring in order, but may reverse the expert IDs."""
    expert = Uint32(slot)
    if reverse != Uint32(0):
        expert = Uint32(Uint32(count - 1) - expert)
    return expert


@cute.jit
def locate_routed_pool_block(counts: cute.Tensor,
                             pool_block: Uint32, lane: Uint32,
                             local_experts: cutlass.Constexpr,
                             block_m: cutlass.Constexpr,
                             cached_starts=None, cached_blocks=None):
    """Reference GCMoEScheduler::create_task from per-lane count registers."""
    expert = Uint32(0)
    m_block = Uint32(0)
    valid_m = Uint32(0)
    block_offset = Uint32(0)
    for wave in cutlass.range_constexpr((local_experts + 31) // 32):
        expert_idx = Uint32(wave * 32) + lane
        received = counts[wave]
        if cutlass.const_expr(cached_starts is None):
            blocks = (received + Uint32(block_m - 1)) // Uint32(block_m)
            inclusive = warp_inclusive_sum(blocks, lane)
            lane_offset = block_offset + inclusive - blocks
            block_offset += cute.arch.shuffle_sync(inclusive, 31)
        else:
            blocks = cached_blocks[wave]
            lane_offset = cached_starts[wave]
        is_owner = expert_idx < Uint32(local_experts) and \
            pool_block >= lane_offset and pool_block < lane_offset + blocks
        owner_mask = Uint32(cute.arch.vote_ballot_sync(is_owner))
        if owner_mask != Uint32(0):
            owner_lane = warp_first_set_bit(owner_mask)
            local_m = pool_block - lane_offset
            local_valid = min(Uint32(block_m),
                              received - local_m * Uint32(block_m))
            expert = cute.arch.shuffle_sync(expert_idx, owner_lane)
            m_block = cute.arch.shuffle_sync(local_m, owner_lane)
            valid_m = cute.arch.shuffle_sync(local_valid, owner_lane)
    return expert, m_block, valid_m


@cute.jit
def locate_decode_pool_block(counts: cute.Tensor, pool_block: Uint32, lane: Uint32,
                              local_experts: cutlass.Constexpr, block_m: cutlass.Constexpr,
                              cached_starts, cached_blocks, single_tile_mask):
    """When each active expert owns one tile, the pool index is its ballot rank."""
    if cutlass.const_expr(single_tile_mask is None):
        return locate_routed_pool_block(counts, pool_block, lane, local_experts, block_m,
                                        cached_starts, cached_blocks)
    expert, m_block, valid_m = Uint32(0), Uint32(0), Uint32(0)
    if single_tile_mask != Uint32(0):
        expert = warp_nth_set_bit(single_tile_mask, pool_block + Uint32(1))
        valid_m = cute.arch.shuffle_sync(counts[0], expert)
    else:
        expert, m_block, valid_m = locate_routed_pool_block(
            counts, pool_block, lane, local_experts, block_m, cached_starts, cached_blocks)
    return expert, m_block, valid_m


@cute.jit
def cached_num_tokens(counts: cute.Tensor, expert: Int32,
                      lane: Uint32,
                      local_experts: cutlass.Constexpr) -> Uint32:
    """GCMoEScheduler::get_num_tokens from expert wave*32+lane registers."""
    selected = Uint32(0)
    for wave in cutlass.range_constexpr((local_experts + 31) // 32):
        if Uint32(expert) == Uint32(wave * 32) + lane:
            selected = counts[wave]
    return cute.arch.shuffle_sync(selected, Uint32(expert) % Uint32(32))


@cute.jit
def advance_expert(counts: cute.Tensor, token_idx: Uint32,
                   current_expert: Int32, expert_start: Uint32,
                   expert_end: Uint32, pool_block_offset: Uint32,
                   lane: Uint32, local_experts: cutlass.Constexpr,
                   block_m: cutlass.Constexpr):
    """Advance the unpadded token interval and its padded pool prefix."""
    while token_idx >= expert_end and current_expert < Int32(local_experts):
        current_expert += Int32(1)
        if current_expert < Int32(local_experts):
            pool_block_offset += (expert_end - expert_start +
                                  Uint32(block_m - 1)) // Uint32(block_m)
            expert_start = expert_end
            expert_end += cached_num_tokens(
                counts, current_expert, lane, local_experts)
    return current_expert, expert_start, expert_end, pool_block_offset


@cute.jit
def first_routed_rows(recv: cute.Pointer, expert: Uint32, lane: Int32,
                      block_m: cutlass.Constexpr, dispatch_sms: cutlass.Constexpr,
                      num_ranks: cutlass.Constexpr) -> Uint32:
    count = Uint64(0)
    if lane == Int32(0):
        count_ptr = recv + expert * Uint32(2)
        while Uint32(count >> Uint64(32)) != Uint32(dispatch_sms * num_ranks):
            count = ld_volatile_u64(count_ptr)
        count = ld_acquire_sys_u64(count_ptr)
    return min(cute.arch.shuffle_sync(Uint32(count), 0), Uint32(block_m))


@cute.jit
def publish_routed_static_waves(task: cute.Pointer, full: cute.Pointer, empty: cute.Pointer,
                                expert_counts: cute.Tensor, iteration: Uint32, lane: Int32,
                                cluster_id: Uint32, routed_m_blocks: Uint32,
                                reverse_experts: Uint32, block_m: cutlass.Constexpr,
                                num_clusters: cutlass.Constexpr, l1_clusters: cutlass.Constexpr,
                                l2_clusters: cutlass.Constexpr, local_experts: cutlass.Constexpr,
                                hidden: cutlass.Constexpr, intermediate: cutlass.Constexpr,
                                wave_m_blocks: cutlass.Constexpr,
                                cached_starts=None, cached_blocks=None,
                                overlap: cutlass.Constexpr = False, single_tile_mask=None,
                                task_slots: cutlass.Constexpr = 2) -> Uint32:
    """Assign each wave's W1 and then W2 tiles without global queue claims."""
    tiles_per_block = l1_clusters + l2_clusters
    tiles_per_wave = wave_m_blocks * tiles_per_block
    index = cluster_id
    prefix_blocks = min(Uint32(wave_m_blocks), routed_m_blocks)
    prefix_tiles = prefix_blocks * Uint32(l1_clusters)
    interleaved_end = prefix_tiles + (routed_m_blocks - prefix_blocks) * Uint32(tiles_per_block)
    while index < routed_m_blocks * Uint32(tiles_per_block):
        if cutlass.const_expr(overlap):
            # Seed W1, then interleave W2 of an older block with W1 of a
            # newer one. Every W2 dependency is earlier in the ticket order.
            first = True
            pool_block = Uint32(0)
            n = Uint32(0)
            if index < prefix_tiles:
                pool_block = index // Uint32(l1_clusters)
                n = index % Uint32(l1_clusters)
            elif index < interleaved_end:
                interleaved = index - prefix_tiles
                pool_block = interleaved // Uint32(tiles_per_block)
                tile = interleaved % Uint32(tiles_per_block)
                first = tile >= Uint32(l2_clusters)
                n = tile
                if first:
                    pool_block += prefix_blocks
                    n -= Uint32(l2_clusters)
            else:
                first = False
                tail = index - interleaved_end
                pool_block = routed_m_blocks - prefix_blocks + tail // Uint32(l2_clusters)
                n = tail % Uint32(l2_clusters)
        else:
            base = index // Uint32(tiles_per_wave) * Uint32(wave_m_blocks)
            within_wave = index % Uint32(tiles_per_wave)
            blocks = min(Uint32(wave_m_blocks), routed_m_blocks - base)
            w1_tiles = blocks * Uint32(l1_clusters)
            first = within_wave < w1_tiles
            tile = within_wave if first else within_wave - w1_tiles
            width = Uint32(l1_clusters) if first else Uint32(l2_clusters)
            pool_block = base + tile // width
            n = tile % width
        expert, m_block, valid_m = locate_decode_pool_block(
            expert_counts, pool_block, Uint32(lane), local_experts, block_m,
            cached_starts, cached_blocks, single_tile_mask)
        phase = Uint32(GCBlockPhase.Linear1) if first else Uint32(GCBlockPhase.Linear2)
        shape_n = Uint32(2 * intermediate) if first else Uint32(hidden)
        shape_k = Uint32(hidden) if first else Uint32(intermediate)
        wait_task_slot(empty, iteration, task_slots)
        publish_iteration(task, full, iteration, lane, phase,
                          physical_expert(expert, reverse_experts, local_experts),
                          m_block, n, pool_block, valid_m, shape_n, shape_k, task_slots=task_slots)
        iteration += Uint32(1)
        index += Uint32(num_clusters)
    return iteration


@cute.jit
def publish_split_routed_phases(task: cute.Pointer, full: cute.Pointer, empty: cute.Pointer,
                                w2_counter: cute.Pointer, expert_counts: cute.Tensor,
                                iteration: Uint32, lane: Int32, cluster: Uint32,
                                routed_blocks: Uint32, reverse_experts: Uint32,
                                block_m: cutlass.Constexpr, w1_clusters: cutlass.Constexpr,
                                l1_clusters: cutlass.Constexpr, l2_clusters: cutlass.Constexpr,
                                local_experts: cutlass.Constexpr, hidden: cutlass.Constexpr,
                                intermediate: cutlass.Constexpr,
                                cached_starts=None, cached_blocks=None,
                                task_slots: cutlass.Constexpr = 2) -> Uint32:
    if cluster < Uint32(w1_clusters):
        # Static W1 ownership lets W2 loaders begin independently. W1
        # workers join the W2 queue as soon as their own tiles are issued.
        for tile_iter in cutlass.range(cluster, routed_blocks * Uint32(l1_clusters), w1_clusters):
            tile = Uint32(tile_iter)
            pool = tile // Uint32(l1_clusters)
            expert, m_block, valid = locate_routed_pool_block(
                expert_counts, pool, Uint32(lane), local_experts, block_m, cached_starts, cached_blocks)
            wait_task_slot(empty, iteration, task_slots=task_slots)
            publish_iteration(task, full, iteration, lane, Uint32(GCBlockPhase.Linear1),
                              physical_expert(expert, reverse_experts, local_experts), m_block,
                              tile % Uint32(l1_clusters), pool, valid, Uint32(2 * intermediate),
                              Uint32(hidden), task_slots=task_slots)
            iteration += Uint32(1)
    fetching = True
    while fetching:
        wait_task_slot(empty, iteration, task_slots=task_slots)
        tile = claim_task(w2_counter, lane)
        if tile >= routed_blocks * Uint32(l2_clusters):
            fetching = False
        else:
            pool = tile // Uint32(l2_clusters)
            expert, m_block, valid = locate_routed_pool_block(
                expert_counts, pool, Uint32(lane), local_experts, block_m, cached_starts, cached_blocks)
            publish_iteration(task, full, iteration, lane, Uint32(GCBlockPhase.Linear2),
                              physical_expert(expert, reverse_experts, local_experts), m_block,
                              tile % Uint32(l2_clusters), pool, valid, Uint32(hidden),
                              Uint32(intermediate), task_slots=task_slots)
            iteration += Uint32(1)
    return iteration


@cute.jit
def prioritized_expert_block(block: Uint32, first: Uint32) -> Uint32:
    # Move one tile to the front and retain the remaining order.
    reordered = block
    if block == Uint32(0):
        reordered = first
    elif block <= first:
        reordered -= Uint32(1)
    return reordered


@cute.jit
def schedule_tasks(task: cute.Pointer, full: cute.Pointer, empty: cute.Pointer,
                   recv_ptr: cute.Pointer, l1_count_ptr: cute.Pointer,
                   l2_count_ptr: cute.Pointer, shared_l1_count_ptr: cute.Pointer,
                   shared_l2_count_ptr: cute.Pointer, expert_counts: cute.Tensor,
                   iteration: Uint32, lane: Int32, block: Int32, num_tokens: Uint32,
                   reverse_experts: Uint32,
                   block_m: cutlass.Constexpr,
                   compute_sms: cutlass.Constexpr,
                   dispatch_sms: cutlass.Constexpr,
                   hidden: cutlass.Constexpr,
                   intermediate: cutlass.Constexpr,
                   l1_clusters: cutlass.Constexpr,
                   l2_clusters: cutlass.Constexpr,
                   local_experts: cutlass.Constexpr,
                   num_ranks: cutlass.Constexpr,
                   shared_experts: cutlass.Constexpr,
                   fixed_expert_span: cutlass.Constexpr = 0,
                   early_experts: cutlass.Constexpr = 0,
                   ready_order=None,
                   static_w1_prefix: cutlass.Constexpr = False,
                   speculative_w1: cutlass.Constexpr = False,
                   static_decode_wave: cutlass.Constexpr = 0,
                   decode_w1_warmup: cutlass.Constexpr = 0,
                   decode_w1_cadence: cutlass.Constexpr = 1,
                   decode_shared_w2: cutlass.Constexpr = 0,
                   decode_w2_clusters: cutlass.Constexpr = 0,
                   route_counts_ready=None, route_epoch=0,
                   w2_ready_masks=None, w2_ready_k: cutlass.Constexpr = 0,
                   ring_blocks: cutlass.Constexpr = 0,
                   local_tasks: cutlass.Constexpr = False,
                   cache_routed_ranges: cutlass.Constexpr = False,
                   decode_w2_n_group: cutlass.Constexpr = 1,
                   single_expert_tile_lookup: cutlass.Constexpr = False,
                   split_route_heads: cutlass.Constexpr = False,
                   prioritize_expert_rows: cutlass.Constexpr = 0,
                   shared_w2_ready=None, decode_w2_early_workers: cutlass.Constexpr = 0,
                   speculative_w1_n_tiles: cutlass.Constexpr = 0,
                   static_w2_overlap: cutlass.Constexpr = False,
                   task_slots: cutlass.Constexpr = 2,
                   pair_shared_w1_rank=None,
) -> None:
    """Publish shared and interleaved routed tasks, then the end sentinel."""
    shared_m_blocks = (num_tokens + Uint32(block_m - 1)) // \
        Uint32(block_m)
    shared_l2_early = (shared_m_blocks * Uint32(l2_clusters) <=
                       Uint32(compute_sms))
    shared_w2_cluster = Uint32(block // 2)
    if cutlass.const_expr(decode_shared_w2 == 2):
        # Give W2's TMA producers a separate cluster range from shared W1.
        cluster_count = Uint32(compute_sms // 2)
        rotate = min(cluster_count, shared_m_blocks * Uint32(l1_clusters * shared_experts))
        shared_w2_cluster = (shared_w2_cluster + cluster_count - rotate) % cluster_count
    if cutlass.const_expr(decode_w2_clusters and not decode_shared_w2):
        if shared_m_blocks * Uint32(l2_clusters) <= Uint32(decode_w2_clusters):
            shared_w2_cluster = (shared_w2_cluster + Uint32(decode_w2_clusters)) % Uint32(compute_sms // 2)
    if cutlass.const_expr(shared_experts > 0):
        shared_l1_clusters = l1_clusters * shared_experts
        if cutlass.const_expr(pair_shared_w1_rank is not None):
            # Pack both ranks into one M tile and assign each rank half N.
            cluster = Uint32(block // 2)
            if cluster < Uint32(shared_l1_clusters // 2):
                n_cluster = cluster + pair_shared_w1_rank * Uint32(shared_l1_clusters // 2)
                wait_task_slot(empty, iteration, task_slots)
                publish_iteration(task, full, iteration, lane, Uint32(GCBlockPhase.SharedLinear1),
                                  Uint32(0), Uint32(0), n_cluster, Uint32(0), num_tokens * Uint32(2),
                                  Uint32(2 * intermediate), Uint32(hidden), task_slots=task_slots)
                iteration += Uint32(1)
        else:
            iteration = publish_shared_phase(
                task, full, empty, shared_l1_count_ptr, iteration,
                Int32(lane), Uint32(block // 2),
                Uint32(GCBlockPhase.SharedLinear1),
                num_tokens, block_m, shared_l1_clusters,
                compute_sms, 2 * intermediate * shared_experts,
                hidden, local_tasks, task_slots=task_slots)
        if cutlass.const_expr(decode_shared_w2 not in (1, 3, 4)):
            if shared_l2_early:
                iteration = publish_shared_phase(
                    task, full, empty, shared_l2_count_ptr, iteration,
                    Int32(lane), shared_w2_cluster,
                    Uint32(GCBlockPhase.SharedLinear2),
                    num_tokens, block_m, l2_clusters,
                    compute_sms, hidden,
                    intermediate * shared_experts, local_tasks, task_slots=task_slots)
    if cutlass.const_expr(early_experts > 0):
        # Use the high-index clusters to keep the first shared W1/W2 CTAs
        # available. Each early cluster owns one N tile of the first M tile.
        cluster_idx = Uint32(block // 2)
        head_clusters = speculative_w1_n_tiles or l1_clusters
        early_base = Uint32(compute_sms // 2 - early_experts * head_clusters)
        if cutlass.const_expr(split_route_heads):
            # Keep the shared work on its original clusters, then use
            # the earliest remaining resident workers for routed heads.
            shared_prefix = shared_m_blocks * Uint32(
                max(l1_clusters * shared_experts, l2_clusters if shared_experts else 0))
            early_base = min(early_base, shared_prefix)
        if cluster_idx >= early_base and cluster_idx < early_base + Uint32(early_experts * head_clusters):
            local_idx = cluster_idx - early_base
            expert = local_idx // Uint32(head_clusters)
            n_cluster = local_idx % Uint32(head_clusters)
            valid = Uint32(0)
            issue = True
            if cutlass.const_expr(speculative_w1 and fixed_expert_span == 0):
                # A positive partial count makes expert zero's pool block
                # permanent before the exact size is known. Empty experts
                # never issue a task against a block owned by another expert.
                partial = Uint64(0)
                if lane == Int32(0):
                    while (Uint32(partial) == Uint32(0) and
                           Uint32(partial >> Uint64(32)) != Uint32(dispatch_sms * num_ranks)):
                        partial = ld_volatile_u64(recv_ptr + expert * Uint32(2))
                issue = cute.arch.shuffle_sync(Uint32(partial), 0) != Uint32(0)
            if cutlass.const_expr(not speculative_w1):
                valid = first_routed_rows(recv_ptr, expert, lane, block_m, dispatch_sms, num_ranks)
            # Zero marks a weight-first task. Activation and output warps
            # resolve its routed row count before using the fixed M slot.
            if issue and (speculative_w1 or valid != Uint32(0)):
                wait_task_slot(empty, iteration, task_slots=task_slots)
                publish_iteration(task, full, iteration, lane, Uint32(GCBlockPhase.Linear1),
                                  expert, Uint32(0), n_cluster,
                                  expert * Uint32(fixed_expert_span), valid,
                                  Uint32(2 * intermediate), Uint32(hidden), task_slots=task_slots)
                iteration += Uint32(1)
    # Each lane owns experts wave*32+lane in registers, as in
    # GCMoEScheduler. The count cache survives routed L1/L2 task
    # claims, without an E-dependent shared prefix allocation.
    if cutlass.const_expr(route_counts_ready is not None):
        wait_route_count_doorbell(route_counts_ready, Uint32(route_epoch), Uint32(lane))
    for wave in cutlass.range_constexpr(
            (local_experts + 31) // 32):
        slot = Uint32(wave * 32 + lane)
        count = Uint64(0)
        if slot < Uint32(local_experts):
            local_expert = physical_expert(slot, reverse_experts, local_experts)
            if cutlass.const_expr(route_counts_ready is None):
                while Uint32(count >> 32) != Uint32(
                        dispatch_sms * num_ranks):
                    count = ld_volatile_u64(recv_ptr + local_expert * 2)
            count = ld_acquire_sys_u64(recv_ptr + local_expert * 2)
        expert_counts[wave] = Uint32(count)
    cute.arch.sync_warp()
    single_tile_mask = None
    first_routed_block = Uint32(0)
    if cutlass.const_expr(single_expert_tile_lookup):
        valid_expert = Uint32(lane) < Uint32(local_experts)
        active = Uint32(cute.arch.vote_ballot_sync(valid_expert and expert_counts[0] > Uint32(0)))
        overflow = Uint32(cute.arch.vote_ballot_sync(valid_expert and expert_counts[0] > Uint32(block_m)))
        single_tile_mask = active if overflow == Uint32(0) else Uint32(0)
        if cutlass.const_expr(prioritize_expert_rows):
            if single_tile_mask != Uint32(0):
                rows = expert_counts[0]
                if cutlass.const_expr(prioritize_expert_rows > 0):
                    score = rows
                else:
                    score = Uint32(block_m + 1) - rows
                # Select the same expert from every CTA using immutable
                # receive counts and a deterministic lane tie-break.
                key = Uint32(0)
                if rows != Uint32(0):
                    key = (score << Uint32(5)) | (Uint32(31) - Uint32(lane))
                chosen = Uint32(31) - (cute.arch.warp_redux_sync(key, 'max') & Uint32(31))
                first_routed_block = Uint32(cute.arch.popc(
                    active & ((Uint32(1) << chosen) - Uint32(1))))
    lane_m_blocks = Uint32(0)
    cached_starts = None
    cached_blocks = None
    if cutlass.const_expr(cache_routed_ranges):
        waves = (local_experts + 31) // 32
        cached_starts = cute.make_rmem_tensor((waves,), Uint32)
        cached_blocks = cute.make_rmem_tensor((waves,), Uint32)
        block_offset = Uint32(0)
    for wave in cutlass.range_constexpr(
            (local_experts + 31) // 32):
        received = expert_counts[wave]
        own_blocks = (received + Uint32(block_m - 1)) // \
            Uint32(block_m)
        if cutlass.const_expr(cache_routed_ranges):
            inclusive = warp_inclusive_sum(own_blocks, Uint32(lane))
            cached_starts[wave] = block_offset + inclusive - own_blocks
            cached_blocks[wave] = own_blocks
            block_offset += cute.arch.shuffle_sync(inclusive, 31)
        else:
            lane_m_blocks += own_blocks
    if cutlass.const_expr(cache_routed_ranges):
        routed_m_blocks = block_offset
    else:
        routed_m_blocks = cute.arch.warp_redux_sync(lane_m_blocks, 'add')
    # Match GCMoEScheduler::get_gc_num_l1_warmup_waves and
    # get_next_task for one locality domain: enough L1 work is
    # issued before the first L2 wave, then one L1 follows each
    # successful L2 claim until the L1 queue is exhausted.
    num_clusters = compute_sms // 2
    if cutlass.const_expr(decode_w2_clusters):
        iteration = publish_split_routed_phases(
            task, full, empty, l2_count_ptr, expert_counts, iteration, lane, Uint32(block // 2),
            routed_m_blocks, reverse_experts, block_m, num_clusters - decode_w2_clusters,
            l1_clusters, l2_clusters, local_experts, hidden, intermediate, cached_starts, cached_blocks,
            task_slots=task_slots)
    elif cutlass.const_expr(static_decode_wave):
        static_cluster = Uint32(block // 2)
        if cutlass.const_expr(static_w2_overlap):
            shared_prefix = min(Uint32(num_clusters), shared_m_blocks * Uint32(
                max(l1_clusters * shared_experts, l2_clusters if shared_experts else 0)))
            static_cluster = (static_cluster + Uint32(num_clusters) - shared_prefix) % Uint32(num_clusters)
        iteration = publish_routed_static_waves(
            task, full, empty, expert_counts, iteration, lane, static_cluster,
            routed_m_blocks, reverse_experts, block_m, num_clusters, l1_clusters,
            l2_clusters, local_experts, hidden, intermediate, static_decode_wave,
            cached_starts, cached_blocks, static_w2_overlap,
            single_tile_mask if cutlass.const_expr(static_w2_overlap) else None, task_slots=task_slots)
    else:
        w1_prefix = Uint32(0)
        if cutlass.const_expr(static_w1_prefix):
            w1_prefix = min(Uint32(num_clusters), routed_m_blocks * Uint32(l1_clusters))
            # The first routed experts go to clusters without a pinned shared tile.
            shared_clusters = max(l1_clusters * shared_experts, l2_clusters if shared_experts else 0)
            if cutlass.const_expr(decode_shared_w2 == 2):
                # Rotated shared W2 starts immediately after shared W1.
                shared_clusters = l1_clusters * shared_experts + l2_clusters
            shared_prefix = min(Uint32(num_clusters), shared_m_blocks * Uint32(shared_clusters))
            prefix_idx = (Uint32(block // 2) + Uint32(num_clusters) - shared_prefix) % Uint32(num_clusters)
            if prefix_idx < w1_prefix:
                pool_block = prefix_idx // Uint32(l1_clusters)
                local_expert, m_block, valid_m = locate_routed_pool_block(
                    expert_counts, pool_block, Uint32(lane), local_experts, block_m,
                    cached_starts, cached_blocks)
                wait_task_slot(empty, iteration, task_slots=task_slots)
                publish_iteration(task, full, iteration, lane, Uint32(GCBlockPhase.Linear1),
                                  physical_expert(local_expert, reverse_experts, local_experts),
                                  m_block, prefix_idx % Uint32(l1_clusters), pool_block, valid_m,
                                  Uint32(2 * intermediate), Uint32(hidden), task_slots=task_slots)
                iteration += Uint32(1)
        first_l2_m_blocks = (num_clusters + l2_clusters - 1) // l2_clusters
        if cutlass.const_expr(decode_w2_n_group > 1):
            first_l2_m_blocks = ((first_l2_m_blocks + decode_w2_n_group - 1) //
                                 decode_w2_n_group) * decode_w2_n_group
        first_l2_warmup = (first_l2_m_blocks * l1_clusters +
                           num_clusters - 1) // num_clusters
        interleave_diff = max(l1_clusters - l2_clusters, 0)
        l1_warmup = Uint32(0)
        if routed_m_blocks > 0:
            interleave_warmup = (l1_clusters * decode_w2_n_group +
                                 (routed_m_blocks - 1) * interleave_diff +
                                 num_clusters - 1) // num_clusters + 1
            total_l1_waves = (routed_m_blocks * l1_clusters +
                              num_clusters - 1) // num_clusters
            l1_warmup = min(total_l1_waves,
                            max(Uint32(first_l2_warmup), interleave_warmup))
            if cutlass.const_expr(decode_w2_early_workers):
                # Give a small group its first W2 claim one W1 task earlier.
                # The other workers still feed W1 and these workers rejoin
                # the default W1/W2 cadence after the first W2 claim.
                worker = Uint32(block // 2)
                if (worker >= Uint32(num_clusters - decode_w2_early_workers) and
                        l1_warmup > Uint32(first_l2_warmup)):
                    l1_warmup -= Uint32(1)
            if cutlass.const_expr(decode_w1_warmup):
                # Ensure all W1 tasks feeding the first full W2 wave can
                # be claimed before every worker is holding a W2 task.
                l1_warmup = min(total_l1_waves,
                                Uint32(max(first_l2_warmup, decode_w1_warmup)))
            if cutlass.const_expr(static_w1_prefix):
                # All clusters already published one W1 tile for a full prefix.
                # Retain one dynamic claim so the normal W1 exhaustion path runs.
                if w1_prefix == Uint32(num_clusters) and l1_warmup > Uint32(1):
                    l1_warmup -= Uint32(1)
        l1_done = Uint32(0xffffffff)
        shared_w2_queued = False
        scheduling = True
        while scheduling:
            wait_task_slot(empty, iteration, task_slots=task_slots)
            fetching = True
            if cutlass.const_expr(shared_w2_ready is not None):
                iteration, issued = publish_one_decode_shared_w2(
                    task, full, shared_l2_count_ptr, shared_w2_ready, iteration,
                    lane, num_tokens, l1_clusters, l2_clusters, hidden, intermediate, True, task_slots=task_slots)
                fetching = issued == Uint32(0)
            while fetching:
                if cutlass.const_expr(w2_ready_masks is not None):
                    if l1_warmup == Uint32(0):
                        defer_w2 = Uint32(0)
                        if lane == Int32(0):
                            w2_pool = ld_volatile_u32(l2_count_ptr) // Uint32(l2_clusters)
                            if (w2_pool < routed_m_blocks and
                                    ld_volatile_u32(l1_count_ptr) < routed_m_blocks * Uint32(l1_clusters)):
                                flags = ld_volatile_u64(
                                    w2_ready_masks + (w2_pool % Uint32(ring_blocks)) * Uint32(2))
                                prefix = Uint64((1 << (w2_ready_k * 2)) - 1)
                                expected = prefix
                                if (w2_pool // Uint32(ring_blocks)) & Uint32(1):
                                    expected = Uint64(0)
                                defer_w2 = Uint32((flags & prefix) != expected)
                        # Prefer another W1 while the head W2 inputs are pending.
                        # A later successful claim still uses the exact K barrier.
                        l1_warmup = cute.arch.shuffle_sync(defer_w2, 0)
                if l1_warmup != l1_done and l1_warmup > 0:
                    l1_warmup -= Uint32(1)
                    task_idx = claim_task(l1_count_ptr, Int32(lane))
                    if cutlass.const_expr(static_w1_prefix):
                        task_idx += w1_prefix
                    if task_idx >= routed_m_blocks * l1_clusters:
                        l1_warmup = l1_done
                    else:
                        pool_block = task_idx // l1_clusters
                        if cutlass.const_expr(prioritize_expert_rows):
                            pool_block = prioritized_expert_block(pool_block, first_routed_block)
                        if cutlass.const_expr(ready_order is not None):
                            pool_block = wait_ready_tile(ready_order, pool_block, lane)
                        n_cluster = task_idx % l1_clusters
                        local_expert, m_block, valid_m = \
                            locate_decode_pool_block(
                                expert_counts, pool_block, Uint32(lane),
                                local_experts, block_m, cached_starts, cached_blocks, single_tile_mask)
                        already_scheduled = False
                        if cutlass.const_expr(early_experts > 0):
                            already_scheduled = (local_expert < Uint32(early_experts) and
                                                 m_block == Uint32(0))
                            if cutlass.const_expr(speculative_w1_n_tiles):
                                already_scheduled = already_scheduled and n_cluster < Uint32(speculative_w1_n_tiles)
                        if not already_scheduled:
                            storage_block = pool_block
                            if cutlass.const_expr(fixed_expert_span != 0):
                                storage_block = local_expert * Uint32(fixed_expert_span) + m_block
                            publish_iteration(task, full, iteration, Int32(lane),
                                              Uint32(GCBlockPhase.Linear1),
                                              physical_expert(local_expert, reverse_experts, local_experts),
                                              m_block, n_cluster, storage_block, valid_m,
                                              Uint32(2 * intermediate), Uint32(hidden), local_tasks, task_slots=task_slots)
                            iteration = Uint32(iteration + Uint32(1))
                            fetching = False
                else:
                    task_idx = claim_task(l2_count_ptr, Int32(lane))
                    if task_idx >= routed_m_blocks * l2_clusters:
                        scheduling = False
                        fetching = False
                    else:
                        if l1_warmup != l1_done:
                            l1_warmup = Uint32(decode_w1_cadence)
                        order_block = task_idx // l2_clusters
                        n_cluster = task_idx % l2_clusters
                        if cutlass.const_expr(decode_w2_n_group > 1):
                            group_tiles = Uint32(decode_w2_n_group * l2_clusters)
                            first_block = task_idx // group_tiles * Uint32(decode_w2_n_group)
                            within_group = task_idx % group_tiles
                            group_blocks = min(Uint32(decode_w2_n_group), routed_m_blocks - first_block)
                            order_block = first_block + within_group % group_blocks
                            n_cluster = within_group // group_blocks
                        pool_block = order_block
                        if cutlass.const_expr(prioritize_expert_rows):
                            pool_block = prioritized_expert_block(pool_block, first_routed_block)
                        if cutlass.const_expr(ready_order is not None):
                            pool_block = wait_ready_tile(ready_order, order_block, lane)
                        local_expert, m_block, valid_m = \
                            locate_decode_pool_block(
                                expert_counts, pool_block, Uint32(lane),
                                local_experts, block_m, cached_starts, cached_blocks, single_tile_mask)
                        required_l1 = (order_block + 1) * l1_clusters
                        if cutlass.const_expr(static_w1_prefix):
                            # Each cluster submits its own prefix before any
                            # dynamic claim. W2's data barrier still waits on W1.
                            required_l1 = max(required_l1, w1_prefix) - w1_prefix
                        while ld_volatile_u32(l1_count_ptr) < required_l1:
                            pass
                        storage_block = pool_block
                        if cutlass.const_expr(fixed_expert_span != 0):
                            storage_block = local_expert * Uint32(fixed_expert_span) + m_block
                        publish_iteration(task, full, iteration, Int32(lane),
                                          Uint32(GCBlockPhase.Linear2),
                                          physical_expert(local_expert, reverse_experts, local_experts),
                                          m_block, n_cluster, storage_block, valid_m,
                                          Uint32(hidden), Uint32(intermediate), local_tasks, task_slots=task_slots)
                        iteration = Uint32(iteration + Uint32(1))
                        fetching = False
            if cutlass.const_expr(decode_shared_w2 == 1 and shared_experts > 0):
                if shared_l2_early and not shared_w2_queued:
                    # Give each cluster its first routed claim before
                    # occupying a pipeline slot with the shared W2 tile.
                    iteration = publish_shared_phase(
                        task, full, empty, shared_l2_count_ptr, iteration,
                        Int32(lane), shared_w2_cluster,
                        Uint32(GCBlockPhase.SharedLinear2),
                        num_tokens, block_m, l2_clusters,
                        compute_sms, hidden, intermediate * shared_experts, local_tasks, task_slots=task_slots)
                    shared_w2_queued = True
    if cutlass.const_expr(shared_w2_ready is not None):
        # All shared W1 producers were queued up front. If routed work is
        # exhausted before they finish, the ordinary A loader can still wait.
        issued = Uint32(1)
        while issued != Uint32(0):
            wait_task_slot(empty, iteration, task_slots=task_slots)
            iteration, issued = publish_one_decode_shared_w2(
                task, full, shared_l2_count_ptr, shared_w2_ready, iteration,
                lane, num_tokens, l1_clusters, l2_clusters, hidden, intermediate, False, task_slots=task_slots)
    elif cutlass.const_expr(shared_experts > 0):
        publish_after_routed = not shared_l2_early
        if cutlass.const_expr(decode_shared_w2 == 3):
            # These epilogues may wait for peer results, so they must be
            # each worker's final jobs, after its routed outputs are posted.
            publish_after_routed = True
        if publish_after_routed:
            iteration = publish_shared_phase(
                task, full, empty, shared_l2_count_ptr, iteration,
                Int32(lane), Uint32(block // 2),
                Uint32(GCBlockPhase.SharedLinear2),
                num_tokens, block_m, l2_clusters,
                compute_sms, hidden,
                intermediate * shared_experts, local_tasks, task_slots=task_slots)
    wait_task_slot(empty, iteration, task_slots=task_slots)
    publish_iteration(task, full, iteration, Int32(lane),
                      Uint32(GCBlockPhase.None_), Uint32(0),
                      Uint32(0), Uint32(0),
                      Uint32(0), Uint32(0), Uint32(0), Uint32(0), local_tasks, task_slots=task_slots)
