"""Two-CTA TensorMap transfers for mixed FP8/FP4 MMA stages."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Uint32, Uint64

from ..ptx.ld_st import EVICT_LAST, ld_global_u32x4, st_shared_u32x4
from ..ptx.tma import (
    tile_gather4_2cta,
    tile_g2s_2cta_2d,
    tile_g2s_2cta_4d,
    tile_g2s_2d,
    tile_g2s_4d,
    tma_store_fence,
)
from ..ptx.utils import elect_one_sync


def use_small_tile_path(shared_experts: int, block_m: int) -> bool:
    """Whether the fused-SE geometry enables valid_m-shrunk routed tiles.

    Gates the prefix activation maps and the K128 descriptor reuse.
    """
    return shared_experts > 0 and block_m // 2 > 16


@cute.jit
def grouped_routed_weight_coords(k_block: Uint32, expert: Uint32, n_block: Uint32,
                                 k_blocks: Uint32, n_blocks: Uint32, inner_k: Uint32):
    group = (expert * (k_blocks // inner_k) + k_block // inner_k) * n_blocks + n_block
    return Uint32(0), k_block % inner_k, group


@cute.jit
def load_routed_scale_codes(address: Uint64, tile: Uint32, lane: Uint32,
                            group: cutlass.Constexpr):
    source = cute.make_ptr(Uint32, address + Uint64(tile) * Uint64(128 * group),
                           cute.AddressSpace.gmem, assumed_align=16)
    if cutlass.const_expr(group == 1):
        codes = (cute.make_tensor(source, cute.make_layout((32,)))[lane],)
    else:
        codes = ld_global_u32x4(source + lane * Uint32(4))
    return codes


@cute.jit
def store_routed_scale_word(smem: cute.Pointer, bits: Uint32,
                            lane: Uint32, base: cutlass.Constexpr) -> None:
    mask = Uint32(0x03030303)
    offset = Uint32(base * 0x01010101)
    st_shared_u32x4(smem + lane * Uint32(4),
                    (bits & mask) + offset,
                    ((bits >> Uint32(2)) & mask) + offset,
                    ((bits >> Uint32(4)) & mask) + offset,
                    ((bits >> Uint32(6)) & mask) + offset)


@cute.jit
def unpack_routed_scale_tile(smem: cute.Pointer, address: Uint64, tile: Uint32,
                             base: cutlass.Constexpr, group: cutlass.Constexpr = 1) -> None:
    lane = Uint32(cute.arch.thread_idx()[0]) % Uint32(32)
    packed = load_routed_scale_codes(address, tile, lane, group)
    for stage in cutlass.range_constexpr(group):
        store_routed_scale_word(smem + Uint32(stage * 128), packed[stage], lane, base)
    tma_store_fence()
    cute.arch.sync_warp()


@cute.jit
def load_mixed_k128_tma(activation_map: cute.Pointer,
                        activation_sf_map: cute.Pointer,
                        weight_map: cute.Pointer,
                        weight_sf_map: cute.Pointer,
                        activation_smem: cute.Pointer,
                        activation_sf_smem: cute.Pointer,
                        weight_smem: cute.Pointer,
                        weight_sf_smem: cute.Pointer,
                        full_bar: cute.Pointer, empty_bar: cute.Pointer,
                        k_block: Uint32, stage_phase: Uint32,
                        block_idx: Uint32, n_block: Uint32,
                        local_expert: Uint32, valid_m: Uint32,
                        activation_rows: Uint32,
                        shape_k: Uint32, is_shared: Uint32,
                        cta_rank: Uint32,
                        loader_warp: cutlass.Constexpr,
                        block_m: cutlass.Constexpr,
                        sf_block_m: cutlass.Constexpr,
                        shared_weight_gran_k: cutlass.Constexpr,
                        has_shared: cutlass.Constexpr,
                        shared_weight_cache_hint: cutlass.Constexpr,
                        routed_cache_hint: Uint64, gather_rows=None,
                        gather_routed_w1=None,
                        local_mma: cutlass.Constexpr = False, weight_k_major=None,
                        activation_tile_major=None,
                        single_loader_lane: cutlass.Constexpr = False,
                        pair_routed_weight_scales: cutlass.Constexpr = False,
                        activation_total_rows=None, pair_routed_weights=None,
                        grouped_n_blocks=None, grouped_weight_phase=None, grouped_weight_inner_k=None,
                        packed_sf_address=None, packed_sf_n_blocks=None,
                        packed_sf_base: cutlass.Constexpr = None, packed_sf_k_group: cutlass.Constexpr = 1,
                        packed_sf_helper: cutlass.Constexpr = False,
                        trim_gather_tail: cutlass.Constexpr = False,
                        transpose_activation=None, activation_in_smem=None, peer_gather_map=None,
                        keep_routed_weight_scales: cutlass.Constexpr = False,
                        activation_cache_hint: cutlass.Constexpr = EVICT_LAST,
                        shared_weight_packet=None, activation_sf_rows=None) -> None:
    """Warp 0/1 stage the same K128 task through 2CTA TensorMaps.

    Both CTAs issue their own TMA transfers into local SMEM. The leader's
    full barrier counts two arrivals per CTA and source-byte transaction
    counts for both CTAs. The caller waits it before issuing mixed MMA.
    """
    cute.arch.mbarrier_wait(empty_bar, stage_phase ^ Uint32(1))
    issue = True
    if cutlass.const_expr(not single_loader_lane):
        issue = elect_one_sync()
    k_idx = k_block * Uint32(128)
    # Every activation tile is read again by the other N-tiles of its
    # expert, so keep the stream resident while the weight stream cycles.
    cache_hint = Uint64(activation_cache_hint)
    if cutlass.const_expr(loader_warp == 0):
        aligned_m = (valid_m + Uint32(15)) & Uint32(~15)
        m_idx = block_idx * Uint32(block_m)
        if cutlass.const_expr(activation_tile_major is not None):
            if activation_tile_major:
                m_idx = (block_idx * (shape_k // Uint32(128)) + k_block) * Uint32(block_m)
                k_idx = Uint32(0)
        if cutlass.const_expr(not local_mma):
            if cta_rank != Uint32(0):
                m_idx += aligned_m // Uint32(2)
        if cutlass.const_expr(transpose_activation is not None):
            if transpose_activation:
                # A K128 plane stores each channel's 32 token bytes
                # contiguously. Each CTA reads 16 tokens for every channel.
                k_idx = cta_rank * Uint32(block_m // 2)
                m_idx = (block_idx * (shape_k // Uint32(128)) + k_block) * Uint32(128)
        if issue:
            if cutlass.const_expr(local_mma):
                tile_g2s_2d(activation_smem, activation_map, full_bar, k_idx, m_idx, cache_hint)
                tile_g2s_2d(activation_sf_smem, activation_sf_map, full_bar,
                            block_idx * Uint32(sf_block_m), k_block, cache_hint)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_bar, activation_rows * Uint32(128) + Uint32(sf_block_m * 4))
            elif cutlass.const_expr(gather_rows is not None):
                if gather_routed_w1:
                    for group in cutlass.range_constexpr(block_m // 8):
                        issue_group = True
                        if cutlass.const_expr(trim_gather_tail):
                            issue_group = Uint32(group * 4) < activation_rows
                        if issue_group:
                            r0, r1, r2, r3 = gather_rows[group]
                            gather_map = (activation_map if cutlass.const_expr(peer_gather_map is None)
                                          else peer_gather_map)
                            tile_gather4_2cta(activation_smem + group * 512,
                                              gather_map, full_bar, k_idx, r0, r1, r2, r3,
                                              cache_hint)
                else:
                    tile_g2s_2cta_2d(activation_smem, activation_map, full_bar,
                                     k_idx, m_idx, cache_hint)
            else:
                has_payload = True
                if cutlass.const_expr(activation_total_rows is not None):
                    has_payload = activation_rows != Uint32(0)
                if cutlass.const_expr(activation_in_smem is not None):
                    has_payload = has_payload and not activation_in_smem
                if has_payload:
                    tile_g2s_2cta_2d(activation_smem, activation_map, full_bar,
                                     k_idx, m_idx, cache_hint)
            if cutlass.const_expr(not local_mma):
                tile_g2s_2cta_2d(activation_sf_smem, activation_sf_map, full_bar,
                                 block_idx * Uint32(sf_block_m), k_block, cache_hint)
                if cta_rank == Uint32(0):
                    total_rows = (2 * activation_rows if cutlass.const_expr(activation_total_rows is None)
                                  else activation_total_rows)
                    if cutlass.const_expr(activation_in_smem is not None):
                        total_rows = Uint32(0) if activation_in_smem else total_rows
                    scale_rows = (Uint32(sf_block_m) if cutlass.const_expr(activation_sf_rows is None)
                                  else activation_sf_rows)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full_bar, total_rows * Uint32(128) + scale_rows * Uint32(8))
                else:
                    cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)
    else:
        n_idx = n_block * Uint32(128)
        # Routed and shared weights can have different eviction policies.
        weight_cache_hint = routed_cache_hint
        gran = Uint32(32)
        if cutlass.const_expr(has_shared):
            weight_cache_hint = (routed_cache_hint if is_shared == Uint32(0)
                                 else Uint64(shared_weight_cache_hint))
            if cutlass.const_expr(shared_weight_gran_k == 128):
                if is_shared != Uint32(0):
                    gran = Uint32(128)
        scale_cache_hint = weight_cache_hint
        if cutlass.const_expr(keep_routed_weight_scales):
            if is_shared == Uint32(0):
                scale_cache_hint = Uint64(EVICT_LAST)
        sf_stages = gran // Uint32(32)
        sf_index = k_block // sf_stages
        sf_packs_per_expert = (shape_k + gran * Uint32(4) - Uint32(1)) // \
            (gran * Uint32(4))
        load_sf = k_block % sf_stages == Uint32(0)
        weight_bytes = Uint32(128 * 128 // 2)
        if cutlass.const_expr(has_shared):
            if is_shared != Uint32(0):
                weight_bytes = Uint32(128 * 128)
        sf_bytes = Uint32(128 * 4) if load_sf else Uint32(0)
        if cutlass.const_expr(packed_sf_base is not None):
            if is_shared == Uint32(0):
                load_sf = False
                sf_bytes = Uint32(0)
        wait_next = False
        if cutlass.const_expr(pair_routed_weight_scales):
            if is_shared == Uint32(0):
                load_sf = (k_block & Uint32(1)) == Uint32(0)
                sf_bytes = Uint32(1024) if load_sf else Uint32(0)
                wait_next = load_sf
        load_weight = True
        if cutlass.const_expr(pair_routed_weights is not None):
            if pair_routed_weights:
                load_weight = (k_block & Uint32(1)) == Uint32(0)
                weight_bytes = Uint32(16384) if load_weight else Uint32(0)
                wait_next = wait_next or load_weight
        if cutlass.const_expr(shared_weight_packet is not None):
            if shared_weight_packet != Uint32(0):
                load_weight = (k_block & Uint32(1)) == Uint32(0)
                weight_bytes = Uint32(33280) if load_weight else Uint32(0)
                load_sf = False
                sf_bytes = Uint32(0)
                wait_next = load_weight
        if cutlass.const_expr(pair_routed_weight_scales or pair_routed_weights is not None or
                              shared_weight_packet is not None):
            if wait_next:
                # The paired transfer owns both destinations until the
                # first stage completes. MMA consumes the stages in order.
                cute.arch.mbarrier_wait(empty_bar + 1, stage_phase ^ Uint32(1))
        tile_coord = k_block
        expert_coord = local_expert
        if cutlass.const_expr(weight_k_major is not None):
            if weight_k_major:
                tile_coord = local_expert
                expert_coord = k_block
        weight_n_idx = n_idx
        if cutlass.const_expr(grouped_weight_phase is not None):
            if grouped_weight_phase:
                weight_n_idx, tile_coord, expert_coord = grouped_routed_weight_coords(
                    k_block, local_expert, n_block, shape_k // Uint32(128),
                    grouped_n_blocks, grouped_weight_inner_k)
        # Each K128 plane contains an 8 KiB FP4 or 16 KiB FP8 weight tile.
        # The paired routed TensorMap writes two adjacent SMEM stages.
        if cutlass.const_expr(local_mma):
            if issue:
                tile_g2s_4d(weight_smem, weight_map, full_bar,
                            Uint32(0), weight_n_idx, tile_coord, expert_coord, weight_cache_hint)
                if load_sf:
                    tile_g2s_2d(weight_sf_smem, weight_sf_map, full_bar,
                                n_idx, local_expert * sf_packs_per_expert + sf_index,
                                scale_cache_hint)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_bar, weight_bytes + sf_bytes)
        else:
            if issue:
                if load_weight:
                    regular_tma = True
                    if cutlass.const_expr(shared_weight_packet is not None):
                        regular_tma = shared_weight_packet == Uint32(0)
                        if not regular_tma:
                            group = n_block * (shape_k // Uint32(256)) + k_block // Uint32(2)
                            tile_g2s_2cta_2d(weight_smem, weight_map, full_bar, Uint32(0),
                                             group * Uint32(130), weight_cache_hint)
                    if regular_tma:
                        tile_g2s_2cta_4d(weight_smem, weight_map, full_bar,
                                         Uint32(0), weight_n_idx, tile_coord, expert_coord, weight_cache_hint)
                if load_sf:
                    tile_g2s_2cta_2d(weight_sf_smem, weight_sf_map, full_bar,
                                     n_idx, local_expert * sf_packs_per_expert + sf_index,
                                     scale_cache_hint)
            if cutlass.const_expr(packed_sf_base is not None and not packed_sf_helper):
                if is_shared == Uint32(0):
                    if cutlass.const_expr(packed_sf_k_group == 1):
                        tile = (local_expert * (shape_k // Uint32(128)) + k_block) * \
                            packed_sf_n_blocks + n_block
                        unpack_routed_scale_tile(weight_sf_smem, packed_sf_address, tile, packed_sf_base)
                    elif k_block % Uint32(4) == Uint32(0):
                        # The first weight transfer is in flight while the
                        # remaining three SMEM scale slots become reusable.
                        for stage in cutlass.range_constexpr(1, 4):
                            cute.arch.mbarrier_wait(empty_bar + stage, stage_phase ^ Uint32(1))
                        tile = (local_expert * packed_sf_n_blocks + n_block) * \
                            (shape_k // Uint32(512)) + k_block // Uint32(4)
                        unpack_routed_scale_tile(weight_sf_smem, packed_sf_address, tile, packed_sf_base, 4)
            if issue:
                if cta_rank == Uint32(0):
                    tx_bytes = (weight_bytes + sf_bytes) * Uint32(2)
                    if cutlass.const_expr(pair_routed_weights is not None or shared_weight_packet is not None):
                        if tx_bytes == Uint32(0):
                            cute.arch.mbarrier_arrive(full_bar)
                        else:
                            cute.arch.mbarrier_arrive_and_expect_tx(full_bar, tx_bytes)
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(full_bar, tx_bytes)
                else:
                    cute.arch.mbarrier_arrive(full_bar, peer_cta_rank_in_cluster=0)
    if cutlass.const_expr(not single_loader_lane):
        cute.arch.sync_warp()
