"""CuTe DSL source for three persistent GC MoE roles and four worker phases."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import math
from cutlass import Float32, Int32, Uint8, Uint16, Uint32, Uint64
from cutlass.experimental import primitives as cutlass_primitives

from ..comm.gc_barrier import (
    DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
    grid_sync,
    wait_route_count_doorbell,
    wait_sys_u64_eq,
)
from ..common.math import (
    bf16x2_high_f32_bits,
    bf16x2_max,
    bf16x2_min,
    bits_float,
    f32_to_bf16_bits,
    f32x2_to_bf16x2_bits,
    fast_exp_neg_f32,
    fast_rcp,
    abs_f32,
    f32x2_to_e4m3x2_bits,
    fmax_f32,
    pack_bf16_pair,
    ue8m0_fp8_inv,
    ue8m0_fp8_scale_exp,
    ue8m0_fp8_scale_exp_pair,
    warp_inclusive_sum,
    warp_group4_max,
    warp_group4_hardware_max,
)
from ..common.tma_copy import (grouped_routed_weight_coords, load_mixed_k128_tma,
                               load_routed_scale_codes, store_routed_scale_word, use_small_tile_path)
from ..common.decode_trace import cta_mark, global_timer, sm_id, task_mark, task_tag
from ..layout.gc_moe import (
    ACCUM_BARRIER_STAGES,
    COMBINE_WARPS,
    FINALIZE_THREADS,
    GCBufferLayout,
    MAX_RANKS,
    PACKED_ROUTED_SCALE_OFFSETS,
    ROUTED_W1_PREFETCH_OFFSET,
    ROUTED_W2_PREFETCH_OFFSET,
    SIGNAL_OFFSETS,
)
from ..layout.sym_buffer import mapped_peer_base
from ..mma.sm100 import (
    MixedMmaTile,
    blockscaled_instr_desc_hi,
    issue_mixed_k128,
    kmajor_smem_desc,
    kmajor_smem_desc_from_start,
    select_weight_format,
)
from ..ptx.ld_st import (
    EVICT_FIRST,
    EVICT_LAST,
    EVICT_NORMAL,
    atom_acq_rel_gpu_add_u64,
    atom_acq_rel_sys_add_u32,
    atom_acq_rel_sys_or_u32,
    atom_add_u64,
    atom_add_u32,
    atom_release_gpu_add_u32,
    atom_shared_add_u32,
    atom_sys_add_u64,
    red_sys_add_u64,
    append_shared_route_index,
    bulk_zero_shared,
    cp_async_raw16,
    cp_async_raw_wait,
    ld_acquire_gpu_u32,
    ld_peer_param_u64,
    multimem_load_add_u64,
    multimem_store_u64,
    multimem_store_release_u64,
    ld_acquire_gpu_u64,
    ld_acquire_sys_u64,
    ld_acquire_sys_u32,
    ld_global_u32x3,
    ld_global_u32x4,
    ld_global_u64,
    ld_shared_u32x4,
    ld_volatile_u32,
    ld_volatile_u64,
    red_gpu_add_s32,
    red_gpu_add_u32,
    red_release_gpu_add_u32,
    red_release_gpu_xor_u64,
    select_fp8_token_bytes,
    st_global_u32x4,
    st_global_u64,
    st_release_sys_u64,
    st_release_gpu_u32,
    st_release_u32,
    st_shared_u32x4,
    stmatrix_bf16x8_trans,
    stmatrix_fp8x4_trans_late,
    transpose_bf16x2,
)
from ..ptx.tcgen05 import (
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tmem_free,
    tmem_load_16dp256b4,
)
from ..ptx.tma import (
    bulk_g2s_1d,
    bulk_prefetch_l2,
    bulk_s2g_1d,
    generic_tensormap_ptr,
    prefetch_g2s_2d,
    prefetch_g2s_4d,
    prefetch_tensormap,
    tile_s2g_2d,
    tma_store_arrive,
    tma_store_fence,
    tma_store_wait_0,
    tma_store_wait_1,
)
from ..ptx.utils import (
    accumulate_bf16_into_f32,
    assume_u32_lt,
    device_trap,
    elect_one_sync,
    fence_acq_rel_sys,
    launch_dependents,
    match_lanes_u32,
    read_clock64,
    sync_aligned,
    sync_unaligned,
    undefined_u32,
    warp_first_set_bit,
    warp_nth_set_bit,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)
from .sm100_decode import (WARP_BYTES, combine_after_shared_w2, combine_flag_index,
                           copy_input_vector, reduce_output_sentinel,
                           reduce_output_vector)

ROUTE_COUNT_BITS = 11
ROUTE_COUNT_MASK = (1 << ROUTE_COUNT_BITS) - 1
ROUTE_GROUP_PAYLOAD_BITS = 4 * ROUTE_COUNT_BITS
ROUTE_OCTET_COUNT_BITS = 6
ROUTE_OCTET_COUNT_MASK = (1 << ROUTE_OCTET_COUNT_BITS) - 1
ROUTE_OCTET_PAYLOAD_BITS = 8 * ROUTE_OCTET_COUNT_BITS
ROUTE_ATOMIC_COUNT_BITS = 16
ROUTE_ATOMIC_COUNT_MASK = (1 << ROUTE_ATOMIC_COUNT_BITS) - 1
ROUTE_ATOMIC_PARTICIPANT_SHIFT = 3 * ROUTE_ATOMIC_COUNT_BITS
ROUTE_MULTICAST_COUNT_BITS = 14
ROUTE_MULTICAST_PAYLOAD_BITS = 4 * ROUTE_MULTICAST_COUNT_BITS
ROUTE_MULTICAST_EPOCH_MASK = 7
from ..scheduler.gc_moe import (
    GCBlockPhase,
    advance_expert,
    first_routed_rows,
    physical_expert,
    release_task_info,
    schedule_tasks,
)


# A fixed small-decode floor, independent of the local expert count: 64 CTAs
# provide up to 512 combine warps before clamping to the compute SM budget.
SMALL_DECODE_FINALIZE_SMS = 64


@cute.jit
def _swiglu_pair(gate0_accum: Float32, gate1_accum: Float32,
                up0_accum: Float32, up1_accum: Float32,
                weight0: Float32, weight1: Float32,
                clamp_value: Float32, alpha: Float32,
                clamp_enabled: cutlass.Constexpr,
                oai_enabled: cutlass.Constexpr,
                fast_math: cutlass.Constexpr, clamp_pair=None):
    """Reference W1 float2 SwiGLU, preserving the BF16x2 and FP32 order."""
    gate_bits = f32x2_to_bf16x2_bits(gate0_accum, gate1_accum)
    up_bits = f32x2_to_bf16x2_bits(up0_accum, up1_accum)
    if cutlass.const_expr(clamp_enabled):
        bound_pair = clamp_pair
        if cutlass.const_expr(clamp_pair is None):
            bound = Uint32(f32_to_bf16_bits(clamp_value))
            bound_pair = bound | (bound << Uint32(16))
        gate_bits = bf16x2_min(gate_bits, bound_pair)
        up_bits = bf16x2_max(up_bits, bound_pair ^ Uint32(0x80008000))
        up_bits = bf16x2_min(up_bits, bound_pair)
    gate0 = bits_float((gate_bits & Uint32(0xFFFF)) << Uint32(16))
    gate1 = bits_float(bf16x2_high_f32_bits(gate_bits))
    sigmoid0, sigmoid1 = gate0, gate1
    if cutlass.const_expr(oai_enabled):
        sigmoid0, sigmoid1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (alpha, alpha), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        exp0 = fast_exp_neg_f32(sigmoid0)
        exp1 = fast_exp_neg_f32(sigmoid1)
    else:
        exp0 = cute.math.exp(-sigmoid0, fastmath=False)
        exp1 = cute.math.exp(-sigmoid1, fastmath=False)
    denom0, denom1 = cute.arch.add_packed_f32x2(
        (Float32(1), Float32(1)), (exp0, exp1), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        gate0, gate1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (fast_rcp(denom0), fast_rcp(denom1)), rnd='rn', ftz=False)
    else:
        gate0 = cute.math.div(
            gate0, denom0, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
        gate1 = cute.math.div(
            gate1, denom1, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
    up0 = bits_float((up_bits & Uint32(0xFFFF)) << Uint32(16))
    up1 = bits_float(bf16x2_high_f32_bits(up_bits))
    if cutlass.const_expr(oai_enabled):
        up0, up1 = cute.arch.add_packed_f32x2(
            (up0, up1), (Float32(1), Float32(1)), rnd='rn', ftz=False)
    activation0, activation1 = cute.arch.mul_packed_f32x2(
        (gate0, gate1), (up0, up1), rnd='rn', ftz=False)
    return cute.arch.mul_packed_f32x2(
        (activation0, activation1), (weight0, weight1), rnd='rn', ftz=False)


@cute.jit
def sync_epilogue_warpgroup(warpgroup: Uint32) -> None:
    """Reference WG-local barrier ID 3 + warpgroup, with 128 callers."""
    sync_aligned(128, Uint32(3) + warpgroup)


@cute.jit
def _load_w2_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Keep both 16dp halves of one 8-row W2 atom in registers."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w2_tmem(accum_empty: cute.Pointer, stage: Uint32,
                     local_mma: cutlass.Constexpr = False) -> None:
    tcgen05_before_thread_sync()
    if cutlass.const_expr(local_mma):
        cute.arch.mbarrier_arrive(accum_empty + stage)
    else:
        cute.arch.mbarrier_arrive(accum_empty + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def _load_w1_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Source-contract 16dp W1 atom: gate/up for two post channels."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w1_tmem(accum_empty: cute.Pointer, stage: Uint32,
                     local_mma: cutlass.Constexpr = False) -> None:
    _release_w2_tmem(accum_empty, stage, local_mma)


@cute.jit
def _w1_pack_fp8x4(a: Float32, b: Float32,
                    c: Float32, d: Float32) -> Uint32:
    """Pack reference channel0/token0, channel0/token1, channel1/token0/1."""
    ab = f32x2_to_e4m3x2_bits(a, b)
    cd = f32x2_to_e4m3x2_bits(c, d)
    return Uint32(ab) | (Uint32(cd) << Uint32(16))


@cute.jit
def _w1_store_direct(output: cute.Pointer, stride: Uint32, row: Uint32,
                     col: Uint32, lane: Uint32, packed: Uint32,
                     register_transpose: cutlass.Constexpr = False,
                     vector_store: cutlass.Constexpr = False) -> None:
    # STSM.m16n8.trans normally transposes these four elements in shared
    # memory. Gather four adjacent channels per lane and store directly.
    if cutlass.const_expr(register_transpose and not vector_store):
        # Transpose 16-bit pairs of tokens. Adjacent lanes now own adjacent
        # channel pairs for the same two tokens. Share the pair and each
        # lane writes the four channels for a different token.
        transposed = transpose_bf16x2(packed)
        partner = cute.arch.shuffle_sync(transposed, lane ^ Uint32(1))
        odd = lane & Uint32(1)
        first = transposed if odd == Uint32(0) else partner
        second = partner if odd == Uint32(0) else transposed
        word = select_fp8_token_bytes(first, second, odd)
        token = (lane // Uint32(4) & Uint32(~1)) | odd
        quarter = (lane >> Uint32(1)) & Uint32(3)
    else:
        token = lane // Uint32(4)
        quarter = lane % Uint32(4)
        src = (quarter % Uint32(2)) * Uint32(16) + token // Uint32(2)
        shift = (quarter // Uint32(2)) * Uint32(16) + (token % Uint32(2)) * Uint32(8)
        word = Uint32(0)
        for byte in cutlass.range_constexpr(4):
            fragment = cute.arch.shuffle_sync(packed, src + Uint32(byte * 4))
            word |= ((fragment >> shift) & Uint32(255)) << Uint32(byte * 8)
    dst = output + Uint64(row + token) * Uint64(stride) + Uint64(col + quarter * Uint32(4))
    if cutlass.const_expr(vector_store):
        # Four adjacent lanes own one 16-byte output segment for a row.
        # Collect the words before issuing one peer write from its first lane.
        first = token * Uint32(4)
        words = ()
        for index in cutlass.range_constexpr(4):
            words += (cute.arch.shuffle_sync(word, first + Uint32(index)),)
        if quarter == Uint32(0):
            st_global_u32x4(cute.recast_ptr(dst, dtype=Uint32), words[0], words[1], words[2], words[3])
    else:
        cute.recast_ptr(dst, dtype=Uint32)[0] = word


@cute.jit
def _w1_store_transposed(output: cute.Pointer, plane: Uint32, row: Uint32,
                          column: Uint32, lane: Uint32, packed: Uint32) -> None:
    """Store each lane's two natural token pairs in an MN-major BM32 plane."""
    token_pair = (lane % Uint32(4)) * Uint32(2)
    channel = column + (lane // Uint32(16)) * Uint32(4) + (lane % Uint32(16)) // Uint32(4)
    offset = Uint64(plane) * Uint64(32 * 128) + Uint64(channel * Uint32(32) + row + token_pair)
    dst = cute.recast_ptr(output + offset, dtype=Uint16)
    dst[0] = Uint16(packed)
    dst[128] = Uint16(packed >> Uint32(16))


@cute.jit
def _finish_dispatch_pull(signals: cute.Tensor, scratch: cute.Pointer,
                          barrier: cute.Pointer, phase: Uint32,
                          address: Uint64, ring_block: Uint32, increment: Uint32,
                          row_bytes: cutlass.Constexpr) -> None:
    cute.arch.mbarrier_wait(barrier, phase)
    dst = cute.make_ptr(Uint32, address, cute.AddressSpace.gmem, assumed_align=16)
    bulk_s2g_1d(dst, scratch, Uint32(row_bytes))
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0)
    full = signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block
    red_release_gpu_add_u32(full, increment)


@cute.jit
def _unpack_w2_metadata(word: Uint32) -> tuple[Uint32, Uint32, Uint32]:
    return word & Uint32(0xff), (word >> Uint32(8)) & Uint32(0xffff), word >> Uint32(24)


@cute.jit
def _load_w2_source_metadata(base: cute.Pointer, row_in_block: Uint32,
                              valid_m: Uint32, pool_token: Uint32,
                              rank_idx: Uint32, is_shared,
                              topk: cutlass.Constexpr,
                             has_shared: cutlass.Constexpr, packed_metadata: cutlass.Constexpr = False):
    """Read valid rows with fresh undefined values for inactive rows."""
    src_rank = undefined_u32()
    src_token = undefined_u32()
    topk_slot = undefined_u32()
    if row_in_block < valid_m:
        if cutlass.const_expr(has_shared):
            src_rank = rank_idx
            src_token = pool_token
            topk_slot = Uint32(topk)
            if not is_shared:
                if cutlass.const_expr(packed_metadata):
                    src_rank, src_token, topk_slot = _unpack_w2_metadata(base[pool_token])
                else:
                    src_rank, src_token, topk_slot = ld_global_u32x3(
                        base + pool_token * Uint32(3))
        else:
            if cutlass.const_expr(packed_metadata):
                src_rank, src_token, topk_slot = _unpack_w2_metadata(base[pool_token])
            else:
                src_rank, src_token, topk_slot = ld_global_u32x3(
                    base + pool_token * Uint32(3))
    return src_rank, src_token, topk_slot


@cute.jit
def _w2_vector_fragment(raw_values: cute.Tensor, atom: cutlass.Constexpr,
                        lane: Uint32):
    # movmatrix transposes each 8x8 register tile. The two lane exchanges
    # then give each lane four consecutive pairs from the same tile.
    a = transpose_bf16x2(pack_bf16_pair(raw_values[0, atom, 0], raw_values[0, atom, 1]))
    b = transpose_bf16x2(pack_bf16_pair(raw_values[0, atom, 2], raw_values[0, atom, 3]))
    c = transpose_bf16x2(pack_bf16_pair(raw_values[0, atom, 4], raw_values[0, atom, 5]))
    d = transpose_bf16x2(pack_bf16_pair(raw_values[0, atom, 6], raw_values[0, atom, 7]))
    even = (lane & Uint32(1)) == Uint32(0)
    ab = cute.arch.shuffle_sync(b if even else a, lane ^ Uint32(1))
    cd = cute.arch.shuffle_sync(d if even else c, lane ^ Uint32(1))
    aa, bb = (a if even else ab), (ab if even else b)
    cc, dd = (c if even else cd), (cd if even else d)
    lower = (lane & Uint32(2)) == Uint32(0)
    ac = cute.arch.shuffle_sync(cc if lower else aa, lane ^ Uint32(2))
    bd = cute.arch.shuffle_sync(dd if lower else bb, lane ^ Uint32(2))
    return ((aa if lower else ac), (bb if lower else bd),
            (ac if lower else cc), (bd if lower else dd))


@cute.jit
def _store_w2_vector(dst: cute.Pointer, words, sentinel: cutlass.Constexpr) -> None:
    if cutlass.const_expr(sentinel):
        # Keep double-NaN payloads distinct from an empty Combine slot.
        escaped = ()
        for index in cutlass.range_constexpr(4):
            word = words[index]
            escaped += (Uint32(0x7fff7fff) if word == Uint32(0xffffffff) else word,)
        st_global_u32x4(dst, escaped[0], escaped[1], escaped[2], escaped[3])
    else:
        st_global_u32x4(dst, words[0], words[1], words[2], words[3])


@cute.jit
def _w2_mailbox_offset(token: Uint32, slot: Uint32, channel_bytes: Uint32,
                       hidden: cutlass.Constexpr, capacity: cutlass.Constexpr,
                       slots: cutlass.Constexpr, token_major: cutlass.Constexpr,
                       interleaved_bytes: cutlass.Constexpr) -> Uint64:
    row_bytes = hidden * 2
    if cutlass.const_expr(interleaved_bytes):
        group = (Uint64(token) * Uint64(row_bytes // interleaved_bytes) +
                 Uint64(channel_bytes // Uint32(interleaved_bytes)))
        return ((group * Uint64(slots) + Uint64(slot)) * Uint64(interleaved_bytes) +
                Uint64(channel_bytes % Uint32(interleaved_bytes)))
    token_stride = slots * row_bytes if token_major else row_bytes
    slot_stride = row_bytes if token_major else capacity * row_bytes
    return (Uint64(token) * Uint64(token_stride) + Uint64(slot) * Uint64(slot_stride) +
            Uint64(channel_bytes))


@cute.jit
def _w2_store_direct(raw_values: cute.Tensor, atom: cutlass.Constexpr,
                     signals: cute.Tensor, peer_ptrs: cute.Tensor,
                     metadata_words: cutlass.Constexpr, combine_words: cutlass.Constexpr,
                     row_in_block: Uint32, pool_token: Uint32, valid_m: Uint32,
                     n_block: Uint32, warp_in_wg: Uint32, lane: Uint32,
                     rank_idx: Uint32, is_shared: Uint32, hidden: cutlass.Constexpr,
                     capacity: cutlass.Constexpr, topk: cutlass.Constexpr,
                     num_ranks: cutlass.Constexpr, has_shared: cutlass.Constexpr,
                     token_major: cutlass.Constexpr, interleaved_bytes: cutlass.Constexpr,
                     sentinel: cutlass.Constexpr = False,
                     packed_metadata: cutlass.Constexpr = False) -> None:
    src_rank, src_token, slot = _load_w2_source_metadata(
        signals.iterator + metadata_words, row_in_block, valid_m, pool_token,
        rank_idx, is_shared, topk, has_shared, packed_metadata)
    a, b, c, d = _w2_vector_fragment(raw_values, atom, lane)
    if row_in_block < valid_m:
        channel_bytes = (n_block * Uint32(256) + warp_in_wg * Uint32(64) +
                         (lane % Uint32(4)) * Uint32(16))
        dst_bytes = _w2_mailbox_offset(src_token, slot, channel_bytes, hidden, capacity,
                                       topk + has_shared, token_major, interleaved_bytes)
        dst = cute.make_ptr(
            Uint32, mapped_peer_base(peer_ptrs, src_rank, num_ranks) +
            Uint64(combine_words * 4) + dst_bytes,
            cute.AddressSpace.gmem, assumed_align=16)
        _store_w2_vector(dst, (a, b, c, d), sentinel)


@cute.jit
def _initialize_private_combine_bar(bar: cute.Pointer) -> None:
    """Publish one warp-owned barrier before that warp's first use."""
    if elect_one_sync():
        cute.arch.mbarrier_init(bar, 1)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_warp()


class GCMoERoleKernel:
    """Keep the role on the Python object so CuTe erases other role paths."""

    def __init__(self, role: int, dispatch_sms: int = 1, compute_sms: int = 2,
                 block_m: int = 16, hidden: int = 256,
                 intermediate: int = 128, num_tokens: int = 16,
                 num_ranks: int = 1, local_experts: int = 1,
                 shared_experts: int = 0, capacity: int = 1920,
                 shared_weight_gran_k: int = 32,
                 finalize_output: bool = False,
                 ring_tokens: int = 1920,
                 sf_ring_tokens: int = 30720,
                 topk: int = 1,
                 activation_clamp: float = math.inf,
                 fast_math: bool = True,
                 has_stats: bool = False,
                 nvlink_timeout_secs: int = DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
                 *, role_config, combine_geometry: tuple[int, int, int] | None = None,
                 routed_weight_linger: bool = False,
                 decode_single_publisher: bool = False,
                 route_publish_ctas: int = 1,
                 split_route_heads: bool = False,
                 decode_vector_pull: bool = False,
                 decode_vector_pull_group: int = 1,
                 decode_vector_combine: bool = False,
                 alternate_experts: bool = False,
                 retain_routed_w2: bool = False,
                 prefetch_experts: int = 0,
                 w1_prefetch_k_tiles: int = 0,
                 w1_prefix_expert_limit: int = 0,
                 keep_w1_prefix: bool = False,
                 w2_prefetch_k_tiles: int = 0, bulk_w2_tail_experts: int = 0,
                 keep_w2_prefix: bool = False,
                 prefetch_weight_scales: bool = False,
                 prefetch_k_first: bool = False,
                 weight_prefix_warps: bool = False,
                 shared_priority_prefix: int = 0,
                 post_init_prefetch: bool = False,
                 relaxed_init_barrier: bool = False,
                 skip_init_prefetch_barrier: bool = False,
                 scheduled_weight_tiles: int = 0,
                 w2_prefetch_from_w1: int = 0,
                 combine_in_compute: int = 0,
                 decode_w2_n_group: int = 1,
                 w2_a_issue_window: int = 0, decode_w2_early_workers: int = 0,
                 w2_b_issue_prefix: int = 0,
                 decode_cluster_stride: int = 0, decode_resident_clusters: int = 0,
                 keep_routed_weight_scales: bool = False,
                 w2_tile_major_input: bool = False,
                 transpose_w2_activations: int = 0,
                 single_tma_lanes: int = 0,
                 pair_routed_weight_scales: bool = False,
                 shared_w1_prefetch_limit: int = 0,
                 asymmetric_activation_rows: int = 0,
                 w1_tile_major_input: bool = False,
                 pair_routed_weights: int = 0,
                 cache_routed_ranges: bool = False,
                 single_expert_tile_lookup: bool = False,
                 prioritize_expert_rows: int = 0,
                 decode_trace_offset: int = 0,
                 fixed_expert_span: int = 0,
                 early_experts: int = 0,
                 release_group: int = 1,
                 direct_w1_store: bool = False,
                 narrow_bm32: bool = False, trim_wide_activations: bool = False,
                 short_activation_sf: bool = False,
                 single_mma_lane: bool = False,
                 trace_stride: int = 1,
                 trace_operand_wait: bool = False,
                 trace_operand_sources: bool = False,
                 reuse_k128_desc: bool | None = None,
                 mma_unroll: int = 4,
                 hoist_routed_sfb_selection: bool = False, descriptor_prefetch: str = 'all',
                 probe_routed_mma: int = 0,
                 w1_register_transpose: bool = False,
                 w1_packed_scale_reduction: bool = False,
                 w1_hardware_scale_reduction: bool = False,
                 w1_early_epi_inputs: bool = False, w1_pair_scale_rounding: bool = False,
                 prefetch_w2_metadata: bool = False,
                 packed_w2_metadata: bool = False,
                 bulk_w1_prefetch_bytes: int = 0, bulk_w1_prefetch_on_dispatch: int = 0,
                 bulk_w1_prefetch_lanes: int = 1,
                 ready_tile_offset: int = 0,
                 programmatic_finalize: bool = False,
                 compute_block_offset: int = 0,
                 static_w1_prefix: bool = False, direct_w2_store: bool = False,
                 speculative_w1: bool = False, compact_compute: bool = False,
                 speculative_w1_n_tiles: int = 0,
                 static_w2_overlap: bool = False,
                 task_slots: int = 2,
                 stripe_combine_offset: int = 0, dedup_activation_offset: int = 0,
                 source_elects_activation_owner: bool = False, weight_lookahead: int = 0,
                 packed_route_counts: bool = False, combine_load_group: int = 1,
                 inline_route_indices: bool = False,
                 dispatch_prefix_scan: bool = False, static_decode_wave: int = 0,
                 token_major_combine: bool = False, pull_route_metadata_offset: int = 0,
                 decode_w1_warmup: int = 0, decode_w1_cadence: int = 1,
                 packed_route_count_group: int = 1, posted_route_sums: bool = False,
                 decode_shared_w2: int = 0, inline_route_scores: bool = False,
                 pair_shared_w1_offset: int = 0, pair_shared_w1_vector: bool = False,
                 rotate_route_publication: int = 0,
                 prune_decode_resets: bool = False, omit_sentinel_peer_finish: bool = False,
                 decode_linear_pool: bool = False,
                 dispatch_pull_rows_per_cta: int = 0,
                 double_buffer_dispatch_pull: bool = False,
                 packed_decode_activations: bool = False, packed_decode_scores: bool = False,
                 route_count_doorbell: bool = False, dispatch_shared_route_counts: bool = False,
                 multicast_route_counts: bool = False, multicast_combine_ready: bool = False,
                 multicast_decode_input_offset: int = 0, push_decode_activations: bool = False,
                 push_overlap_reservation: bool = False, push_reserve_spans: bool = False,
                 decode_w2_clusters: int = 0, rank_route_doorbell: bool = False,
                 peer_decode_input_offset: int = 0, stripe_active_peers: bool = False,
                 stripe_launch_delay: int = 0, stripe_tiles: int = 1,
                 activation_prefix_bytes: int = 0, activation_prefix_offset: int = 0,
                 peer_activation_prefix_tiles: int = 0, peer_activation_rank_rows: int = 0,
                 peer_activation_head_clusters: int = 0,
                 parallel_activation_prefix: bool = False,
                 accumulator_stages: int = ACCUM_BARRIER_STAGES, peer_decode_input_direct: bool = False,
                 peer_decode_token_fanout: int = 0,
                 peer_prefetch_expert_heads: int = 0, peer_prefetch_opportunistic: bool = False,
                 trim_decode_gather_tail: bool = False,
                 compute_completion_offset: int = 0, decode_input_stride: int = 0,
                 multicast_pull_routes: bool = False, push_vector_io: bool = False,
                 warp_pull_routes: bool = False, decode_w2_ready_k: int = 0,
                 push_early_pool_offset: int = 0, push_row_ready_offset: int = 0,
                 dual_weight_warps: bool = False, push_scale_receiver_sms: int = 0,
                 push_packed_scales_offset: int = 0, local_mma: bool = False,
                 narrow_local_mma: bool = False, independent_compute_ctas: bool = False,
                 local_mma_n8: int = 0,
                 routed_k_major: tuple[bool, bool] = (False, False),
                 grouped_routed_weights: int = 0,
                 grouped_weight_inner_k: tuple[int, int] = (2, 2),
                 packed_routed_scale_base: int | None = None,
                 expert_combine_offset: int = 0, dispatch_backfill: bool = False,
                 combine_token_warps: int = 0, combine_interleaved_bytes: int = 0,
                 streaming_combine: bool = False, direct_combine_output: bool = False,
                 sentinel_combine: bool = False, sentinel_combine_sms: int = 0,
                 reuse_finalize_warps: bool = False,
                 sentinel_half_vector: bool = False,
                 sentinel_token_order: bool = False, sentinel_poll: str = 'full',
                 sentinel_combine_warps: int = COMBINE_WARPS, sentinel_poll_delay_ns: int = 0,
                 combine_ready_flags_offset: int = 0, sentinel_adaptive_poll: bool = False,
                 combine_ready_mask: bool = False, reset_decode_ring_on_dispatch: bool = False,
                 single_cleanup_cta: bool = False,
                 decode_operand_cache: str = 'auto', paired_shared_weights: bool = False,
                 shared_weight_packet_mask: int = 0, decode_compute_max_registers: int = 0,
                 packed_routed_scale_k_group: int = 1, packed_scale_helper: bool = False,
                 follower_scheduler: bool = False, packed_scale_release_group: int = 4):
        if role_config.role != role or role_config.block_m != block_m:
            raise ValueError('kernel role/block M differs from its host configuration')
        math_stages = role_config.num_stages
        if dual_weight_warps and (role != 2 or math_stages % 2 or hidden % 256 or intermediate % 256):
            raise ValueError('dual weight warps require even K128 iterations and stages')
        self.dual_weight_warps = dual_weight_warps
        self.local_mma = local_mma
        self.narrow_local_mma = narrow_local_mma
        self.local_mma_n8 = local_mma_n8
        self.independent_compute_ctas = independent_compute_ctas
        self.compute_cluster_size = 1 if independent_compute_ctas else 2
        self.dispatch_backfill = dispatch_backfill
        self.reset_decode_ring_on_dispatch = reset_decode_ring_on_dispatch
        self.paired_shared_weights = paired_shared_weights
        self.shared_weight_packet_mask = shared_weight_packet_mask
        self.scheduler_scale = 2 // self.compute_cluster_size
        self.routed_k_major = routed_k_major
        self.grouped_routed_weights = grouped_routed_weights
        self.grouped_weight_inner_k = grouped_weight_inner_k
        self.packed_routed_scale_base = packed_routed_scale_base
        self.packed_routed_scale_k_group = packed_routed_scale_k_group
        self.packed_scale_helper = packed_scale_helper
        self.packed_scale_release_group = packed_scale_release_group
        self.mma_cta_group = 1 if local_mma else 2
        self.activation_stage_rows = role_config.load_m
        self.task_scheduler_cta = 1 if dual_weight_warps or follower_scheduler else 0
        if role not in (1, 2, 3):
            raise ValueError('role must be dispatch, compute or finalize')
        if local_experts < 1 or local_experts > 512 or \
                num_ranks * local_experts > 2048 or topk < 1 or \
                topk + bool(shared_experts) > 32:
            raise NotImplementedError('device scheduler requires 1..512 local experts, <=2048 global experts, topk>=1 and topk+shared slot<=32')
        if num_ranks < 1 or num_ranks > MAX_RANKS or not 0 <= num_tokens <= capacity:
            raise NotImplementedError('dispatch routing requires 1..72 ranks within capacity')
        if math_stages < 2 or math_stages > 32:
            raise ValueError('GC math pipeline needs 2..32 TMA stages')
        if not 0 < nvlink_timeout_secs <= 2_147_483_647:
            raise ValueError('NVLink barrier timeout must fit positive INT_MAX seconds')
        self.role = role
        self.fused_dispatch_role = None
        self.fused_finalize_role = None
        self.fused_finalize_sentinel = 0
        self.compute_completion_offset = compute_completion_offset
        self.fused_worker_reuse = False
        self.decode_single_publisher = role == 1 and decode_single_publisher
        self.route_publish_ctas = route_publish_ctas
        self.split_route_heads = split_route_heads
        self.decode_vector_pull = role == 1 and decode_vector_pull
        self.decode_vector_pull_group = decode_vector_pull_group
        self.decode_vector_combine = role == 3 and decode_vector_combine
        self.alternate_experts = role != 3 and alternate_experts
        self.retain_routed_w2 = role == 2 and retain_routed_w2
        self.prefetch_experts = (min(prefetch_experts, local_experts) if role == 2 or
                                 (role == 1 and bulk_w1_prefetch_on_dispatch) else 0)
        self.w1_prefetch_k_tiles = w1_prefetch_k_tiles
        self.w1_prefix_experts = w1_prefix_expert_limit or local_experts
        self.keep_w1_prefix = keep_w1_prefix
        self.w2_prefetch_k_tiles = w2_prefetch_k_tiles
        self.bulk_w2_tail_experts = bulk_w2_tail_experts
        self.keep_w2_prefix = keep_w2_prefix
        self.prefetch_weight_scales = prefetch_weight_scales
        self.prefetch_k_first = prefetch_k_first
        self.weight_prefix_warps = weight_prefix_warps
        self.post_init_prefetch = post_init_prefetch
        self.relaxed_init_barrier = relaxed_init_barrier
        self.skip_init_prefetch_barrier = skip_init_prefetch_barrier
        self.scheduled_weight_tiles = scheduled_weight_tiles
        self.w2_prefetch_from_w1 = w2_prefetch_from_w1
        self.combine_in_compute = combine_in_compute
        self.decode_w2_n_group = decode_w2_n_group
        self.w2_a_issue_window = w2_a_issue_window
        self.w2_b_issue_prefix = w2_b_issue_prefix
        self.decode_cluster_stride = decode_cluster_stride
        self.decode_resident_clusters = decode_resident_clusters
        self.decode_w2_early_workers = decode_w2_early_workers
        self.keep_routed_weight_scales = keep_routed_weight_scales
        self.w2_tile_major_input = w2_tile_major_input
        self.transpose_w2_activations = transpose_w2_activations
        self.single_tma_lanes = single_tma_lanes
        self.pair_routed_weight_scales = pair_routed_weight_scales
        self.shared_w1_prefetch_limit = shared_w1_prefetch_limit
        self.asymmetric_activation_rows = asymmetric_activation_rows
        self.w1_tile_major_input = w1_tile_major_input
        self.pair_routed_weights = pair_routed_weights
        self.cache_routed_ranges = cache_routed_ranges
        self.single_expert_tile_lookup = single_expert_tile_lookup
        self.prioritize_expert_rows = prioritize_expert_rows
        self.decode_trace_offset = decode_trace_offset
        self.fixed_expert_span = fixed_expert_span
        self.early_experts = early_experts
        self.release_group = release_group
        self.direct_w1_store = direct_w1_store
        self.narrow_bm32 = narrow_bm32
        self.short_activation_sf = short_activation_sf
        self.single_mma_lane = single_mma_lane
        self.trace_stride = trace_stride
        self.trace_operand_wait = trace_operand_wait
        self.trace_operand_sources = trace_operand_sources
        self.ready_tile_offset = ready_tile_offset
        self.programmatic_finalize = programmatic_finalize
        self.compute_block_offset = compute_block_offset
        self.static_w1_prefix = static_w1_prefix
        self.direct_w2_store = direct_w2_store
        self.speculative_w1 = speculative_w1
        self.speculative_w1_n_tiles = speculative_w1_n_tiles
        self.static_w2_overlap = static_w2_overlap
        self.task_slots = task_slots
        self.compact_compute = compact_compute
        self.stripe_combine_offset = stripe_combine_offset
        self.expert_combine_offset = expert_combine_offset
        self.expert_ready_offset = expert_combine_offset + (local_experts + 1) * 8
        self.dedup_activation_offset = dedup_activation_offset
        self.source_elects_activation_owner = source_elects_activation_owner
        self.weight_lookahead = weight_lookahead
        self.packed_route_counts = packed_route_counts
        self.inline_route_indices = inline_route_indices
        self.packed_route_count_group = packed_route_count_group
        self.rank_route_doorbell = rank_route_doorbell
        self.posted_route_sums = posted_route_sums
        self.inline_route_scores = inline_route_scores
        self.rotate_route_publication = rotate_route_publication
        self.prune_decode_resets = prune_decode_resets
        self.decode_linear_pool = decode_linear_pool
        self.dispatch_pull_rows_per_cta = dispatch_pull_rows_per_cta
        self.double_buffer_dispatch_pull = double_buffer_dispatch_pull
        self.omit_sentinel_peer_finish = omit_sentinel_peer_finish
        self.packed_decode_activations = role == 1 and packed_decode_activations
        self.packed_decode_scores = role == 1 and packed_decode_scores
        self.route_count_doorbell = route_count_doorbell
        self.dispatch_shared_route_counts = dispatch_shared_route_counts
        self.multicast_route_counts = multicast_route_counts
        self.multicast_combine_ready = multicast_combine_ready
        self.stripe_active_peers = stripe_active_peers
        self.stripe_launch_delay = stripe_launch_delay
        self.stripe_tiles = stripe_tiles
        self.activation_prefix_bytes = activation_prefix_bytes
        self.activation_prefix_offset = activation_prefix_offset
        self.peer_activation_prefix_tiles = peer_activation_prefix_tiles
        self.peer_activation_rank_rows = peer_activation_rank_rows
        self.peer_activation_head_clusters = peer_activation_head_clusters
        self.parallel_activation_prefix = parallel_activation_prefix
        self.multicast_decode_input_offset = multicast_decode_input_offset
        self.peer_decode_input_offset = peer_decode_input_offset
        self.peer_decode_input_direct = peer_decode_input_direct
        self.peer_decode_token_fanout = peer_decode_token_fanout
        self.peer_prefetch_expert_heads = peer_prefetch_expert_heads
        self.peer_prefetch_opportunistic = peer_prefetch_opportunistic
        self.trim_decode_gather_tail = trim_decode_gather_tail
        self.decode_input_inbox_offset = multicast_decode_input_offset or peer_decode_input_offset
        self.push_decode_activations = push_decode_activations
        self.push_vector_io = push_vector_io
        self.push_overlap_reservation = push_overlap_reservation
        self.push_reserve_spans = push_reserve_spans
        self.push_early_pool_offset = push_early_pool_offset
        self.push_row_ready_offset = push_row_ready_offset
        self.push_scale_receiver_sms = push_scale_receiver_sms
        self.push_packed_scales_offset = push_packed_scales_offset
        self.pull_route_metadata_offset = pull_route_metadata_offset
        self.multicast_pull_routes = multicast_pull_routes
        self.warp_pull_routes = warp_pull_routes
        self.dispatch_prefix_scan = dispatch_prefix_scan
        self.static_decode_wave = static_decode_wave
        self.decode_w1_warmup = decode_w1_warmup
        self.decode_w1_cadence = decode_w1_cadence
        self.decode_shared_w2 = decode_shared_w2
        self.pair_shared_w1_offset = pair_shared_w1_offset
        self.pair_shared_w1_vector = pair_shared_w1_vector
        self.decode_w2_clusters = decode_w2_clusters
        self.decode_w2_ready_k = decode_w2_ready_k
        self.combine_load_group = combine_load_group
        self.combine_token_warps = combine_token_warps
        self.combine_interleaved_bytes = combine_interleaved_bytes
        self.streaming_combine = streaming_combine
        self.direct_combine_output = direct_combine_output
        self.sentinel_combine = sentinel_combine
        self.sentinel_token_order = sentinel_token_order
        self.sentinel_poll = sentinel_poll
        self.sentinel_poll_delay_ns = sentinel_poll_delay_ns
        self.sentinel_adaptive_poll = sentinel_adaptive_poll
        self.combine_ready_flags_offset = combine_ready_flags_offset
        self.combine_ready_mask = combine_ready_mask
        self.combine_warps = sentinel_combine_warps if sentinel_combine else COMBINE_WARPS
        self.reuse_finalize_warps = reuse_finalize_warps
        self.single_cleanup_cta = single_cleanup_cta
        self.sentinel_half_vector = sentinel_half_vector
        self.finalize_threads = self.combine_warps * 32 + (0 if reuse_finalize_warps else 128)
        self.token_major_combine = token_major_combine
        self.dispatch_sms = dispatch_sms
        self.compute_sms = compute_sms
        self.block_m = block_m
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.intermediate = intermediate
        # Base grid; only role3 with finalize_output overrides it below.
        self.finalize_sms = (min(compute_sms, SMALL_DECODE_FINALIZE_SMS)
                             if num_tokens <= 128 else compute_sms)
        self.num_ranks = num_ranks
        self.local_experts = local_experts
        self.topk = topk
        self.has_activation_clamp = not math.isinf(activation_clamp)
        self.activation_clamp = activation_clamp if self.has_activation_clamp else 0.0
        self.fast_math = fast_math
        self.has_stats = has_stats
        self.shared_experts = shared_experts
        self.use_small_tile_path = not local_mma and use_small_tile_path(shared_experts, block_m)
        self.trim_wide_activations = trim_wide_activations
        self.reuse_k128_desc = self.use_small_tile_path if reuse_k128_desc is None else reuse_k128_desc
        self.mma_unroll = mma_unroll
        self.hoist_routed_sfb_selection = hoist_routed_sfb_selection
        self.descriptor_prefetch = descriptor_prefetch
        self.probe_routed_mma = probe_routed_mma
        self.w1_register_transpose = w1_register_transpose
        self.w1_packed_scale_reduction = w1_packed_scale_reduction
        self.w1_hardware_scale_reduction = w1_hardware_scale_reduction
        self.w1_early_epi_inputs = w1_early_epi_inputs
        self.w1_pair_scale_rounding = w1_pair_scale_rounding
        self.prefetch_w2_metadata = prefetch_w2_metadata
        self.packed_w2_metadata = packed_w2_metadata
        self.bulk_w1_prefetch_bytes = bulk_w1_prefetch_bytes
        self.bulk_w1_prefetch_on_dispatch = bulk_w1_prefetch_on_dispatch
        self.bulk_w1_prefetch_lanes = bulk_w1_prefetch_lanes
        self.capacity = capacity
        self.finalize_output = finalize_output
        self.shared_weight_gran_k = shared_weight_gran_k
        # Use FIRST for shared weights/SFs with larger decode tiles and
        # at most one routed block per expert on average. Keep BM16 NORMAL.
        self.shared_weight_cache_hint = (
            EVICT_FIRST if block_m >= 32 and
            num_tokens * topk <= block_m * local_experts
            else EVICT_NORMAL)
        self.activation_cache_hint = {
            'activation-normal': EVICT_NORMAL, 'activation-first': EVICT_FIRST,
        }.get(decode_operand_cache, EVICT_LAST)
        self.routed_weight_cache_hint = {
            'weights-normal': EVICT_NORMAL, 'weights-last': EVICT_LAST,
        }.get(decode_operand_cache, EVICT_NORMAL if routed_weight_linger else EVICT_FIRST)
        if decode_operand_cache in ('weights-normal', 'weights-last'):
            self.shared_weight_cache_hint = self.routed_weight_cache_hint
        self.math_stages = math_stages
        self.nvlink_timeout_secs = nvlink_timeout_secs
        # Block configuration and the shared C/D storage union.
        self.store_m_l1 = role_config.store_m_l1
        self.store_m_l2 = role_config.store_m_l2
        self.compute_threads = role_config.threads
        self.epilogue_threads = role_config.threads - 128 if role == 2 else 256
        self.epilogue_rows = block_m // (self.epilogue_threads // 128)
        self.cd_l1_bytes = 2 * 2 * self.store_m_l1 * 64
        self.cd_l2_bytes = 2 * self.store_m_l2 * 128 * 2
        self.cd_bytes = max(self.cd_l1_bytes, self.cd_l2_bytes)
        # float2[kNumEpilogueWarps][STORE_BLOCK_M_L1 / 2].
        self.amax_float_count = 8 * self.store_m_l1
        self.non_epi_registers = 40 if local_experts <= 64 else 72
        if role == 2 and combine_in_compute in (1, 2):
            self.non_epi_registers = 64 if sentinel_poll == 'ordered' else 88
        self.epi_registers = 208 if local_experts <= 64 else 168
        self.decode_compute_max_registers = decode_compute_max_registers
        if decode_compute_max_registers:
            # Keep the runtime warpgroup allocation inside the static CTA
            # register reservation, leaving registers for four-warp Combine.
            epilogue_budget = (decode_compute_max_registers * self.compute_threads -
                               128 * self.non_epi_registers) // self.epilogue_threads
            self.epi_registers = min(self.epi_registers, epilogue_budget // 8 * 8)
        tile = MixedMmaTile(block_m, hidden, intermediate, epilogue_stages=accumulator_stages)
        self.l1_clusters = tile.l1_clusters
        self.l2_clusters = tile.l2_clusters
        self.shared_prefix_ctas = 0
        if shared_priority_prefix:
            cluster_count = self.l1_clusters * shared_experts
            if shared_priority_prefix == 2:
                cluster_count = max(cluster_count, self.l2_clusters)
            self.shared_prefix_ctas = cluster_count * 2
            if self.shared_prefix_ctas >= compute_sms:
                raise ValueError('Shared priority must leave Routed prefetch workers')
        self.stripe_groups = self.l2_clusters // stripe_tiles
        self.stripe_ready_offset = stripe_combine_offset + self.stripe_groups * 8
        self.l1_full_mask = (1 << tile.l1_n_blocks) - 1
        self.tmem_cols = tile.tmem_cols
        self.probe_mma_offset = 0
        if probe_routed_mma:
            used_columns = tile.accum_tmem_cols + tile.sfa_tmem_cols + tile.sfb_tmem_cols
            self.probe_mma_offset = (used_columns + 15) // 16 * 16
            if self.probe_mma_offset + block_m > self.tmem_cols:
                raise ValueError('MMA probe needs one unused accumulator in the existing TMEM allocation')
        self.accum_stages = tile.epilogue_stages
        self.sf_block_m = role_config.sf_block_m
        self.sfa_tmem_cols = tile.sfa_tmem_cols
        self.dispatch_warps = role_config.routing_warps
        dispatch_registers = 48 if local_experts <= 64 else 96
        role_registers = (
            self.dispatch_warps * 32 * dispatch_registers if role == 1 else
            128 * self.non_epi_registers + self.epilogue_threads * self.epi_registers if role == 2 else
            128 * dispatch_registers + (COMBINE_WARPS * 32) * self.epi_registers)
        if role_registers > 64512:
            raise ValueError('role register allocation exceeds the SM budget')
        self.pull_bytes = role_config.pull_bytes
        self.decode_input_stride = decode_input_stride or self.pull_bytes
        self.pull_chunks = 1 if self.packed_decode_activations else hidden // self.pull_bytes
        if role == 3 and finalize_output:
            if combine_geometry is None:
                raise ValueError('finalize requires host combine geometry')
            (self.combine_chunks, self.combine_chunk_bytes,
             self.combine_uint4_per_lane) = combine_geometry
            # Size fused output work independently of the expert-cleanup grid.
            self.finalize_sms = min(
                compute_sms,
                max(SMALL_DECODE_FINALIZE_SMS if num_tokens <= 128 else
                    min(compute_sms, local_experts + 1),
                    (num_tokens * self.combine_chunks + self.combine_warps - 1) //
                    self.combine_warps))
            if stripe_combine_offset:
                self.finalize_sms = self.stripe_groups * (2 if num_tokens > 16 else 1)
            if combine_token_warps:
                self.finalize_sms = num_tokens * self.combine_chunks // combine_token_warps
            if sentinel_combine_sms:
                self.finalize_sms = min(self.finalize_sms, sentinel_combine_sms)
        if role == 3 and combine_in_compute:
            self.finalize_threads = 128
            self.finalize_sms = 1
        self.shared_layout = role_config.shared_layout
        if self.shared_layout.accum_empty - self.shared_layout.accum_full != self.accum_stages * 8:
            raise ValueError('accumulator queue depth differs from its barrier layout')
        layout = GCBufferLayout(num_ranks, num_ranks * local_experts, capacity, topk,
                                hidden, intermediate, ring_tokens, sf_ring_tokens,
                                shared_experts, packed_decode_activations, packed_decode_scores)
        self.topk_offset_words = layout.views['topk_idx'].offset // 4
        self.input_x_words = layout.views['x'].offset // 4
        self.input_sf_words = layout.views['x_sf'].offset // 4
        self.input_weight_words = layout.views['topk_weights'].offset // 4
        self.l1_x_words = layout.views['l1_acts'].offset // 4
        self.l1_sf_words = layout.views['l1_acts_sf'].offset // 4
        self.l1_weight_words = layout.views['l1_topk_weights'].offset // 4
        self.l2_x_words = layout.views['l2_acts'].offset // 4
        self.l2_sf_words = layout.views['l2_acts_sf'].offset // 4
        self.shared_l1_sf_words = layout.views['shared_l1_acts_sf'].offset // 4
        self.shared_l2_x_words = layout.views['shared_l2_acts'].offset // 4
        self.shared_l2_sf_words = layout.views['shared_l2_acts_sf'].offset // 4
        self.shared_sf_tokens = ((capacity + 7) // 8) * 128
        self.combine_words = layout.views['combine_tokens'].offset // 4
        self.metadata_words = layout.workspace_offsets['token_src_metadata'] // 4
        self.ring_tokens = layout.ring_tokens
        self.sf_ring_tokens = layout.sf_ring_tokens
        self.ring_blocks = layout.ring_tokens // block_m
        sf_block_m = self.sf_block_m
        sf_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                         for t in range(block_m))
        sf_even_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                              for t in range(0, block_m, 2))
        last_ring_sf_row = (self.ring_blocks - 1) * sf_block_m
        l1_sf_local_max = ((hidden // 128 - 1) * self.sf_ring_tokens +
                           last_ring_sf_row + sf_row_max)
        routed_w1_sf_local_max = 4 * (
            (intermediate // 128 - 1) * self.sf_ring_tokens +
            last_ring_sf_row + sf_even_row_max) + 19
        self.wide_l1_sf = (l1_sf_local_max > 0xffffffff or
                           self.l1_sf_words + l1_sf_local_max > 0xffffffff)
        self.wide_routed_w1_sf = routed_w1_sf_local_max > 0xffffffff
        self.wide_shared_w1_sf = False
        if shared_experts > 0:
            last_shared_sf_row = ((capacity + block_m - 1) // block_m - 1) * sf_block_m
            shared_w1_sf_local_max = 4 * (
                (shared_experts * intermediate // 128 - 1) *
                self.shared_sf_tokens + last_shared_sf_row + sf_even_row_max) + 19
            self.wide_shared_w1_sf = shared_w1_sf_local_max > 0xffffffff

    @cute.jit
    def _initialize_shared_storage(self, slab: cute.Pointer,
                                    physical_warp: Uint32, lane: Uint32,
                                    trace_signals=None, trace_block: Uint32 = 0,
                                    trace_start: Uint64 = 0) -> None:
        if cutlass.const_expr(trace_signals is not None and self.decode_trace_offset != 0):
            if physical_warp == Uint32(3) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(4),
                         global_timer() - trace_start)
        if cutlass.const_expr(not (self.independent_compute_ctas or
                                   self.skip_init_prefetch_barrier)):
            # Both warps initialize CTA-local state only, so the post-init
            # rendezvous below is what publishes it to the peer. skip_init_
            # prefetch_barrier drops this pre-init arrive.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        if cutlass.const_expr(trace_signals is not None and self.decode_trace_offset != 0):
            if physical_warp == Uint32(3) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(5),
                         global_timer() - trace_start)
        if physical_warp == 0:
            if elect_one_sync():
                if cutlass.const_expr(self.role == 1):
                    bulk_zero_shared(
                        slab + self.shared_layout.expert_counts,
                        Uint32(((self.num_ranks * self.local_experts * 4 + 1023) //
                                1024) * 1024))
        elif physical_warp == 1:
            if cutlass.const_expr(self.role == 1 and not (self.decode_vector_pull or self.push_vector_io)):
                bars = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
                for index in cutlass.range(
                        Int32(lane), Int32(self.dispatch_warps), Int32(32), unroll_full=True):
                    cute.arch.mbarrier_init(bars + index, 1)
                    if cutlass.const_expr(self.parallel_activation_prefix or
                                          self.double_buffer_dispatch_pull):
                        cute.arch.mbarrier_init(bars + index + self.dispatch_warps, 1)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 2:
            # CuTe 4.8.1's single-thread-per-CTA docstring is conservative
            # for disjoint barriers; the wrapper does not elect a lane.
            # Each object below has exactly one initializer. Every such
            # lane executes the init fence before the cluster rendezvous
            # publishes initialization to consumers. The constructor's
            # math_stages <= 32 bound keeps all stages within one warp.
            if cutlass.const_expr(self.role == 2):
                cta_rank, _, _ = cute.arch.block_in_cluster_idx()
                data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
                weight_full = cute.recast_ptr(slab + self.shared_layout.weight_full, dtype=Uint64)
                data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
                if lane < self.math_stages:
                    if cta_rank == 0 or cutlass.const_expr(self.local_mma):
                        if cutlass.const_expr(self.trace_operand_sources):
                            cute.arch.mbarrier_init(data_full + lane, self.mma_cta_group)
                            cute.arch.mbarrier_init(weight_full + lane, self.mma_cta_group)
                        else:
                            producers = 3 if self.packed_scale_helper else 2
                            cute.arch.mbarrier_init(data_full + lane, producers * self.mma_cta_group)
                    cute.arch.mbarrier_init(data_empty + lane, 1)
                accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
                accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
                if lane < self.accum_stages:
                    cute.arch.mbarrier_init(accum_full + lane, 1)
                    if cta_rank == 0 or cutlass.const_expr(self.local_mma):
                        cute.arch.mbarrier_init(accum_empty + lane, self.mma_cta_group * self.epilogue_threads)
                full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
                empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
                if lane < self.task_slots:
                    cute.arch.mbarrier_init(full + lane, 1)
                    if cta_rank == self.task_scheduler_cta:
                        readers = self.compute_cluster_size * self.epilogue_threads
                        if cutlass.const_expr(self.packed_scale_helper):
                            readers += self.compute_cluster_size
                        if cutlass.const_expr(self.scheduled_weight_tiles or self.w2_prefetch_from_w1):
                            readers += 1
                        cute.arch.mbarrier_init(empty + lane, readers)
            elif cutlass.const_expr(self.role == 3):
                combine = cute.recast_ptr(slab + self.shared_layout.combine_bars, dtype=Uint64)
                if lane < COMBINE_WARPS:
                    cute.arch.mbarrier_init(combine + lane, 1)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 3:
            if cutlass.const_expr(self.role == 2):
                tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
                cute.arch.alloc_tmem(self.tmem_cols, tptr, is_two_cta=not self.local_mma)
        if cutlass.const_expr(trace_signals is not None and self.decode_trace_offset != 0):
            if physical_warp == Uint32(0) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(14),
                         global_timer() - trace_start)
            elif physical_warp == Uint32(1) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(15),
                         global_timer() - trace_start)
            elif physical_warp == Uint32(2) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(13),
                         global_timer() - trace_start)
            elif physical_warp == Uint32(3) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(6),
                         global_timer() - trace_start)
        if cutlass.const_expr(self.independent_compute_ctas):
            cute.arch.sync_threads()
        elif cutlass.const_expr(self.relaxed_init_barrier):
            # Every initializer already released through mbarrier_init_fence,
            # so the second rendezvous only needs the acquire of the wait.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        if cutlass.const_expr(trace_signals is not None and self.decode_trace_offset != 0):
            if physical_warp == Uint32(3) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(7),
                         global_timer() - trace_start)
        if cutlass.const_expr(not (self.role == 2 and self.dispatch_backfill)):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(trace_signals is not None and self.decode_trace_offset != 0):
            if physical_warp == Uint32(3) and lane == Uint32(0):
                cta_mark(trace_signals, self.decode_trace_offset, 2, trace_block, Uint32(8),
                         global_timer() - trace_start)

    @cute.jit
    def _route_publication_order(self, dst_rank: Uint32, dst_local: Uint32,
                                  src_rank: Uint32) -> tuple[Uint32, Uint32]:
        if cutlass.const_expr(self.rotate_route_publication):
            # Start each source at a different recipient. Rotate the expert
            # within the rank as well when comparing atomic contention.
            dst_rank = (dst_rank + src_rank) % Uint32(self.num_ranks)
            if cutlass.const_expr(self.rotate_route_publication == 2):
                dst_local = (dst_local + src_rank) % Uint32(self.local_experts)
        return dst_rank, dst_local

    @cute.jit
    def _split_route_destination(self, record: Uint32, publisher: Uint32) -> tuple[Uint32, Uint32]:
        tail_width = self.local_experts - self.early_experts
        rank = record // Uint32(tail_width)
        expert = Uint32(self.early_experts) + record % Uint32(tail_width)
        if publisher == Uint32(0):
            rank = record // Uint32(self.early_experts)
            expert = record % Uint32(self.early_experts)
        return rank, expert

    @cute.jit
    def _load_raw_peer_activation(self, dst: cute.Pointer, first_row: Uint32,
                                  physical_rows: Uint32, valid_rows: Uint32,
                                  lane: Uint32, peer_info: cute.Tensor, source_rank: Uint32,
                                  source_token: Uint32, k_tile: Uint32):
        for batch in cutlass.range_constexpr(self.block_m // 8):
            # A group of eight lanes reads one row of 128 FP8 channels.
            row = Uint32(batch * 4) + lane // Uint32(8)
            if row < physical_rows:
                rank = cute.arch.shuffle_sync(source_rank, row)
                token = cute.arch.shuffle_sync(source_token, row)
                valid = first_row + row < valid_rows
                src = mapped_peer_base(peer_info, rank, self.num_ranks) + Uint64(self.input_x_words * 4) + \
                    Uint64(token) * Uint64(self.hidden) + Uint64(k_tile * Uint32(128))
                segment = lane & Uint32(7)
                swizzled = segment ^ (row & Uint32(7))
                cp_async_raw16(dst + row * Uint32(128) + swizzled * Uint32(16),
                               src + Uint64(segment * Uint32(16)),
                               Uint32(16) if valid else Uint32(0))
        cp_async_raw_wait()
        cute.arch.sync_warp()
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _prefetch_routed_w1_bulk(self, peer_ptrs: cute.Tensor, first: Uint32,
                                 stride: cutlass.Constexpr) -> None:
        base = ld_peer_param_u64(peer_ptrs.iterator + ROUTED_W1_PREFETCH_OFFSET + 1)
        expert_bytes = self.hidden * self.intermediate
        prefix_bytes = (self.w1_prefetch_k_tiles * 128 or self.hidden) * self.intermediate
        chunks_per_expert = prefix_bytes // self.bulk_w1_prefetch_bytes
        experts = self.w1_prefix_experts if self.w1_prefetch_k_tiles else self.prefetch_experts
        chunks = experts * chunks_per_expert
        for chunk in cutlass.range(first, Uint32(chunks), stride):
            offset = Uint64(chunk) * Uint64(self.bulk_w1_prefetch_bytes)
            if cutlass.const_expr(self.w1_prefetch_k_tiles):
                expert = chunk // Uint32(chunks_per_expert)
                stripe = chunk % Uint32(chunks_per_expert)
                offset = (Uint64(expert) * Uint64(expert_bytes) +
                          Uint64(stripe) * Uint64(self.bulk_w1_prefetch_bytes))
            bulk_prefetch_l2(
                base + offset, Uint32(self.bulk_w1_prefetch_bytes),
                cache_hint=EVICT_LAST if self.keep_w1_prefix else None)

    @cute.jit
    def _dispatch_prefetch_routed_w1(self, peer_ptrs: cute.Tensor, block: Uint32,
                                     warp: Uint32) -> None:
        prefetch_warps = min(8, self.dispatch_warps)
        if block > Uint32(0) and warp < prefetch_warps and elect_one_sync():
            first = (block - Uint32(1)) * Uint32(prefetch_warps) + warp
            self._prefetch_routed_w1_bulk(peer_ptrs, first, (self.dispatch_sms - 1) * prefetch_warps)

    @cute.jit
    def _dispatch_prefetch_w2_tail(self, peer_ptrs: cute.Tensor, block: Uint32,
                                  warp: Uint32) -> None:
        request_bytes = 64 * 1024
        expert_bytes = self.hidden * self.intermediate // 2
        chunks_per_expert = self.w2_prefetch_k_tiles * 128 * self.hidden // (2 * request_bytes)
        if elect_one_sync():
            base = ld_peer_param_u64(peer_ptrs.iterator + ROUTED_W2_PREFETCH_OFFSET + 1)
            for chunk in cutlass.range(warp * Uint32(self.dispatch_sms) + block,
                    Uint32(chunks_per_expert * self.bulk_w2_tail_experts),
                    self.dispatch_sms * self.dispatch_warps):
                expert, stripe = chunk // Uint32(chunks_per_expert), chunk % Uint32(chunks_per_expert)
                address = base + Uint64(expert) * Uint64(expert_bytes) + Uint64(stripe) * Uint64(request_bytes)
                bulk_prefetch_l2(address, Uint32(request_bytes))

    @cute.jit
    def _prefetch_decode_weights(self, signals: cute.Tensor, first_tile: Uint32,
                                 tile_stride: cutlass.Constexpr, w1_map: cute.Pointer,
                                 w1_sf_map: cute.Pointer, w2_map: cute.Pointer,
                                 w2_sf_map: cute.Pointer) -> None:
        if cutlass.const_expr(not self.bulk_w1_prefetch_bytes and
                              (self.prefetch_experts or self.w1_prefetch_k_tiles)):
            # Prefetch a disjoint set of W1 input tiles while Dispatch
            # publishes counts. A short prefix across all experts
            # avoids guessing which expert's activations arrive first.
            reverse = Uint32(0)
            if cutlass.const_expr(self.alternate_experts):
                reverse = ld_volatile_u32(signals.iterator +
                    SIGNAL_OFFSETS['recv_parity'] // 4) & Uint32(1)
            n_tiles = self.l1_clusters * 2
            k_step = 2 if self.pair_routed_weights & 1 else 1
            k_tiles = (self.w1_prefetch_k_tiles or self.hidden // 128) // k_step
            experts = self.w1_prefix_experts if self.w1_prefetch_k_tiles else self.prefetch_experts
            tiles_per_expert = n_tiles * k_tiles
            if elect_one_sync():
                for tile in cutlass.range(first_tile,
                        Uint32(tiles_per_expert * experts), tile_stride):
                    tile_group = tile // n_tiles
                    if cutlass.const_expr(self.prefetch_k_first):
                        expert_slot = tile_group % experts
                        k_tile = tile_group // experts
                    else:
                        expert_slot = tile_group // k_tiles
                        k_tile = tile_group % k_tiles
                    k_tile *= Uint32(k_step)
                    expert = physical_expert(expert_slot, reverse, self.local_experts)
                    outer = (k_tile, expert)
                    if cutlass.const_expr(self.routed_k_major[0]):
                        outer = (expert, k_tile)
                    n_coord = tile % n_tiles * Uint32(128)
                    if cutlass.const_expr(self.grouped_routed_weights & 1):
                        n_coord, tile_coord, pair = grouped_routed_weight_coords(
                            k_tile, expert, tile % n_tiles, Uint32(self.hidden // 128), Uint32(n_tiles),
                            Uint32(self.grouped_weight_inner_k[0]))
                        outer = (tile_coord, pair)
                    prefetch_g2s_4d(w1_map, Uint32(0), n_coord,
                                    outer[0], outer[1],
                                    cache_hint=EVICT_LAST if self.keep_w1_prefix else None)
                    if cutlass.const_expr(self.prefetch_weight_scales):
                        scale_k = expert * Uint32(self.hidden // 128) + k_tile
                        prefetch_g2s_2d(w1_sf_map, tile % n_tiles * Uint32(128), scale_k)
        if cutlass.const_expr(self.w2_prefetch_k_tiles):
            # Prime the first W2 tiles while W1 and Dispatch start.
            n_tiles = self.l2_clusters * 2
            k_step = 2 if self.pair_routed_weights & 2 else 1
            k_tiles = self.w2_prefetch_k_tiles // k_step
            tiles_per_expert = n_tiles * k_tiles
            if elect_one_sync():
                for tile in cutlass.range(first_tile,
                        Uint32(tiles_per_expert * self.local_experts), tile_stride):
                    tile_group = tile // n_tiles
                    if cutlass.const_expr(self.prefetch_k_first):
                        expert = tile_group % self.local_experts
                        k_tile = tile_group // self.local_experts
                    else:
                        expert = tile_group // k_tiles
                        k_tile = tile_group % k_tiles
                    k_tile *= Uint32(k_step)
                    outer = (expert, k_tile) if self.routed_k_major[1] else (k_tile, expert)
                    n_coord = tile % n_tiles * Uint32(128)
                    if cutlass.const_expr(self.grouped_routed_weights & 2):
                        n_coord, tile_coord, pair = grouped_routed_weight_coords(
                            k_tile, expert, tile % n_tiles, Uint32(self.intermediate // 128), Uint32(n_tiles),
                            Uint32(self.grouped_weight_inner_k[1]))
                        outer = (tile_coord, pair)
                    prefetch_g2s_4d(w2_map, Uint32(0), n_coord,
                                    outer[0], outer[1],
                                    cache_hint=EVICT_LAST if self.keep_w2_prefix else None)
                    if cutlass.const_expr(self.prefetch_weight_scales):
                        scale_k = expert * Uint32(self.intermediate // 128) + k_tile
                        prefetch_g2s_2d(w2_sf_map, tile % n_tiles * Uint32(128), scale_k)

    @cute.jit
    def _store_dispatch_to_w1_tiles(self, signals: cute.Tensor, warp_pull: cute.Pointer,
                                     ring_block: Uint32, row: Uint32, lane: Uint32) -> None:
        # Four K128 planes share a warp's 512-byte source vector.
        row_offset = (Uint64(ring_block) * Uint64(self.hidden // 128 * self.block_m) +
                      Uint64(row)) * Uint64(128)
        base = (cute.recast_ptr(signals.iterator, dtype=Uint8) +
                Uint64(self.l1_x_words * 4) + row_offset)
        lane_plane = lane // Uint32(8)
        column = lane % Uint32(8) * Uint32(16)
        for strip in cutlass.range_constexpr(self.hidden // 512):
            data = ld_shared_u32x4(warp_pull + Uint32(strip * 128) + lane * Uint32(4))
            plane = Uint32(strip * 4) + lane_plane
            dst = cute.recast_ptr(base + Uint64(plane * Uint32(self.block_m * 128) + column),
                                  dtype=Uint32)
            st_global_u32x4(dst, data[0], data[1], data[2], data[3])

    @cute.jit
    def _prefetch_shared_w1(self, first_tile: Uint32, weight_map: cute.Pointer) -> None:
        # Leave the resident TMA pipe's initial window to the demand loads.
        n_tiles = self.l1_clusters * self.shared_experts * 2
        end = min(self.shared_w1_prefetch_limit, self.hidden // 128)
        if elect_one_sync():
            for tile in cutlass.range(first_tile, Uint32((end - self.math_stages) * n_tiles),
                                      self.compute_sms):
                k = Uint32(self.math_stages) + tile // Uint32(n_tiles)
                n = (tile % Uint32(n_tiles)) * Uint32(128)
                prefetch_g2s_4d(weight_map, Uint32(0), n, k, Uint32(0))

    @cute.jit
    def _prefetch_task_weight_range(self, weight_map: cute.Pointer, expert: Uint32,
                                     n_cluster: Uint32, start: cutlass.Constexpr,
                                     end: cutlass.Constexpr, k_major: cutlass.Constexpr) -> None:
        n = n_cluster * Uint32(256)
        for k in cutlass.range(Uint32(start), Uint32(end)):
            outer = (expert, k) if cutlass.const_expr(k_major) else (k, expert)
            prefetch_g2s_4d(weight_map, Uint32(0), n, outer[0], outer[1])
            prefetch_g2s_4d(weight_map, Uint32(0), n + Uint32(128), outer[0], outer[1])

    @cute.jit
    def _prefetch_scheduled_weights(self, task: cute.Pointer, full: cute.Pointer,
                                     empty: cute.Pointer, w1_map: cute.Pointer,
                                     w2_map: cute.Pointer) -> None:
        # The follower owns one reader credit. Return it after snapshotting
        # the task so issuing prefetches cannot hold the scheduler slot.
        if elect_one_sync():
            words = cute.make_tensor(task, cute.make_layout((8 * self.task_slots,)))
            iteration = Uint32(0)
            running = True
            while running:
                stage = iteration % Uint32(self.task_slots)
                cute.arch.mbarrier_wait(full + stage, (iteration // Uint32(self.task_slots)) % Uint32(2))
                phase = words[stage * 8]
                running = phase != Uint32(GCBlockPhase.None_)
                if running:
                    expert = words[stage * 8 + 1]
                    m_block = words[stage * 8 + 2]
                    n_cluster = words[stage * 8 + 3]
                    release_task_info(empty, Int32(stage), self.task_scheduler_cta)
                    if cutlass.const_expr(self.w2_prefetch_from_w1):
                        if phase == Uint32(GCBlockPhase.Linear1) and m_block == Uint32(0):
                            # Partition W2's N tiles over the first W1 tile's
                            # workers. Every requested W2 weight tile is unique.
                            for w2_n in cutlass.range(n_cluster, Uint32(self.l2_clusters),
                                                      self.l1_clusters):
                                self._prefetch_task_weight_range(
                                    w2_map, expert, Uint32(w2_n), 0, self.w2_prefetch_from_w1,
                                    self.routed_k_major[1])
                    elif phase == Uint32(GCBlockPhase.Linear1):
                        self._prefetch_task_weight_range(
                            w1_map, expert, n_cluster, self.w1_prefetch_k_tiles,
                            min(self.scheduled_weight_tiles, self.hidden // 128),
                            self.routed_k_major[0])
                    elif phase == Uint32(GCBlockPhase.Linear2):
                        self._prefetch_task_weight_range(
                            w2_map, expert, n_cluster, self.w2_prefetch_k_tiles,
                            min(self.scheduled_weight_tiles, self.intermediate // 128),
                            self.routed_k_major[1])
                iteration += Uint32(1)

    @cute.jit
    def _signal_stripe(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                       rank: Uint32, stripe: Uint32, increment: Uint64):
        count = signals.iterator + self.stripe_combine_offset // 4 + stripe * Uint32(2)
        observed = atom_acq_rel_gpu_add_u64(count, increment) + increment
        # The high half is the total dependency count plus one. Knowing
        # zero dependencies must be distinguishable from no count yet.
        target = Uint32(observed >> Uint64(32))
        if target != Uint32(0) and Uint32(observed) + Uint32(1) == target:
            # The target contribution follows role1's epoch store. Only
            # read the epoch after acquiring that contribution.
            epoch = Uint64(ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['role3_epoch'] // 4))
            # The local RMW chain collects the producer CTAs' writes. Only
            # the last contributor publishes the stripe across GPUs.
            fence_acq_rel_sys()
            slot = stripe * Uint32(self.num_ranks) + rank
            if cutlass.const_expr(self.multicast_combine_ready):
                base = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS)
                ready = cute.make_ptr(
                    Uint64, base + Uint64(self.stripe_ready_offset) + Uint64(slot) * Uint64(8),
                    cute.AddressSpace.gmem, assumed_align=8)
                multimem_store_release_u64(ready, epoch)
            else:
                for peer in cutlass.range_constexpr(self.num_ranks):
                    base = mapped_peer_base(peer_ptrs, Uint32(peer), self.num_ranks)
                    ready = cute.make_ptr(
                        Uint64, base + Uint64(self.stripe_ready_offset) + Uint64(slot) * Uint64(8),
                        cute.AddressSpace.gmem, assumed_align=8)
                    st_release_sys_u64(ready, epoch)

    @cute.jit
    def _signal_expert_ready(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                             recv_ptr: cute.Pointer, expert: Uint32, shared: Uint32,
                             rank: Uint32, tokens: Uint32, epoch: Uint64) -> None:
        local_slot = expert
        global_slot = rank * Uint32(self.local_experts) + expert
        rows = tokens
        if shared != Uint32(0):
            local_slot = Uint32(self.local_experts)
            global_slot = Uint32(self.num_ranks * self.local_experts) + rank
        else:
            rows = Uint32(ld_acquire_sys_u64(recv_ptr + expert * Uint32(2)))
        m_tiles = (rows + Uint32(self.block_m - 1)) // Uint32(self.block_m)
        target = Uint64(m_tiles * Uint32(self.l2_clusters * 2))
        count = signals.iterator + self.expert_combine_offset // 4 + local_slot * Uint32(2)
        arrived = atom_acq_rel_gpu_add_u64(count, Uint64(1)) + Uint64(1)
        if arrived == target:
            fence_acq_rel_sys()
            if shared != Uint32(0):
                ready = signals.iterator + self.expert_ready_offset // 4 + global_slot * Uint32(2)
                st_release_sys_u64(ready, epoch)
            else:
                multicast = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS)
                ready = cute.make_ptr(
                    Uint64, multicast + Uint64(self.expert_ready_offset) + Uint64(global_slot) * Uint64(8),
                    cute.AddressSpace.gmem, assumed_align=8)
                multimem_store_release_u64(ready, epoch)

    @cute.jit
    def _role3_combine_stripes(self, signals: cute.Tensor, output: cute.Pointer,
                               rank: Uint32, num_tokens: Uint32, block: Uint32,
                               tid: Uint32, physical_warp: Uint32, lane: Uint32,
                               epoch: Uint64):
        stripe = block % Uint32(self.stripe_groups)
        group = block // Uint32(self.stripe_groups)
        ready = signals.iterator + self.stripe_ready_offset // 4
        if cutlass.const_expr(self.decode_trace_offset != 0):
            if tid == Uint32(0):
                cta_mark(signals, self.decode_trace_offset, 3, block, Uint32(4), Uint64(stripe))
        if cutlass.const_expr(not self.stripe_active_peers):
            if tid < Uint32(self.num_ranks):
                slot = stripe * Uint32(self.num_ranks) + tid
                wait_sys_u64_eq(ready + slot * Uint32(2), epoch, rank, tid)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.decode_trace_offset != 0):
                if tid == Uint32(0):
                    cta_mark(signals, self.decode_trace_offset, 3, block, Uint32(12), Uint64(stripe))
        combine = (cute.recast_ptr(signals.iterator, dtype=Uint8) +
                   Uint64(self.combine_words * 4))
        out = cute.recast_ptr(output, dtype=Uint8)
        routes = cute.recast_ptr(signals.iterator + self.topk_offset_words, dtype=Uint64)
        token = group * Uint32(FINALIZE_THREADS // 32) + physical_warp
        step = Uint32(self.finalize_sms // self.stripe_groups * (FINALIZE_THREADS // 32))
        while token < num_tokens:
            valid = False
            owner = rank
            if lane < Uint32(self.topk):
                expert = routes[token * Uint32(self.topk) + lane]
                valid = expert < Uint64(self.num_ranks * self.local_experts)
                if cutlass.const_expr(self.stripe_active_peers):
                    owner = Uint32(expert) // Uint32(self.local_experts)
            if cutlass.const_expr(self.shared_experts > 0):
                if lane == Uint32(self.topk):
                    valid = True
            if cutlass.const_expr(self.stripe_active_peers):
                if valid:
                    slot = stripe * Uint32(self.num_ranks) + owner
                    wait_sys_u64_eq(ready + slot * Uint32(2), epoch, rank, owner)
                cute.arch.sync_warp()
            mask = Uint32(cute.arch.vote_ballot_sync(valid))
            for tile in cutlass.range_constexpr(self.stripe_tiles):
                reduce_output_vector(combine, out, token,
                                     stripe * Uint32(self.stripe_tiles) + Uint32(tile),
                                     lane, mask, self.hidden, self.capacity, self.topk,
                                     self.shared_experts > 0)
            token += step
        if cutlass.const_expr(self.decode_trace_offset != 0):
            if tid == Uint32(0):
                cta_mark(signals, self.decode_trace_offset, 3, block, Uint32(20), Uint64(stripe))

    @cute.jit
    def _pull_peer_routes(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                          routes: cute.Tensor, source_idx: cute.Tensor,
                          expert_counts: cute.Pointer, totals: cute.Pointer,
                          block: Uint32, tid: Uint32, warp: Uint32, lane: Uint32, rank: Uint32,
                          num_tokens: Uint32, bank: Uint32, epoch: Uint32) -> None:
        flags = self.pull_route_metadata_offset // 4
        route_rows = flags + 4 * self.num_ranks
        row_bank = bank * Uint32(32 * self.topk)
        if block == Uint32(0):
            expert_limit = Uint64(self.num_ranks * self.local_experts)
            if cutlass.const_expr(self.multicast_pull_routes):
                multicast = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS)
                source_row = (bank * Uint32(self.num_ranks) + rank) * Uint32(32 * self.topk)
                dst = cute.make_ptr(
                    Uint64, multicast + Uint64(route_rows + source_row) * Uint64(4),
                    cute.AddressSpace.gmem, assumed_align=8)
                for pair in cutlass.range(tid, num_tokens * Uint32(self.topk // 2),
                                          self.dispatch_warps * 32):
                    first = Uint64(routes[Uint32(pair) * Uint32(2)])
                    second = Uint64(routes[Uint32(pair) * Uint32(2) + Uint32(1)])
                    lo = Uint32(first) if first < expert_limit else Uint32(0xffffffff)
                    hi = Uint32(second) if second < expert_limit else Uint32(0xffffffff)
                    multimem_store_u64(dst + Uint32(pair), Uint64(lo) | (Uint64(hi) << Uint64(32)))
                fence_acq_rel_sys()
                cute.arch.sync_threads()
                if tid == Uint32(0):
                    ready = cute.make_ptr(
                        Uint64, multicast + Uint64(self.pull_route_metadata_offset) +
                        Uint64(bank * Uint32(self.num_ranks) + rank) * Uint64(8),
                        cute.AddressSpace.gmem, assumed_align=8)
                    multimem_store_release_u64(ready, (Uint64(epoch) << Uint64(32)) | Uint64(num_tokens))
            else:
                for i in cutlass.range(tid, num_tokens * Uint32(self.topk), self.dispatch_warps * 32):
                    expert = Uint64(routes[i])
                    encoded = Uint32(expert) if expert < expert_limit else Uint32(0xffffffff)
                    signals[route_rows + row_bank + Uint32(i)] = encoded
                cute.arch.sync_threads()
                if tid < Uint32(self.num_ranks):
                    dst = cute.make_ptr(
                        Uint64, mapped_peer_base(peer_ptrs, tid, self.num_ranks) +
                        Uint64(self.pull_route_metadata_offset) +
                        Uint64(bank * Uint32(self.num_ranks) + rank) * Uint64(8),
                        cute.AddressSpace.gmem, assumed_align=8)
                    fence_acq_rel_sys()
                    st_release_sys_u64(dst, (Uint64(epoch) << Uint64(32)) | Uint64(num_tokens))

        source = block
        scan_lane = tid
        scan_threads = self.dispatch_warps * 32
        if cutlass.const_expr(self.warp_pull_routes):
            source = block + warp * Uint32(self.dispatch_sms)
            scan_lane = lane
            scan_threads = 32
            expert_counts += warp * Uint32(self.local_experts)
        if source < Uint32(self.num_ranks):
            peer_tokens = Uint32(0)
            if lane == Uint32(0):
                flag = signals.iterator + flags + \
                    (bank * Uint32(self.num_ranks) + source) * Uint32(2)
                start = read_clock64()
                status = ld_volatile_u64(flag)
                while Uint32(status >> Uint64(32)) != epoch:
                    status = ld_volatile_u64(flag)
                    if Uint64(read_clock64() - start) >= Uint64(
                            self.nvlink_timeout_secs * 2000000000):
                        device_trap()
                peer_tokens = Uint32(ld_acquire_sys_u64(flag))
            cute.arch.sync_warp()
            peer_tokens = cute.arch.shuffle_sync(peer_tokens, 0)
            if cutlass.const_expr(self.multicast_pull_routes):
                row = (bank * Uint32(self.num_ranks) + source) * Uint32(32 * self.topk)
                peer_routes = signals.iterator + route_rows + row
            else:
                peer_routes = cute.make_ptr(
                    Uint32, mapped_peer_base(peer_ptrs, source, self.num_ranks) +
                    Uint64(route_rows * 4) + Uint64(row_bank) * Uint64(4),
                    cute.AddressSpace.gmem, assumed_align=16)
            for index in cutlass.range(scan_lane, peer_tokens * Uint32(self.topk), scan_threads):
                expert = peer_routes[index] - rank * Uint32(self.local_experts)
                if expert < Uint32(self.local_experts):
                    slot = atom_shared_add_u32(expert_counts + expert, Uint32(1))
                    source_idx[(expert * Uint32(self.num_ranks) + source) *
                               Uint32(self.capacity) + slot] = Uint32(index)
            if cutlass.const_expr(self.warp_pull_routes):
                cute.arch.sync_warp()
            else:
                cute.arch.sync_threads()
            if scan_lane < Uint32(self.local_experts):
                count = Uint64(expert_counts[scan_lane])
                record = ((bank * Uint32(self.num_ranks) + source) *
                          Uint32(self.local_experts) + scan_lane) * Uint32(2)
                count_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count'] // 4 + record
                if cutlass.const_expr(self.packed_route_counts):
                    st_release_sys_u64(count_ptr, (Uint64(epoch) << Uint64(32)) | count)
                else:
                    st_global_u64(count_ptr, count)
                    fence_acq_rel_sys()
                    participants = Uint64(self.dispatch_sms) << Uint64(32)
                    atom_sys_add_u64(totals + scan_lane * Uint32(2), participants | count)

    @cute.jit
    def _sum_packed_route_counts(self, signals: cute.Tensor, totals: cute.Pointer,
                                 block: Uint32, warp: Uint32, lane: Uint32,
                                 bank: Uint32, epoch: Uint32, rank: Uint32) -> None:
        expert = block + warp * Uint32(self.dispatch_sms)
        if expert < Uint32(self.local_experts):
            count = Uint32(0)
            if lane < Uint32(self.num_ranks):
                # One lane acquires each source's unique count/epoch record.
                record = signals.iterator + SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                    ((bank * Uint32(self.num_ranks) + lane) * Uint32(self.local_experts) +
                     expert) * Uint32(2)
                start = read_clock64()
                stamp = ld_volatile_u64(record)
                while Uint32(stamp >> Uint64(32)) != epoch:
                    stamp = ld_volatile_u64(record)
                    if Uint64(read_clock64() - start) >= Uint64(
                            self.nvlink_timeout_secs * 2000000000):
                        cute.printf('GC packed route count timeout rank={} source={} expert={}',
                                    rank, lane, expert)
                        device_trap()
                count = Uint32(ld_acquire_sys_u64(record))
                if cutlass.const_expr(self.inline_route_indices):
                    count >>= Uint32(16)
            cute.arch.sync_warp()
            total = cute.arch.warp_redux_sync(count, 'add')
            if lane == Uint32(0):
                ready = Uint64(self.dispatch_sms * self.num_ranks) << Uint64(32)
                st_release_sys_u64(totals + expert * Uint32(2), ready | Uint64(total))

    @cute.jit
    def _wait_grouped_route_packet(self, record: cute.Pointer, epoch: Uint32,
                                   rank: Uint32, lane: Uint32, first: Uint32,
                                   payload_bits: cutlass.Constexpr = ROUTE_GROUP_PAYLOAD_BITS) -> Uint64:
        expected = epoch & Uint32((1 << (64 - payload_bits)) - 1)
        start = read_clock64()
        stamp = ld_volatile_u64(record)
        while Uint32(stamp >> Uint64(payload_bits)) != expected:
            stamp = ld_volatile_u64(record)
            if Uint64(read_clock64() - start) >= Uint64(
                    self.nvlink_timeout_secs * 2000000000):
                cute.printf('GC grouped route timeout rank={} source={} first_expert={}',
                            rank, lane, first)
                device_trap()
        return ld_acquire_sys_u64(record)

    @cute.jit
    def _octet_route_count(self, record: cute.Pointer, packet: Uint64,
                           expert: Uint32) -> Uint32:
        count = Uint32(packet >> Uint64(expert * Uint32(ROUTE_OCTET_COUNT_BITS))) & \
            Uint32(ROUTE_OCTET_COUNT_MASK)
        if count == Uint32(ROUTE_OCTET_COUNT_MASK):
            spill = ld_global_u64(record + Uint32(self.num_ranks * 2))
            count += Uint32(spill >> Uint64(expert * Uint32(8))) & Uint32(0xff)
        return count

    @cute.jit
    def _sum_octet_route_counts(self, signals: cute.Tensor, totals: cute.Pointer,
                                block: Uint32, warp: Uint32, lane: Uint32,
                                bank: Uint32, epoch: Uint32, rank: Uint32) -> None:
        expert = block + warp * Uint32(self.dispatch_sms)
        if expert < Uint32(8):
            count = Uint32(0)
            if lane < Uint32(self.num_ranks):
                index = bank * Uint32(self.local_experts * self.num_ranks) + lane
                record = signals.iterator + SIGNAL_OFFSETS['expert_recv_count'] // 4 + index * Uint32(2)
                packet = self._wait_grouped_route_packet(
                    record, epoch, rank, lane, expert, ROUTE_OCTET_PAYLOAD_BITS)
                count = self._octet_route_count(record, packet, expert)
            cute.arch.sync_warp()
            total = cute.arch.warp_redux_sync(count, 'add')
            if lane == Uint32(0):
                ready = Uint64(self.dispatch_sms * self.num_ranks) << Uint64(32)
                st_release_sys_u64(totals + expert * Uint32(2), ready | Uint64(total))

    @cute.jit
    def _sum_grouped_route_counts(self, signals: cute.Tensor, totals: cute.Pointer,
                                  block: Uint32, warp: Uint32, lane: Uint32,
                                  bank: Uint32, epoch: Uint32, rank: Uint32) -> None:
        first = (block + warp * Uint32(self.dispatch_sms)) * Uint32(4)
        if cutlass.const_expr(self.rank_route_doorbell):
            first = warp * Uint32(4)
            if warp == Uint32(0) and lane < Uint32(self.num_ranks):
                last = Uint32((self.local_experts - 1) // 4 * 4)
                index = ((bank * Uint32(self.num_ranks) + lane) * Uint32(self.local_experts) +
                         last)
                flag = signals.iterator + SIGNAL_OFFSETS['expert_recv_count'] // 4 + index * Uint32(2)
                self._wait_grouped_route_packet(flag, epoch, rank, lane, last)
            # Import the releasing source stores before any warp reduces counts.
            cute.arch.sync_threads()
        if first < Uint32(self.local_experts):
            packet = Uint64(0)
            if lane < Uint32(self.num_ranks):
                record = signals.iterator + SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                    ((bank * Uint32(self.num_ranks) + lane) * Uint32(self.local_experts) +
                     first) * Uint32(2)
                if cutlass.const_expr(self.rank_route_doorbell):
                    packet = ld_global_u64(record)
                else:
                    packet = self._wait_grouped_route_packet(record, epoch, rank, lane, first)
            cute.arch.sync_warp()
            for member in cutlass.range_constexpr(4):
                expert = first + Uint32(member)
                if expert < Uint32(self.local_experts):
                    count = Uint32(packet >> Uint64(member * ROUTE_COUNT_BITS)) & \
                        Uint32(ROUTE_COUNT_MASK)
                    total = cute.arch.warp_redux_sync(count, 'add')
                    if lane == Uint32(0):
                        ready = Uint64(self.dispatch_sms * self.num_ranks) << Uint64(32)
                        st_release_sys_u64(totals + expert * Uint32(2), ready | Uint64(total))

    @cute.jit
    def _unpack_atomic_route_counts(self, signals: cute.Tensor, totals: cute.Pointer,
                                    block: Uint32, warp: Uint32, lane: Uint32,
                                    bank: Uint32, rank: Uint32) -> None:
        groups = (self.local_experts + 2) // 3
        group = block + warp * Uint32(self.dispatch_sms)
        if group < Uint32(groups):
            value = Uint64(0)
            if lane == Uint32(0):
                # The normal per-expert totals occupy the first two banks.
                record = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                    (Uint32(2 * self.local_experts) + bank * Uint32(groups) + group) * Uint32(2)
                start = read_clock64()
                while Uint32(value >> Uint64(ROUTE_ATOMIC_PARTICIPANT_SHIFT)) != Uint32(self.num_ranks):
                    value = ld_volatile_u64(record)
                    if Uint64(read_clock64() - start) >= Uint64(
                            self.nvlink_timeout_secs * 2000000000):
                        cute.printf('GC grouped atomic route timeout rank={} group={}', rank, group)
                        device_trap()
                value = ld_acquire_sys_u64(record)
            cute.arch.sync_warp()
            low = cute.arch.shuffle_sync(Uint32(value), 0)
            high = cute.arch.shuffle_sync(Uint32(value >> Uint64(32)), 0)
            expert = group * Uint32(3) + lane
            if lane < Uint32(3) and expert < Uint32(self.local_experts):
                count = (high if lane == Uint32(2) else low >> (lane * Uint32(16))) & \
                    Uint32(ROUTE_ATOMIC_COUNT_MASK)
                ready = Uint64(self.dispatch_sms * self.num_ranks) << Uint64(32)
                st_release_sys_u64(totals + expert * Uint32(2), ready | Uint64(count))

    @cute.jit
    def _sum_multicast_route_counts(self, peer_ptrs: cute.Tensor, totals: cute.Pointer,
                                    bank: Uint32, epoch: Uint32, rank: Uint32, lane: Uint32) -> None:
        group = self.packed_route_count_group
        packets_per_rank = self.local_experts // group
        if lane < Uint32(packets_per_rank):
            experts = self.local_experts * self.num_ranks
            index = (Uint32(experts) * (Uint32(1) + bank) +
                     rank * Uint32(packets_per_rank) + lane)
            multicast = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS)
            packet = cute.make_ptr(
                Uint64, multicast + Uint64(SIGNAL_OFFSETS['expert_send_count']) +
                Uint64(index) * Uint64(8), cute.AddressSpace.gmem, assumed_align=8)
            expected_epoch = epoch
            payload_bits = 32
            if cutlass.const_expr(group == 4):
                expected_epoch &= Uint32(ROUTE_MULTICAST_EPOCH_MASK)
                payload_bits = ROUTE_MULTICAST_PAYLOAD_BITS
            expected = expected_epoch * Uint32(self.num_ranks)
            start = read_clock64()
            observed = multimem_load_add_u64(packet)
            # A rank cannot publish N+2 until it has observed all peers in
            # N+1. Thus a reader of N sees only N or the retired same-bank tag.
            while Uint32(observed >> Uint64(payload_bits)) != expected:
                observed = multimem_load_add_u64(packet)
                if Uint64(read_clock64() - start) >= Uint64(
                        self.nvlink_timeout_secs * 2000000000):
                    cute.printf('GC multicast route timeout rank={} expert={}', rank, lane)
                    device_trap()
            ready = Uint64(self.dispatch_sms * self.num_ranks) << Uint64(32)
            if cutlass.const_expr(group == 4):
                mask = (1 << ROUTE_MULTICAST_COUNT_BITS) - 1
                for member in cutlass.range_constexpr(4):
                    count = (observed >> Uint64(member * ROUTE_MULTICAST_COUNT_BITS)) & Uint64(mask)
                    expert = lane * Uint32(4) + Uint32(member)
                    st_release_sys_u64(totals + expert * Uint32(2), ready | count)
            else:
                st_release_sys_u64(totals + lane * Uint32(2), ready | Uint64(Uint32(observed)))

    @cute.jit
    def _broadcast_decode_input(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                                 rank: Uint32, bank: Uint32, epoch: Uint32,
                                 token: Uint32, lane: Uint32) -> None:
        row = (bank * Uint32(self.num_ranks) + rank) * Uint32(32) + token
        base = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS) + \
            Uint64(self.multicast_decode_input_offset)
        output = cute.make_ptr(Uint64, base + Uint64(row) * Uint64(self.pull_bytes),
                               cute.AddressSpace.gmem, assumed_align=8)
        src = signals.iterator + self.input_x_words + token * Uint32(self.pull_bytes // 4)
        for word_iter in cutlass.range(lane, Uint32(self.pull_bytes // 8), 32):
            word = Uint32(word_iter)
            multimem_store_u64(output + word, ld_global_u64(src + word * Uint32(2)))
        cute.arch.sync_warp()
        if elect_one_sync():
            fence_acq_rel_sys()
            ready_base = base + Uint64(2 * self.num_ranks * 32 * self.pull_bytes)
            flag = cute.make_ptr(Uint64, ready_base + Uint64(row) * Uint64(8),
                                 cute.AddressSpace.gmem, assumed_align=8)
            multimem_store_release_u64(flag, Uint64(epoch))
        cute.arch.sync_warp()

    @cute.jit
    def _prefetches_peer_input(self, expert: Uint32) -> bool:
        if cutlass.const_expr(self.peer_prefetch_expert_heads):
            return expert % Uint32(self.local_experts) < Uint32(self.peer_prefetch_expert_heads)
        return True

    @cute.jit
    def _push_decode_token_route(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                                 scratch: cute.Pointer, barrier: cute.Pointer, phase: Uint32,
                                 rank: Uint32, bank: Uint32, epoch: Uint32,
                                 route: Uint32, lane: Uint32) -> Uint32:
        token = route // Uint32(self.topk)
        slot = route % Uint32(self.topk)
        routes = signals.iterator + self.topk_offset_words
        expert = ld_global_u64(routes + route * Uint32(2))
        expert_count = Uint64(self.num_ranks * self.local_experts)
        peer = Uint32(expert) // Uint32(self.local_experts)
        duplicate = False
        if lane < slot:
            previous = ld_global_u64(routes + (token * Uint32(self.topk) + lane) * Uint32(2))
            duplicate = (previous < expert_count and
                         Uint32(previous) // Uint32(self.local_experts) == peer and
                         self._prefetches_peer_input(Uint32(previous)))
        earlier = Uint32(cute.arch.vote_ballot_sync(duplicate))
        # One warp owns each distinct (token, destination rank) mailbox.
        if (expert < expert_count and self._prefetches_peer_input(Uint32(expert)) and
                earlier == Uint32(0) and
                (cutlass.const_expr(self.peer_decode_input_direct) or peer != rank)):
            if elect_one_sync():
                src = signals.iterator + self.input_x_words + token * Uint32(self.pull_bytes // 4)
                bulk_g2s_1d(scratch, src, Uint32(self.pull_bytes), barrier)
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.pull_bytes)
                cute.arch.mbarrier_wait(barrier, phase)
                row = (bank * Uint32(self.num_ranks) + rank) * Uint32(32) + token
                base = mapped_peer_base(peer_ptrs, peer, self.num_ranks) + \
                    Uint64(self.peer_decode_input_offset)
                dst = cute.make_ptr(Uint32, base + Uint64(row) * Uint64(self.decode_input_stride),
                                    cute.AddressSpace.gmem, assumed_align=16)
                bulk_s2g_1d(dst, scratch, Uint32(self.pull_bytes))
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                flag = cute.make_ptr(
                    Uint64, base + Uint64(2 * self.num_ranks * 32 * self.decode_input_stride) +
                    Uint64(row) * Uint64(8), cute.AddressSpace.gmem, assumed_align=8)
                st_release_sys_u64(flag, Uint64(epoch))
            phase ^= Uint32(1)
        cute.arch.sync_warp()
        return phase

    @cute.jit
    def _push_decode_token_fanout(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                                  scratch: cute.Pointer, barrier: cute.Pointer, phase: Uint32,
                                  rank: Uint32, bank: Uint32, epoch: Uint32,
                                  token: Uint32, lane: Uint32) -> Uint32:
        routes = signals.iterator + self.topk_offset_words
        dst_bit = Uint32(0)
        candidate_peer = Uint32(self.num_ranks)
        if lane < Uint32(self.topk):
            route = token * Uint32(self.topk) + lane
            expert = ld_global_u64(routes + route * Uint32(2))
            if (expert < Uint64(self.num_ranks * self.local_experts) and
                    self._prefetches_peer_input(Uint32(expert))):
                peer = Uint32(expert) // Uint32(self.local_experts)
                if cutlass.const_expr(self.peer_decode_input_direct) or peer != rank:
                    candidate_peer = peer
                    dst_bit = Uint32(1) << peer
        destinations = cute.arch.warp_redux_sync(dst_bit, 'or')
        if destinations != Uint32(0):
            row = (bank * Uint32(self.num_ranks) + rank) * Uint32(32) + token
            row_bytes = Uint64(row) * Uint64(self.decode_input_stride)
            flag_bytes = (Uint64(2 * self.num_ranks * 32 * self.decode_input_stride) +
                          Uint64(row) * Uint64(8))
            issuer = elect_one_sync()
            if issuer:
                src = signals.iterator + self.input_x_words + token * Uint32(self.pull_bytes // 4)
                bulk_g2s_1d(scratch, src, Uint32(self.pull_bytes), barrier)
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.pull_bytes)
                cute.arch.mbarrier_wait(barrier, phase)
            if cutlass.const_expr(self.peer_decode_token_fanout == 2):
                # Every destination has its own issuer and may publish as
                # soon as its transfer completes. All issuers share one read.
                cute.arch.sync_warp()
                owners = match_lanes_u32(candidate_peer)
                if dst_bit != Uint32(0) and warp_first_set_bit(owners) == lane:
                    tma_store_fence()
                    base = mapped_peer_base(peer_ptrs, candidate_peer, self.num_ranks) + \
                        Uint64(self.peer_decode_input_offset)
                    dst = cute.make_ptr(Uint32, base + row_bytes, cute.AddressSpace.gmem,
                                        assumed_align=16)
                    bulk_s2g_1d(dst, scratch, Uint32(self.pull_bytes))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                    flag = cute.make_ptr(Uint64, base + flag_bytes, cute.AddressSpace.gmem,
                                         assumed_align=8)
                    st_release_sys_u64(flag, Uint64(epoch))
            elif issuer:
                remaining = destinations
                while remaining != Uint32(0):
                    peer = warp_first_set_bit(remaining)
                    base = mapped_peer_base(peer_ptrs, peer, self.num_ranks) + \
                        Uint64(self.peer_decode_input_offset)
                    dst = cute.make_ptr(Uint32, base + row_bytes, cute.AddressSpace.gmem,
                                        assumed_align=16)
                    bulk_s2g_1d(dst, scratch, Uint32(self.pull_bytes))
                    remaining &= remaining - Uint32(1)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                remaining = destinations
                while remaining != Uint32(0):
                    peer = warp_first_set_bit(remaining)
                    base = mapped_peer_base(peer_ptrs, peer, self.num_ranks) + \
                        Uint64(self.peer_decode_input_offset)
                    flag = cute.make_ptr(Uint64, base + flag_bytes, cute.AddressSpace.gmem,
                                         assumed_align=8)
                    st_release_sys_u64(flag, Uint64(epoch))
                    remaining &= remaining - Uint32(1)
            phase ^= Uint32(1)
        cute.arch.sync_warp()
        return phase

    @cute.jit
    def _push_count_ptr(self, base: Uint64, gen: Uint32, local_expert: Uint32) -> cute.Pointer:
        index = gen * Uint32(self.local_experts) + local_expert
        return cute.make_ptr(Uint64, base + Uint64(SIGNAL_OFFSETS['expert_recv_count_sum']) +
                             Uint64(index) * Uint64(8), cute.AddressSpace.gmem, assumed_align=8)

    @cute.jit
    def _reserve_decode_spans(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                              slab: cute.Pointer, routes: cute.Pointer, num_tokens: Uint32,
                              tid: Uint32, gen: Uint32, epoch: Uint32) -> None:
        counts = cute.recast_ptr(slab + self.shared_layout.expert_counts, dtype=Uint32)
        local_slots = signals.iterator + SIGNAL_OFFSETS['size'] // 4
        bases = signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4
        experts = Uint32(self.num_ranks * self.local_experts)
        threads = self.dispatch_warps * 32
        for route in cutlass.range(tid, num_tokens * Uint32(self.topk), threads):
            item = Uint32(route)
            expert = ld_global_u64(routes + item * Uint32(2))
            if expert < Uint64(experts):
                local_slots[item] = atom_shared_add_u32(counts + Uint32(expert), Uint32(1))
        cute.arch.sync_threads()
        for expert_iter in cutlass.range(tid, experts, threads):
            expert = Uint32(expert_iter)
            rows = counts[expert]
            if rows != Uint32(0):
                rank = expert // Uint32(self.local_experts)
                local = expert % Uint32(self.local_experts)
                base = mapped_peer_base(peer_ptrs, rank, self.num_ranks)
                peer_count = self._push_count_ptr(base, gen, local)
                st_global_u64(bases + expert * Uint32(2),
                              Uint64(Uint32(atom_sys_add_u64(peer_count, Uint64(rows)))))
        cute.arch.sync_threads()
        if tid == Uint32(0):
            # This mode uses the local epoch word to publish sender spans.
            ready = signals.iterator + SIGNAL_OFFSETS['routed_counts_ready'] // 4
            st_release_gpu_u32(ready, epoch)

    @cute.jit
    def _publish_push_counts(self, peer_ptrs: cute.Tensor, tid: Uint32, gen: Uint32) -> None:
        fence_acq_rel_sys()
        for expert_iter in cutlass.range(tid, Uint32(self.num_ranks * self.local_experts),
                                          self.dispatch_warps * 32):
            expert = Uint32(expert_iter)
            dst_rank = expert // Uint32(self.local_experts)
            dst_expert = expert % Uint32(self.local_experts)
            dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
            count = self._push_count_ptr(dst_base, gen, dst_expert)
            red_sys_add_u64(count, Uint64(self.dispatch_sms) << Uint64(32))
        cute.arch.sync_threads()

    @cute.jit
    def _receive_push_scales(self, signals: cute.Tensor, block: Uint32, warp: Uint32,
                             lane: Uint32, gen: Uint32, parity: Uint32) -> None:
        count = Uint32(0)
        if lane < Uint32(self.local_experts):
            ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                (gen * Uint32(self.local_experts) + lane) * Uint32(2)
            expected = Uint32(self.dispatch_sms * self.num_ranks)
            stamp = ld_acquire_sys_u64(ptr)
            while Uint32(stamp >> Uint64(32)) != expected:
                stamp = ld_acquire_sys_u64(ptr)
            count = Uint32(stamp)
        inclusive = warp_inclusive_sum(count, lane)
        begin = inclusive - count
        total = cute.arch.shuffle_sync(inclusive, self.local_experts - 1)
        sender_sms = self.dispatch_sms - self.push_scale_receiver_sms
        first = (block - Uint32(sender_sms)) * Uint32(self.dispatch_warps) + warp
        for index in cutlass.range(first, total, self.push_scale_receiver_sms * self.dispatch_warps):
            members = Uint32(cute.arch.vote_ballot_sync(index >= begin and index < inclusive))
            expert = warp_first_set_bit(members)
            slot = Uint32(index) - cute.arch.shuffle_sync(begin, expert)
            pool = expert * Uint32(self.fixed_expert_span) + slot // Uint32(self.block_m)
            row = slot % Uint32(self.block_m)
            pool_token = pool * Uint32(self.block_m) + row
            flag = signals.iterator + self.push_row_ready_offset // 4 + pool_token
            if elect_one_sync():
                while ld_acquire_sys_u32(flag) != parity * Uint32(2) + Uint32(1):
                    pass
            cute.arch.sync_warp()
            sf_row = pool * Uint32(128) + (row % Uint32(32)) * Uint32(4) + row // Uint32(32)
            src = signals.iterator + self.push_packed_scales_offset // 4 + \
                pool_token * Uint32(self.hidden // 128)
            for vec in cutlass.range(lane, Uint32(self.hidden // 512), 32):
                values = ld_global_u32x4(src + Uint32(vec) * Uint32(4))
                for part in cutlass.range_constexpr(4):
                    column = Uint32(vec) * Uint32(4) + Uint32(part)
                    offset = Uint64(self.l1_sf_words) + \
                        Uint64(column) * Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                    dst = cute.make_ptr(Uint32, Uint64(signals.iterator.toint()) + offset * Uint64(4),
                                        cute.AddressSpace.gmem, assumed_align=4)
                    dst[0] = values[part]
            cute.arch.sync_warp()
            if elect_one_sync():
                st_release_u32(flag, parity * Uint32(2) + Uint32(2), True)
            cute.arch.sync_warp()

    @cute.jit
    def _push_decode_rows(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                          rank: Uint32, num_tokens: Uint32, block: Uint32,
                          tid: Uint32, warp: Uint32, lane: Uint32) -> None:
        smem = cutlass.utils.SmemAllocator()
        slab = cute.recast_ptr(smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
        self._initialize_shared_storage(slab, warp, lane)
        warpgroup_reg_dealloc(48)
        row_scratch = cute.recast_ptr(slab + self.shared_layout.pull, dtype=Uint32) + \
            warp * Uint32(self.pull_bytes // 4)
        row_bar = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64) + warp
        parity = ld_volatile_u32(signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)
        gen = parity & Uint32(1)
        if block == Uint32(0):
            if tid == Uint32(0):
                signals[SIGNAL_OFFSETS['role3_epoch'] // 4] = parity + Uint32(1)
                if cutlass.const_expr(self.push_early_pool_offset):
                    pool_ready = signals.iterator + self.push_early_pool_offset // 4
                    st_release_sys_u64(pool_ready, Uint64(parity + Uint32(1)))
            if tid < Uint32(self.local_experts):
                retired = gen ^ Uint32(1)
                old = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                    (retired * Uint32(self.local_experts) + tid) * Uint32(2)
                st_global_u64(old, Uint64(0))
        routes = signals.iterator + self.topk_offset_words
        if cutlass.const_expr(self.push_reserve_spans):
            if block == Uint32(0):
                self._reserve_decode_spans(signals, peer_ptrs, slab, routes, num_tokens, tid,
                                           gen, parity + Uint32(1))
                if cutlass.const_expr(self.push_early_pool_offset):
                    self._publish_push_counts(peer_ptrs, tid, gen)
        phase = Uint32(0)
        sender_sms = self.dispatch_sms - self.push_scale_receiver_sms
        first = block * Uint32(self.dispatch_warps) + warp
        if cutlass.const_expr(self.push_scale_receiver_sms):
            if block >= Uint32(sender_sms):
                first = num_tokens * Uint32(self.topk)
        for route in cutlass.range(first, num_tokens * Uint32(self.topk),
                                   sender_sms * self.dispatch_warps):
            item = Uint32(route)
            expert = ld_global_u64(routes + item * Uint32(2))
            if expert < Uint64(self.num_ranks * self.local_experts):
                dst_rank = Uint32(expert) // Uint32(self.local_experts)
                dst_expert = Uint32(expert) % Uint32(self.local_experts)
                dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                token = item // Uint32(self.topk)
                src = signals.iterator + self.input_x_words + token * Uint32(self.pull_bytes // 4)
                if cutlass.const_expr(self.push_overlap_reservation):
                    if elect_one_sync():
                        bulk_g2s_1d(row_scratch, src, Uint32(self.pull_bytes), row_bar)
                        cute.arch.mbarrier_arrive_and_expect_tx(row_bar, self.pull_bytes)
                slot = Uint32(0)
                if cutlass.const_expr(self.push_reserve_spans):
                    ready = signals.iterator + SIGNAL_OFFSETS['routed_counts_ready'] // 4
                    wait_route_count_doorbell(ready, parity + Uint32(1), lane)
                    base = signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 + \
                        Uint32(expert) * Uint32(2)
                    local_slot = signals[SIGNAL_OFFSETS['size'] // 4 + item]
                    slot = Uint32(ld_global_u64(base)) + local_slot
                else:
                    count = self._push_count_ptr(dst_base, gen, dst_expert)
                    if lane == Uint32(0):
                        slot = Uint32(atom_sys_add_u64(count, Uint64(1)))
                    slot = cute.arch.shuffle_sync(slot, 0)
                if slot >= Uint32(self.fixed_expert_span * self.block_m):
                    device_trap()
                pool = dst_expert * Uint32(self.fixed_expert_span) + slot // Uint32(self.block_m)
                row = slot % Uint32(self.block_m)
                pool_token = pool * Uint32(self.block_m) + row
                if cutlass.const_expr(self.push_early_pool_offset):
                    if elect_one_sync():
                        pool_ready = cute.make_ptr(
                            Uint64, dst_base + Uint64(self.push_early_pool_offset),
                            cute.AddressSpace.gmem, assumed_align=8)
                        wait_sys_u64_eq(pool_ready, Uint64(parity + Uint32(1)), rank, dst_rank)
                    cute.arch.sync_warp()
                dst = cute.make_ptr(
                    Uint32, dst_base + Uint64(self.l1_x_words * 4) +
                    Uint64(pool_token) * Uint64(self.hidden), cute.AddressSpace.gmem, assumed_align=16)
                if cutlass.const_expr(self.push_vector_io):
                    copy_input_vector(dst, src, lane, self.hidden)
                if elect_one_sync():
                    if cutlass.const_expr(not self.push_vector_io):
                        if cutlass.const_expr(not self.push_overlap_reservation):
                            bulk_g2s_1d(row_scratch, src, Uint32(self.pull_bytes), row_bar)
                            cute.arch.mbarrier_arrive_and_expect_tx(row_bar, self.pull_bytes)
                        cute.arch.mbarrier_wait(row_bar, phase)
                        phase ^= Uint32(1)
                        bulk_s2g_1d(dst, row_scratch, Uint32(self.hidden))
                        if cutlass.const_expr(self.push_scale_receiver_sms):
                            scales = cute.make_ptr(
                                Uint32, dst_base + Uint64(self.push_packed_scales_offset) +
                                Uint64(pool_token) * Uint64(self.hidden // 32),
                                cute.AddressSpace.gmem, assumed_align=16)
                            bulk_s2g_1d(scales, row_scratch + self.hidden // 4,
                                        Uint32(self.hidden // 32))
                        cute.arch.cp_async_bulk_commit_group()
                    meta = cute.make_ptr(Uint32, dst_base + Uint64(self.metadata_words * 4) +
                                         Uint64(pool_token) * Uint64(12), cute.AddressSpace.gmem)
                    meta[0] = rank
                    meta[1] = token
                    meta[2] = item % Uint32(self.topk)
                    route_weight = cute.make_ptr(Uint32, dst_base + Uint64(self.l1_weight_words * 4) +
                                                 Uint64(pool_token) * Uint64(4), cute.AddressSpace.gmem)
                    route_weight[0] = signals[Uint32(self.input_weight_words) + item]
                cute.arch.sync_warp()
                if cutlass.const_expr(not self.push_vector_io):
                    cute.arch.fence_view_async_shared()
                if cutlass.const_expr(not self.push_scale_receiver_sms):
                    sf_row = pool * Uint32(128) + (row % Uint32(32)) * Uint32(4) + row // Uint32(32)
                    for sf_iter in cutlass.range(lane, Uint32(self.hidden // 128), 32):
                        column = Uint32(sf_iter)
                        index = Uint64(column) * Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                        dst_sf = cute.make_ptr(Uint32, dst_base + Uint64(self.l1_sf_words * 4) +
                                               index * Uint64(4), cute.AddressSpace.gmem)
                        sf = src if cutlass.const_expr(self.push_vector_io) else row_scratch
                        dst_sf[0] = sf[Uint32(self.hidden // 4) + column]
                if cutlass.const_expr(not self.push_vector_io):
                    if elect_one_sync():
                        cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.push_early_pool_offset):
                    if cutlass.const_expr(not self.push_scale_receiver_sms):
                        fence_acq_rel_sys()
                        cute.arch.sync_warp()
                    if elect_one_sync():
                        if cutlass.const_expr(self.push_row_ready_offset):
                            ready = cute.make_ptr(
                                Uint32, dst_base + Uint64(self.push_row_ready_offset) +
                                Uint64(pool_token) * Uint64(4), cute.AddressSpace.gmem, assumed_align=4)
                            epoch = parity + Uint32(1)
                            if cutlass.const_expr(self.push_scale_receiver_sms):
                                epoch = parity * Uint32(2) + Uint32(1)
                            st_release_u32(ready, epoch, True)
                        else:
                            ready = cute.make_ptr(
                                Uint32, dst_base + Uint64(SIGNAL_OFFSETS['l1_full_count']) +
                                Uint64(pool) * Uint64(4), cute.AddressSpace.gmem, assumed_align=4)
                            atom_acq_rel_sys_add_u32(ready, Uint32(1))
        if cutlass.const_expr(self.push_scale_receiver_sms):
            if block >= Uint32(sender_sms):
                self._receive_push_scales(signals, block, warp, lane, gen, parity)
        if cutlass.const_expr(self.push_vector_io):
            # Publish ordinary peer stores before block 0 posts its counts.
            fence_acq_rel_sys()
        if cutlass.const_expr(not self.push_early_pool_offset):
            grid_sync(signals, block, tid, self.dispatch_sms, 0)
        if cutlass.const_expr(self.decode_trace_offset != 0):
            if tid == Uint32(0):
                cta_mark(signals, self.decode_trace_offset, 1, block, Uint32(1), Uint64(0))
        if block == Uint32(0):
            if cutlass.const_expr(not self.push_early_pool_offset):
                self._publish_push_counts(peer_ptrs, tid, gen)
            if tid < Uint32(self.local_experts):
                count = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                    (gen * Uint32(self.local_experts) + tid) * Uint32(2)
                expected = Uint32(self.dispatch_sms * self.num_ranks)
                start = read_clock64()
                observed = ld_volatile_u64(count)
                while Uint32(observed >> Uint64(32)) != expected:
                    observed = ld_volatile_u64(count)
                    if Uint64(read_clock64() - start) >= Uint64(
                            self.nvlink_timeout_secs * 2000000000):
                        cute.printf('GC source push timeout rank={} expert={}', rank, tid)
                        device_trap()
                rows = Uint32(ld_acquire_sys_u64(count))
                used = (rows + Uint32(self.block_m - 1)) // Uint32(self.block_m)
                full = signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + \
                    tid * Uint32(self.fixed_expert_span)
                if cutlass.const_expr(self.push_early_pool_offset):
                    if cutlass.const_expr(not self.push_row_ready_offset):
                        tail = rows % Uint32(self.block_m)
                        if tail != Uint32(0):
                            padding = Uint32(self.block_m) - tail
                            atom_acq_rel_sys_add_u32(full + used - Uint32(1), padding)
                else:
                    for tile in cutlass.range(Uint32(0), used):
                        st_release_gpu_u32(full + Uint32(tile), Uint32(self.block_m))
            if cutlass.const_expr(self.decode_trace_offset != 0):
                cute.arch.sync_threads()
                if tid == Uint32(0):
                    cta_mark(signals, self.decode_trace_offset, 1, block, Uint32(2), Uint64(0))

    @cute.jit
    def _load_packed_scale_tasks(self, peer_ptrs: cute.Tensor, task: cute.Pointer,
                                 full: cute.Pointer, empty: cute.Pointer, data_sfb: cute.Pointer,
                                 data_full: cute.Pointer, data_empty: cute.Pointer,
                                 cta_rank: Uint32, lane: Uint32) -> None:
        words = cute.make_tensor(task, cute.make_layout((8 * self.task_slots,)))
        task_iter = Uint32(0)
        data_stage = Uint32(0)
        data_phase = Uint32(0)
        running = True
        while running:
            stage = task_iter % Uint32(self.task_slots)
            cute.arch.mbarrier_wait(full + stage, (task_iter // Uint32(self.task_slots)) % Uint32(2))
            phase = words[stage * 8]
            running = phase != Uint32(GCBlockPhase.None_)
            if running:
                expert = Uint32(words[stage * 8 + 1])
                n_block = Uint32(words[stage * 8 + 3]) * Uint32(2) + cta_rank
                n_blocks = Uint32(words[stage * 8 + 6]) // Uint32(128)
                k_groups = Uint32(words[stage * 8 + 7]) // Uint32(512)
                cute.arch.sync_warp()
                if lane == Uint32(0):
                    release_task_info(empty, Int32(stage), self.task_scheduler_cta)
                routed = phase < Uint32(GCBlockPhase.SharedLinear1)
                address = Uint64(0)
                if routed:
                    pointer = Uint32(PACKED_ROUTED_SCALE_OFFSETS[0] + 1)
                    if phase == Uint32(GCBlockPhase.Linear2):
                        pointer = Uint32(PACKED_ROUTED_SCALE_OFFSETS[1] + 1)
                    address = ld_peer_param_u64(peer_ptrs.iterator + pointer)
                for group in cutlass.range(Uint32(0), k_groups):
                    tile = (expert * n_blocks + n_block) * k_groups + Uint32(group)
                    codes = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    if cutlass.const_expr(self.packed_scale_release_group < 4):
                        if routed:
                            codes = load_routed_scale_codes(address, tile, lane, 4)
                    for first in cutlass.range_constexpr(0, 4, self.packed_scale_release_group):
                        slots = ()
                        for index in cutlass.range_constexpr(self.packed_scale_release_group):
                            raw_slot = data_stage + Uint32(first + index)
                            wrapped = raw_slot >= Uint32(self.math_stages)
                            slot = raw_slot - Uint32(self.math_stages) if wrapped else raw_slot
                            ready_phase = data_phase ^ Uint32(wrapped)
                            cute.arch.mbarrier_wait(data_empty + slot, ready_phase ^ Uint32(1))
                            slots += (slot,)
                        if routed:
                            if cutlass.const_expr(self.packed_scale_release_group == 4):
                                codes = load_routed_scale_codes(address, tile, lane, 4)
                            for index in cutlass.range_constexpr(self.packed_scale_release_group):
                                store_routed_scale_word(data_sfb + slots[index] * Uint32(128),
                                                        codes[first + index], lane, self.packed_routed_scale_base)
                            tma_store_fence()
                            cute.arch.sync_warp()
                        if lane == Uint32(0):
                            for index in cutlass.range_constexpr(self.packed_scale_release_group):
                                if cta_rank == Uint32(0):
                                    cute.arch.mbarrier_arrive(data_full + slots[index])
                                else:
                                    cute.arch.mbarrier_arrive(data_full + slots[index], peer_cta_rank_in_cluster=0)
                    data_stage += Uint32(4)
                    if data_stage >= Uint32(self.math_stages):
                        data_stage -= Uint32(self.math_stages)
                        data_phase ^= Uint32(1)
            task_iter += Uint32(1)

    @cute.jit
    def _clear_routed_ring(self, signals: cute.Tensor, ring_block: Uint32,
                           stride: Uint32) -> None:
        # The padded expert ranges concatenate to [0,P). These are the slots
        # used by the replay, including when the pool wraps around the ring.
        pool_blocks = signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4]
        if cutlass.const_expr(self.fixed_expert_span != 0):
            pool_blocks = Uint32(self.fixed_expert_span * self.local_experts)
        while ring_block < min(pool_blocks, Uint32(self.ring_blocks)):
            signals[SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block] = Uint32(0)
            if cutlass.const_expr(self.activation_prefix_offset != 0):
                signals[self.activation_prefix_offset // 4 + ring_block] = Uint32(0)
            if cutlass.const_expr(not self.decode_linear_pool):
                signals[SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block] = Uint32(0)
            st_global_u64(signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                          ring_block * Uint32(2), Uint64(0))
            if cutlass.const_expr(not self.decode_linear_pool):
                signals[SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block] = Uint32(0)
            if cutlass.const_expr(self.ready_tile_offset != 0):
                signals[self.ready_tile_offset // 4 + 1 + ring_block] = Uint32(0)
            ring_block += stride

    @cute.jit
    def _role3_dispatch(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                        rank_idx: Uint32, num_tokens: Uint32, tid: Uint32,
                        block: Uint32, recv_ptr: cute.Pointer, stats: cute.Tensor,
                        ready_epoch: Uint64) -> None:
        # Publish completed peer FC2 stores even with no routed tokens.
        cleanup_ctas = 1 if cutlass.const_expr(self.single_cleanup_cta) else self.finalize_sms
        # Epochs distinguish late old publications without resetting ready.
        if cutlass.const_expr(self.multicast_combine_ready):
            if block == Uint32(0) and tid == Uint32(0):
                multicast = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS)
                dst = cute.make_ptr(
                    Uint64, multicast + Uint64(SIGNAL_OFFSETS['combine_ready_grid_idx']) +
                    Uint64(rank_idx * Uint32(8)), cute.AddressSpace.gmem, assumed_align=8)
                multimem_store_release_u64(dst, ready_epoch)
        elif cutlass.const_expr(not self.omit_sentinel_peer_finish):
            if block == 0 and tid < self.num_ranks:
                dst = cute.make_ptr(
                    Uint64, mapped_peer_base(peer_ptrs, Uint32(tid), self.num_ranks) +
                    Uint64(SIGNAL_OFFSETS['combine_ready_grid_idx']) +
                    Uint64(rank_idx * 8), cute.AddressSpace.gmem,
                    assumed_align=8)
                st_release_sys_u64(dst, ready_epoch)
        # Dual-buffered receive sums: this replay consumes the generation
        # selected by the persistent parity word; the bump below flips it
        # for the next replay, and role 1 of the replay that retires this
        # generation clears it at entry.
        gen = Uint32(ready_epoch - Uint64(1)) & Uint32(1)
        recv_ptr = recv_ptr + gen * Uint32(self.local_experts * 2)
        # Role1 acquired every expert sum, including zero-count experts,
        # before publishing this summary. Normal graph completion dependencies
        # also retire every role2 ring access before this footer executes.
        # All CTAs use the immutable role1 epoch, so block0 may flip parity
        # without a grid-wide cache-completion barrier. The next dispatch
        # clears this bank only after the entire local finalize grid ends.
        # That clear precedes count publication; all-rank sum acquires in
        # the intervening call order it before any peer can publish into
        # the same bank again. Zero-count experts must join too.
        if block == Uint32(0) and tid == Uint32(0):
            red_gpu_add_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4,
                Uint32(1))
            if cutlass.const_expr(self.ready_tile_offset != 0):
                signals[self.ready_tile_offset // 4] = Uint32(0)
        if cutlass.const_expr(not (self.decode_vector_combine or self.expert_combine_offset)):
            warpgroup_reg_dealloc(48 if self.local_experts <= 64 else 96)

        if block == 0:
            if cutlass.const_expr(self.expert_combine_offset):
                if tid < Uint32(self.local_experts + 1):
                    st_global_u64(signals.iterator + self.expert_combine_offset // 4 +
                                  tid * Uint32(2), Uint64(0))
            if cutlass.const_expr(self.dedup_activation_offset != 0):
                slot = Uint32(tid)
                while slot < Uint32(self.num_ranks * 32):
                    signals[self.dedup_activation_offset // 4 + slot] = Uint32(0)
                    slot += Uint32(128)
            if cutlass.const_expr(self.stripe_combine_offset != 0):
                if tid < Uint32(self.stripe_groups):
                    st_global_u64(signals.iterator + self.stripe_combine_offset // 4 +
                                  tid * Uint32(2), Uint64(0))
            if cutlass.const_expr(not self.prune_decode_resets):
                send_off = SIGNAL_OFFSETS['expert_send_count'] // 4
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        128, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    st_global_u64(signals.iterator + send_off + expert_idx * Uint32(2),
                                  Uint64(0))
            if tid // Uint32(32) == Uint32(0):
                if elect_one_sync():
                    for locality in cutlass.range(2, unroll_full=True):
                        signals[SIGNAL_OFFSETS['l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['l2_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l2_task_count'] // 4 + locality] = Uint32(0)
            cute.arch.sync_warp()
            shared_block = Uint32(tid)
            shared_end = Uint32((self.capacity + 7) // 8)
            if cutlass.const_expr(self.prune_decode_resets):
                shared_end = ((num_tokens + Uint32(self.block_m - 1)) // Uint32(self.block_m)
                              if self.shared_experts else Uint32(0))
            while shared_block < shared_end:
                signals[SIGNAL_OFFSETS['shared_l2_full_count'] // 4 +
                        shared_block] = Uint32(0)
                shared_block += Uint32(128)
            cute.arch.sync_warp()
        if cutlass.const_expr(self.has_stats):
            local_expert = block * Uint32(128) + tid
            while local_expert < Uint32(self.local_experts):
                received = Uint32(ld_global_u64(recv_ptr + local_expert * Uint32(2)))
                red_gpu_add_s32(stats.iterator + local_expert, Int32(received))
                local_expert += Uint32(cleanup_ctas * 128)

        if cutlass.const_expr(not self.reset_decode_ring_on_dispatch):
            self._clear_routed_ring(signals, block * Uint32(128) + tid,
                                    Uint32(cleanup_ctas * 128))

        # The former tail NVLink barrier (tag 2) is gone: the receive
        # signals are dual-buffered by recv_parity, so peers' next publish
        # targets the alternate generation and ordering is carried by the
        # all-expert sum acquires in the intervening call. Selected-peer
        # ready acquires protect each source's single-buffer inputs/indices.

    @cute.jit
    def _combine_compute_warp(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                               rank: Uint32, num_tokens: Uint32,
                               token_chunk: Uint32, lane: Uint32, epoch: Uint32) -> None:
        chunks = self.hidden * 2 // WARP_BYTES
        if cutlass.const_expr(self.combine_in_compute == 2):
            if token_chunk < num_tokens * Uint32(chunks):
                first_w2 = signals.iterator + SIGNAL_OFFSETS['compute_combine_epoch'] // 4
                wait_route_count_doorbell(first_w2, epoch, lane)
        combine = cute.recast_ptr(signals.iterator, dtype=Uint8) + self.combine_words * 4
        routes = cute.recast_ptr(signals.iterator + self.topk_offset_words, dtype=Uint64)
        output_address = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS - 1)
        output = cute.make_ptr(Uint8, output_address, cute.AddressSpace.gmem, assumed_align=16)
        while token_chunk < num_tokens * Uint32(chunks):
            token = token_chunk // Uint32(chunks)
            chunk = token_chunk % Uint32(chunks)
            active = lane == Uint32(self.topk)
            if lane < Uint32(self.topk):
                expert = routes[token * Uint32(self.topk) + lane]
                active = expert < Uint64(self.num_ranks * self.local_experts)
            mask = Uint32(cute.arch.vote_ballot_sync(active))
            reduce_output_sentinel(combine, output, token, chunk, lane, mask, rank,
                                   self.hidden, self.capacity, self.topk, True,
                                   1, self.sentinel_poll, self.token_major_combine,
                                   self.combine_interleaved_bytes != 0)
            token_chunk += Uint32(self.compute_sms)

    @cute.jit
    def _finish_shared_w2_vector(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                                  token: Uint32, channel_bytes: Uint32, words) -> None:
        row_bytes = self.hidden * 2
        mailbox_offset = _w2_mailbox_offset(
            token, Uint32(0), channel_bytes, self.hidden, self.capacity, self.topk + 1,
            self.token_major_combine, self.combine_interleaved_bytes)
        mailbox = (signals.iterator + self.combine_words + mailbox_offset // Uint64(4))
        slot_stride = row_bytes if self.token_major_combine else self.capacity * row_bytes
        output_base = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS - 1)
        dst = cute.make_ptr(Uint32, output_base + Uint64(token) * Uint64(row_bytes) + Uint64(channel_bytes),
                            cute.AddressSpace.gmem, assumed_align=16)
        routes = cute.recast_ptr(signals.iterator + self.topk_offset_words, dtype=Uint64)
        combine_after_shared_w2(mailbox, dst, routes, words, token, slot_stride // 4,
                                 self.topk, self.local_experts * self.num_ranks)

    @cute.jit
    def _role3_combine(self, signals: cute.Tensor, rank_idx: Uint32,
                       num_tokens: Uint32, y_out: cute.Tensor,
                       block, physical_warp, lane,
                       combine_smem: cute.Pointer,
                       combine_bars: cute.Pointer, ready_epoch: Uint64) -> None:
        combine_chunks = self.combine_chunks
        chunk_bytes = self.combine_chunk_bytes
        u4_per_lane = self.combine_uint4_per_lane
        if cutlass.const_expr(not self.decode_vector_combine):
            warpgroup_reg_alloc(self.epi_registers)
        epi_warp = Uint32(physical_warp - 4)
        ready = signals.iterator + SIGNAL_OFFSETS['combine_ready_grid_idx'] // 4
        if cutlass.const_expr(self.expert_combine_offset):
            ready = signals.iterator + self.expert_ready_offset // 4
        if cutlass.const_expr(not self.finalize_output):
            # No token reduction exists to carry selected-peer acquires.
            peer = epi_warp
            while peer < Uint32(self.num_ranks):
                wait_sys_u64_eq(ready + peer * 2, ready_epoch, rank_idx, peer)
                peer += Uint32(COMBINE_WARPS)
            sync_aligned(COMBINE_WARPS * 32, 2)
        if cutlass.const_expr(self.finalize_output):
            # One warp owns a token chunk. Every routed slot of the chunk is
            # fetched into its own plane under a single barrier so the
            # reduction exposes one TMA latency instead of one per slot. The
            # output plane aliases slot 0 and is retired before the next
            # chunk overwrites it.
            combine = cute.recast_ptr(
                signals.iterator, dtype=Uint8) + \
                Uint64(self.combine_words) * Uint64(4)
            output = cute.recast_ptr(y_out.iterator, dtype=Uint8)
            routes = cute.recast_ptr(
                signals.iterator + self.topk_offset_words, dtype=Uint64)
            combine_slots = self.topk + (1 if self.shared_experts > 0 else 0)
            # Keep the historical reduction order: shared slot, then routes.
            slot_of = ([self.topk] if self.shared_experts > 0 else []) + \
                list(range(self.topk))
            if cutlass.const_expr(not self.decode_vector_combine):
                warp_stride = chunk_bytes * combine_slots if self.combine_interleaved_bytes else chunk_bytes
                slot_stride = chunk_bytes if self.combine_interleaved_bytes else COMBINE_WARPS * chunk_bytes
                store_smem = combine_smem + Uint32(epi_warp * warp_stride)
                store_words = cute.recast_ptr(store_smem, dtype=Uint32)
                bar = combine_bars + epi_warp
            phase = Uint32(0)
            token_chunk = epi_warp * Uint32(self.finalize_sms) + block
            if cutlass.const_expr(self.sentinel_token_order):
                token_chunk = Uint32(block) * Uint32(self.combine_warps) + epi_warp
            if cutlass.const_expr(self.combine_token_warps):
                token_chunk = Uint32(block) * Uint32(self.combine_token_warps) + epi_warp
                if epi_warp >= Uint32(self.combine_token_warps):
                    token_chunk = num_tokens * Uint32(combine_chunks)
            if cutlass.const_expr(self.shared_experts > 0 and not self.decode_vector_combine):
                if token_chunk < num_tokens * Uint32(combine_chunks):
                    _initialize_private_combine_bar(bar)
            while token_chunk < num_tokens * Uint32(combine_chunks):
                token = token_chunk // Uint32(combine_chunks)
                chunk = token_chunk % Uint32(combine_chunks)
                chunk_offset = chunk * Uint32(chunk_bytes)
                lane_valid = False
                routed_peer = Uint32(0)
                if lane < self.topk:
                    routed_expert = routes[token * Uint32(self.topk) + lane]
                    lane_valid = routed_expert < \
                        Uint64(self.num_ranks * self.local_experts)
                    if lane_valid:
                        routed_peer = Uint32(routed_expert) // Uint32(self.local_experts)
                        if cutlass.const_expr(self.expert_combine_offset):
                            routed_peer = Uint32(routed_expert)
                if cutlass.const_expr(self.expert_combine_offset and self.shared_experts > 0):
                    if lane == Uint32(self.topk):
                        lane_valid = True
                        routed_peer = Uint32(self.num_ranks * self.local_experts) + rank_idx
                # All lanes vote; invalid routes never form a peer pointer.
                # Duplicate owners may acquire the same publication twice.
                if cutlass.const_expr(self.decode_trace_offset != 0):
                    if lane == Uint32(0) and token_chunk < Uint32(
                            self.finalize_sms * self.combine_warps):
                        cta_mark(signals, self.decode_trace_offset, 3, Uint32(block),
                                 Uint32(4) + epi_warp, Uint64(token_chunk))
                if cutlass.const_expr(not (self.streaming_combine or self.sentinel_combine)):
                    peer_ready = not lane_valid
                    if cutlass.const_expr(self.combine_token_warps):
                        peer_ready = peer_ready or epi_warp != Uint32(0)
                    wait_start = read_clock64()
                    while cute.arch.vote_ballot_sync(peer_ready) != Uint32(0xffffffff):
                        if lane_valid and not peer_ready:
                            peer_ready = ld_acquire_sys_u64(
                                ready + routed_peer * Uint32(2)) == ready_epoch
                        if Uint64(read_clock64() - wait_start) >= Uint64(60 * 2000000000):
                            if not peer_ready:
                                cute.printf('GC token readiness timeout rank={} ready_slot={}',
                                            rank_idx, routed_peer)
                                device_trap()
                    # Import the selected peers' writes before the TMA issue.
                    if cutlass.const_expr(self.combine_token_warps):
                        sync_aligned(self.combine_token_warps * 32, 2)
                    else:
                        cute.arch.sync_warp()
                    if cutlass.const_expr(self.decode_trace_offset != 0):
                        if lane == Uint32(0) and token_chunk < Uint32(
                                self.finalize_sms * self.combine_warps):
                            cta_mark(signals, self.decode_trace_offset, 3, Uint32(block),
                                     Uint32(12) + epi_warp, Uint64(token_chunk))
                if cutlass.const_expr(self.shared_experts > 0):
                    if lane == self.topk:
                        lane_valid = True
                mask = Uint32(cute.arch.vote_ballot_sync(lane_valid))
                if cutlass.const_expr(self.sentinel_combine):
                    ready_flags = None
                    if cutlass.const_expr(self.combine_ready_flags_offset):
                        ready_flags = signals.iterator + self.combine_ready_flags_offset // 4
                    reduce_output_sentinel(combine, output, token, chunk, lane, mask, rank_idx,
                                           self.hidden, self.capacity, self.topk, self.shared_experts > 0,
                                           u4_per_lane, self.sentinel_poll, self.token_major_combine,
                                           self.combine_interleaved_bytes != 0,
                                           self.sentinel_poll_delay_ns, ready_flags,
                                           self.sentinel_adaptive_poll, self.combine_ready_mask,
                                           self.sentinel_half_vector)
                elif cutlass.const_expr(self.decode_vector_combine):
                    reduce_output_vector(combine, output, token, chunk, lane, mask,
                                         self.hidden, self.capacity, self.topk,
                                         self.shared_experts > 0, self.combine_load_group,
                                         self.token_major_combine)
                else:
                    # Retire the previous chunk's output before any fetch plane
                    # reuses the aliased slot-0 region, also when this chunk has
                    # no valid route and only writes zeros. The wait is per-thread
                    # and only the elected lane issued the store, so converge the
                    # warp before any lane touches the plane.
                    if cutlass.const_expr(not self.direct_combine_output):
                        tma_store_wait_0()
                    cute.arch.sync_warp()
                    if mask != Uint32(0):
                        if cutlass.const_expr(self.streaming_combine):
                            if elect_one_sync():
                                bytes_expected = Uint32(cute.arch.popc(mask)) * Uint32(chunk_bytes)
                                cute.arch.mbarrier_arrive_and_expect_tx(bar, bytes_expected)
                            cute.arch.sync_warp()
                            pending = lane_valid
                            wait_start = read_clock64()
                            while cute.arch.vote_ballot_sync(pending) != Uint32(0):
                                if pending:
                                    slot_ready = (lane == Uint32(self.topk) if
                                                  cutlass.const_expr(self.shared_experts > 0) else False)
                                    if not slot_ready:
                                        slot_ready = ld_acquire_sys_u64(
                                            ready + routed_peer * Uint32(2)) == ready_epoch
                                    if slot_ready:
                                        src = combine + (Uint64(lane) * Uint64(self.capacity) +
                                                         Uint64(token)) * Uint64(self.hidden * 2) + \
                                            Uint64(chunk_offset)
                                        bulk_g2s_1d(store_smem + Uint32(lane) * Uint32(slot_stride),
                                                    src, Uint32(chunk_bytes), bar)
                                        pending = False
                                    if pending and Uint64(read_clock64() - wait_start) >= Uint64(
                                            60 * 2000000000):
                                        cute.printf('GC streaming Combine timeout rank={} peer={}',
                                                    rank_idx, routed_peer)
                                        device_trap()
                            if cutlass.const_expr(self.decode_trace_offset != 0):
                                if lane == Uint32(0) and token_chunk < Uint32(
                                        self.finalize_sms * self.combine_warps):
                                    cta_mark(signals, self.decode_trace_offset, 3, Uint32(block),
                                             Uint32(12) + epi_warp, Uint64(token_chunk))
                        elif elect_one_sync():
                            if cutlass.const_expr(self.combine_interleaved_bytes):
                                # Adjacent W2 slots share one transfer. Invalid
                                # slots are ignored by the reduction mask.
                                fetch_bytes = combine_slots * chunk_bytes
                                src = combine + Uint64(token_chunk) * Uint64(fetch_bytes)
                                cute.arch.mbarrier_arrive_and_expect_tx(bar, Uint32(fetch_bytes))
                                bulk_g2s_1d(store_smem, src, Uint32(fetch_bytes), bar)
                            else:
                                expect = Uint32(0)
                                for g in cutlass.range_constexpr(combine_slots):
                                    if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                                        expect += Uint32(chunk_bytes)
                                cute.arch.mbarrier_arrive_and_expect_tx(bar, expect)
                                for g in cutlass.range_constexpr(combine_slots):
                                    if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                                        src = combine + \
                                            (Uint64(slot_of[g]) * Uint64(self.capacity) +
                                             Uint64(token)) * Uint64(self.hidden * 2) + \
                                            Uint64(chunk_offset)
                                        bulk_g2s_1d(store_smem + Uint32(slot_of[g] * slot_stride), src,
                                                    Uint32(chunk_bytes), bar)
                        cute.arch.sync_warp()
                        cute.arch.mbarrier_wait(bar, phase)
                        phase ^= Uint32(1)
                    reduced = cute.make_rmem_tensor((u4_per_lane * 8,), Float32)
                    for index in cutlass.range_constexpr(u4_per_lane * 8):
                        reduced[index] = Float32(0)
                    for g in cutlass.range_constexpr(combine_slots):
                        if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                            load_words = cute.recast_ptr(
                                store_smem + Uint32(slot_of[g] * slot_stride), dtype=Uint32)
                            for j in cutlass.range_constexpr(u4_per_lane):
                                words = ld_shared_u32x4(
                                    load_words + (j * 32 + lane) * 4)
                                for q in cutlass.range_constexpr(4):
                                    packed = words[q]
                                    reduced[j * 8 + q * 2] = \
                                        accumulate_bf16_into_f32(
                                            reduced[j * 8 + q * 2],
                                            Uint16(packed & Uint32(0xffff)))
                                    reduced[j * 8 + q * 2 + 1] = \
                                        accumulate_bf16_into_f32(
                                            reduced[j * 8 + q * 2 + 1],
                                            Uint16(packed >> Uint32(16)))
                    if cutlass.const_expr(not self.direct_combine_output):
                        cute.arch.fence_view_async_shared()

                    dst = output + Uint64(token) * Uint64(self.hidden * 2) + Uint64(chunk_offset)
                    for j in cutlass.range_constexpr(u4_per_lane):
                        packed_words = cute.make_rmem_tensor((4,), Uint32)
                        for q in cutlass.range_constexpr(4):
                            packed_words[q] = pack_bf16_pair(
                                reduced[j * 8 + q * 2],
                                reduced[j * 8 + q * 2 + 1])
                        if cutlass.const_expr(self.direct_combine_output):
                            out_words = cute.recast_ptr(dst, dtype=Uint32)
                            st_global_u32x4(
                                out_words + (j * 32 + lane) * 4,
                                packed_words[0], packed_words[1], packed_words[2], packed_words[3])
                        else:
                            st_shared_u32x4(
                                store_words + (j * 32 + lane) * 4,
                                packed_words[0], packed_words[1], packed_words[2], packed_words[3])
                    if cutlass.const_expr(not self.direct_combine_output):
                        cute.arch.sync_warp()
                        if elect_one_sync():
                            tma_store_fence()
                            bulk_s2g_1d(dst, store_smem, Uint32(chunk_bytes))
                            tma_store_arrive()
                        cute.arch.sync_warp()
                if cutlass.const_expr(self.decode_trace_offset != 0):
                    if lane == Uint32(0):
                        # Reuse the warp's slot for the latest chunk. The
                        # TMA path records output submission, not arrival.
                        cta_mark(signals, self.decode_trace_offset, 3, Uint32(block),
                                 Uint32(20) + epi_warp, Uint64(token_chunk))
                token_chunk += Uint32(self.finalize_sms * self.combine_warps)


    @cute.jit
    def _join_compute_completion(self, signals: cute.Tensor, block: Uint32,
                                 tid: Uint32, rank: Uint32, epoch: Uint32) -> None:
        flags = signals.iterator + self.compute_completion_offset // 4
        grid_ready = flags + self.compute_sms
        if block == Uint32(0):
            if tid < Uint32(32):
                for worker in cutlass.range(tid, Uint32(self.compute_sms), 32):
                    flag = flags + worker
                    start = read_clock64()
                    while ld_volatile_u32(flag) != epoch:
                        if Uint64(read_clock64() - start) >= Uint64(
                                self.nvlink_timeout_secs * 2000000000):
                            cute.printf('GC compute completion timeout rank={} worker={}', rank, worker)
                            device_trap()
                    ld_acquire_gpu_u32(flag)
            cute.arch.sync_threads()
            if tid == Uint32(0):
                st_release_gpu_u32(grid_ready, epoch)
        if block < Uint32(self.fused_finalize_role.finalize_sms):
            if tid < Uint32(32):
                wait_route_count_doorbell(grid_ready, epoch, tid)
            cute.arch.sync_threads()

    @cute.kernel
    def kernel(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
               rank_idx: Uint32, num_tokens: Uint32,
               tensor_maps=None, y_out=None, stats=None,
               activation_clamp=None, activation_alpha=None,
               tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
               tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
               tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
               tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
               tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
               tensor_map_16=None, tensor_map_17=None,
               tensor_map_18=None, tensor_map_19=None,
               tensor_map_20=None, tensor_map_21=None):
        block, _, _ = cute.arch.block_idx()
        run_body = True
        shared_slab = None
        if cutlass.const_expr(self.fused_finalize_role is not None):
            if cutlass.const_expr(not self.fused_finalize_role.decode_vector_combine):
                allocator = cutlass.utils.SmemAllocator()
                size = max(self.shared_layout.field_end, self.fused_finalize_role.shared_layout.field_end)
                shared_slab = cute.recast_ptr(allocator.allocate(size, 1024), dtype=Uint8)
        if cutlass.const_expr(self.fused_dispatch_role is not None):
            smem = cutlass.utils.SmemAllocator()
            size = max(self.shared_layout.field_end, self.fused_dispatch_role.shared_layout.field_end)
            shared_slab = cute.recast_ptr(smem.allocate(size, 1024), dtype=Uint8)
            run_body = block >= self.dispatch_sms
            if not run_body:
                self.fused_dispatch_role._role_body(signals, peer_ptrs, rank_idx, num_tokens, block,
                                                    shared_slab=shared_slab)
            if cutlass.const_expr(self.fused_worker_reuse):
                if run_body:
                    block -= self.dispatch_sms
                else:
                    # The retired Dispatch CTAs join the compute queue at
                    # the end of the logical cluster index space.
                    cute.arch.sync_threads()
                    block += self.compute_sms - self.dispatch_sms
                run_body = True
            else:
                block -= self.dispatch_sms
        if run_body:
            self._role_body(signals, peer_ptrs, rank_idx, num_tokens, block,
                            tensor_maps, y_out, stats, activation_clamp, activation_alpha,
                            tensor_map_1, tensor_map_2, tensor_map_3, tensor_map_4,
                            tensor_map_5, tensor_map_6, tensor_map_7, tensor_map_8,
                            tensor_map_9, tensor_map_10, tensor_map_11, tensor_map_12,
                            tensor_map_13, tensor_map_14, tensor_map_15, tensor_map_16,
                            tensor_map_17, tensor_map_18, tensor_map_19, tensor_map_20, tensor_map_21,
                            shared_slab)
        if cutlass.const_expr(self.fused_finalize_role is not None):
            tid, _, _ = cute.arch.thread_idx()
            # The cooperative launch keeps every compute CTA resident.
            # Import its W2 scatters before the local join and peer release.
            cute.arch.sync_threads()
            if tid == Uint32(0):
                fence_acq_rel_sys()
            if cutlass.const_expr(self.compute_completion_offset != 0):
                epoch = ld_volatile_u32(signals.iterator + SIGNAL_OFFSETS['role3_epoch'] // 4)
                if tid == Uint32(0):
                    flags = signals.iterator + self.compute_completion_offset // 4
                    st_release_gpu_u32(flags + block, epoch)
                if cutlass.const_expr(self.fused_finalize_sentinel != 2):
                    self._join_compute_completion(signals, Uint32(block), Uint32(tid), rank_idx, epoch)
            else:
                grid_sync(signals, Uint32(block), Uint32(tid), self.compute_sms, 1)
            if block < self.fused_finalize_role.finalize_sms:
                output_address = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS - 1)
                stats_address = ld_peer_param_u64(peer_ptrs.iterator + MAX_RANKS - 2)
                output = cute.make_tensor(
                    cute.make_ptr(Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((1,)))
                statistics = cute.make_tensor(
                    cute.make_ptr(Int32, stats_address, cute.AddressSpace.gmem, assumed_align=4),
                    cute.make_layout((self.local_experts,)))
                if cutlass.const_expr(self.fused_finalize_sentinel):
                    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    if warp >= 4 and warp < 8:
                        self.fused_finalize_role._role3_combine(
                            signals, rank_idx, num_tokens, output, block, warp, tid % 32,
                            signals.iterator, signals.iterator, Uint64(epoch))
                    if cutlass.const_expr(self.fused_finalize_sentinel == 2):
                        self._join_compute_completion(signals, Uint32(block), Uint32(tid), rank_idx, epoch)
                    if cutlass.const_expr(self.decode_trace_offset != 0):
                        if tid == Uint32(0):
                            cta_mark(signals, self.decode_trace_offset, 2,
                                     Uint32(block), Uint32(2), Uint64(sm_id()))
                    if tid < Uint32(128):
                        recv = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4
                        self.fused_finalize_role._role3_dispatch(
                            signals, peer_ptrs, rank_idx, num_tokens, Uint32(tid), Uint32(block),
                            recv, statistics, Uint64(epoch))
                    if cutlass.const_expr(self.decode_trace_offset != 0):
                        if tid == Uint32(0):
                            cta_mark(signals, self.decode_trace_offset, 2,
                                     Uint32(block), Uint32(3), Uint64(sm_id()))
                else:
                    self.fused_finalize_role._role_body(
                        signals, peer_ptrs, rank_idx, num_tokens, block, y_out=output, stats=statistics,
                        shared_slab=shared_slab)

    @cute.jit
    def _stage_pair_w1_input(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                             block: Uint32, warp: Uint32, lane: Uint32,
                             rank: Uint32, tokens: Uint32, epoch: Uint32) -> None:
        flags = signals.iterator + self.pair_shared_w1_offset // 4
        if block == Uint32(0) and warp == Uint32(0) and lane == Uint32(0):
            st_release_sys_u64(flags, Uint64(epoch))
        if block < tokens and warp == Uint32(0):
            peer = rank ^ Uint32(1)
            peer_base = mapped_peer_base(peer_ptrs, peer, self.num_ranks)
            remote_flags = cute.make_ptr(Uint64, peer_base + Uint64(self.pair_shared_w1_offset),
                                         cute.AddressSpace.gmem, assumed_align=8)
            if lane == Uint32(0):
                wait_sys_u64_eq(remote_flags, Uint64(epoch), rank, peer)
            cute.arch.sync_warp()
            src = cute.make_ptr(Uint32, peer_base + Uint64(self.input_x_words * 4) +
                                Uint64(block) * Uint64(self.hidden), cute.AddressSpace.gmem, assumed_align=16)
            dst = signals.iterator + self.input_x_words + (block + tokens) * Uint32(self.hidden // 4)
            copy_input_vector(dst, src, lane, self.hidden, 4)
            peer_scales = cute.make_ptr(
                Uint32, peer_base + Uint64(self.input_sf_words * 4) +
                Uint64(block) * Uint64(self.hidden // 32), cute.AddressSpace.gmem, assumed_align=4)
            target = cute.make_tensor(signals.iterator + self.shared_l1_sf_words,
                                      cute.make_layout((self.shared_sf_tokens * (self.hidden // 128),)))
            source = cute.make_tensor(peer_scales, cute.make_layout((self.hidden // 128,)))
            row = block + tokens
            for column in cutlass.range(lane, Uint32(self.hidden // 128), 32):
                target[column * Uint32(self.shared_sf_tokens) + row * Uint32(4)] = source[column]
            cute.arch.sync_warp()
            if lane == Uint32(0):
                if atom_acq_rel_sys_add_u32(flags + 3, Uint32(1)) + Uint32(1) == tokens:
                    signals[self.pair_shared_w1_offset // 4 + 3] = Uint32(0)
                    st_release_u32(flags + 2, epoch, False)

    @cute.jit
    def _role_body(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                   rank_idx: Uint32, num_tokens: Uint32, block: Int32,
                   tensor_maps=None, y_out=None, stats=None,
                   activation_clamp=None, activation_alpha=None,
                   tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
                   tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
                   tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
                   tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
                   tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
                   tensor_map_16=None, tensor_map_17=None, tensor_map_18=None,
                   tensor_map_19=None, tensor_map_20=None, tensor_map_21=None, shared_slab=None):
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.compute_block_offset != 0):
            block += self.compute_block_offset
        physical_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        if cutlass.const_expr(self.role == 1 and self.dispatch_backfill):
            if tid == Uint32(0):
                launch_dependents()
        if cutlass.const_expr(self.role == 2 and self.programmatic_finalize and not self.stripe_launch_delay):
            if tid == Uint32(0):
                launch_dependents()
        if cutlass.const_expr(self.decode_trace_offset != 0):
            if tid == Uint32(0):
                cta_mark(signals, self.decode_trace_offset, self.role,
                         Uint32(block), Uint32(0),
                         Uint64(sm_id()) if cutlass.const_expr(self.role == 2) else Uint64(0))

        if cutlass.const_expr(self.role == 1 and self.push_decode_activations):
            self._push_decode_rows(signals, peer_ptrs, rank_idx, num_tokens, Uint32(block),
                                   Uint32(tid), Uint32(physical_warp), Uint32(lane))
        elif cutlass.const_expr(self.role == 1):
            # First count local routes, reserve a distinct source-index span,
            # then publish the receive count only after writing that span.
            # Route each global expert to its owner rank and local expert.
            # Each local expert has an independent receive count and a padded
            # pool-block prefix; ring slots remain shared by all experts.
            slab = shared_slab
            if cutlass.const_expr(shared_slab is None):
                dispatch_smem = cutlass.utils.SmemAllocator()
                slab = cute.recast_ptr(
                    dispatch_smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            expert_token_count = cute.recast_ptr(
                slab + self.shared_layout.expert_counts, dtype=Uint32)
            pull = cute.recast_ptr(slab + self.shared_layout.pull, dtype=Uint32)
            pull_bar = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane))
            warpgroup_reg_dealloc(48 if self.local_experts <= 64 else 96)
            if cutlass.const_expr(self.bulk_w1_prefetch_on_dispatch == 1):
                self._dispatch_prefetch_routed_w1(peer_ptrs, Uint32(block), Uint32(physical_warp))
            # Dual-buffered receive signals: replay N publishes and reads
            # generation parity & 1; role 3 flips the persistent parity
            # word after consuming each replay, and the replay that retires
            # a generation clears it here, so no cross-rank tail barrier is
            # needed.  Role 3 reads the epoch published below instead of a
            # grid barrier.
            parity = ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)
            if cutlass.const_expr(self.pair_shared_w1_offset):
                self._stage_pair_w1_input(signals, peer_ptrs, Uint32(block), Uint32(physical_warp),
                                           Uint32(lane), rank_idx, num_tokens, parity + Uint32(1))
            retired = (parity + Uint32(1)) & Uint32(1)
            gen = parity & Uint32(1)
            pull_phase = Uint32(0)
            if cutlass.const_expr(self.peer_decode_input_offset != 0):
                if block != Uint32(0):
                    scratch = pull + physical_warp * (self.pull_bytes // 4)
                    barrier = pull_bar + physical_warp
                    stride = (self.dispatch_sms - 1) * self.dispatch_warps
                    if cutlass.const_expr(self.peer_decode_token_fanout):
                        first = Uint32(block) - Uint32(1) + \
                            Uint32(physical_warp) * Uint32(self.dispatch_sms - 1)
                        for token in cutlass.range(first, num_tokens, stride):
                            pull_phase = self._push_decode_token_fanout(
                                signals, peer_ptrs, scratch, barrier, pull_phase, rank_idx,
                                gen, parity + Uint32(1), Uint32(token), Uint32(lane))
                    else:
                        first = (Uint32(block) - Uint32(1)) * Uint32(self.dispatch_warps) + Uint32(physical_warp)
                        for route in cutlass.range(first, num_tokens * Uint32(self.topk), stride):
                            pull_phase = self._push_decode_token_route(
                                signals, peer_ptrs, scratch, barrier, pull_phase, rank_idx,
                                gen, parity + Uint32(1), Uint32(route), Uint32(lane))
            if cutlass.const_expr(self.multicast_decode_input_offset != 0):
                first = Uint32(block) + Uint32(physical_warp) * Uint32(self.dispatch_sms)
                for source_token in cutlass.range(first, num_tokens,
                                                  self.dispatch_sms * self.dispatch_warps):
                    self._broadcast_decode_input(signals, peer_ptrs, rank_idx, gen,
                                                  parity + Uint32(1), Uint32(source_token), Uint32(lane))
            reverse_experts = gen if cutlass.const_expr(self.alternate_experts) else Uint32(0)
            if block == Uint32(0):
                if cutlass.const_expr(self.reset_decode_ring_on_dispatch):
                    # Both routed consumers wait on the coordinator's later
                    # counts-ready release. Shared W1 uses separate storage.
                    self._clear_routed_ring(signals, Uint32(tid), Uint32(self.dispatch_warps * 32))
                    cute.arch.sync_threads()
                if tid == Uint32(0):
                    # The finalize footer reads this after the Dispatch
                    # completion dependency or the fused grid's PDL wait.
                    signals[SIGNAL_OFFSETS['role3_epoch'] // 4] = parity + Uint32(1)
                retired_sum = signals.iterator + \
                    SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                    retired * Uint32(self.local_experts * 2)
                for retired_iter in cutlass.range(
                        Uint32(tid), Uint32(self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    st_global_u64(
                        retired_sum + Uint32(retired_iter) * Uint32(2), Uint64(0))
                if cutlass.const_expr(self.packed_route_count_group == 3):
                    groups = (self.local_experts + 2) // 3
                    if tid < Uint32(groups):
                        retired_packets = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                            (Uint32(2 * self.local_experts) + retired * Uint32(groups) + tid) * Uint32(2)
                        st_global_u64(retired_packets, Uint64(0))
                if cutlass.const_expr(not self.packed_route_counts and not self.prune_decode_resets):
                    retired_recv = signals.iterator + \
                        SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                        retired * Uint32(self.num_ranks * self.local_experts * 2)
                    for retired_iter in cutlass.range(
                            Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                            self.dispatch_warps * 32, unroll_full=True):
                        st_global_u64(
                            retired_recv + Uint32(retired_iter) * Uint32(2), Uint64(0))
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            send_ptr = signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4
            routes = cute.make_tensor(
                cute.recast_ptr(signals.iterator + self.topk_offset_words, dtype=Uint64),
                cute.make_layout((self.capacity * self.topk,)))
            source_idx = cute.make_tensor(
                cute.recast_ptr(signals.iterator + SIGNAL_OFFSETS['size'] // 4,
                                dtype=Uint32),
                cute.make_layout((self.num_ranks * self.local_experts * self.capacity,)))
            # The existing source-index allocation has room for half as
            # many 64-bit records. The host checks the small-batch bound.
            route_capacity = self.capacity // 2 if self.inline_route_scores else self.capacity
            route_record_bytes = 8 if self.inline_route_scores else 4
            if cutlass.const_expr(self.pull_route_metadata_offset != 0):
                self._pull_peer_routes(signals, peer_ptrs, routes, source_idx,
                                       expert_token_count, recv_ptr, Uint32(block),
                                       Uint32(tid), Uint32(physical_warp), Uint32(lane), rank_idx, num_tokens,
                                       gen, parity + Uint32(1))
            else:
                # Reference read_topk_idx: adjacent lanes own adjacent topk
                # entries, so one warp processes floor(32 / topk) tokens per
                # iteration. Lanes past that whole-token group stay inactive.
                tokens_per_warp = 32 // self.topk
                active_route_lanes = tokens_per_warp * self.topk
                route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                    tokens_per_warp)
                route_stride = self.dispatch_sms * self.dispatch_warps * tokens_per_warp
                if cutlass.const_expr(not self.decode_single_publisher):
                    for route_iter in cutlass.range(
                            route_base, num_tokens, route_stride, unroll_full=True):
                        route_base = Uint32(route_iter)
                        route_token = route_base + Uint32(lane // self.topk)
                        token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                        if lane < active_route_lanes and route_token < num_tokens:
                            expert = routes[token_topk]
                            if expert < Uint64(self.num_ranks * self.local_experts):
                                atom_shared_add_u32(expert_token_count + Uint32(expert),
                                                    Uint32(1))
                        cute.arch.sync_warp()
                    cute.arch.sync_threads()
                    for expert_iter in cutlass.range(
                            Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                            self.dispatch_warps * 32, unroll_full=True):
                        expert_idx = Uint32(expert_iter)
                        old_send = atom_add_u64(
                            send_ptr + expert_idx * Uint32(2),
                            (Uint64(1) << 32) | Uint64(expert_token_count[expert_idx]))
                        expert_token_count[expert_idx] = Uint32(old_send)
                    cute.arch.sync_threads()
                route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                    tokens_per_warp)
                publishes_counts = block == Uint32(0)
                first_expert = Uint32(0)
                last_expert = Uint32(self.num_ranks * self.local_experts)
                experts_per_publisher = self.num_ranks * self.local_experts // self.route_publish_ctas
                publisher_items = Uint32(experts_per_publisher)
                if cutlass.const_expr(self.split_route_heads):
                    # CTA 0 publishes the head experts on every rank.
                    # CTA 1 independently publishes the remaining experts.
                    publishes_counts = block < Uint32(2)
                    publisher_items = Uint32(self.num_ranks) * (
                        Uint32(self.early_experts) if block == Uint32(0) else
                        Uint32(self.local_experts - self.early_experts))
                    route_base = (Uint32(physical_warp * tokens_per_warp)
                                  if publishes_counts else num_tokens)
                elif cutlass.const_expr(self.route_publish_ctas > 1):
                    # Each publisher scans the short local top-k row and owns
                    # a disjoint range of destination ranks. No publisher waits
                    # for another local CTA before releasing those counts.
                    publishes_counts = block < Uint32(self.route_publish_ctas)
                    first_expert = Uint32(block) * Uint32(experts_per_publisher)
                    last_expert = first_expert + Uint32(experts_per_publisher)
                    route_base = (Uint32(physical_warp * tokens_per_warp)
                                  if publishes_counts else num_tokens)
                for route_iter in cutlass.range(
                        route_base, num_tokens, route_stride, unroll_full=True):
                    route_base = Uint32(route_iter)
                    route_token = route_base + Uint32(lane // self.topk)
                    token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                    expert = Uint64(0xffffffffffffffff)
                    if lane < active_route_lanes and route_token < num_tokens:
                        expert = Uint64(routes[token_topk])
                    valid_expert = expert < Uint64(self.num_ranks * self.local_experts)
                    if cutlass.const_expr(self.split_route_heads):
                        is_head = Uint32(expert) % Uint32(self.local_experts) < Uint32(self.early_experts)
                        valid_expert = valid_expert and (is_head == (block == Uint32(0)))
                    if cutlass.const_expr(self.route_publish_ctas > 1):
                        valid_expert = expert >= Uint64(first_expert) and expert < Uint64(last_expert)
                    dst_rank = Uint32(expert) // Uint32(self.local_experts)
                    if cutlass.const_expr(self.source_elects_activation_owner):
                        key_rank = dst_rank if valid_expert else Uint32(self.num_ranks)
                        key = (Uint32(lane // self.topk) << Uint32(8)) | key_rank
                        owners = match_lanes_u32(key)
                        if warp_first_set_bit(owners) == Uint32(lane):
                            token_topk |= Uint32(0x80000000)
                    if valid_expert:
                        dst_local = Uint32(expert) % Uint32(self.local_experts)
                        counter = expert_token_count + Uint32(expert)
                        separate_record = True
                        slot = Uint32(0)
                        if cutlass.const_expr(self.inline_route_indices):
                            # Count in the high half; two byte-sized route
                            # indices in their assigned slots in the low half.
                            slot = append_shared_route_index(counter, token_topk)
                            separate_record = slot >= Uint32(2)
                        else:
                            slot = atom_shared_add_u32(counter, Uint32(1))
                        if separate_record:
                            dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                            dst_index = cute.make_ptr(
                                Uint32, dst_base + Uint64(SIGNAL_OFFSETS['size']) +
                                Uint64(((dst_local * Uint32(self.num_ranks) + rank_idx) *
                                        Uint32(route_capacity) + slot) * Uint32(route_record_bytes)),
                                cute.AddressSpace.gmem, assumed_align=route_record_bytes)
                            if cutlass.const_expr(self.inline_route_scores):
                                weight_index = token_topk & Uint32(0x7fffffff)
                                score_bits = signals[Uint32(self.input_weight_words) + weight_index]
                                st_global_u64(dst_index, Uint64(token_topk) | (Uint64(score_bits) << Uint64(32)))
                            else:
                                dst_index[0] = token_topk
                    cute.arch.sync_warp()
                if cutlass.const_expr(not self.decode_single_publisher):
                    grid_sync(signals, Uint32(block), Uint32(tid), self.dispatch_sms, 0)
                if publishes_counts:
                    if cutlass.const_expr(self.decode_single_publisher):
                        # All local routes fit in CTA 0. Its starting counters
                        # are zero and the scatter above already counted routes.
                        cute.arch.sync_threads()
                    if cutlass.const_expr(self.packed_route_counts):
                        fence_acq_rel_sys()
                    count_group = self.packed_route_count_group
                    packets_per_rank = (self.local_experts + count_group - 1) // count_group
                    last_packet = Uint32(self.num_ranks * packets_per_rank)
                    if cutlass.const_expr(self.route_publish_ctas > 1 or self.split_route_heads):
                        last_packet = publisher_items
                    for expert_iter in cutlass.range(
                            Uint32(tid), last_packet,
                            self.dispatch_warps * 32, unroll_full=True):
                        packet_index = first_expert + Uint32(expert_iter)
                        dst_rank = packet_index // Uint32(packets_per_rank)
                        dst_local = packet_index % Uint32(packets_per_rank) * Uint32(count_group)
                        if cutlass.const_expr(self.split_route_heads):
                            dst_rank, dst_local = self._split_route_destination(packet_index, Uint32(block))
                        if cutlass.const_expr(self.rotate_route_publication):
                            dst_rank, dst_local = self._route_publication_order(dst_rank, dst_local, rank_idx)
                        expert_idx = dst_rank * Uint32(self.local_experts) + dst_local
                        dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                        if cutlass.const_expr(self.decode_single_publisher):
                            # Keep the participant count compatible with ranks
                            # whose local batch falls back to the stock dispatch.
                            status = (Uint64(self.dispatch_sms) << 32) | \
                                Uint64(expert_token_count[expert_idx])
                        else:
                            status = ld_global_u64(send_ptr + expert_idx * Uint32(2))
                        count_index = rank_idx * Uint32(self.local_experts) + dst_local
                        if cutlass.const_expr(count_group == 8):
                            count_index = rank_idx
                        remote_count = cute.make_ptr(
                            Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count']) +
                            Uint64(gen * Uint32(self.num_ranks * self.local_experts * 8)) +
                            Uint64(count_index * Uint32(8)), cute.AddressSpace.gmem,
                            assumed_align=8)
                        count = status & Uint64(0xffffffff)
                        if cutlass.const_expr(count_group == 8):
                            packet = Uint64(parity + Uint32(1)) << Uint64(ROUTE_OCTET_PAYLOAD_BITS)
                            spill = Uint64(0)
                            needs_spill = False
                            for member in cutlass.range_constexpr(8):
                                value = expert_token_count[expert_idx + Uint32(member)]
                                packed = min(value, Uint32(ROUTE_OCTET_COUNT_MASK))
                                packet |= Uint64(packed) << Uint64(member * ROUTE_OCTET_COUNT_BITS)
                                spill |= Uint64(value - packed) << Uint64(member * 8)
                                needs_spill = needs_spill or value >= Uint32(ROUTE_OCTET_COUNT_MASK)
                            # Top-k at most eight leaves at most 193 above
                            # the sentinel in any count, even with duplicates.
                            if needs_spill:
                                st_global_u64(remote_count + Uint32(self.num_ranks), spill)
                            st_release_sys_u64(remote_count, packet)
                        elif cutlass.const_expr(self.packed_route_count_group > 1):
                            # At most 32 tokens times 32 top-k slots per source.
                            # A bank is overwritten only after its old readers finish.
                            packet = Uint64(0)
                            bits = ROUTE_ATOMIC_COUNT_BITS if count_group == 3 else ROUTE_COUNT_BITS
                            if cutlass.const_expr(self.multicast_route_counts):
                                # The aggregate can reach 32 ranks * 32 tokens * 8
                                # slots. Four 14-bit counts and 32 summed 3-bit
                                # epoch tags still leave every field independent.
                                bits = ROUTE_MULTICAST_COUNT_BITS
                                epoch = (parity + Uint32(1)) & Uint32(ROUTE_MULTICAST_EPOCH_MASK)
                                packet = Uint64(epoch) << Uint64(ROUTE_MULTICAST_PAYLOAD_BITS)
                            elif cutlass.const_expr(self.packed_route_count_group == 4):
                                packet = Uint64(parity + Uint32(1)) << Uint64(ROUTE_GROUP_PAYLOAD_BITS)
                            for member in cutlass.range_constexpr(self.packed_route_count_group):
                                if dst_local + Uint32(member) < Uint32(self.local_experts):
                                    value = expert_token_count[expert_idx + Uint32(member)]
                                    packet |= Uint64(value) << Uint64(member * bits)
                            if cutlass.const_expr(self.packed_route_count_group == 3):
                                st_global_u64(remote_count, packet)
                                fence_acq_rel_sys()
                                group = dst_local // Uint32(3)
                                offset = Uint32(2 * self.local_experts) + \
                                    gen * Uint32(packets_per_rank) + group
                                remote = cute.make_ptr(
                                    Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count_sum']) +
                                    Uint64(offset) * Uint64(8), cute.AddressSpace.gmem, assumed_align=8)
                                # The 16-bit fields cannot carry across at
                                # most 32 sources with 1,024 routes each.
                                participants = Uint64(1) << Uint64(ROUTE_ATOMIC_PARTICIPANT_SHIFT)
                                if cutlass.const_expr(self.posted_route_sums):
                                    red_sys_add_u64(remote, packet | participants)
                                else:
                                    atom_sys_add_u64(remote, packet | participants)
                            elif cutlass.const_expr(self.multicast_route_counts):
                                st_global_u64(remote_count, packet)
                                index = Uint32(self.num_ranks * self.local_experts) * \
                                    (Uint32(1) + gen) + packet_index
                                st_release_sys_u64(send_ptr + index * Uint32(2), packet)
                            elif cutlass.const_expr(self.rank_route_doorbell):
                                if dst_local != Uint32((self.local_experts - 1) // 4 * 4):
                                    st_global_u64(remote_count, packet)
                            else:
                                st_release_sys_u64(remote_count, packet)
                        elif cutlass.const_expr(self.packed_route_counts):
                            epoch = Uint64(parity + Uint32(1)) << Uint64(32)
                            st_release_sys_u64(remote_count, epoch | count)
                        else:
                            st_global_u64(remote_count, count)
                    if cutlass.const_expr(self.rank_route_doorbell):
                        # The last packet carries the receipt for all earlier
                        # packet and route scatter stores from this CTA.
                        cute.arch.sync_threads()
                        if tid < self.num_ranks:
                            last = (self.local_experts - 1) // 4 * 4
                            expert = Uint32(tid * self.local_experts + last)
                            packet = Uint64(parity + Uint32(1)) << Uint64(ROUTE_GROUP_PAYLOAD_BITS)
                            for member in cutlass.range_constexpr(self.local_experts - last):
                                packet |= Uint64(expert_token_count[expert + Uint32(member)]) << \
                                    Uint64(member * ROUTE_COUNT_BITS)
                            dst_base = mapped_peer_base(peer_ptrs, Uint32(tid), self.num_ranks)
                            index = (gen * Uint32(self.num_ranks) + rank_idx) * \
                                Uint32(self.local_experts) + Uint32(last)
                            dst = cute.make_ptr(
                                Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count']) +
                                Uint64(index) * Uint64(8), cute.AddressSpace.gmem, assumed_align=8)
                            fence_acq_rel_sys()
                            st_release_sys_u64(dst, packet)
                    elif cutlass.const_expr(not self.packed_route_counts):
                        # Bring every route and per-rank count store into the
                        # publisher's order before the system release fence.
                        cute.arch.sync_threads()
                        fence_acq_rel_sys()
                        for expert_iter in cutlass.range(
                                Uint32(tid), publisher_items,
                                self.dispatch_warps * 32, unroll_full=True):
                            expert_idx = first_expert + Uint32(expert_iter)
                            dst_rank = expert_idx // Uint32(self.local_experts)
                            dst_local = expert_idx % Uint32(self.local_experts)
                            if cutlass.const_expr(self.split_route_heads):
                                dst_rank, dst_local = self._split_route_destination(expert_idx, Uint32(block))
                                expert_idx = dst_rank * Uint32(self.local_experts) + dst_local
                            if cutlass.const_expr(self.rotate_route_publication):
                                dst_rank, dst_local = self._route_publication_order(dst_rank, dst_local, rank_idx)
                                expert_idx = dst_rank * Uint32(self.local_experts) + dst_local
                            dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                            if cutlass.const_expr(self.decode_single_publisher):
                                # Keep the participant count compatible with ranks
                                # whose local batch falls back to the stock dispatch.
                                expert_count = Uint32(expert_token_count[expert_idx])
                                if cutlass.const_expr(self.inline_route_indices):
                                    expert_count >>= Uint32(16)
                                status = (Uint64(self.dispatch_sms) << 32) | Uint64(expert_count)
                            else:
                                status = ld_global_u64(send_ptr + expert_idx * Uint32(2))
                            if cutlass.const_expr(self.multicast_route_counts):
                                index = Uint32(self.num_ranks * self.local_experts) * \
                                    (Uint32(1) + gen) + expert_idx
                                epoch = Uint64(parity + Uint32(1)) << Uint64(32)
                                st_release_sys_u64(send_ptr + index * Uint32(2),
                                                   epoch | Uint64(Uint32(status)))
                            else:
                                remote_sum = cute.make_ptr(
                                    Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count_sum']) +
                                    Uint64(gen * Uint32(self.local_experts * 8)) +
                                    Uint64(dst_local * Uint32(8)),
                                    cute.AddressSpace.gmem, assumed_align=8)
                                if cutlass.const_expr(self.posted_route_sums):
                                    red_sys_add_u64(remote_sum, status)
                                else:
                                    atom_sys_add_u64(remote_sum, status)
            cute.arch.sync_threads()
            # Each per-expert acquire below imports every rank's route/count
            # stores, replacing the dispatch rank barrier. Include zero-token
            # experts: this join orders prior combine reads before new FC2
            # stores, and generation N's clear before N+2 can publish.
            if cutlass.const_expr(self.decode_trace_offset != 0):
                if tid == Uint32(0):
                    cta_mark(signals, self.decode_trace_offset, 1,
                             Uint32(block), Uint32(1), Uint64(0))
            if cutlass.const_expr(self.bulk_w1_prefetch_on_dispatch == 2):
                # Use the interval before the incoming route counts are joined.
                self._dispatch_prefetch_routed_w1(peer_ptrs, Uint32(block), Uint32(physical_warp))
            if cutlass.const_expr(self.multicast_route_counts):
                if block == Uint32(0) and physical_warp == Uint32(0):
                    self._sum_multicast_route_counts(peer_ptrs, recv_ptr, gen, parity + Uint32(1),
                                                     rank_idx, Uint32(lane))
            elif cutlass.const_expr(self.packed_route_count_group == 8):
                self._sum_octet_route_counts(signals, recv_ptr, Uint32(block),
                                             Uint32(physical_warp), Uint32(lane),
                                             gen, parity + Uint32(1), rank_idx)
            elif cutlass.const_expr(self.packed_route_count_group == 4):
                if cutlass.const_expr(not self.rank_route_doorbell) or block == Uint32(0):
                    self._sum_grouped_route_counts(signals, recv_ptr, Uint32(block),
                                                   Uint32(physical_warp), Uint32(lane),
                                                   gen, parity + Uint32(1), rank_idx)
            elif cutlass.const_expr(self.packed_route_count_group == 3):
                self._unpack_atomic_route_counts(signals, recv_ptr, Uint32(block),
                                                 Uint32(physical_warp), Uint32(lane),
                                                 gen, rank_idx)
            elif cutlass.const_expr(self.packed_route_counts):
                self._sum_packed_route_counts(signals, recv_ptr, Uint32(block),
                                              Uint32(physical_warp), Uint32(lane),
                                              gen, parity + Uint32(1), rank_idx)
            rank_waves = (self.num_ranks + 31) // 32
            if cutlass.const_expr(self.dispatch_pull_rows_per_cta):
                # Spread the first expert's short M tile across Dispatch SMs.
                # Every warp still visits a distinct residue modulo the
                # total pull warp count, and pool row order stays unchanged.
                group = self.dispatch_pull_rows_per_cta
                token_idx = (
                    Uint32(physical_warp // group) * Uint32(self.dispatch_sms * group) +
                    Uint32(block) * Uint32(group) + Uint32(physical_warp % group))
            else:
                token_idx = Uint32(block) * Uint32(self.dispatch_warps) + Uint32(physical_warp)
            expert_stride = Uint32(0)
            if cutlass.const_expr(self.fixed_expert_span != 0):
                # Fixed, disjoint M slots let each warp start from its own
                # expert count without reading the other experts' counts.
                slot = token_idx % Uint32(self.local_experts)
                observed = Uint64(0)
                if lane == Uint32(0):
                    count_ptr = recv_ptr + slot * Uint32(2)
                    wait_start = read_clock64()
                    while Uint32(observed >> 32) != Uint32(self.dispatch_sms * self.num_ranks):
                        observed = ld_volatile_u64(count_ptr)
                        if Uint64(read_clock64() - wait_start) >= Uint64(
                                self.nvlink_timeout_secs * 2000000000):
                            device_trap()
                    observed = ld_acquire_sys_u64(count_ptr)
                cute.arch.sync_warp()
                expert_end = cute.arch.shuffle_sync(Uint32(observed), 0)
                if expert_end > Uint32(self.fixed_expert_span * self.block_m):
                    device_trap()
                expert_start = Uint32(0)
                expert_stride = (Uint32(self.dispatch_sms * self.dispatch_warps - 1) -
                                 slot) // Uint32(self.local_experts) + Uint32(1)
                token_idx = token_idx // Uint32(self.local_experts)
                pool_block_offset = slot * Uint32(self.fixed_expert_span)
                current_expert = (Int32(slot) if token_idx < expert_end else
                                  Int32(self.local_experts))
            else:
                # GCMoEScheduler::fetch_expert_recv_count: each lane caches
                # expert wave*32+lane, including zero for padded lanes.
                count_coordinator = block == Uint32(0) and physical_warp == Uint32(0)
                counts_ready = signals.iterator + SIGNAL_OFFSETS['routed_counts_ready'] // 4
                read_counts = (cutlass.const_expr(not self.dispatch_shared_route_counts) or
                               physical_warp == Uint32(0))
                poll_counts = True
                if cutlass.const_expr(self.route_count_doorbell):
                    poll_counts = count_coordinator
                    if read_counts and not count_coordinator:
                        wait_route_count_doorbell(counts_ready, parity + Uint32(1), Uint32(lane))
                expert_counts = cute.make_rmem_tensor(
                    ((self.local_experts + 31) // 32,), Uint32)
                if read_counts:
                    for wave in cutlass.range_constexpr((self.local_experts + 31) // 32):
                        slot = Uint32(wave * 32) + lane
                        observed = Uint64(0)
                        if slot < Uint32(self.local_experts):
                            expert_idx = physical_expert(slot, reverse_experts, self.local_experts)
                            wait_start = read_clock64()
                            while poll_counts and Uint32(observed >> 32) != Uint32(
                                    self.dispatch_sms * self.num_ranks):
                                observed = ld_volatile_u64(recv_ptr + expert_idx * Uint32(2))
                                if Uint64(read_clock64() - wait_start) >= Uint64(
                                        self.nvlink_timeout_secs * 2000000000):
                                    if block == Uint32(0):
                                        cute.printf(
                                            'GC recv-count timeout role=1 rank={} expert={} gen={} observed_hi={} expected={}',
                                            rank_idx, expert_idx, gen, Uint32(observed >> 32),
                                            self.dispatch_sms * self.num_ranks)
                                    device_trap()
                            observed = ld_acquire_sys_u64(recv_ptr + expert_idx * Uint32(2))
                            if cutlass.const_expr(self.dispatch_shared_route_counts):
                                # The local route publication no longer reads this shared array.
                                expert_token_count[slot] = Uint32(observed)
                        expert_counts[wave] = Uint32(observed)
                if cutlass.const_expr(self.dispatch_shared_route_counts):
                    cute.arch.sync_threads()
                    expert_counts[0] = Uint32(0)
                    if lane < Uint32(self.local_experts):
                        expert_counts[0] = expert_token_count[lane]
                else:
                    cute.arch.sync_warp()

                if count_coordinator:
                    if cutlass.const_expr(self.route_count_doorbell):
                        if lane == Uint32(0):
                            st_release_gpu_u32(counts_ready, parity + Uint32(1))
                    own_pool_blocks = Uint32(0)
                    for wave in cutlass.range_constexpr(
                            (self.local_experts + 31) // 32):
                        own_pool_blocks += (expert_counts[wave] +
                                            Uint32(self.block_m - 1)) // Uint32(self.block_m)
                    total_pool_blocks = cute.arch.warp_redux_sync(own_pool_blocks, 'add')
                    if lane == Uint32(0):
                        signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4] = total_pool_blocks
                    if cutlass.const_expr(self.stripe_combine_offset != 0):
                        shared_blocks = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            shared_blocks = ((num_tokens + Uint32(self.block_m - 1)) //
                                             Uint32(self.block_m))
                        count = Uint64((total_pool_blocks + shared_blocks) *
                                       Uint32(2 * self.stripe_tiles) + Uint32(1)) << Uint64(32)
                        stripe = Uint32(lane)
                        while stripe < Uint32(self.stripe_groups):
                            self._signal_stripe(signals, peer_ptrs, rank_idx, stripe, count)
                            stripe += Uint32(32)


                # One global warp cursor traverses concatenated, unpadded expert
                # tokens; the pool-block prefix advances by each expert's padded
                # M-block count. Only the TMA and final L1 release elect a lane.
                current_expert = Int32(-1)
                expert_start = Uint32(0)
                expert_end = Uint32(0)
                pool_block_offset = Uint32(0)
                current_expert, expert_start, expert_end, pool_block_offset = \
                    advance_expert(
                        expert_counts, token_idx, current_expert, expert_start,
                        expert_end, pool_block_offset, lane, self.local_experts,
                        self.block_m)
            if cutlass.const_expr(self.decode_trace_offset != 0):
                if tid == Uint32(0):
                    cta_mark(signals, self.decode_trace_offset, 1,
                             Uint32(block), Uint32(2), Uint64(0))

            if cutlass.const_expr(self.double_buffer_dispatch_pull):
                pull_slot = Uint32(0)
                slot_phase = Uint32(0)
                pending_phase = Uint32(0)
                pending_addr = Uint64(0)
                pending_block = Uint32(0)
                pending_increment = Uint32(0)
                pending_pull = False
                slot_words = self.dispatch_warps * (self.pull_bytes // 4)
            while current_expert < Int32(self.local_experts):
                expert_id = physical_expert(Uint32(current_expert), reverse_experts, self.local_experts)
                received = expert_end - expert_start
                pull_token = token_idx - expert_start
                pool_block = pool_block_offset + pull_token // self.block_m
                pool_token = pool_block * Uint32(self.block_m) + pull_token % Uint32(self.block_m)
                ring_block = pool_block
                generation = Uint32(0)
                if cutlass.const_expr(not self.decode_linear_pool):
                    ring_block = pool_block % self.ring_blocks
                    generation = pool_block // self.ring_blocks

                # Each lane owns one source rank in each 32-rank wave.
                # Reduce across its register waves before the warp reduction.
                remaining = cute.make_rmem_tensor((rank_waves,), Uint32)
                packed_indices = Uint32(0)
                # Reload per token to isolate the immutable source count from
                # the destructively peeled loop-carried remainder.
                for wave in cutlass.range_constexpr(rank_waves):
                    source_rank = Uint32(wave * 32) + lane
                    remaining[wave] = Uint32(0)
                    if source_rank < Uint32(self.num_ranks):
                        count_expert = expert_id
                        if cutlass.const_expr(self.packed_route_count_group > 1):
                            count_group = Uint32(self.packed_route_count_group)
                            count_expert = expert_id // count_group * count_group
                        count_index = source_rank * Uint32(self.local_experts) + count_expert
                        if cutlass.const_expr(self.packed_route_count_group == 8):
                            count_index = source_rank
                        count_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                            gen * Uint32(self.num_ranks * self.local_experts * 2) + \
                            count_index * Uint32(2)
                        packet = ld_global_u64(count_ptr)
                        if cutlass.const_expr(self.packed_route_count_group == 8):
                            remaining[wave] = self._octet_route_count(count_ptr, packet, expert_id)
                        elif cutlass.const_expr(self.packed_route_count_group == 4):
                            bits = (ROUTE_MULTICAST_COUNT_BITS if self.multicast_route_counts else
                                    ROUTE_COUNT_BITS)
                            shift = (expert_id % Uint32(4)) * Uint32(bits)
                            remaining[wave] = Uint32(packet >> Uint64(shift)) & Uint32((1 << bits) - 1)
                        elif cutlass.const_expr(self.packed_route_count_group == 3):
                            shift = (expert_id % Uint32(3)) * Uint32(ROUTE_ATOMIC_COUNT_BITS)
                            remaining[wave] = Uint32(packet >> Uint64(shift)) & Uint32(ROUTE_ATOMIC_COUNT_MASK)
                        elif cutlass.const_expr(self.inline_route_indices):
                            packed_indices = Uint32(packet)
                            remaining[wave] = packed_indices >> Uint32(16)
                        else:
                            remaining[wave] = Uint32(packet)
                src_rank = Uint32(0)
                src_slot = Uint32(0)
                if cutlass.const_expr(self.dispatch_prefix_scan):
                    # Short decode intervals can be located in parallel. The
                    # original source rank, token and top-k slot still travel
                    # with each row, so Combine's accumulation order is fixed.
                    inclusive = warp_inclusive_sum(remaining[0], Uint32(lane))
                    begin = inclusive - remaining[0]
                    owners = Uint32(cute.arch.vote_ballot_sync(
                        pull_token >= begin and pull_token < inclusive))
                    src_rank = warp_first_set_bit(owners)
                    src_slot = pull_token - cute.arch.shuffle_sync(begin, src_rank)
                elif cutlass.const_expr(self.num_ranks <= 8):
                    # Concatenate source-rank intervals without changing the
                    # source-index payload or its output metadata encoding.
                    prefix = Uint32(0)
                    for source_rank in cutlass.range_constexpr(self.num_ranks):
                        count = cute.arch.shuffle_sync(remaining[0], source_rank)
                        if pull_token >= prefix and pull_token < prefix + count:
                            src_rank = Uint32(source_rank)
                            src_slot = pull_token - prefix
                        prefix += count
                else:
                    peel_slot = Uint32(pull_token)
                    peel_offset = Uint32(0)
                    selected = False
                    while not selected:
                        lane_active = Uint32(0)
                        lane_minimum = Uint32(0xffffffff)
                        for wave in cutlass.range_constexpr(rank_waves):
                            count = remaining[wave]
                            lane_active += Uint32(1) if count > Uint32(0) else Uint32(0)
                            if count > Uint32(0):
                                lane_minimum = min(lane_minimum, count)
                        active = cute.arch.warp_redux_sync(lane_active, 'add')
                        minimum = cute.arch.warp_redux_sync(lane_minimum, 'min')
                        round_tokens = active * minimum
                        if peel_slot < round_tokens:
                            position = peel_slot % active
                            seen = Uint32(0)
                            for wave in cutlass.range_constexpr(rank_waves):
                                mask = Uint32(cute.arch.vote_ballot_sync(
                                    remaining[wave] > Uint32(0)))
                                wave_active = Uint32(cute.arch.popc(mask))
                                if position >= seen and position < seen + wave_active:
                                    src_rank = Uint32(wave * 32) + warp_nth_set_bit(
                                        mask, position - seen + Uint32(1))
                                seen += wave_active
                            src_slot = peel_offset + peel_slot // active
                            selected = True
                        else:
                            peel_slot -= round_tokens
                            peel_offset += minimum
                            for wave in cutlass.range_constexpr(rank_waves):
                                remaining[wave] -= min(remaining[wave], minimum)
                route_slot = ((expert_id * Uint32(self.num_ranks) + src_rank) *
                              Uint32(route_capacity) + src_slot)
                token_topk = Uint32(0)
                weight = Uint32(0)
                if cutlass.const_expr(self.inline_route_scores):
                    record = ld_global_u64(signals.iterator + SIGNAL_OFFSETS['size'] // 4 +
                                           route_slot * Uint32(2))
                    token_topk = Uint32(record)
                    weight = Uint32(record >> Uint64(32))
                elif cutlass.const_expr(self.inline_route_indices):
                    selected_indices = cute.arch.shuffle_sync(packed_indices, src_rank)
                    token_topk = (selected_indices >> ((src_slot & Uint32(1)) * Uint32(8))) & Uint32(0xff)
                    if src_slot >= Uint32(2):
                        token_topk = Uint32(source_idx[route_slot])
                else:
                    token_topk = source_idx[route_slot]
                copy_row = cutlass.const_expr(not self.peer_decode_input_direct)
                if cutlass.const_expr(self.source_elects_activation_owner):
                    copy_row = (token_topk & Uint32(0x80000000)) != Uint32(0)
                    token_topk &= Uint32(0x7fffffff)
                src_token = token_topk // Uint32(self.topk)
                topk_slot = token_topk % Uint32(self.topk)
                if cutlass.const_expr(not self.decode_linear_pool):
                    if generation > 0:
                        empty_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block
                        target = generation * (2 * self.intermediate // 128)
                        while ld_acquire_gpu_u32(empty_ptr) < target:
                            pass
                src_base = mapped_peer_base(peer_ptrs, src_rank, self.num_ranks)
                remote_row_bytes = self.pull_bytes if self.packed_decode_activations else self.hidden
                src_address = src_base + Uint64(self.input_x_words * 4) + \
                    Uint64(src_token) * Uint64(remote_row_bytes)
                if cutlass.const_expr(self.decode_input_inbox_offset != 0):
                    needs_mailbox = (cutlass.const_expr(self.peer_decode_input_offset == 0 or
                                                        self.peer_decode_input_direct) or src_rank != rank_idx)
                    if needs_mailbox and self._prefetches_peer_input(expert_id):
                        inbox = signals.iterator + self.decode_input_inbox_offset // 4
                        row = (gen * Uint32(self.num_ranks) + src_rank) * Uint32(32) + src_token
                        ready = inbox + (2 * self.num_ranks * 32 * self.decode_input_stride) // 4 + row * Uint32(2)
                        epoch = Uint64(parity + Uint32(1))
                        use_mailbox = True
                        if cutlass.const_expr(self.peer_prefetch_opportunistic):
                            available = Uint32(0)
                            if lane == Uint32(0):
                                available = Uint32(ld_acquire_sys_u64(ready) == epoch)
                            use_mailbox = cute.arch.shuffle_sync(available, 0) != Uint32(0)
                        else:
                            if elect_one_sync():
                                wait_sys_u64_eq(ready, epoch, rank_idx, src_rank)
                        cute.arch.sync_warp()
                        if use_mailbox:
                            src_address = Uint64((inbox + row * Uint32(self.decode_input_stride // 4)).toint())
                src_x = cute.make_ptr(Uint32, src_address, cute.AddressSpace.gmem, assumed_align=16)
                dst_row = (pool_token if cutlass.const_expr(self.decode_linear_pool) else
                           pool_token % self.ring_tokens)
                row_flag = signals.iterator
                if cutlass.const_expr(self.dedup_activation_offset != 0):
                    if src_token >= Uint32(32):
                        device_trap()
                    dst_row = src_rank * Uint32(32) + src_token
                    row_flag = signals.iterator + self.dedup_activation_offset // 4 + dst_row
                    if cutlass.const_expr(not self.source_elects_activation_owner):
                        previous = Uint32(0)
                        if lane == Uint32(0):
                            previous = atom_add_u32(row_flag, Uint32(1))
                        copy_row = cute.arch.shuffle_sync(previous, 0) == Uint32(0)
                dst_x = cute.recast_ptr(
                    cute.recast_ptr(signals.iterator, dtype=Uint8) +
                    Uint64(self.l1_x_words) * Uint64(4) +
                    Uint64(dst_row) * Uint64(self.hidden),
                    dtype=Uint32)
                if cutlass.const_expr(self.decode_vector_pull):
                    if copy_row:
                        copy_input_vector(dst_x, src_x, lane, self.hidden, self.decode_vector_pull_group)
                else:
                    warp_pull = pull + physical_warp * (self.pull_bytes // 4)
                    warp_bar = pull_bar + physical_warp
                    if cutlass.const_expr(self.double_buffer_dispatch_pull):
                        warp_pull += pull_slot * Uint32(slot_words)
                        warp_bar += pull_slot * Uint32(self.dispatch_warps)
                    if copy_row and elect_one_sync():
                        if cutlass.const_expr(self.peer_activation_prefix_tiles and not self.peer_activation_head_clusters):
                            prefix_bytes = 128 * self.peer_activation_prefix_tiles
                            suffix_bytes = self.hidden - prefix_bytes
                            bulk_g2s_1d(warp_pull, src_x + prefix_bytes // 4, Uint32(suffix_bytes), warp_bar)
                            cute.arch.mbarrier_arrive_and_expect_tx(warp_bar, suffix_bytes)
                        elif cutlass.const_expr(self.activation_prefix_bytes):
                            bulk_g2s_1d(warp_pull, src_x, Uint32(self.activation_prefix_bytes), warp_bar)
                            cute.arch.mbarrier_arrive_and_expect_tx(warp_bar, self.activation_prefix_bytes)
                            if cutlass.const_expr(self.parallel_activation_prefix):
                                prefix_words = self.activation_prefix_bytes // 4
                                tail_bytes = self.hidden - self.activation_prefix_bytes
                                tail_bar = warp_bar + self.dispatch_warps
                                bulk_g2s_1d(warp_pull + prefix_words, src_x + prefix_words,
                                             Uint32(tail_bytes), tail_bar)
                                cute.arch.mbarrier_arrive_and_expect_tx(tail_bar, tail_bytes)
                        else:
                            for chunk in cutlass.range_constexpr(self.pull_chunks):
                                byte_words = chunk * (self.pull_bytes // 4)
                                bulk_g2s_1d(warp_pull, src_x + byte_words,
                                             Uint32(self.pull_bytes), warp_bar)
                                cute.arch.mbarrier_arrive_and_expect_tx(warp_bar, self.pull_bytes)
                                if chunk + 1 < self.pull_chunks:
                                    cute.arch.mbarrier_wait(warp_bar, pull_phase)
                                    pull_phase ^= Uint32(1)
                                    bulk_s2g_1d(dst_x + byte_words, warp_pull,
                                                 Uint32(self.pull_bytes))
                                    cute.arch.cp_async_bulk_commit_group()
                                    cute.arch.cp_async_bulk_wait_group(0)
                    if cutlass.const_expr(self.double_buffer_dispatch_pull):
                        if pending_pull and elect_one_sync():
                            previous_slot = pull_slot ^ Uint32(1)
                            _finish_dispatch_pull(
                                signals, pull + physical_warp * (self.pull_bytes // 4) +
                                previous_slot * Uint32(slot_words),
                                pull_bar + physical_warp + previous_slot * Uint32(self.dispatch_warps),
                                pending_phase, pending_addr, pending_block, pending_increment,
                                self.pull_bytes)
                cute.arch.sync_warp()

                sf_row = ring_block * ((self.block_m + 127) // 128 * 128) + \
                    ((pull_token % self.block_m) & ~127) + \
                    ((pull_token % self.block_m) & 31) * 4 + \
                    (((pull_token % self.block_m) >> 5) & 3)
                remote_sf_base = src_base + Uint64(self.input_sf_words * 4) + \
                    Uint64(src_token) * Uint64(self.hidden // 128 * 4)
                if cutlass.const_expr(self.peer_decode_input_direct):
                    remote_sf_base = src_address + Uint64(self.hidden)
                if cutlass.const_expr(not (self.inline_route_scores or self.packed_decode_scores)):
                    remote_weight = cute.make_ptr(
                        Uint32, src_base + Uint64(self.input_weight_words * 4) +
                        Uint64(token_topk) * Uint64(4), cute.AddressSpace.gmem)
                    weight = remote_weight[0]
                if cutlass.const_expr(self.packed_decode_activations and not self.peer_decode_input_direct):
                    if elect_one_sync():
                        cute.arch.mbarrier_wait(warp_bar, pull_phase)
                        pull_phase ^= Uint32(1)
                        if cutlass.const_expr(not self.w1_tile_major_input):
                            bulk_s2g_1d(dst_x, warp_pull, Uint32(self.hidden))
                            cute.arch.cp_async_bulk_commit_group()
                    cute.arch.sync_warp()
                    cute.arch.fence_view_async_shared()
                    if cutlass.const_expr(self.w1_tile_major_input):
                        self._store_dispatch_to_w1_tiles(
                            signals, warp_pull, ring_block, pull_token % Uint32(self.block_m), Uint32(lane))
                if cutlass.const_expr((self.hidden // 128) % 4 == 0):
                    sf_vectors = self.hidden // 512
                    for q_round in cutlass.range(
                            (sf_vectors + 31) // 32, unroll_full=True):
                        q = q_round * 32 + lane
                        if q < sf_vectors:
                            if cutlass.const_expr(self.packed_decode_activations and not self.peer_decode_input_direct):
                                local_sf = warp_pull + Uint32(self.hidden // 4) + q * Uint32(4)
                                values = (local_sf[0], local_sf[1], local_sf[2], local_sf[3])
                            else:
                                remote_vec = cute.make_ptr(
                                    Uint32, remote_sf_base + Uint64(q * Uint32(16)),
                                    cute.AddressSpace.gmem, assumed_align=16)
                                values = ld_global_u32x4(remote_vec)
                            for component in cutlass.range_constexpr(4):
                                sf_col = q * 4 + component
                                if cutlass.const_expr(self.wide_l1_sf):
                                    l1_sf_bytes = cute.recast_ptr(
                                        signals.iterator, dtype=Uint8) + \
                                        Uint64(self.l1_sf_words) * Uint64(4)
                                    sf_word = Uint64(sf_col) * \
                                        Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                    cute.recast_ptr(
                                        l1_sf_bytes + sf_word * Uint64(4),
                                        dtype=Uint32)[0] = values[component]
                                else:
                                    signals[self.l1_sf_words +
                                            sf_col * self.sf_ring_tokens + sf_row] = \
                                        values[component]
                else:
                    for sf_round in cutlass.range(
                            (self.hidden // 128 + 31) // 32, unroll_full=True):
                        sf_col = sf_round * 32 + lane
                        if sf_col < self.hidden // 128:
                            remote_word = cute.make_ptr(
                                Uint32, remote_sf_base + Uint64(sf_col * Uint32(4)),
                                cute.AddressSpace.gmem, assumed_align=4)
                            if cutlass.const_expr(self.wide_l1_sf):
                                l1_sf_bytes = cute.recast_ptr(
                                    signals.iterator, dtype=Uint8) + \
                                    Uint64(self.l1_sf_words) * Uint64(4)
                                sf_word = Uint64(sf_col) * \
                                    Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                cute.recast_ptr(
                                    l1_sf_bytes + sf_word * Uint64(4),
                                    dtype=Uint32)[0] = remote_word[0]
                            else:
                                signals[self.l1_sf_words +
                                        sf_col * self.sf_ring_tokens + sf_row] = \
                                    remote_word[0]
                cute.arch.sync_warp()
                if elect_one_sync():
                    weight_row = (pool_token if cutlass.const_expr(self.decode_linear_pool) else
                                  pool_token % self.ring_tokens)
                    weight_dst = signals.iterator + self.l1_weight_words + weight_row
                    if cutlass.const_expr(self.packed_decode_scores):
                        score = warp_pull + Uint32(self.hidden // 4 + self.hidden // 128) + topk_slot
                        weight_dst[0] = score[0]
                    else:
                        weight_dst[0] = weight
                    if cutlass.const_expr(self.packed_w2_metadata):
                        signals[self.metadata_words + pool_token] = (
                            src_rank | (src_token << Uint32(8)) | (topk_slot << Uint32(24)))
                    else:
                        signals[self.metadata_words + pool_token * 3] = src_rank
                        signals[self.metadata_words + pool_token * 3 + 1] = src_token
                        signals[self.metadata_words + pool_token * 3 + 2] = topk_slot
                    is_last_token = token_idx == expert_end - Uint32(1)
                    full_increment = Uint32(self.block_m) - (pull_token % self.block_m) \
                        if is_last_token else Uint32(1)
                    if cutlass.const_expr(self.peer_activation_prefix_tiles):
                        prefix_count = signals.iterator + self.activation_prefix_offset // 4 + ring_block
                        red_release_gpu_add_u32(prefix_count, Uint32(full_increment))
                    if cutlass.const_expr(self.peer_activation_prefix_tiles and not self.peer_activation_head_clusters):
                        cute.arch.mbarrier_wait(warp_bar, pull_phase)
                        pull_phase ^= Uint32(1)
                        prefix_bytes = 128 * self.peer_activation_prefix_tiles
                        bulk_s2g_1d(dst_x + prefix_bytes // 4, warp_pull, Uint32(self.hidden - prefix_bytes))
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                    elif cutlass.const_expr(self.activation_prefix_bytes):
                        prefix_words = self.activation_prefix_bytes // 4
                        tail_bytes = self.hidden - self.activation_prefix_bytes
                        cute.arch.mbarrier_wait(warp_bar, pull_phase)
                        tail_bar = warp_bar
                        if cutlass.const_expr(self.parallel_activation_prefix):
                            tail_bar = warp_bar + self.dispatch_warps
                        else:
                            pull_phase ^= Uint32(1)
                            # The sequential version reuses the first barrier.
                            bulk_g2s_1d(warp_pull + prefix_words, src_x + prefix_words,
                                         Uint32(tail_bytes), tail_bar)
                            cute.arch.mbarrier_arrive_and_expect_tx(tail_bar, tail_bytes)
                        bulk_s2g_1d(dst_x, warp_pull, Uint32(self.activation_prefix_bytes))
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        prefix_count = signals.iterator + self.activation_prefix_offset // 4 + ring_block
                        red_release_gpu_add_u32(prefix_count, Uint32(full_increment))
                        cute.arch.mbarrier_wait(tail_bar, pull_phase)
                        pull_phase ^= Uint32(1)
                        bulk_s2g_1d(dst_x + prefix_words, warp_pull + prefix_words, Uint32(tail_bytes))
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                    elif cutlass.const_expr(self.packed_decode_activations):
                        if cutlass.const_expr(not (self.peer_decode_input_direct or self.w1_tile_major_input)):
                            cute.arch.cp_async_bulk_wait_group(0)
                    elif cutlass.const_expr(not (self.decode_vector_pull or
                                                  self.double_buffer_dispatch_pull)):
                        # The last remote G2S overlapped weight/SF work. Drain
                        # it, store the chunk, then publish W1 after S2G finishes.
                        if copy_row:
                            cute.arch.mbarrier_wait(warp_bar, pull_phase)
                            pull_phase ^= Uint32(1)
                            final_byte_words = (self.pull_chunks - 1) * (self.pull_bytes // 4)
                            bulk_s2g_1d(dst_x + final_byte_words, warp_pull,
                                         Uint32(self.pull_bytes))
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
                    if cutlass.const_expr(self.dedup_activation_offset != 0):
                        ready = Uint32(0x80000000)
                        if cutlass.const_expr(self.source_elects_activation_owner):
                            if copy_row:
                                st_release_gpu_u32(row_flag, ready)
                        else:
                            if copy_row:
                                red_release_gpu_add_u32(row_flag, ready)
                            else:
                                while (ld_acquire_gpu_u32(row_flag) & ready) == Uint32(0):
                                    pass
                    full_ptr = signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block
                    if cutlass.const_expr(self.ready_tile_offset != 0):
                        previous = atom_release_gpu_add_u32(full_ptr, Uint32(full_increment))
                        if previous + Uint32(full_increment) == Uint32(self.block_m):
                            queue = signals.iterator + self.ready_tile_offset // 4
                            position = atom_add_u32(queue, Uint32(1))
                            st_release_gpu_u32(queue + position + Uint32(1),
                                               pool_block + Uint32(1))
                    elif cutlass.const_expr(not self.double_buffer_dispatch_pull):
                        red_release_gpu_add_u32(full_ptr, Uint32(full_increment))
                cute.arch.sync_warp()
                if cutlass.const_expr(self.double_buffer_dispatch_pull):
                    pending_phase = slot_phase
                    pending_addr = Uint64(dst_x.toint())
                    pending_block = ring_block
                    pending_increment = (Uint32(self.block_m) - pull_token % self.block_m
                                         if token_idx == expert_end - Uint32(1) else Uint32(1))
                    pending_pull = True
                    pull_slot ^= Uint32(1)
                    if pull_slot == Uint32(0):
                        slot_phase ^= Uint32(1)
                if cutlass.const_expr(self.fixed_expert_span != 0):
                    token_idx += expert_stride
                    if token_idx >= expert_end:
                        current_expert = Int32(self.local_experts)
                else:
                    token_idx += Uint32(self.dispatch_sms * self.dispatch_warps)
                    current_expert, expert_start, expert_end, pool_block_offset = \
                        advance_expert(
                            expert_counts, token_idx, current_expert, expert_start,
                            expert_end, pool_block_offset, lane, self.local_experts,
                            self.block_m)
            if cutlass.const_expr(self.double_buffer_dispatch_pull):
                if pending_pull and elect_one_sync():
                    previous_slot = pull_slot ^ Uint32(1)
                    _finish_dispatch_pull(
                        signals, pull + physical_warp * (self.pull_bytes // 4) +
                        previous_slot * Uint32(slot_words),
                        pull_bar + physical_warp + previous_slot * Uint32(self.dispatch_warps),
                        pending_phase, pending_addr, pending_block, pending_increment,
                        self.pull_bytes)
            if cutlass.const_expr(self.decode_trace_offset != 0):
                if tid == Uint32(0):
                    cta_mark(signals, self.decode_trace_offset, 1,
                             Uint32(block), Uint32(3), Uint64(0))
            if cutlass.const_expr(self.bulk_w2_tail_experts):
                self._dispatch_prefetch_w2_tail(peer_ptrs, Uint32(block), Uint32(physical_warp))
        elif cutlass.const_expr(self.role == 2):
            trace_cta = Uint32(block)
            if cutlass.const_expr(self.decode_trace_offset != 0):
                # One open-CTA timestamp per CTA separates the leader from the
                # follower of each cluster without touching the task marks.
                if physical_warp == 3 and lane == 0:
                    cta_mark(signals, self.decode_trace_offset, 2, trace_cta, Uint32(12),
                             global_timer())
            startup_start = global_timer()
            if cutlass.const_expr(self.bulk_w1_prefetch_bytes and not self.bulk_w1_prefetch_on_dispatch):
                if physical_warp == 1:
                    if cutlass.const_expr(self.bulk_w1_prefetch_lanes == 1):
                        if elect_one_sync():
                            self._prefetch_routed_w1_bulk(peer_ptrs, Uint32(block), self.compute_sms)
                    elif lane < self.bulk_w1_prefetch_lanes:
                        # Each issuing lane owns distinct aligned requests.
                        self._prefetch_routed_w1_bulk(
                            peer_ptrs, Uint32(block * self.bulk_w1_prefetch_lanes + lane),
                            self.compute_sms * self.bulk_w1_prefetch_lanes)
            if cutlass.const_expr(self.descriptor_prefetch == 'weights'):
                if physical_warp == 0 and lane == Uint32(0):
                    prefetch_tensormap(tensor_map_2.iterator)
                    prefetch_tensormap(tensor_map_3.iterator)
                    prefetch_tensormap(tensor_map_7.iterator)
                    prefetch_tensormap(tensor_map_8.iterator)
                    if cutlass.const_expr(self.shared_experts):
                        prefetch_tensormap(tensor_map_11.iterator)
                        prefetch_tensormap(tensor_map_12.iterator)
                        prefetch_tensormap(tensor_map_16.iterator)
                        prefetch_tensormap(tensor_map_17.iterator)
            if cutlass.const_expr(self.descriptor_prefetch == 'all') and physical_warp == 0:
                # Each descriptor has its own aligned by-value kernel
                # parameter, in the configuration's map order.
                prefetch_tensormap(tensor_maps.iterator)
                prefetch_tensormap(tensor_map_1.iterator)
                prefetch_tensormap(tensor_map_2.iterator)
                prefetch_tensormap(tensor_map_3.iterator)
                prefetch_tensormap(tensor_map_4.iterator)
                prefetch_tensormap(tensor_map_5.iterator)
                prefetch_tensormap(tensor_map_6.iterator)
                prefetch_tensormap(tensor_map_7.iterator)
                prefetch_tensormap(tensor_map_8.iterator)
                prefetch_tensormap(tensor_map_9.iterator)
                prefetch_tensormap(tensor_map_10.iterator)
                prefetch_tensormap(tensor_map_11.iterator)
                prefetch_tensormap(tensor_map_12.iterator)
                prefetch_tensormap(tensor_map_13.iterator)
                prefetch_tensormap(tensor_map_14.iterator)
                prefetch_tensormap(tensor_map_15.iterator)
                prefetch_tensormap(tensor_map_16.iterator)
                prefetch_tensormap(tensor_map_17.iterator)
                if cutlass.const_expr(self.short_activation_sf or self.use_small_tile_path or self.narrow_bm32 or
                                      self.narrow_local_mma or self.trim_wide_activations):
                    prefetch_tensormap(tensor_map_18.iterator)
                    prefetch_tensormap(tensor_map_19.iterator)
                    prefetch_tensormap(tensor_map_20.iterator)
                    prefetch_tensormap(tensor_map_21.iterator)
            if cutlass.const_expr(self.decode_trace_offset != 0):
                if physical_warp == 0 and lane == 0:
                    cta_mark(signals, self.decode_trace_offset, 2, trace_cta, Uint32(9),
                             global_timer() - startup_start)
            if cutlass.const_expr(not (self.weight_prefix_warps or self.post_init_prefetch)):
                if physical_warp == 0 and Uint32(block) >= Uint32(self.shared_prefix_ctas):
                    # Give Shared's initial clusters a short startup path.
                    # Compress the remaining worker IDs so no prefix tiles are dropped.
                    self._prefetch_decode_weights(
                        signals, Uint32(block) - Uint32(self.shared_prefix_ctas),
                        self.compute_sms - self.shared_prefix_ctas,
                        tensor_map_2.iterator, tensor_map_3.iterator,
                        tensor_map_7.iterator, tensor_map_8.iterator)
            if cutlass.const_expr(self.decode_trace_offset != 0):
                if physical_warp == 0 and lane == 0:
                    cta_mark(signals, self.decode_trace_offset, 2, trace_cta, Uint32(10),
                             global_timer() - startup_start)
            slab = shared_slab
            if cutlass.const_expr(shared_slab is None):
                smem = cutlass.utils.SmemAllocator()
                slab = cute.recast_ptr(
                    smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
            empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
            task = cute.recast_ptr(slab + self.shared_layout.tasks, dtype=Uint32)
            cd_smem = slab + self.shared_layout.cd
            data_a = slab + self.shared_layout.data_a
            data_b = slab + self.shared_layout.data_b
            data_sfa = cute.recast_ptr(slab + self.shared_layout.data_sfa, dtype=Uint32)
            data_sfb = cute.recast_ptr(slab + self.shared_layout.data_sfb, dtype=Uint32)
            epi_amax = cute.recast_ptr(slab + self.shared_layout.amax, dtype=Float32)
            data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
            weight_full = data_full
            if cutlass.const_expr(self.trace_operand_sources):
                weight_full = cute.recast_ptr(slab + self.shared_layout.weight_full, dtype=Uint64)
            data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
            accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
            accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
            tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
            cta_rank, _, _ = cute.arch.block_in_cluster_idx()
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane),
                                            signals, trace_cta, startup_start)
            tmem_base = (tptr[0] if cutlass.const_expr(self.compact_compute or self.local_mma)
                         else Uint32(0))
            # Warpgroup-uniform deallocation/allocation. The first WG
            # runs load/MMA/scheduler; the other two run epilogue.
            if cutlass.const_expr(not self.compact_compute):
                if physical_warp < 4:
                    warpgroup_reg_dealloc(self.non_epi_registers)
                else:
                    warpgroup_reg_alloc(self.epi_registers)
            if cutlass.const_expr(self.weight_prefix_warps):
                # The follower's other two warps are free after TMEM setup.
                # Their prefetch loop runs alongside task publication and loads.
                if cta_rank != Uint32(0) and (physical_warp == 2 or physical_warp == 3):
                    first_tile = Uint32(block) - cta_rank + Uint32(physical_warp - 2)
                    self._prefetch_decode_weights(
                        signals, first_tile, self.compute_sms,
                        tensor_map_2.iterator, tensor_map_3.iterator,
                        tensor_map_7.iterator, tensor_map_8.iterator)
            if cutlass.const_expr(self.post_init_prefetch):
                # Move the weight prefix past the rendezvous onto the follower's
                # two post-TMEM warps. That warp pair also issues the follower's
                # extra weight loads, so this trades startup for load latency
                # and measured slower than the warp 0 prefix.
                if cta_rank != Uint32(0) and (physical_warp == 2 or physical_warp == 3):
                    first_tile = Uint32(block) - cta_rank + Uint32(physical_warp - 2)
                    self._prefetch_decode_weights(
                        signals, first_tile, self.compute_sms,
                        tensor_map_2.iterator, tensor_map_3.iterator,
                        tensor_map_7.iterator, tensor_map_8.iterator)
            if cutlass.const_expr(
                    min(self.shared_w1_prefetch_limit, self.hidden // 128) > self.math_stages):
                if cta_rank != Uint32(0) and (physical_warp == 2 or physical_warp == 3):
                    first_tile = Uint32(block) - cta_rank + Uint32(physical_warp - 2)
                    self._prefetch_shared_w1(first_tile, tensor_map_11.iterator)
            iteration = Uint32(0)
            # Dual-buffered receive sums: poll the generation selected by
            # the persistent parity word (volatile read, fresh from L2).
            parity = ld_volatile_u32(signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)
            gen = parity & Uint32(1)
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['l1_task_count'] // 4
            l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['l2_task_count'] // 4
            shared_l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l1_task_count'] // 4
            shared_l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l2_task_count'] // 4
            expert_counts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            if cutlass.const_expr(self.decode_trace_offset != 0):
                if cta_rank == self.task_scheduler_cta and physical_warp == 3 and lane == 0:
                    cta_mark(signals, self.decode_trace_offset, 2, trace_cta, Uint32(11),
                             global_timer() - startup_start)
            if cta_rank == self.task_scheduler_cta and physical_warp == 3:
                schedule_block = Int32(block)
                if cutlass.const_expr(self.decode_cluster_stride):
                    cluster = Uint32(block) // Uint32(2)
                    if cluster < Uint32(self.decode_resident_clusters):
                        # Reorder initially resident clusters. Backfill clusters
                        # keep their later ownership and can arrive after Dispatch.
                        mapped = (cluster * Uint32(self.decode_cluster_stride) %
                                  Uint32(self.decode_resident_clusters))
                        schedule_block = Int32(mapped * Uint32(2) + (Uint32(block) & Uint32(1)))
                schedule_tasks(
                    task, full, empty, recv_ptr, l1_count_ptr, l2_count_ptr,
                    shared_l1_count_ptr, shared_l2_count_ptr, expert_counts,
                    iteration, Int32(lane), schedule_block * self.scheduler_scale, num_tokens,
                    gen if cutlass.const_expr(self.alternate_experts) else Uint32(0),
                    self.block_m,
                    self.compute_sms * self.scheduler_scale,
                    self.dispatch_sms,
                    self.hidden,
                    self.intermediate,
                    self.l1_clusters * self.scheduler_scale,
                    self.l2_clusters * self.scheduler_scale,
                    self.local_experts,
                    self.num_ranks,
                    self.shared_experts,
                    self.fixed_expert_span,
                    self.early_experts,
                    signals.iterator + self.ready_tile_offset // 4 + 1
                    if cutlass.const_expr(self.ready_tile_offset != 0) else None,
                    self.static_w1_prefix,
                    self.speculative_w1,
                    self.static_decode_wave,
                    self.decode_w1_warmup, self.decode_w1_cadence,
                    self.decode_shared_w2,
                    self.decode_w2_clusters,
                    signals.iterator + SIGNAL_OFFSETS['routed_counts_ready'] // 4
                    if cutlass.const_expr(self.route_count_doorbell) else None,
                    parity + Uint32(1) if cutlass.const_expr(self.route_count_doorbell) else Uint32(0),
                    signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4
                    if cutlass.const_expr(self.decode_w2_ready_k) else None,
                    self.decode_w2_ready_k, self.ring_blocks, self.independent_compute_ctas,
                    self.cache_routed_ranges,
                    self.decode_w2_n_group,
                    self.single_expert_tile_lookup,
                    split_route_heads=self.split_route_heads,
                    prioritize_expert_rows=self.prioritize_expert_rows,
                    shared_w2_ready=(signals.iterator + SIGNAL_OFFSETS['shared_l2_full_count'] // 4
                                     if cutlass.const_expr(self.decode_shared_w2 == 4) else None),
                    decode_w2_early_workers=self.decode_w2_early_workers,
                    speculative_w1_n_tiles=self.speculative_w1_n_tiles,
                    static_w2_overlap=self.static_w2_overlap,
                    task_slots=self.task_slots,
                    pair_shared_w1_rank=(rank_idx & Uint32(1)
                                         if cutlass.const_expr(self.pair_shared_w1_offset) else None),
                )
            if cutlass.const_expr(self.scheduled_weight_tiles or self.w2_prefetch_from_w1):
                if cta_rank != Uint32(0) and physical_warp == 2:
                    self._prefetch_scheduled_weights(
                        task, full, empty, tensor_map_2.iterator, tensor_map_7.iterator)
            if cutlass.const_expr(self.packed_scale_helper):
                if ((cta_rank == Uint32(0) and physical_warp == Uint32(3)) or
                        (cta_rank == Uint32(1) and physical_warp == Uint32(2))):
                    self._load_packed_scale_tasks(peer_ptrs, task, full, empty, data_sfb, data_full,
                                                  data_empty, cta_rank, Uint32(lane))
            if cutlass.const_expr(self.combine_in_compute in (1, 2)):
                # The follower's two unused warps cover the same chunk space
                # as the separate Combine grid, without waiting for local W2.
                if cta_rank != Uint32(0) and (physical_warp == 2 or physical_warp == 3):
                    first_chunk = Uint32(block) - cta_rank + Uint32(physical_warp - 2)
                    self._combine_compute_warp(
                        signals, peer_ptrs, rank_idx, num_tokens, first_chunk,
                        Uint32(lane), parity + Uint32(1))
            # The same task stream drives both TMA producer warps
            # and the leader MMA warp. All three consume every valid phase.
            extra_weight_warp = False
            if cutlass.const_expr(self.dual_weight_warps):
                # The follower runs the scheduler. Only the leader needs an MMA warp.
                extra_weight_warp = ((cta_rank == 0 and physical_warp == 3) or
                                     (cta_rank != 0 and physical_warp == 2))
            activation_issuer = True
            weight_issuer = True
            if cutlass.const_expr(self.single_tma_lanes == 2):
                activation_issuer = lane == Uint32(0)
            if cutlass.const_expr(self.single_tma_lanes):
                weight_issuer = lane == Uint32(0)
            if physical_warp == 0 and activation_issuer:
                task_words = cute.make_tensor(task, cute.make_layout((8 * self.task_slots,)))
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                launch_pending = True
                running = True
                while running:
                    task_stage = task_iter % Uint32(self.task_slots)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(self.task_slots)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        if cutlass.const_expr(self.stripe_launch_delay == 1):
                            # Both cluster CTAs enter the first W2 activation task.
                            # CTAs without W2 trigger implicitly when they exit.
                            if (lane == Uint32(0) and launch_pending and
                                    (task_phase == Uint32(GCBlockPhase.Linear2) or
                                     task_phase == Uint32(GCBlockPhase.SharedLinear2))):
                                launch_dependents()
                                launch_pending = False
                        if cutlass.const_expr(self.decode_trace_offset != 0):
                            trace_tag = task_tag(task_words, task_stage, task_phase)
                            if lane == Uint32(0):
                                task_mark(signals, self.decode_trace_offset, Uint32(block),
                                          task_iter, 0, trace_tag, self.trace_stride)
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        if cutlass.const_expr(self.speculative_w1):
                            if valid_m == Uint32(0):
                                valid_m = first_routed_rows(
                                    recv_ptr, local_expert, Int32(lane), self.block_m,
                                    self.dispatch_sms, self.num_ranks)
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        block_idx = (pool_block if cutlass.const_expr(self.decode_linear_pool) else
                                     pool_block % Uint32(self.ring_blocks))
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        # The MMA B operand consumes N/2 rows from each CTA.
                        # Keep the original stage pitch and full shared/fallback maps.
                        activation_rows = Uint32(self.activation_stage_rows)
                        activation_total_rows = None
                        if cutlass.const_expr(self.narrow_bm32 or self.narrow_local_mma):
                            narrow_rows = valid_m <= Uint32(8 if self.local_mma_n8 == 2 else 16)
                            if cutlass.const_expr(self.asymmetric_activation_rows):
                                short_tail = valid_m > Uint32(16) and valid_m <= Uint32(24)
                                activation_total_rows = (Uint32(16) if narrow_rows else
                                                         Uint32(24) if short_tail else Uint32(32))
                                narrow_rows = narrow_rows or (short_tail and cta_rank != Uint32(0))
                            if narrow_rows:
                                activation_rows = Uint32(
                                    16 if self.narrow_local_mma and self.local_mma_n8 != 2 else 8)
                                activation_map = tensor_map_18.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_19.iterator
                                elif task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                    activation_map = tensor_map_20.iterator
                                elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                    activation_map = tensor_map_21.iterator
                            if cutlass.const_expr(self.asymmetric_activation_rows == 2):
                                if valid_m <= Uint32(8):
                                    activation_total_rows = Uint32(8)
                                    if cta_rank != Uint32(0):
                                        activation_rows = Uint32(0)
                        if cutlass.const_expr(self.use_small_tile_path):
                            if is_shared == Uint32(0) and valid_m <= Uint32(32):
                                activation_rows = Uint32(16)
                                activation_map = tensor_map_18.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_19.iterator
                        if cutlass.const_expr(self.use_small_tile_path and self.block_m > 48):
                            if is_shared == Uint32(0) and valid_m > Uint32(32) and valid_m <= Uint32(48):
                                activation_rows = Uint32(24)
                                activation_map = tensor_map_20.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_21.iterator
                        if cutlass.const_expr(self.trim_wide_activations):
                            if is_shared != Uint32(0):
                                activation_rows = Uint32(((self.num_tokens + 15) // 16) * 8)
                                activation_map = tensor_map_18.iterator
                                if task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                    activation_map = tensor_map_19.iterator
                            elif valid_m <= Uint32(48):
                                activation_rows = Uint32(24)
                                activation_map = tensor_map_20.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_21.iterator
                        if cutlass.const_expr(self.peer_decode_input_direct):
                            if task_phase == Uint32(GCBlockPhase.Linear1):
                                activation_rows = Uint32(self.block_m // 2)
                                if cutlass.const_expr(self.trim_decode_gather_tail):
                                    activation_rows = ((valid_m + Uint32(15)) & Uint32(~15)) // Uint32(2)
                                activation_map = tensor_maps.iterator
                        load_transposed_w2 = None
                        if cutlass.const_expr(self.transpose_w2_activations):
                            load_transposed_w2 = (
                                task_phase == Uint32(GCBlockPhase.Linear2) or
                                task_phase == Uint32(GCBlockPhase.SharedLinear2))
                            if cutlass.const_expr(self.transpose_w2_activations == 2):
                                load_transposed_w2 = load_transposed_w2 and valid_m > Uint32(16)
                            if load_transposed_w2:
                                activation_map = tensor_map_4.iterator
                                if task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                    activation_map = tensor_map_13.iterator
                        # Keep this pair generic for the entire K loop.
                        # The conversion is full-warp, before elected TMA.
                        activation_sf_rows = None
                        if cutlass.const_expr(self.short_activation_sf):
                            activation_sf_rows = Uint32(self.sf_block_m)
                            if valid_m <= Uint32(16):
                                activation_sf_rows = Uint32(64)
                                activation_sf_map = tensor_map_18.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_sf_map = tensor_map_19.iterator
                                elif task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                    activation_sf_map = tensor_map_20.iterator
                                elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                    activation_sf_map = tensor_map_21.iterator
                        activation_map = generic_tensormap_ptr(activation_map)
                        activation_sf_map = generic_tensormap_ptr(activation_sf_map)
                        routed_w1 = task_phase == Uint32(GCBlockPhase.Linear1) and \
                            (not self.speculative_w1 or valid_m != Uint32(0))
                        l1_full_ptr = signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + block_idx
                        peer_prefix_task = routed_w1
                        if cutlass.const_expr(self.peer_activation_head_clusters):
                            peer_prefix_task = routed_w1 and n_cluster < Uint32(self.peer_activation_head_clusters)
                        l1_first_ptr = l1_full_ptr
                        if cutlass.const_expr(self.activation_prefix_offset != 0):
                            if peer_prefix_task:
                                l1_first_ptr = signals.iterator + self.activation_prefix_offset // 4 + block_idx
                        required = Uint32(self.block_m)
                        if cutlass.const_expr(not self.decode_linear_pool):
                            required *= pool_block // Uint32(self.ring_blocks) + Uint32(1)
                        if routed_w1:
                            if cutlass.const_expr(self.push_row_ready_offset):
                                ready = signals.iterator + self.push_row_ready_offset // 4 + \
                                    pool_block * Uint32(self.block_m)
                                epoch = parity + Uint32(1)
                                if cutlass.const_expr(self.push_scale_receiver_sms):
                                    epoch = parity * Uint32(2) + Uint32(2)
                                for row in cutlass.range(lane, valid_m, 32):
                                    while ld_acquire_sys_u32(ready + Uint32(row)) != epoch:
                                        pass
                                cute.arch.sync_warp()
                            elif cutlass.const_expr(self.push_early_pool_offset):
                                while ld_acquire_sys_u32(l1_first_ptr) != required:
                                    pass
                            else:
                                while ld_acquire_gpu_u32(l1_first_ptr) != required:
                                    pass
                        if cutlass.const_expr(self.shared_experts > 0):
                            if cutlass.const_expr(self.pair_shared_w1_offset):
                                if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                    ready = signals.iterator + self.pair_shared_w1_offset // 4 + 2
                                    wait_route_count_doorbell(ready, parity + Uint32(1), Uint32(lane))
                            if task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                shared_full_ptr = signals.iterator + \
                                    SIGNAL_OFFSETS['shared_l2_full_count'] // 4 + block_idx
                                shared_w1_writers = self.l1_clusters * self.shared_experts * 2
                                if cutlass.const_expr(self.pair_shared_w1_offset):
                                    start = read_clock64()
                                    while ld_acquire_sys_u32(shared_full_ptr) != shared_w1_writers:
                                        if read_clock64() - start >= Uint64(
                                                self.nvlink_timeout_secs * 2000000000):
                                            cute.printf('GC paired shared W1 timeout rank={}', rank_idx)
                                            device_trap()
                                else:
                                    while ld_acquire_gpu_u32(shared_full_ptr) != shared_w1_writers:
                                        pass
                        peer_prefix_rank = Uint32(0)
                        peer_prefix_token = Uint32(0)
                        peer_first_row = Uint32(0)
                        if cutlass.const_expr(self.peer_activation_prefix_tiles):
                            if peer_prefix_task:
                                aligned_rows = (valid_m + Uint32(15)) & Uint32(~15)
                                peer_first_row = Uint32(cta_rank * (aligned_rows // Uint32(2)))
                                row = peer_first_row + lane
                                if lane < activation_rows and row < valid_m:
                                    source = pool_block * Uint32(self.block_m) + row
                                    if cutlass.const_expr(self.packed_w2_metadata):
                                        record = Uint32(signals[self.metadata_words + source])
                                        peer_prefix_rank = record & Uint32(0xff)
                                        peer_prefix_token = (record >> Uint32(8)) & Uint32(0xffff)
                                    else:
                                        source *= Uint32(3)
                                        peer_prefix_rank = Uint32(signals[self.metadata_words + source])
                                        peer_prefix_token = Uint32(signals[self.metadata_words + source + 1])
                        gather_rows = None
                        gather_routed_w1 = Uint32(0)
                        peer_activation_map = None
                        if cutlass.const_expr(self.peer_activation_rank_rows):
                            peer_row = Uint32(0xffffffff)
                            if peer_prefix_task and lane < activation_rows and peer_first_row + lane < valid_m:
                                peer_row = peer_prefix_rank * Uint32(self.peer_activation_rank_rows) + peer_prefix_token
                            gather_rows = ()
                            for group in cutlass.range_constexpr(self.block_m // 8):
                                rows = ()
                                for row in cutlass.range_constexpr(4):
                                    rows += (cute.arch.shuffle_sync(peer_row, group * 4 + row),)
                                gather_rows += (rows,)
                            peer_activation_map = generic_tensormap_ptr(tensor_map_4.iterator)
                        if cutlass.const_expr(self.dedup_activation_offset != 0 or self.peer_decode_input_direct):
                            gather_routed_w1 = Uint32(task_phase == Uint32(GCBlockPhase.Linear1))
                            cta_rows = ((valid_m + Uint32(15)) & Uint32(~15)) // Uint32(2)
                            row = cta_rank * cta_rows + lane
                            key = Uint32(0xffffffff)
                            gather_lanes = (cta_rows if cutlass.const_expr(self.peer_decode_input_direct)
                                            else Uint32(self.block_m // 2))
                            if gather_routed_w1 and lane < gather_lanes and row < valid_m:
                                record = self.metadata_words + (pool_block * Uint32(self.block_m) +
                                                                row) * Uint32(3)
                                key = signals[record] * Uint32(32) + signals[record + 1]
                                if cutlass.const_expr(self.peer_decode_input_direct):
                                    key += gen * Uint32(self.num_ranks * 32)
                                if cutlass.const_expr(self.source_elects_activation_owner):
                                    # Dispatch publishes metadata for nonowners without
                                    # waiting for a copy scheduled on another pull warp.
                                    flag = signals.iterator + self.dedup_activation_offset // 4 + key
                                    while ld_acquire_gpu_u32(flag) == Uint32(0):
                                        pass
                            if cutlass.const_expr(self.source_elects_activation_owner):
                                cute.arch.sync_warp()
                            gather_rows = ()
                            for group in cutlass.range_constexpr(self.block_m // 8):
                                rows = ()
                                for r in cutlass.range_constexpr(4):
                                    rows += (cute.arch.shuffle_sync(key, group * 4 + r),)
                                gather_rows += (rows,)
                        # GCL2KBlockDependency belongs to this task, so
                        # the observed XOR-normalized mask survives all
                        # K128 waits and is reset for the next task.
                        mask_ptr = signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 + block_idx * 2
                        expected = Uint64(self.l1_full_mask)
                        if cutlass.const_expr(not self.decode_linear_pool):
                            generation = pool_block // Uint32(self.ring_blocks)
                            if generation & Uint32(1):
                                expected = Uint64(0)
                        pending = Uint64(self.l1_full_mask)
                        load_valid_m = valid_m
                        if cutlass.const_expr(self.speculative_w1):
                            # Empty experts still retire the speculative
                            # pipeline using one padded MMA tile.
                            load_valid_m = max(valid_m, Uint32(1))
                        activation_tile_major = None
                        if cutlass.const_expr(self.w1_tile_major_input or self.w2_tile_major_input):
                            activation_tile_major = False
                            if cutlass.const_expr(self.w1_tile_major_input):
                                activation_tile_major = task_phase == Uint32(GCBlockPhase.Linear1)
                            if cutlass.const_expr(self.w2_tile_major_input):
                                activation_tile_major = activation_tile_major or (
                                    task_phase == Uint32(GCBlockPhase.Linear2) or
                                    task_phase == Uint32(GCBlockPhase.SharedLinear2))
                        k_block = Uint32(0)
                        window_issued = Uint32(0)
                        window_mask = Uint32(0x55555555 & ((1 << (2 * self.w2_a_issue_window)) - 1))
                        while k_block < shape_k // Uint32(128):
                            load_k, load_stage, load_phase = k_block, data_stage, data_phase
                            if cutlass.const_expr(self.w2_a_issue_window):
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    # Only the A issuer may reorder. The B issuer and
                                    # MMA consume the same logical and physical K slots.
                                    available = Uint32(0)
                                    if lane == Uint32(0):
                                        while available == Uint32(0):
                                            ready_bits = ~pending
                                            pair_ready = ready_bits & (ready_bits >> Uint64(1))
                                            available = Uint32(pair_ready >> (k_block * Uint32(2))) & \
                                                window_mask & ~window_issued
                                            if available == Uint32(0):
                                                pending = ld_acquire_gpu_u64(mask_ptr) ^ expected
                                    cute.arch.sync_warp()
                                    available = cute.arch.shuffle_sync(available, 0)
                                    offset = warp_first_set_bit(available) // Uint32(2)
                                    load_k = k_block + offset
                                    load_stage = data_stage + offset
                                    if load_stage >= Uint32(self.math_stages):
                                        load_stage -= Uint32(self.math_stages)
                                        load_phase ^= Uint32(1)
                                    window_issued |= Uint32(1) << (offset * Uint32(2))
                            peer_activation_ready = None
                            if cutlass.const_expr(self.peer_activation_prefix_tiles and not self.peer_activation_rank_rows):
                                peer_activation_ready = routed_w1 and k_block < Uint32(self.peer_activation_prefix_tiles)
                                if peer_activation_ready:
                                    cute.arch.mbarrier_wait(data_empty + data_stage // self.release_group,
                                                            data_phase ^ Uint32(1))
                                    self._load_raw_peer_activation(
                                        data_a + data_stage * (self.activation_stage_rows * 128),
                                        peer_first_row, activation_rows, valid_m, Uint32(lane),
                                        peer_ptrs, peer_prefix_rank, peer_prefix_token, k_block)
                            if cutlass.const_expr(self.activation_prefix_offset != 0):
                                prefix_tiles = self.peer_activation_prefix_tiles or self.activation_prefix_bytes // 128
                                if peer_prefix_task and k_block == Uint32(prefix_tiles):
                                    while ld_acquire_gpu_u32(l1_full_ptr) != required:
                                        pass
                            if cutlass.const_expr(not self.w2_a_issue_window):
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    required_mask = Uint64(3) << (k_block * Uint32(2))
                                    while (pending & required_mask) != Uint64(0):
                                        pending = ld_acquire_gpu_u64(mask_ptr) ^ expected
                            if cutlass.const_expr(self.decode_trace_offset != 0):
                                if load_k == Uint32(0) and lane == Uint32(0):
                                    task_mark(signals, self.decode_trace_offset, Uint32(block),
                                              task_iter, 1, trace_tag, self.trace_stride)
                            if cutlass.const_expr(self.peer_activation_rank_rows):
                                gather_routed_w1 = Uint32(
                                    peer_prefix_task and k_block < Uint32(self.peer_activation_prefix_tiles))
                            load_mixed_k128_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + load_stage * (self.activation_stage_rows * 128),
                                data_sfa + load_stage * self.sf_block_m,
                                data_b + load_stage * (128 * 128),
                                data_sfb + load_stage * 128,
                                data_full + load_stage, data_empty + load_stage // self.release_group,
                                load_k, load_phase, block_idx,
                                n_cluster * Uint32(self.compute_cluster_size) + cta_rank,
                                local_expert, load_valid_m, activation_rows, shape_k,
                                is_shared, cta_rank, 0,
                                self.block_m, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.shared_weight_cache_hint,
                                Uint64(EVICT_FIRST), gather_rows, gather_routed_w1, self.local_mma,
                                activation_tile_major=activation_tile_major,
                                single_loader_lane=self.single_tma_lanes == 2,
                                activation_total_rows=activation_total_rows,
                                trim_gather_tail=self.trim_decode_gather_tail or bool(self.peer_activation_rank_rows),
                                transpose_activation=load_transposed_w2,
                                activation_in_smem=peer_activation_ready,
                                peer_gather_map=peer_activation_map,
                                activation_cache_hint=self.activation_cache_hint,
                                activation_sf_rows=activation_sf_rows)
                            if cutlass.const_expr(self.decode_trace_offset != 0):
                                if lane == Uint32(0):
                                    if load_k == Uint32(0):
                                        task_mark(signals, self.decode_trace_offset, Uint32(block),
                                                  task_iter, 2, trace_tag, self.trace_stride)
                                    if load_k + Uint32(1) == shape_k // Uint32(128):
                                        task_mark(signals, self.decode_trace_offset, Uint32(block),
                                                  task_iter, 15, trace_tag, self.trace_stride)
                            if cutlass.const_expr(self.w2_a_issue_window):
                                step = Uint32(1)
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    step = Uint32(0)
                                    if window_issued == window_mask:
                                        step = Uint32(self.w2_a_issue_window)
                                        window_issued = Uint32(0)
                                data_stage += step
                                if data_stage >= Uint32(self.math_stages):
                                    data_stage -= Uint32(self.math_stages)
                                    data_phase ^= Uint32(1)
                                k_block += step
                            else:
                                data_stage = Uint32(0) if data_stage == \
                                    Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                                data_phase ^= Uint32(data_stage == Uint32(0))
                                k_block += Uint32(1)
                    task_iter += Uint32(1)
            elif (physical_warp == 1 or extra_weight_warp) and weight_issuer:
                task_words = cute.make_tensor(task, cute.make_layout((8 * self.task_slots,)))
                task_iter = Uint32(0)
                weight_stream = Uint32(extra_weight_warp)
                data_stage = weight_stream
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(self.task_slots)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(self.task_slots)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        if cutlass.const_expr(self.decode_trace_offset != 0):
                            trace_tag = task_tag(task_words, task_stage, task_phase)
                            if lane == Uint32(0):
                                task_mark(signals, self.decode_trace_offset, Uint32(block),
                                          task_iter, 3, trace_tag, self.trace_stride)
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        block_idx = (pool_block if cutlass.const_expr(self.decode_linear_pool) else
                                     pool_block % Uint32(self.ring_blocks))
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        weight_map = generic_tensormap_ptr(weight_map)
                        weight_sf_map = generic_tensormap_ptr(weight_sf_map)
                        routed_hint = Uint64(self.routed_weight_cache_hint)
                        if cutlass.const_expr(self.retain_routed_w2):
                            if task_phase == Uint32(GCBlockPhase.Linear2):
                                routed_hint = Uint64(EVICT_LAST)
                        weight_k_major = None
                        if cutlass.const_expr(any(self.routed_k_major)):
                            weight_k_major = (
                                self.routed_k_major[0] and task_phase == Uint32(GCBlockPhase.Linear1) or
                                self.routed_k_major[1] and task_phase == Uint32(GCBlockPhase.Linear2))
                        paired_weight_phase = None
                        if cutlass.const_expr(self.pair_routed_weights):
                            paired_weight_phase = (
                                bool(self.pair_routed_weights & 1) and task_phase == Uint32(GCBlockPhase.Linear1) or
                                bool(self.pair_routed_weights & 2) and task_phase == Uint32(GCBlockPhase.Linear2))
                        grouped_weight_phase = None
                        grouped_n_blocks = None
                        grouped_inner_k = None
                        if cutlass.const_expr(self.grouped_routed_weights):
                            grouped_weight_phase = (
                                bool(self.grouped_routed_weights & 1) and task_phase == Uint32(GCBlockPhase.Linear1) or
                                bool(self.grouped_routed_weights & 2) and task_phase == Uint32(GCBlockPhase.Linear2))
                            grouped_n_blocks = Uint32(task_words[task_stage * 8 + 6]) // Uint32(128)
                            if cutlass.const_expr(self.grouped_routed_weights == 2):
                                grouped_inner_k = Uint32(self.grouped_weight_inner_k[1])
                            else:
                                grouped_inner_k = Uint32(self.grouped_weight_inner_k[0])
                                if cutlass.const_expr(self.grouped_routed_weights == 3 and
                                                      self.grouped_weight_inner_k[0] != self.grouped_weight_inner_k[1]):
                                    if task_phase == Uint32(GCBlockPhase.Linear2):
                                        grouped_inner_k = Uint32(self.grouped_weight_inner_k[1])
                        packed_sf_address = None
                        packed_sf_n_blocks = None
                        if cutlass.const_expr(self.packed_routed_scale_base is not None and not self.packed_scale_helper):
                            sf_slot = Uint32(PACKED_ROUTED_SCALE_OFFSETS[0] + 1)
                            if task_phase == Uint32(GCBlockPhase.Linear2):
                                sf_slot = Uint32(PACKED_ROUTED_SCALE_OFFSETS[1] + 1)
                            packed_sf_address = ld_peer_param_u64(peer_ptrs.iterator + sf_slot)
                            packed_sf_n_blocks = Uint32(task_words[task_stage * 8 + 6]) // Uint32(128)
                        k_block = weight_stream
                        while k_block < shape_k // Uint32(128):
                            if cutlass.const_expr(self.w2_b_issue_prefix):
                                if (task_phase == Uint32(GCBlockPhase.Linear2) and
                                        k_block == Uint32(self.w2_b_issue_prefix)):
                                    # Reserve the initial weight stages for overlap. Let
                                    # W1 use the next requests while W2's first A tile is pending.
                                    if lane == Uint32(0):
                                        mask = (signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                                                block_idx * Uint32(2))
                                        wanted = Uint64(3)
                                        if cutlass.const_expr(not self.decode_linear_pool):
                                            gen_bit = (pool_block // Uint32(self.ring_blocks)) & Uint32(1)
                                            wanted = Uint64(0) if gen_bit else Uint64(3)
                                        while (ld_acquire_gpu_u64(mask) & Uint64(3)) != wanted:
                                            pass
                                    if cutlass.const_expr(not self.single_tma_lanes):
                                        cute.arch.sync_warp()
                            if cutlass.const_expr(self.weight_lookahead > 0):
                                future = k_block + Uint32(self.weight_lookahead)
                                if is_shared == Uint32(0) and future < shape_k // Uint32(128):
                                    if elect_one_sync():
                                        tile_coord = future
                                        expert_coord = local_expert
                                        if cutlass.const_expr(weight_k_major is not None):
                                            if weight_k_major:
                                                tile_coord = local_expert
                                                expert_coord = future
                                        prefetch_g2s_4d(
                                            weight_map, Uint32(0),
                                            (n_cluster * Uint32(self.compute_cluster_size) + cta_rank) * Uint32(128),
                                            tile_coord, expert_coord)
                            weight_stage = data_b + data_stage * Uint32(16384)
                            scale_stage = data_sfb + data_stage * Uint32(128)
                            shared_weight_packet = None
                            if cutlass.const_expr(self.shared_weight_packet_mask == 3):
                                shared_weight_packet = is_shared
                            elif cutlass.const_expr(self.shared_weight_packet_mask == 1):
                                shared_weight_packet = Uint32(task_phase == Uint32(GCBlockPhase.SharedLinear1))
                            elif cutlass.const_expr(self.shared_weight_packet_mask == 2):
                                shared_weight_packet = Uint32(task_phase == Uint32(GCBlockPhase.SharedLinear2))
                            if cutlass.const_expr(self.paired_shared_weights):
                                group = data_stage // Uint32(2)
                                weight_stage += group * Uint32(1024)
                                scale_stage = cute.recast_ptr(
                                    data_b + group * Uint32(33792) + Uint32(32768), dtype=Uint32)
                                scale_stage += (data_stage & Uint32(1)) * Uint32(128)
                            # The weight loader ignores the activation geometry.
                            load_mixed_k128_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + data_stage * (self.activation_stage_rows * 128),
                                data_sfa + data_stage * self.sf_block_m,
                                weight_stage, scale_stage,
                                weight_full + data_stage, data_empty + data_stage // self.release_group,
                                k_block, data_phase, block_idx,
                                n_cluster * Uint32(self.compute_cluster_size) + cta_rank,
                                local_expert, valid_m, Uint32(0), shape_k,
                                is_shared, cta_rank, 1,
                                self.block_m, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.shared_weight_cache_hint,
                                routed_hint, local_mma=self.local_mma, weight_k_major=weight_k_major,
                                single_loader_lane=self.single_tma_lanes != 0,
                                pair_routed_weight_scales=self.pair_routed_weight_scales,
                                pair_routed_weights=paired_weight_phase,
                                grouped_n_blocks=grouped_n_blocks,
                                grouped_weight_phase=grouped_weight_phase,
                                grouped_weight_inner_k=grouped_inner_k,
                                packed_sf_address=packed_sf_address,
                                packed_sf_n_blocks=packed_sf_n_blocks,
                                packed_sf_base=self.packed_routed_scale_base,
                                packed_sf_k_group=self.packed_routed_scale_k_group,
                                packed_sf_helper=self.packed_scale_helper,
                                keep_routed_weight_scales=self.keep_routed_weight_scales,
                                shared_weight_packet=shared_weight_packet)
                            if cutlass.const_expr(self.decode_trace_offset != 0):
                                if lane == Uint32(0):
                                    if k_block == Uint32(0):
                                        task_mark(signals, self.decode_trace_offset, Uint32(block),
                                                  task_iter, 4, trace_tag, self.trace_stride)
                                    if k_block + Uint32(1) == shape_k // Uint32(128):
                                        task_mark(signals, self.decode_trace_offset, Uint32(block),
                                                  task_iter, 14, trace_tag, self.trace_stride)
                            if cutlass.const_expr(self.dual_weight_warps):
                                data_stage += Uint32(2)
                                if data_stage >= Uint32(self.math_stages):
                                    data_stage -= Uint32(self.math_stages)
                                    data_phase ^= Uint32(1)
                                k_block += Uint32(2)
                            else:
                                data_stage = Uint32(0) if data_stage == \
                                    Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                                data_phase ^= Uint32(data_stage == Uint32(0))
                                k_block += Uint32(1)
                    task_iter += Uint32(1)
            mma_issuer = True
            if cutlass.const_expr(self.single_mma_lane):
                mma_issuer = lane == Uint32(0)
            if (cta_rank == 0 or cutlass.const_expr(self.local_mma)) and physical_warp == 2 and mma_issuer:
                task_words = cute.make_tensor(task, cute.make_layout((8 * self.task_slots,)))
                # Create the two swapped K-major data descriptor bases
                # once for the persistent leader MMA warp.
                weight_desc_base = kmajor_smem_desc(
                    data_b, Uint32(1024), True, Uint32(0))
                activation_desc_base = kmajor_smem_desc(
                    data_a, Uint32(1024), True, Uint32(0))
                data_desc_hi = (weight_desc_base >> Uint64(32)) << Uint64(32)
                sf_desc_hi = kmajor_smem_desc_from_start(
                    Uint32(0), Uint32(128), False)
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(self.task_slots)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(self.task_slots)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        if cutlass.const_expr(self.decode_trace_offset != 0):
                            trace_tag = task_tag(task_words, task_stage, task_phase)
                            if lane == Uint32(0):
                                task_mark(signals, self.decode_trace_offset, Uint32(block),
                                          task_iter, 5, trace_tag, self.trace_stride)
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.speculative_w1):
                            if valid_m == Uint32(0):
                                valid_m = first_routed_rows(
                                    recv_ptr, Uint32(task_words[task_stage * 8 + 1]),
                                    Int32(lane), self.block_m, self.dispatch_sms, self.num_ranks)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        num_k_blocks = shape_k // Uint32(128)
                        weight_format = Uint32(5)
                        if cutlass.const_expr(self.shared_experts > 0):
                            weight_format = select_weight_format(is_shared)
                        if cutlass.const_expr(self.speculative_w1):
                            valid_m = max(valid_m, Uint32(1))
                        token_alignment = 8 if cutlass.const_expr(self.local_mma_n8) else 16
                        aligned_n = (valid_m + Uint32(token_alignment - 1)) & Uint32(~(token_alignment - 1))
                        mma_transposed_w2 = None
                        if cutlass.const_expr(self.transpose_w2_activations):
                            mma_transposed_w2 = (
                                task_phase == Uint32(GCBlockPhase.Linear2) or
                                task_phase == Uint32(GCBlockPhase.SharedLinear2))
                            if cutlass.const_expr(self.transpose_w2_activations == 2):
                                mma_transposed_w2 = mma_transposed_w2 and valid_m > Uint32(16)
                            if mma_transposed_w2:
                                aligned_n = Uint32(32)
                        instr_desc_base = blockscaled_instr_desc_hi(
                            aligned_n, weight_format, Uint32(0), Uint32(0), self.mma_cta_group)
                        if cutlass.const_expr(self.transpose_w2_activations):
                            if mma_transposed_w2:
                                instr_desc_base |= Uint32(1 << 16)
                        routed_sfb_mask = None
                        if cutlass.const_expr(self.hoist_routed_sfb_selection and
                                             self.shared_experts > 0 and self.shared_weight_gran_k == 128):
                            routed_sfb_mask = Uint32(0) - Uint32(is_shared == Uint32(0))
                        accum_stage = task_iter % Uint32(self.accum_stages)
                        accum_phase = (task_iter // Uint32(self.accum_stages)) % Uint32(2)
                        shadow_tmem = None
                        replay_mma = False
                        if cutlass.const_expr(self.probe_routed_mma):
                            shadow_tmem = tmem_base + Uint32(self.probe_mma_offset)
                            if cutlass.const_expr(self.probe_routed_mma & 1):
                                replay_mma = task_phase == Uint32(GCBlockPhase.Linear1)
                            if cutlass.const_expr(self.probe_routed_mma & 2):
                                replay_mma = replay_mma or task_phase == Uint32(GCBlockPhase.Linear2)
                        if cutlass.const_expr(self.trace_operand_wait):
                            track_wait = (lane == Uint32(0) and
                                          (Uint32(block) // Uint32(2)) %
                                          Uint32(self.trace_stride) == Uint32(0))
                            accum_wait_start = Uint64(0)
                            if track_wait:
                                accum_wait_start = global_timer()
                        cute.arch.mbarrier_wait(accum_empty + accum_stage,
                                                accum_phase ^ Uint32(1))
                        tcgen05_after_thread_sync()
                        # Unroll K128 MMA stages by the configured group size.
                        if cutlass.const_expr(self.trace_operand_wait):
                            if track_wait:
                                task_mark(signals, self.decode_trace_offset, Uint32(block),
                                          task_iter, 13,
                                          ((global_timer() - accum_wait_start) << Uint64(8)) |
                                          Uint64(task_phase), self.trace_stride)
                            total_operand_wait = Uint64(0)
                        if cutlass.const_expr(self.trace_operand_sources):
                            readiness_counts = Uint64(0)
                        for k_iter in cutlass.range(
                                Int32(0), Int32(num_k_blocks),
                                Int32(1), unroll=self.mma_unroll):
                            k_block = Uint32(k_iter)
                            if cutlass.const_expr(self.trace_operand_sources):
                                if track_wait:
                                    test = cutlass_primitives.MBarrierWait.TEST
                                    a_ready = cutlass_primitives.mbarrier_wait_parity(
                                        data_full + data_stage, data_phase, test)
                                    b_ready = cutlass_primitives.mbarrier_wait_parity(
                                        weight_full + data_stage, data_phase, test)
                                    # Counts for both ready, A late, B late and both late.
                                    index = Uint64(not a_ready) + (Uint64(not b_ready) << Uint64(1))
                                    readiness_counts += Uint64(1) << (index * Uint64(12))
                            if cutlass.const_expr(self.trace_operand_wait):
                                wait_start = Uint64(0)
                                if track_wait:
                                    wait_start = global_timer()
                            cute.arch.mbarrier_wait(data_full + data_stage,
                                                    data_phase)
                            if cutlass.const_expr(self.trace_operand_sources):
                                cute.arch.mbarrier_wait(weight_full + data_stage, data_phase)
                            if cutlass.const_expr(self.trace_operand_wait):
                                if track_wait:
                                    total_operand_wait += global_timer() - wait_start
                            if cutlass.const_expr(self.decode_trace_offset != 0):
                                if lane == Uint32(0):
                                    if k_block == Uint32(0):
                                        task_mark(signals, self.decode_trace_offset,
                                                  Uint32(block), task_iter, 6, trace_tag, self.trace_stride)
                                    if k_block == num_k_blocks - Uint32(1):
                                        task_mark(signals, self.decode_trace_offset,
                                                  Uint32(block), task_iter, 7, trace_tag, self.trace_stride)
                            # Keep descriptor layout bits fixed and
                            # advance only the 16-byte SMEM address field.
                            weight_desc_start = Uint32(
                                Uint32(weight_desc_base) + data_stage * Uint32(1024))
                            weight_sf_stage = None
                            if cutlass.const_expr(self.paired_shared_weights):
                                group = data_stage // Uint32(2)
                                weight_desc_start += group * Uint32(64)
                                weight_sf_stage = cute.recast_ptr(
                                    data_b + group * Uint32(33792) + Uint32(32768), dtype=Uint32)
                                weight_sf_stage += (data_stage & Uint32(1)) * Uint32(128)
                            activation_desc_start = Uint32(
                                Uint32(activation_desc_base) + data_stage *
                                Uint32((self.activation_stage_rows * 128) // 16))
                            issue_mixed_k128(
                                weight_desc_start, activation_desc_start,
                                data_sfb, data_sfa, data_stage,
                                data_desc_hi, sf_desc_hi,
                                tmem_base + accum_stage * Uint32(self.block_m),
                                tmem_base + Uint32(self.accum_stages * self.block_m +
                                                  self.sfa_tmem_cols),
                                tmem_base + Uint32(self.accum_stages * self.block_m),
                                data_empty + data_stage // self.release_group, accum_full + accum_stage,
                                instr_desc_base, k_block,
                                Uint32(k_block == num_k_blocks - Uint32(1)),
                                is_shared, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.reuse_k128_desc, self.release_group,
                                self.single_mma_lane, self.mma_cta_group,
                                routed_sfb_mask=routed_sfb_mask, shadow_tmem=shadow_tmem,
                                weight_sf_stage=weight_sf_stage,
                                replay_mma=replay_mma, transpose_activation=mma_transposed_w2)
                            if cutlass.const_expr(self.decode_trace_offset != 0):
                                if k_block == num_k_blocks - Uint32(1) and lane == Uint32(0):
                                    task_mark(signals, self.decode_trace_offset, Uint32(block),
                                              task_iter, 8, trace_tag, self.trace_stride)
                            data_stage = Uint32(0) if data_stage == \
                                Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                        if cutlass.const_expr(self.trace_operand_wait):
                            if track_wait:
                                task_mark(signals, self.decode_trace_offset, Uint32(block),
                                          task_iter, 11, (total_operand_wait << Uint64(8)) |
                                          Uint64(task_phase), self.trace_stride)
                        if cutlass.const_expr(self.trace_operand_sources):
                            if track_wait:
                                task_mark(signals, self.decode_trace_offset, Uint32(block),
                                          task_iter, 12, (readiness_counts << Uint64(8)) |
                                          Uint64(task_phase), self.trace_stride)
                    task_iter += Uint32(1)
                # The terminal None task increments task_iter once. Drain
                # the final valid accumulator stage before leaving the
                # producer warp, as the reference does after its loop.
                if task_iter > Uint32(1):
                    last_valid_iter = task_iter - Uint32(2)
                    last_stage = last_valid_iter % Uint32(self.accum_stages)
                    last_phase = (last_valid_iter // Uint32(self.accum_stages)) % Uint32(2)
                    cute.arch.mbarrier_wait(accum_empty + last_stage,
                                            last_phase)
            if physical_warp >= 4:
                if cutlass.const_expr(not (self.compact_compute or self.local_mma)):
                    if tptr[0] != Uint32(0):
                        device_trap()
                task_words = cute.make_tensor(task, cute.make_layout((8 * self.task_slots,)))
                iteration = Uint32(0)
                launch_pending = True
                running = True
                while running:
                    stage = iteration % self.task_slots
                    generation_phase = (iteration // self.task_slots) % 2
                    cute.arch.mbarrier_wait(full + stage, generation_phase)
                    task_phase = task_words[stage * 8]
                    n_cluster = Uint32(task_words[stage * 8 + 3])
                    pool_block = Uint32(task_words[stage * 8 + 4])
                    valid_m = Uint32(task_words[stage * 8 + 5])
                    if cutlass.const_expr(self.expert_combine_offset):
                        # The scheduler may reuse this slot after the early
                        # epilogue release, before the peer writes retire.
                        ready_expert = Uint32(task_words[stage * 8 + 1])
                    if cutlass.const_expr(self.decode_trace_offset != 0):
                        trace_tag = Uint64(0)
                    if task_phase != Uint32(GCBlockPhase.None_):
                        if cutlass.const_expr(self.prefetch_w2_metadata):
                            early_w2_metadata = cute.make_rmem_tensor((self.epilogue_rows // 8,), Uint32)
                            if task_phase == Uint32(GCBlockPhase.Linear2):
                                # The W2 task can be posted before the pool
                                # is filled. Dispatch releases this counter
                                # after writing every row's source metadata.
                                ring = pool_block
                                required = Uint32(self.block_m)
                                if cutlass.const_expr(not self.decode_linear_pool):
                                    ring %= Uint32(self.ring_blocks)
                                    required *= pool_block // Uint32(self.ring_blocks) + Uint32(1)
                                l1_ready = signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + ring
                                while ld_acquire_gpu_u32(l1_ready) < required:
                                    pass
                                epi_warp = Uint32(physical_warp - 4)
                                row = (epi_warp // Uint32(4)) * Uint32(self.epilogue_rows) + \
                                    (epi_warp % Uint32(4)) * Uint32(2) + Uint32(lane // 16)
                                for atom in cutlass.range_constexpr(self.epilogue_rows // 8):
                                    row_in_block = row + Uint32(atom * 8)
                                    packed = Uint32(0)
                                    if row_in_block < valid_m:
                                        if lane % 16 == 0:
                                            source_row = pool_block * Uint32(self.block_m) + row_in_block
                                            if cutlass.const_expr(self.packed_w2_metadata):
                                                packed = signals[self.metadata_words + source_row]
                                            else:
                                                source_rank, source_token, slot = ld_global_u32x3(
                                                    signals.iterator + self.metadata_words + source_row * Uint32(3))
                                                packed = source_rank | (source_token << Uint32(8)) | (slot << Uint32(24))
                                    early_w2_metadata[atom] = cute.arch.shuffle_sync(
                                        packed, Uint32(lane // 16) * Uint32(16))
                        if cutlass.const_expr(self.speculative_w1):
                            if valid_m == Uint32(0):
                                valid_m = first_routed_rows(
                                    recv_ptr, Uint32(task_words[stage * 8 + 1]),
                                    Int32(lane), self.block_m, self.dispatch_sms, self.num_ranks)
                        accum_stage = iteration % Uint32(self.accum_stages)
                        accum_phase = (iteration // Uint32(self.accum_stages)) % Uint32(2)
                        cute.arch.mbarrier_wait(accum_full + accum_stage,
                                                accum_phase)
                        tcgen05_after_thread_sync()
                        if cutlass.const_expr(self.decode_trace_offset != 0):
                            trace_tag = task_tag(task_words, stage, task_phase)
                            if physical_warp == 4 and lane == Uint32(0):
                                task_mark(signals, self.decode_trace_offset, Uint32(block),
                                          iteration, 9, trace_tag, self.trace_stride)
                        release_task_info(empty, Int32(stage), self.task_scheduler_cta,
                                          self.independent_compute_ctas)
                        # Match the reference's lane-0 exchange so valid-M
                        # controls the epilogue uniformly within each warp.
                        valid_m = cute.arch.shuffle_sync(valid_m, 0)
                        if cutlass.const_expr(not self.decode_linear_pool):
                            if physical_warp == 4 and task_phase == Uint32(GCBlockPhase.Linear2):
                                if elect_one_sync():
                                    # MMA has finished reading the input.
                                    # The scatter uses TMEM, so recycle the ring slot.
                                    ring_block = pool_block % Uint32(self.ring_blocks)
                                    red_gpu_add_u32(
                                        signals.iterator +
                                        SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block,
                                        Uint32(1))
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            # The reference joins every epilogue warp after
                            # the elected routed L2 ring release.
                            cute.arch.sync_warp()
                        is_l1 = task_phase == Uint32(GCBlockPhase.Linear1)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_l1 = is_l1 or task_phase == Uint32(GCBlockPhase.SharedLinear1)
                        if is_l1:
                            # W1 epilogue: the 16dp register mapping below
                            # predicates stores with this task's valid rows.
                            is_shared_l1 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l1 = Uint32(1) if task_phase == \
                                    Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                            ring_block = pool_block
                            if cutlass.const_expr(not self.decode_linear_pool):
                                ring_block = pool_block % Uint32(self.ring_blocks)
                                generation = pool_block // Uint32(self.ring_blocks)
                                if not is_shared_l1:
                                    l2_empty_ptr = signals.iterator + \
                                        SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block
                                    while ld_acquire_gpu_u32(l2_empty_ptr) != \
                                            generation * Uint32(self.l2_clusters * 2):
                                        pass
                            output_block = ring_block
                            if is_shared_l1:
                                output_block = pool_block
                            route_weights = cute.recast_ptr(
                                signals.iterator + self.l1_weight_words,
                                dtype=Float32)
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            wg_start_m = warpgroup * Uint32(self.epilogue_rows)
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > Uint32(self.epilogue_rows):
                                    valid_wg_rows = Uint32(self.epilogue_rows)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l1 - 1)) // \
                                Uint32(self.store_m_l1)
                            n_block = n_cluster * Uint32(self.compute_cluster_size) + cta_rank
                            cd_wg = cd_smem + warpgroup * Uint32(
                                2 * self.store_m_l1 * 64)
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l1 // 8, 8), Float32)
                            stored_cached_weight = Float32(1)
                            clamp_value = activation_clamp
                            if cutlass.const_expr(activation_clamp is None):
                                # Preserve the old direct diagnostic call
                                # while production supplies the FP32 ABI arg.
                                clamp_value = Float32(self.activation_clamp)
                            clamp_pair = None
                            if cutlass.const_expr(self.w1_early_epi_inputs):
                                # The weights are already visible after MMA waited for A.
                                # Issue this load before the first TMEM fragment load.
                                if not is_shared_l1 and Uint32(lane) < valid_wg_rows:
                                    weight_token = ring_block * Uint32(self.block_m) + wg_start_m + Uint32(lane)
                                    stored_cached_weight = Float32(route_weights[weight_token])
                                if cutlass.const_expr(self.has_activation_clamp):
                                    bound = Uint32(f32_to_bf16_bits(clamp_value))
                                    clamp_pair = bound | (bound << Uint32(16))
                            if valid_stores == Uint32(0):
                                _release_w1_tmem(accum_empty, accum_stage, self.local_mma)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        base = tmem_base + accum_stage * Uint32(self.block_m) + \
                                            wg_start_m + Uint32(atom * 8)
                                        _load_w1_atom(raw_values, 0, atom, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w1_tmem(accum_empty, accum_stage, self.local_mma)
                            for store in cutlass.range_constexpr(
                                    self.epilogue_rows // self.store_m_l1):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l1 // 8):
                                            base = tmem_base + accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l1 + atom * 8)
                                            _load_w1_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l1 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage, self.local_mma)
                                    # BM240 overlaps the next register
                                    # block's loads with this block's math.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            for atom in cutlass.range_constexpr(
                                                    self.store_m_l1 // 8):
                                                next_base = tmem_base + accum_stage * Uint32(self.block_m) + \
                                                    wg_start_m + Uint32(
                                                        (store + 1) * self.store_m_l1 + atom * 8)
                                                _load_w1_atom(
                                                    raw_values, (store + 1) % 2,
                                                    atom, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage, self.local_mma)
                                    raw_buffer = store % 2 if self.block_m == 240 else 0
                                    activation_values = cute.make_rmem_tensor(
                                        (self.store_m_l1 // 8, 4), Float32)
                                    if cutlass.const_expr(self.w1_packed_scale_reduction):
                                        reduced_scale = cute.make_rmem_tensor((self.store_m_l1 // 8,), Uint32)
                                        epi_scales = cute.recast_ptr(epi_amax, dtype=Uint32)
                                    else:
                                        reduced_amax = cute.make_rmem_tensor(
                                            (self.store_m_l1 // 8, 2), Float32)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        j = store * (self.store_m_l1 // 8) + atom
                                        if cutlass.const_expr(not self.w1_early_epi_inputs):
                                            if not is_shared_l1 and j % 4 == 0:
                                                if self.epilogue_rows % 32 == 0 or \
                                                        Uint32(j * 8 + lane) < \
                                                        Uint32(self.epilogue_rows):
                                                    weight_token = ring_block * \
                                                        Uint32(self.block_m) + wg_start_m + \
                                                        Uint32(j * 8 + lane)
                                                    stored_cached_weight = route_weights[weight_token]
                                        source_lane = Uint32((j * 8) % 32) + \
                                            (Uint32(lane) % Uint32(4)) * Uint32(2)
                                        weight0 = cute.arch.shuffle_sync(
                                            stored_cached_weight, source_lane)
                                        weight1 = cute.arch.shuffle_sync(
                                            stored_cached_weight, source_lane + Uint32(1))
                                        value0, value1 = _swiglu_pair(
                                            raw_values[raw_buffer, atom, 0],
                                            raw_values[raw_buffer, atom, 1],
                                            raw_values[raw_buffer, atom, 2],
                                            raw_values[raw_buffer, atom, 3],
                                            weight0, weight1, clamp_value,
                                            Float32(1), self.has_activation_clamp,
                                            False, self.fast_math, clamp_pair)
                                        value2, value3 = _swiglu_pair(
                                            raw_values[raw_buffer, atom, 4],
                                            raw_values[raw_buffer, atom, 5],
                                            raw_values[raw_buffer, atom, 6],
                                            raw_values[raw_buffer, atom, 7],
                                            weight0, weight1, clamp_value,
                                            Float32(1), self.has_activation_clamp,
                                            False, self.fast_math, clamp_pair)
                                        activation_values[atom, 0] = value0
                                        activation_values[atom, 1] = value1
                                        activation_values[atom, 2] = value2
                                        activation_values[atom, 3] = value3
                                        amax0 = fmax_f32(fmax_f32(Float32(0), abs_f32(value0)), abs_f32(value2))
                                        amax1 = fmax_f32(fmax_f32(Float32(0), abs_f32(value1)), abs_f32(value3))
                                        if cutlass.const_expr(self.w1_packed_scale_reduction):
                                            # UE8M0 rounding is monotone. Put the two
                                            # exponents in normal positive BF16 encodings
                                            # so one packed max also reduces both tokens.
                                            if cutlass.const_expr(self.w1_pair_scale_rounding):
                                                exponent0, exponent1 = ue8m0_fp8_scale_exp_pair(amax0, amax1)
                                            else:
                                                exponent0 = ue8m0_fp8_scale_exp(amax0)
                                                exponent1 = ue8m0_fp8_scale_exp(amax1)
                                            pair = Uint32(0x3f003f00) | exponent0 | (exponent1 << Uint32(16))
                                            for index in cutlass.range_constexpr(3):
                                                pair = bf16x2_max(pair, cute.arch.shuffle_sync_bfly(pair, 4 << index))
                                            reduced_scale[atom] = pair
                                        else:
                                            if cutlass.const_expr(self.w1_hardware_scale_reduction):
                                                amax0 = warp_group4_hardware_max(amax0, Uint32(lane))
                                                amax1 = warp_group4_hardware_max(amax1, Uint32(lane))
                                            else:
                                                amax0 = warp_group4_max(amax0)
                                                amax1 = warp_group4_max(amax1)
                                            reduced_amax[atom, 0] = amax0
                                            reduced_amax[atom, 1] = amax1
                                        if lane < 4:
                                            amax_offset = epi_warp * \
                                                Uint32(self.store_m_l1) + \
                                                Uint32(atom * 8 + lane * 2)
                                            if cutlass.const_expr(self.w1_packed_scale_reduction):
                                                # The two-token packet frees the adjacent
                                                # word for the next store's scale reduction.
                                                epi_scales[amax_offset + Uint32(store % 2)] = reduced_scale[atom]
                                            else:
                                                epi_amax[amax_offset] = amax0
                                                epi_amax[amax_offset + 1] = amax1
                                        cute.arch.sync_warp()
                                    if cutlass.const_expr(not self.direct_w1_store):
                                        tma_store_wait_1()
                                    if cutlass.const_expr(self.direct_w1_store and self.w1_packed_scale_reduction):
                                        sync_aligned(64, Uint32(3) + epi_warp // Uint32(2))
                                    else:
                                        sync_epilogue_warpgroup(warpgroup)
                                    tma_stage = store % 2
                                    cd_stage = cd_wg + Uint32(
                                        tma_stage * self.store_m_l1 * 64)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        partner_offset = (epi_warp ^ Uint32(1)) * \
                                            Uint32(self.store_m_l1) + \
                                            Uint32(atom * 8) + \
                                            (Uint32(lane) % Uint32(4)) * Uint32(2)
                                        if cutlass.const_expr(self.w1_packed_scale_reduction):
                                            pair = bf16x2_max(
                                                reduced_scale[atom],
                                                epi_scales[partner_offset + Uint32(store % 2)])
                                            sf_exp0 = pair & Uint32(0xff)
                                            sf_exp1 = (pair >> Uint32(16)) & Uint32(0xff)
                                        else:
                                            sf_amax0 = fmax_f32(
                                                reduced_amax[atom, 0],
                                                epi_amax[partner_offset])
                                            sf_amax1 = fmax_f32(
                                                reduced_amax[atom, 1],
                                                epi_amax[partner_offset + 1])
                                            if cutlass.const_expr(self.w1_pair_scale_rounding):
                                                sf_exp0, sf_exp1 = ue8m0_fp8_scale_exp_pair(sf_amax0, sf_amax1)
                                            else:
                                                sf_exp0 = ue8m0_fp8_scale_exp(sf_amax0)
                                                sf_exp1 = ue8m0_fp8_scale_exp(sf_amax1)
                                        inv0 = ue8m0_fp8_inv(sf_exp0)
                                        inv1 = ue8m0_fp8_inv(sf_exp1)
                                        upper0, upper1 = cute.arch.mul_packed_f32x2(
                                            (activation_values[atom, 0],
                                             activation_values[atom, 1]), (inv0, inv1), rnd='rn', ftz=False)
                                        lower0, lower1 = cute.arch.mul_packed_f32x2(
                                            (activation_values[atom, 2],
                                             activation_values[atom, 3]), (inv0, inv1), rnd='rn', ftz=False)
                                        packed = _w1_pack_fp8x4(
                                            upper0, upper1, lower0, lower1)
                                        row = Uint32(lane)
                                        smem_offset = Uint32(atom * 8 * 64) + \
                                            row * Uint32(64) + \
                                            (warp_in_wg ^ (row // Uint32(2))) * Uint32(16)
                                        output_base = cute.recast_ptr(signals.iterator, dtype=Uint8)
                                        row_in_block = wg_start_m + Uint32(store * self.store_m_l1 + atom * 8)
                                        if cutlass.const_expr(self.pair_shared_w1_offset):
                                            if is_shared_l1 and row_in_block >= num_tokens:
                                                peer = rank_idx ^ Uint32(1)
                                                output_base = cute.make_ptr(
                                                    Uint8, mapped_peer_base(peer_ptrs, peer, self.num_ranks),
                                                    cute.AddressSpace.gmem, assumed_align=4)
                                                row_in_block -= num_tokens
                                        if cutlass.const_expr(self.direct_w1_store):
                                            output_words = Uint64(self.l2_x_words)
                                            stride = Uint32(self.intermediate)
                                            if is_shared_l1:
                                                output_words = Uint64(self.shared_l2_x_words)
                                                stride = Uint32(self.shared_experts * self.intermediate)
                                            output = output_base + output_words * Uint64(4)
                                            row_base = output_block * Uint32(self.block_m) + row_in_block
                                            col_base = n_block * Uint32(64) + warp_in_wg * Uint32(16)
                                            plane = Uint32(0)
                                            if cutlass.const_expr(self.w2_tile_major_input):
                                                # W1's adjacent N64 tiles form one contiguous W2 K128 plane.
                                                plane = output_block * (stride // Uint32(128)) + n_block // Uint32(2)
                                                row_base = plane * Uint32(self.block_m) + row_in_block
                                                col_base = n_block % Uint32(2) * Uint32(64) + warp_in_wg * Uint32(16)
                                                stride = Uint32(128)
                                            use_transposed_plane = self.transpose_w2_activations == 1
                                            if cutlass.const_expr(self.transpose_w2_activations == 2):
                                                use_transposed_plane = valid_m > Uint32(16)
                                            if use_transposed_plane:
                                                _w1_store_transposed(output, plane, row_in_block,
                                                                     col_base, Uint32(lane), packed)
                                            else:
                                                if cutlass.const_expr(self.pair_shared_w1_vector):
                                                    if is_shared_l1:
                                                        _w1_store_direct(output, stride, row_base, col_base,
                                                                         Uint32(lane), packed, vector_store=True)
                                                    else:
                                                        _w1_store_direct(output, stride, row_base, col_base,
                                                                         Uint32(lane), packed, self.w1_register_transpose)
                                                else:
                                                    _w1_store_direct(output, stride, row_base, col_base,
                                                                     Uint32(lane), packed, self.w1_register_transpose)
                                        else:
                                            stmatrix_fp8x4_trans_late(cd_stage, smem_offset, packed)
                                        if warp_in_wg % Uint32(2) == Uint32(0) and lane < 4:
                                            output_sf_words = Uint64(self.l2_sf_words)
                                            sf_stride = Uint32(self.sf_ring_tokens)
                                            if is_shared_l1:
                                                output_sf_words = Uint64(
                                                    self.shared_l2_sf_words)
                                                sf_stride = Uint32(
                                                    self.shared_sf_tokens)
                                            l2_sf_bytes = output_base + output_sf_words * Uint64(4)
                                            k_sf = n_block * Uint32(2) + \
                                                warp_in_wg // Uint32(2)
                                            token_base = row_in_block
                                            assume_u32_lt(
                                                token_base, Uint32(self.block_m))
                                            sf_token = output_block * Uint32(self.sf_block_m) + \
                                                (token_base & Uint32(~127)) + \
                                                (token_base & Uint32(31)) * Uint32(4) + \
                                                ((token_base >> Uint32(5)) & Uint32(3)) + \
                                                Uint32(lane * 8)
                                            if cutlass.const_expr(
                                                    self.wide_routed_w1_sf or
                                                    self.wide_shared_w1_sf):
                                                sf_addr = Uint64(k_sf // Uint32(4)) * \
                                                    (Uint64(sf_stride) * Uint64(4)) + \
                                                    Uint64(sf_token) * Uint64(4) + \
                                                    Uint64(k_sf % Uint32(4))
                                                l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                l2_sf_bytes[sf_addr + Uint64(16)] = \
                                                    Uint8(sf_exp1)
                                            else:
                                                sf_addr = (k_sf // Uint32(4)) * \
                                                    (sf_stride * Uint32(4)) + \
                                                    sf_token * Uint32(4) + k_sf % Uint32(4)
                                                l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                l2_sf_bytes[sf_addr + Uint32(16)] = \
                                                    Uint8(sf_exp1)
                                        cute.arch.sync_warp()
                                    if cutlass.const_expr(not self.direct_w1_store):
                                        sync_epilogue_warpgroup(warpgroup)
                                        if warp_in_wg == Uint32(0):
                                            with cute.arch.elect_one():
                                                output_map = tensor_map_4.iterator
                                                if is_shared_l1:
                                                    output_map = tensor_map_13.iterator
                                                tma_store_fence()
                                                tile_s2g_2d(
                                                    output_map, cd_stage,
                                                    n_block * Uint32(64),
                                                    output_block * Uint32(self.block_m) +
                                                    wg_start_m + Uint32(store * self.store_m_l1))
                                                tma_store_arrive()
                                        cute.arch.sync_warp()
                                    elif cutlass.const_expr(self.epilogue_rows > self.store_m_l1 and
                                                            not self.w1_packed_scale_reduction):
                                        if Uint32(store + 1) < valid_stores:
                                            # All readers must retire before the next
                                            # direct store reuses the single FP32 plane.
                                            sync_epilogue_warpgroup(warpgroup)
                            if cutlass.const_expr(self.direct_w1_store):
                                # Import both WGs' ordinary stores before
                                # the elected warp publishes W2's K bit.
                                sync_aligned(self.epilogue_threads, 2)
                            else:
                                tma_store_wait_0()
                        else:
                            # Source-contract W2 vector epilogue. The
                            # 16dp register-to-value bijection still
                            # requires validM>8 device verification.
                            is_shared_l2 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l2 = Uint32(
                                    task_phase == Uint32(GCBlockPhase.SharedLinear2))
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            wg_start_m = warpgroup * Uint32(self.epilogue_rows)
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > Uint32(self.epilogue_rows):
                                    valid_wg_rows = Uint32(self.epilogue_rows)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l2 - 1)) // \
                                Uint32(self.store_m_l2)
                            n_block = n_cluster * Uint32(self.compute_cluster_size) + cta_rank
                            cd_wg = cd_smem + warpgroup * Uint32(
                                self.store_m_l2 * 128 * 2)
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l2 // 8, 8), Float32)
                            if valid_stores == Uint32(0):
                                _release_w2_tmem(accum_empty, accum_stage, self.local_mma)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    base = tmem_base + accum_stage * Uint32(self.block_m) + \
                                        wg_start_m
                                    _load_w2_atom(raw_values, 0, 0, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w2_tmem(accum_empty, accum_stage, self.local_mma)
                            for store in cutlass.range_constexpr(
                                    self.epilogue_rows // self.store_m_l2):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            base = tmem_base + accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l2 + atom * 8)
                                            _load_w2_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l2 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage, self.local_mma)
                                    # Wait for the current atom before
                                    # issuing the BM240 next-store prefetch.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            next_base = tmem_base + accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32((store + 1) * 8)
                                            _load_w2_atom(
                                                raw_values, (store + 1) % 2,
                                                0, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage, self.local_mma)
                                    if cutlass.const_expr(self.direct_w2_store):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            row = (wg_start_m +
                                                   Uint32(store * self.store_m_l2 + atom * 8) +
                                                   lane // Uint32(4))
                                            _w2_store_direct(
                                                raw_values, atom, signals, peer_ptrs,
                                                self.metadata_words, self.combine_words, row,
                                                pool_block * Uint32(self.block_m) + row,
                                                valid_m, n_block, warp_in_wg, lane,
                                                rank_idx, is_shared_l2, self.hidden,
                                                self.capacity, self.topk, self.num_ranks,
                                                self.shared_experts > 0, self.token_major_combine,
                                                self.combine_interleaved_bytes, self.sentinel_combine,
                                                self.packed_w2_metadata)
                                    else:
                                        # One metadata record for each lane's
                                        # row in every 8-row atom.
                                        src_metadata = ()
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            token_in_block = wg_start_m + \
                                                Uint32(store * self.store_m_l2 + atom * 8) + \
                                                warp_in_wg * Uint32(2) + Uint32(lane // 16)
                                            pool_token = pool_block * Uint32(self.block_m) + \
                                                token_in_block
                                            if cutlass.const_expr(self.prefetch_w2_metadata):
                                                source_rank, source_token = rank_idx, pool_token
                                                slot = Uint32(self.topk)
                                                if not is_shared_l2:
                                                    row_atom = store * (self.store_m_l2 // 8) + atom
                                                    packed = early_w2_metadata[row_atom]
                                                    source_rank, source_token, slot = _unpack_w2_metadata(packed)
                                                record = (source_rank, source_token, slot)
                                            else:
                                                record = _load_w2_source_metadata(
                                                    signals.iterator + self.metadata_words,
                                                    token_in_block, valid_m, pool_token,
                                                    rank_idx, is_shared_l2, self.topk,
                                                    self.shared_experts > 0, self.packed_w2_metadata)
                                            src_metadata += (record,)
                                        cute.arch.sync_warp()
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            if atom == 0 and store > 0:
                                                sync_epilogue_warpgroup(warpgroup)
                                            row = Uint32(lane % 8)
                                            col = (epi_warp % Uint32(2)) * Uint32(4) + \
                                                Uint32(lane // 8)
                                            smem_store_offset = (warp_in_wg // Uint32(2)) * \
                                                Uint32(self.store_m_l2 * 128) + \
                                                Uint32(atom * 8 * 128) + row * Uint32(128) + \
                                                (col ^ row) * Uint32(16)
                                            buffer = store % 2 if self.block_m == 240 else 0
                                            stmatrix_bf16x8_trans(
                                                cd_wg + smem_store_offset,
                                                pack_bf16_pair(raw_values[buffer, atom, 0],
                                                                raw_values[buffer, atom, 1]),
                                                pack_bf16_pair(raw_values[buffer, atom, 2],
                                                                raw_values[buffer, atom, 3]),
                                                pack_bf16_pair(raw_values[buffer, atom, 4],
                                                                raw_values[buffer, atom, 5]),
                                                pack_bf16_pair(raw_values[buffer, atom, 6],
                                                                raw_values[buffer, atom, 7]))
                                        sync_epilogue_warpgroup(warpgroup)
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            row_in_atom = warp_in_wg * Uint32(2) + \
                                                Uint32(lane // 16)
                                            row_in_store = Uint32(atom * 8) + row_in_atom
                                            token_in_block = wg_start_m + \
                                                Uint32(store * self.store_m_l2) + row_in_store
                                            if token_in_block < valid_m:
                                                bank_group = Uint32(lane % 8)
                                                half = (Uint32(lane) % Uint32(16)) // Uint32(8)
                                                smem_load_offset = half * \
                                                    Uint32(self.store_m_l2 * 128) + \
                                                    row_in_store * Uint32(128) + \
                                                    (bank_group ^ row_in_atom) * Uint32(16)
                                                words = ld_shared_u32x4(
                                                    cd_wg + smem_load_offset)
                                                channel_bytes = n_block * Uint32(256) + \
                                                    (Uint32(lane) % Uint32(16)) * Uint32(16)
                                                dst_bytes = _w2_mailbox_offset(
                                                    src_metadata[atom][1], src_metadata[atom][2], channel_bytes,
                                                    self.hidden, self.capacity, self.topk + (self.shared_experts > 0),
                                                    self.token_major_combine, self.combine_interleaved_bytes)
                                                dst = cute.make_ptr(
                                                    Uint32, mapped_peer_base(
                                                        peer_ptrs, src_metadata[atom][0],
                                                        self.num_ranks) +
                                                    Uint64(self.combine_words * 4) + \
                                                    dst_bytes,
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=16)
                                                epilogue_combines = False
                                                if cutlass.const_expr(self.combine_in_compute == 3):
                                                    epilogue_combines = is_shared_l2 != Uint32(0)
                                                if epilogue_combines:
                                                    self._finish_shared_w2_vector(
                                                        signals, peer_ptrs, src_metadata[atom][1],
                                                        channel_bytes, words)
                                                else:
                                                    _store_w2_vector(dst, words, self.sentinel_combine)
                                                if cutlass.const_expr(self.combine_ready_flags_offset):
                                                    fence_acq_rel_sys()
                                            if cutlass.const_expr(self.combine_ready_flags_offset):
                                                # Each half-warp posts one flag after its 256 bytes.
                                                cute.arch.sync_warp()
                                                if token_in_block < valid_m and lane % 16 == 0:
                                                    ready_slot = src_metadata[atom][2]
                                                    if cutlass.const_expr(self.combine_ready_mask):
                                                        flag = combine_flag_index(
                                                            src_metadata[atom][1], Uint32(0),
                                                            n_block, self.hidden, 1)
                                                    else:
                                                        flag = combine_flag_index(
                                                            src_metadata[atom][1], ready_slot,
                                                            n_block, self.hidden,
                                                            self.topk + (self.shared_experts > 0))
                                                    flag_dst = cute.make_ptr(
                                                        Uint32, mapped_peer_base(
                                                            peer_ptrs, src_metadata[atom][0], self.num_ranks) +
                                                        Uint64(self.combine_ready_flags_offset) + flag * Uint64(4),
                                                        cute.AddressSpace.gmem, assumed_align=4)
                                                    if cutlass.const_expr(self.combine_ready_mask):
                                                        # RMWs import earlier slots before adding
                                                        # their own ready bit for the consumer.
                                                        atom_acq_rel_sys_or_u32(flag_dst, Uint32(1) << ready_slot)
                                                    else:
                                                        st_release_u32(flag_dst, Uint32(1), system=True)
                        # Publish all epilogue stores to the stripe counter's
                        # single producer lane before it releases the CTA.
                        sync_aligned(self.epilogue_threads, 2)
                        if cutlass.const_expr(self.expert_combine_offset):
                            if not is_l1 and physical_warp == Uint32(4) and lane == Uint32(0):
                                self._signal_expert_ready(
                                    signals, peer_ptrs, recv_ptr, ready_expert,
                                    Uint32(task_phase == Uint32(GCBlockPhase.SharedLinear2)),
                                    rank_idx, num_tokens,
                                    Uint64(parity + Uint32(1)))
                        if cutlass.const_expr(self.stripe_combine_offset != 0):
                            if not is_l1 and physical_warp == Uint32(4) and lane == Uint32(0):
                                self._signal_stripe(
                                    signals, peer_ptrs, rank_idx,
                                    n_cluster // Uint32(self.stripe_tiles), Uint64(1))
                        if cutlass.const_expr(self.stripe_launch_delay == 2):
                            if not is_l1 and physical_warp == Uint32(4) and lane == Uint32(0):
                                if launch_pending:
                                    launch_dependents()
                                    launch_pending = False
                        if cutlass.const_expr(self.combine_in_compute == 2):
                            if not is_l1 and physical_warp == Uint32(4) and lane == Uint32(0):
                                if launch_pending:
                                    first_w2 = signals.iterator + SIGNAL_OFFSETS['compute_combine_epoch'] // 4
                                    st_release_gpu_u32(first_w2, parity + Uint32(1))
                                    launch_pending = False
                    is_w1_release = task_phase == Uint32(GCBlockPhase.Linear1)
                    if cutlass.const_expr(self.shared_experts > 0):
                        is_w1_release = is_w1_release or \
                            task_phase == Uint32(GCBlockPhase.SharedLinear1)
                    if is_w1_release:
                        if physical_warp == 4:
                            if elect_one_sync():
                                ring_block = (pool_block if cutlass.const_expr(self.decode_linear_pool) else
                                              pool_block % self.ring_blocks)
                                n_block = n_cluster * self.compute_cluster_size + cta_rank
                                if task_phase == Uint32(GCBlockPhase.Linear1):
                                    red_release_gpu_xor_u64(
                                        signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                                        ring_block * 2,
                                        Uint64(1) << n_block)
                                    if cutlass.const_expr(not self.decode_linear_pool):
                                        red_gpu_add_u32(
                                            signals.iterator + SIGNAL_OFFSETS['l1_empty_count'] // 4 +
                                            ring_block, Uint32(1))
                                else:
                                    if cutlass.const_expr(self.shared_experts > 0):
                                        if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                            flag_offset = Uint32(SIGNAL_OFFSETS['shared_l2_full_count']) + pool_block * 4
                                            local = signals.iterator + flag_offset // 4
                                            if cutlass.const_expr(self.pair_shared_w1_offset):
                                                peer = rank_idx ^ Uint32(1)
                                                peer_base = mapped_peer_base(peer_ptrs, peer, self.num_ranks)
                                                remote = cute.make_ptr(Uint32, peer_base + Uint64(flag_offset),
                                                                       cute.AddressSpace.gmem, assumed_align=4)
                                                fence_acq_rel_sys()
                                                atom_acq_rel_sys_add_u32(local, Uint32(1))
                                                atom_acq_rel_sys_add_u32(remote, Uint32(1))
                                            else:
                                                red_release_gpu_add_u32(local, Uint32(1))
                        cute.arch.sync_warp()
                    if cutlass.const_expr(self.decode_trace_offset != 0):
                        if (task_phase != Uint32(GCBlockPhase.None_) and
                                physical_warp == 4 and lane == Uint32(0)):
                            task_mark(signals, self.decode_trace_offset, Uint32(block),
                                      iteration, 10, trace_tag, self.trace_stride)
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    iteration = Uint32(iteration + Uint32(1))
                # The same full epilogue warp in both CTAs retires TMEM.
                if physical_warp == 4:
                    tmem_free(tptr[0], Uint32(self.tmem_cols), self.mma_cta_group)
                    if cutlass.const_expr(self.decode_trace_offset != 0):
                        if lane == Uint32(0):
                            cta_mark(signals, self.decode_trace_offset, 2,
                                     Uint32(block), Uint32(1), Uint64(sm_id()))
        elif cutlass.const_expr(self.combine_in_compute):
            # Wait for the compute-side consumers as well as W2 producers
            # before recycling any signals into the next invocation.
            cute.arch.griddepcontrol_wait()
            ready_epoch = Uint64(ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['role3_epoch'] // 4))
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4
            self._role3_dispatch(signals, peer_ptrs, rank_idx, num_tokens,
                                 Uint32(tid), Uint32(block), recv_ptr, stats, ready_epoch)
        else:
            # Both graph workers precede role 3. Fused finalization uses
            # warp-owned barriers; legacy paths retain the whole-CTA join.
            combine_smem = cute.recast_ptr(signals.iterator, dtype=Uint8)
            combine_bars = cute.recast_ptr(signals.iterator, dtype=Uint64)
            if cutlass.const_expr(self.finalize_output and not self.decode_vector_combine):
                # One fetch plane per combine slot; the output plane aliases
                # slot 0. Both live inside the role3 reusable prefix.
                combine_smem = shared_slab
                if cutlass.const_expr(shared_slab is None):
                    combine_alloc = cutlass.utils.SmemAllocator()
                    combine_smem = cute.recast_ptr(
                        combine_alloc.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
                combine_bars = cute.recast_ptr(
                    combine_smem + self.shared_layout.combine_bars, dtype=Uint64)
                if cutlass.const_expr(self.expert_combine_offset):
                    if physical_warp < 4:
                        warpgroup_reg_dealloc(48 if self.local_experts <= 64 else 96)
                elif cutlass.const_expr(self.shared_experts > 0):
                    cute.arch.griddepcontrol_wait()
                else:
                    self._initialize_shared_storage(
                        combine_smem, Uint32(physical_warp), Uint32(lane))
            early_sentinel = self.sentinel_combine and self.programmatic_finalize and self.finalize_output
            # Early sentinel reduction acquires each chunk independently.
            # The fused Dispatch grid can still own the epoch until the PDL wait.
            ready_epoch = Uint64(0)
            if cutlass.const_expr(not early_sentinel):
                ready_epoch = Uint64(ld_volatile_u32(
                    signals.iterator + SIGNAL_OFFSETS['role3_epoch'] // 4))
            if cutlass.const_expr(self.stripe_combine_offset != 0):
                self._role3_combine_stripes(
                    signals, y_out.iterator, rank_idx, num_tokens, Uint32(block),
                    Uint32(tid), Uint32(physical_warp), Uint32(lane), ready_epoch)
            if cutlass.const_expr(self.expert_combine_offset or
                                  (self.sentinel_combine and self.programmatic_finalize)):
                if cutlass.const_expr(self.reuse_finalize_warps):
                    # These warps later recycle signals after the local W2 fence.
                    self._role3_combine(signals, rank_idx, num_tokens, y_out,
                                        block, physical_warp + 4, lane, combine_smem, combine_bars, ready_epoch)
                elif physical_warp >= 4:
                    self._role3_combine(signals, rank_idx, num_tokens, y_out,
                                        block, physical_warp, lane, combine_smem, combine_bars, ready_epoch)
            if cutlass.const_expr(self.programmatic_finalize):
                # Wait for all local W2 stores before recycling workspace
                # or announcing readiness to peers. Independent sentinel
                # consumers can retire while the one cleanup CTA waits.
                if not self.single_cleanup_cta or block == Uint32(0):
                    cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(early_sentinel):
                ready_epoch = Uint64(ld_volatile_u32(
                    signals.iterator + SIGNAL_OFFSETS['role3_epoch'] // 4))
            if cutlass.const_expr(not (self.finalize_output and self.shared_experts > 0)):
                sync_unaligned(FINALIZE_THREADS, 1)
            recv_off = SIGNAL_OFFSETS['expert_recv_count_sum'] // 4
            recv_ptr = signals.iterator + recv_off
            if physical_warp < 4 and (not self.single_cleanup_cta or block == Uint32(0)):
                self._role3_dispatch(signals, peer_ptrs, rank_idx, num_tokens,
                                     Uint32(tid), Uint32(block),
                                     recv_ptr, stats, ready_epoch)
            elif cutlass.const_expr(self.stripe_combine_offset == 0 and not self.expert_combine_offset and
                                    not (self.sentinel_combine and self.programmatic_finalize)):
                self._role3_combine(signals, rank_idx, num_tokens, y_out,
                                    block, physical_warp, lane,
                                    combine_smem, combine_bars, ready_epoch)

    @cute.jit
    def __call__(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                 rank_idx: Uint32, num_tokens: Uint32,
                 tensor_maps=None, y_out=None, stats=None,
                 activation_clamp=None, activation_alpha=None,
                 tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
                 tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
                 tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
                 tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
                 tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
                 tensor_map_16=None, tensor_map_17=None,
                 tensor_map_18=None, tensor_map_19=None,
                 tensor_map_20=None, tensor_map_21=None):
        if cutlass.const_expr(self.role == 1):
            if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens).launch(
                    grid=(self.dispatch_sms, 1, 1),
                    block=(self.dispatch_warps * 32, 1, 1),
                    max_number_threads=(self.dispatch_warps * 32, 1, 1),
                    min_blocks_per_mp=1)
            else:
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                            None, None, None,
                            activation_clamp, activation_alpha).launch(
                    grid=(self.dispatch_sms, 1, 1),
                    block=(self.dispatch_warps * 32, 1, 1),
                    max_number_threads=(self.dispatch_warps * 32, 1, 1),
                    min_blocks_per_mp=1)
        elif cutlass.const_expr(self.role == 2):
            grid_count = self.compute_sms
            if cutlass.const_expr(self.fused_dispatch_role is not None and not self.fused_worker_reuse):
                grid_count += self.dispatch_sms
            if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                            tensor_maps, None, None, None, None,
                            tensor_map_1, tensor_map_2, tensor_map_3,
                            tensor_map_4, tensor_map_5, tensor_map_6,
                            tensor_map_7, tensor_map_8, tensor_map_9,
                            tensor_map_10, tensor_map_11, tensor_map_12,
                            tensor_map_13, tensor_map_14, tensor_map_15,
                            tensor_map_16, tensor_map_17,
                            tensor_map_18, tensor_map_19,
                            tensor_map_20, tensor_map_21).launch(
                    grid=(grid_count, 1, 1), block=(self.compute_threads, 1, 1),
                    max_number_threads=(self.compute_threads, 1, 1),
                    cluster=(self.compute_cluster_size, 1, 1),
                    min_blocks_per_mp=2 if self.compact_compute else 1)
            else:
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                            None, None,
                            activation_clamp, activation_alpha,
                            tensor_map_1, tensor_map_2, tensor_map_3,
                            tensor_map_4, tensor_map_5, tensor_map_6,
                            tensor_map_7, tensor_map_8, tensor_map_9,
                            tensor_map_10, tensor_map_11, tensor_map_12,
                            tensor_map_13, tensor_map_14, tensor_map_15,
                            tensor_map_16, tensor_map_17,
                            tensor_map_18, tensor_map_19,
                            tensor_map_20, tensor_map_21).launch(
                    grid=(grid_count, 1, 1), block=(self.compute_threads, 1, 1),
                    max_number_threads=(self.compute_threads, 1, 1),
                    cluster=(self.compute_cluster_size, 1, 1),
                    min_blocks_per_mp=2 if self.compact_compute else 1)
        else:
            if cutlass.const_expr(self.finalize_output):
                if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                                y_out, stats).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(self.finalize_threads, 1, 1),
                        max_number_threads=(self.finalize_threads, 1, 1), min_blocks_per_mp=1)
                else:
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                                y_out, stats,
                                activation_clamp, activation_alpha).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(self.finalize_threads, 1, 1),
                        max_number_threads=(self.finalize_threads, 1, 1), min_blocks_per_mp=1)
            else:
                if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
                else:
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                                None, None, None,
                                activation_clamp, activation_alpha).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
