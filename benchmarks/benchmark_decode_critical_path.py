"""Profile SM100 MoE decode with optional distinct weight sets between replays."""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

import gc_moe
from gc_moe.common.decode_trace import (
    TRACE_BYTES, TRACE_CTAS, TRACE_CTA_FIELDS, TRACE_TASK_BASE, TRACE_TASK_FIELDS,
    TRACE_TASKS,
)
from gc_moe.gc import GCSymmBuffer
from gc_moe.jit_kernels.impls.runtime_utils import colocate_routed_scales
from benchmarks.benchmark_gc_moe_se import (
    ROUTED_RECIPE, allocate_buffer, build_weights, prepare_shared_sf, stage_inputs,
)
from benchmarks.benchmark_gc_moe import build_tokens, per_token_cast_to_fp8
from benchmarks.weight_vmm import WeightVmm


TASK_PHASES = ('none', 'routed_w1', 'routed_w2', 'shared_w1', 'shared_w2')
READINESS = ('both_ready', 'activation_late', 'weight_late', 'both_late')
K_MAJOR_MODES = {'routed_w1_k_major': (0,), 'routed_w2_k_major': (1,),
                 'routed_both_k_major': (0, 1)}
W1_SCHEDULES = {'w1_warmup_1': (1, 1), 'w1_warmup_3': (3, 1), 'w1_warmup_4': (4, 1),
                'w1_cadence_2': (0, 2), 'w1_cadence_3': (0, 3)}
SHARED_W2_SCHEDULES = {'shared_w2_after_first_routed': 1, 'shared_w2_rotated': 2}
DECODE_COMBINATIONS = {
    'route_only': 'route',
    'reduce_only': 'reduce',
    'vector_dispatch': 'vector',
    'direct_w1_stock': 'off',
    'inline_direct_w1': 'off',
    'inline_direct_narrow': 'off',
    'inline_multicast_combine': 'off',
    'inline_direct_narrow_multicast': 'off',
    'inline_reduce_multicast': 'reduce',
}
PRIMARY_CONTEXT_VARIANTS = {
    'primary_stock': 'off', 'primary_route_only': 'route',
    'primary_inline_scores': 'off', 'primary_direct_narrow': 'off',
}
ROUTE_STACK_VARIANTS = (
    'route_only', 'inline_route_only', 'inline_route_prefix',
    'inline_route_prefix_multicast', 'inline_route_prefix_direct',
    'inline_route_prefix_weight_order',
)
WEIGHT_CACHE_MODES = {'weight_cache_alternate': 'alternate', 'weight_cache_w2': 'w2',
                      'weight_cache_alternate_w2': 'alternate-w2'}
MULTICAST_STRIPE_VARIANTS = {
    'multicast_stripes': 0, 'multicast_stripes_active': 0,
    'multicast_stripes_w2_issue': 1, 'multicast_stripes_w2_epilogue': 2,
}
VMM_WEIGHT_MODES = {
    'vmm_plain': (False, False), 'vmm_compressed_payload': (True, False),
    'vmm_compressed_scales': (False, True), 'vmm_compressed_all': (True, True),
}
FUSED_FINALIZE_VARIANTS = {
    'fused_compute_finalize': 'route-reduce',
    'fused_compute_finalize_tma': 'route',
    'fused_compute_finalize_flags': 'route-reduce',
    'fused_compute_finalize_flags_tma': 'route',
}
BACKFILL_OPTIONS = (
    'inline_scores', 'multicast_ready', 'packed_input', 'direct_w2',
    'w1_first', 'posted_counts', 'single_epilogue', 'combine_w2', 'combine_w4', 'combine_w8',
    'weight_prefix_k2', 'weight_prefix_k4', 'weight_prefix_k8',
    'publish_ctas2', 'publish_ctas4', 'publish_ctas8',
    'combine_small', 'combine_contiguous', 'combine_contiguous_wide',
    'combine_streaming', 'combine_streaming_small',
    'combine_direct_output', 'combine_contiguous_direct',
    'front_k4', 'front_k8', 'front_k8_inline', 'front_k8_contiguous_direct',
    'vector_combine', 'sentinel_combine', 'sentinel_early_combine',
    'sentinel_early_wide', 'sentinel_early_s32', 'sentinel_early_s48',
    'sentinel_early_token_order', 'sentinel_early_front8',
    'sentinel_early_probe', 'sentinel_early_defer_reset', 'sentinel_early_warp_probe',
    'sentinel_early_w2k2', 'sentinel_early_w2k4', 'sentinel_early_w1k4_w2k2',
    'sentinel_early_w1k4', 'sentinel_early_w2_ready_k4', 'sentinel_early_w2_ready_k16',
    'sentinel_early_w1k4_w2_ready_k16',
    'sentinel_early_w1k4_w2_ready_k16_sf', 'sentinel_early_w1k4_w2k2_sf',
    'sentinel_early_w1k4_w2_ready_k16_warm1',
    'sentinel_early_w1k4_w2_ready_k16_kfirst', 'sentinel_early_w1k4_w2k2_kfirst',
    'sentinel_early_w1k4_w2_ready_k16_ranges', 'sentinel_early_w1k4_w2k2_ranges',
    'sentinel_early_w1k4_w2_ready_k16_token_major', 'sentinel_early_w1k4_w2k2_token_major',
    'sentinel_early_w1k4_w2_ready_k16_chunk_major', 'sentinel_early_w1k4_w2k2_chunk_major',
    'sentinel_early_w1k4_w2k2_keep_w1_chunk_major',
    'sentinel_early_w1k2_w2k2_chunk_major', 'sentinel_early_w1k8_w2k2_chunk_major',
    'sentinel_early_w1k4_w2k2_relaxed_chunk_major',
    'sentinel_early_w1k4_w2k2_relaxed_delay64_chunk_major',
    'sentinel_early_w1k4_w2k2_relaxed_delay256_chunk_major',
    'sentinel_early_w1k4_w2k2_relaxed_delay1024_chunk_major',
    'sentinel_early_w1k4_w2k2_flags_chunk_major', 'sentinel_early_w1k4_w2k2_flags_wide_chunk_major',
    'sentinel_early_w1k2_w2k2_relaxed_chunk_major', 'sentinel_early_w1k4_w2k2_relaxed_token_major',
    'sentinel_early_w1k4_w2k2_relaxed_wide_chunk_major',
    'sentinel_early_w1k4_w2_ready_k16_relaxed_chunk_major',
    'sentinel_early_w1k4_w2k4_relaxed_chunk_major', 'sentinel_early_w1k4_w2k8_relaxed_chunk_major',
    'sentinel_early_w1k2_w2k8_keep_w2_relaxed_chunk_major',
    'sentinel_early_w1k4_w2k8_keep_w2_relaxed_chunk_major',
    'sentinel_early_w1k4_w2_ready_k16_bm32', 'sentinel_early_w1k4_w2k2_bm32',
    'sentinel_early_on_w2', 'sentinel_early_on_w2_store',
    'sentinel_early_w4', 'sentinel_early_w4_wide',
    'sentinel_early_w1k4_w2k2_keep',
    'sentinel_early_direct_w1', 'sentinel_early_direct_w2', 'sentinel_early_direct_both',
    'sentinel_early_narrow_direct_w1',
    'w2_prefix_k2', 'w2_prefix_k4', 'w2_prefix_k8', 'weights_prefix_4_2',
)
COMBINE_TOKEN_WARPS = {'combine_w2': 2, 'combine_w4': 4, 'combine_w8': 8}
FRONT_STACKS = {'front_k4': 4, 'front_k8': 8, 'front_k8_inline': 8,
                'front_k8_contiguous_direct': 8, 'sentinel_early_front8': 8}
WEIGHT_PREFIX_TILES = {'weight_prefix_k2': 2, 'weight_prefix_k4': 4, 'weight_prefix_k8': 8,
                       'weights_prefix_4_2': 4, 'sentinel_early_w1k4_w2k2': 4,
                       'sentinel_early_w1k4': 4, 'sentinel_early_w1k4_w2_ready_k16': 4,
                       'sentinel_early_w1k4_w2_ready_k16_sf': 4, 'sentinel_early_w1k4_w2k2_sf': 4,
                       'sentinel_early_w1k4_w2_ready_k16_warm1': 4,
                       'sentinel_early_w1k4_w2_ready_k16_kfirst': 4, 'sentinel_early_w1k4_w2k2_kfirst': 4,
                       'sentinel_early_w1k4_w2_ready_k16_ranges': 4, 'sentinel_early_w1k4_w2k2_ranges': 4,
                       'sentinel_early_w1k4_w2_ready_k16_token_major': 4,
                       'sentinel_early_w1k4_w2k2_token_major': 4,
                       'sentinel_early_w1k4_w2_ready_k16_chunk_major': 4,
                       'sentinel_early_w1k4_w2k2_chunk_major': 4,
                       'sentinel_early_w1k4_w2k2_keep_w1_chunk_major': 4,
                       'sentinel_early_w1k2_w2k2_chunk_major': 2,
                       'sentinel_early_w1k8_w2k2_chunk_major': 8,
                       'sentinel_early_w1k4_w2k2_relaxed_chunk_major': 4,
                       'sentinel_early_w1k4_w2k2_relaxed_delay64_chunk_major': 4,
                       'sentinel_early_w1k4_w2k2_relaxed_delay256_chunk_major': 4,
                       'sentinel_early_w1k4_w2k2_relaxed_delay1024_chunk_major': 4,
                       'sentinel_early_w1k4_w2k2_flags_chunk_major': 4,
                       'sentinel_early_w1k4_w2k2_flags_wide_chunk_major': 4,
                       'sentinel_early_w1k2_w2k2_relaxed_chunk_major': 2,
                       'sentinel_early_w1k4_w2k2_relaxed_token_major': 4,
                       'sentinel_early_w1k4_w2k2_relaxed_wide_chunk_major': 4,
                       'sentinel_early_w1k4_w2_ready_k16_relaxed_chunk_major': 4,
                       'sentinel_early_w1k4_w2k4_relaxed_chunk_major': 4,
                       'sentinel_early_w1k4_w2k8_relaxed_chunk_major': 4,
                       'sentinel_early_w1k2_w2k8_keep_w2_relaxed_chunk_major': 2,
                       'sentinel_early_w1k4_w2k8_keep_w2_relaxed_chunk_major': 4,
                       'sentinel_early_w1k4_w2_ready_k16_bm32': 4, 'sentinel_early_w1k4_w2k2_bm32': 4,
                       'sentinel_early_w1k4_w2k2_keep': 4} | FRONT_STACKS
W2_PREFIX_TILES = {'w2_prefix_k2': 2, 'w2_prefix_k4': 4, 'w2_prefix_k8': 8,
                   'weights_prefix_4_2': 2, 'sentinel_early_w2k2': 2,
                   'sentinel_early_w2k4': 4, 'sentinel_early_w1k4_w2k2': 2,
                   'sentinel_early_w1k4_w2k2_sf': 2, 'sentinel_early_w1k4_w2k2_kfirst': 2,
                   'sentinel_early_w1k4_w2k2_ranges': 2,
                   'sentinel_early_w1k4_w2k2_token_major': 2,
                   'sentinel_early_w1k4_w2k2_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_keep_w1_chunk_major': 2,
                   'sentinel_early_w1k2_w2k2_chunk_major': 2,
                   'sentinel_early_w1k8_w2k2_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_relaxed_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_relaxed_delay64_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_relaxed_delay256_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_relaxed_delay1024_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_flags_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_flags_wide_chunk_major': 2,
                   'sentinel_early_w1k2_w2k2_relaxed_chunk_major': 2,
                   'sentinel_early_w1k4_w2k2_relaxed_token_major': 2,
                   'sentinel_early_w1k4_w2k2_relaxed_wide_chunk_major': 2,
                   'sentinel_early_w1k4_w2k4_relaxed_chunk_major': 4,
                   'sentinel_early_w1k4_w2k8_relaxed_chunk_major': 8,
                   'sentinel_early_w1k2_w2k8_keep_w2_relaxed_chunk_major': 8,
                   'sentinel_early_w1k4_w2k8_keep_w2_relaxed_chunk_major': 8,
                   'sentinel_early_w1k4_w2k2_bm32': 2,
                   'sentinel_early_w1k4_w2k2_keep': 2}
ROUTE_PUBLISH_CTAS = {'publish_ctas2': 2, 'publish_ctas4': 4, 'publish_ctas8': 8}
COMBINE_FETCH_MODES = {'combine_small': 'small', 'combine_contiguous': 'contiguous',
                       'combine_contiguous_wide': 'contiguous-wide',
                       'combine_streaming': 'streaming', 'combine_streaming_small': 'streaming-small',
                       'combine_contiguous_direct': 'contiguous',
                       'front_k8_contiguous_direct': 'contiguous'}
DIRECT_COMBINE_OUTPUT = {'combine_direct_output', 'combine_contiguous_direct', 'front_k8_contiguous_direct'}
SENTINEL_OPTIONS = {option for option in BACKFILL_OPTIONS if option.startswith('sentinel_')}
SENTINEL_SMS = {'sentinel_early_s32': 32, 'sentinel_early_s48': 48}
SENTINEL_POLL = {'sentinel_early_probe': 'probe', 'sentinel_early_defer_reset': 'defer-reset',
                 'sentinel_early_warp_probe': 'warp-probe'}
SENTINEL_FOUR_WIDE = {'sentinel_early_w4_wide', 'sentinel_early_w1k4_w2k2_relaxed_wide_chunk_major',
                      'sentinel_early_w1k4_w2k2_flags_wide_chunk_major'}
SENTINEL_POLL_DELAYS = {f'sentinel_early_w1k4_w2k2_relaxed_delay{ns}_chunk_major': ns
                        for ns in (64, 256, 1024)}
SENTINEL_LAUNCH_DELAY = {'sentinel_early_on_w2': 1, 'sentinel_early_on_w2_store': 2}
BACKFILL_W2_READY_K = {'sentinel_early_w2_ready_k4': 4, 'sentinel_early_w2_ready_k16': 16,
                      'sentinel_early_w1k4_w2_ready_k16': 16,
                      'sentinel_early_w1k4_w2_ready_k16_sf': 16,
                      'sentinel_early_w1k4_w2_ready_k16_warm1': 16,
                      'sentinel_early_w1k4_w2_ready_k16_kfirst': 16,
                      'sentinel_early_w1k4_w2_ready_k16_ranges': 16,
                      'sentinel_early_w1k4_w2_ready_k16_token_major': 16,
                      'sentinel_early_w1k4_w2_ready_k16_chunk_major': 16,
                      'sentinel_early_w1k4_w2_ready_k16_relaxed_chunk_major': 16,
                      'sentinel_early_w1k4_w2_ready_k16_bm32': 16}
PREFETCH_WEIGHT_SCALES = {'sentinel_early_w1k4_w2_ready_k16_sf', 'sentinel_early_w1k4_w2k2_sf'}

BACKFILL_TUNING = {
    f'{base}_{setting}': (base, pipeline)
    for base in (
        'sentinel_early_w1k4_w2k2_bm32',
        'sentinel_early_w1k4_w2k8_keep_w2_relaxed_chunk_major',
    )
    for setting, pipeline in {
        'release2': {'release_group': 2},
        'release4': {'release_group': 4},
        'stages4': {'decode_tma_stages': 4},
        'stages6': {'decode_tma_stages': 6},
        'acc3': {'decode_accumulator_stages': 3},
        'lookahead4': {'weight_lookahead': 4},
        'lookahead8': {'weight_lookahead': 8},
        'prefix_warps': {'weight_prefix_warps': True},
        'task_k8': {'scheduled_weight_tiles': 8},
        'task_k16': {'scheduled_weight_tiles': 16},
    }.items()
} | {
    f'sentinel_early_combine_w1_prefetch_{experts}_{mode}': (
        'sentinel_early_combine',
        {'prefetch_experts': experts, 'bulk_w1_prefetch_bytes': chunk_bytes})
    for experts in (1, 2, 4)
    for mode, chunk_bytes in (('tensor', 0), ('bulk16k', 16 * 1024), ('bulk64k', 64 * 1024))
} | {
    f'sentinel_early_combine_staged_bm{rows}': (
        'sentinel_early_combine', {'decode_block_m': rows, 'mma_unroll': 8})
    for rows in (64, 96, 128)
} | {
    f'sentinel_early_combine_staged_bm{rows}_trim': (
        'sentinel_early_combine', {'decode_block_m': rows, 'mma_unroll': 8,
                                   'trim_wide_decode_activations': True})
    for rows in (96, 128)
} | {
    f'{base}_{setting}': (base, dispatch)
    for base in ('sentinel_early_w1k4_w2k2_bm32', 'sentinel_early_combine')
    for setting, dispatch in {
        'packed': {'packed_decode_activations': True},
        'packed_scores': {'packed_decode_scores': True},
        'shared_w2_ready': {'decode_shared_w2': 4},
        'posted_indices_control': {'posted_route_sums': True},
        'posted_indices_inline': {'posted_route_sums': True, 'inline_route_indices': True},
        'posted_indices_stripe2': {
            'posted_route_sums': True, 'dispatch_pull_rows_per_cta': 2,
            'dispatch_prefix_scan': True},
        'posted_indices_inline_stripe2': {
            'posted_route_sums': True, 'inline_route_indices': True,
            'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True},
        'inline': {'inline_route_scores': True},
        'packed_inline': {'packed_decode_activations': True, 'inline_route_scores': True},
        'publish2': {'route_publish_ctas': 2},
        'publish4': {'route_publish_ctas': 4},
        'posted': {'posted_route_sums': True},
        'w2_n_group2': {'decode_w2_n_group': 2},
        'w2_n_group4': {'decode_w2_n_group': 4},
        'w2_tile_major': {'w2_tile_major_input': True, 'direct_w1_store': True},
        'w2_tile_major_mma_u8': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                 'mma_unroll': 8},
        'w2_tile_major_mma_u8_transpose': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                           'mma_unroll': 8, 'w1_register_transpose': True},
        'w2_tile_major_mma_u8_packed_scale': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                              'mma_unroll': 8, 'w1_packed_scale_reduction': True},
        'w2_tile_major_bm48': {'w2_tile_major_input': True, 'direct_w1_store': True,
                               'mma_unroll': 8, 'decode_block_m': 48},
        'w2_tile_major_bm64': {'w2_tile_major_input': True, 'direct_w1_store': True,
                               'mma_unroll': 8, 'decode_block_m': 64},
        'w2_tile_major_bm48_single_epi': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                          'mma_unroll': 8, 'decode_block_m': 48,
                                          'single_epilogue_warpgroup': True},
        'w2_tile_major_bm48_lookup': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                      'mma_unroll': 8, 'decode_block_m': 48,
                                      'single_expert_tile_lookup': True},
        'w2_tile_major_bm48_prefix': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                      'mma_unroll': 8, 'decode_block_m': 48,
                                      'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2},
        'w2_tile_major_bm48_prefix_lookup': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                             'mma_unroll': 8, 'decode_block_m': 48,
                                             'single_expert_tile_lookup': True,
                                             'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2},
        'w2_tile_major_bm48_hoisted_acc3_u8': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                               'mma_unroll': 8, 'decode_block_m': 48,
                                               'hoist_routed_sfb_selection': True,
                                               'decode_accumulator_stages': 3},
        'w2_tile_major_narrow_auto': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                     'mma_unroll': 8, 'narrow_bm32': True},
        'w2_tile_major_hoisted_acc3_u8': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                          'mma_unroll': 8, 'hoist_routed_sfb_selection': True,
                                          'decode_accumulator_stages': 3},
        'w2_tile_major_ranges': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                 'mma_unroll': 8, 'cache_routed_ranges': True},
        'one_tile_lookup': {'single_expert_tile_lookup': True},
        'route_octet': {'packed_route_count_group': 8, 'inline_route_scores': True},
        'route_octet_prefix': {'packed_route_count_group': 8, 'inline_route_scores': True,
                               'dispatch_prefix_scan': True},
        'rotate_route_ranks': {'rotate_route_publication': 1},
        'rotate_route_ranks_experts': {'rotate_route_publication': 2},
        'dispatch_pull_rows_2': {'dispatch_pull_rows_per_cta': 2},
        'dispatch_pull_rows_4': {'dispatch_pull_rows_per_cta': 4},
        'double_buffer_pulls': {'double_buffer_dispatch_pull': True},
        'double_buffer_pulls_rows_2': {'double_buffer_dispatch_pull': True,
                                       'dispatch_pull_rows_per_cta': 2},
        'dispatch_pull_rows_2_prefix': {'dispatch_pull_rows_per_cta': 2,
                                        'dispatch_prefix_scan': True},
        'dispatch_pull_rows_2_prefix_w1_k8': {
            'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True,
            'w1_prefetch_k_tiles': 8, 'w2_prefetch_k_tiles': 2,
            'bulk_w1_prefetch_bytes': 64 * 1024, 'bulk_w1_prefetch_lanes': 4},
        'dispatch_warps12': {'decode_dispatch_warps': 12},
        'dispatch_warps16': {'decode_dispatch_warps': 16},
        'dispatch_warps12_pull4_prefix': {'decode_dispatch_warps': 12,
                                           'dispatch_pull_rows_per_cta': 4,
                                           'dispatch_prefix_scan': True},
        'fixed_expert_slots': {'early_experts': 0},
        'native_dispatch': {'early_experts': 0, 'native_decode_dispatch': 1},
        'native_dispatch_arrival': {'early_experts': 0, 'native_decode_dispatch': 2},
        'native_dispatch_plan': {'early_experts': 0, 'native_decode_dispatch': 3},
        'multicast_counts': {'multicast_route_counts': True},
        'multicast_group4': {'multicast_route_counts': True, 'packed_route_count_group': 4},
        'prune_resets': {'prune_decode_resets': True},
        'omit_finish': {'omit_sentinel_peer_finish': True},
        'prune_resets_omit_finish': {'prune_decode_resets': True, 'omit_sentinel_peer_finish': True},
        'pruned_lookup': {'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
                          'single_expert_tile_lookup': True},
        'pruned_w1_bulk16k': {'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
                              'prefetch_experts': 1, 'bulk_w1_prefetch_bytes': 16 * 1024,
                              'single_expert_tile_lookup': True},
        'linear_pool_pruned_bulk16k': {
            'decode_linear_pool': True, 'prune_decode_resets': True,
            'omit_sentinel_peer_finish': True, 'prefetch_experts': 1,
            'bulk_w1_prefetch_bytes': 16 * 1024, 'single_expert_tile_lookup': True},
        'vector_pull': {'vector_pull': True},
        'vector_pull2': {'vector_pull': True, 'decode_vector_pull_group': 2},
        'vector_pull4': {'vector_pull': True, 'decode_vector_pull_group': 4},
        'w1_warmup1': {'decode_w1_warmup': 1},
        'w1_warmup3': {'decode_w1_warmup': 3},
        'w2_ready_k4': {'decode_w2_ready_k': 4},
        'probe_extra_w1_mma': {'probe_routed_mma': 1},
        'probe_extra_w2_mma': {'probe_routed_mma': 2},
        'probe_extra_w1_w2_mma': {'probe_routed_mma': 3},
        'w1_prefetch_1_dispatch16k': {
            'prefetch_experts': 1, 'bulk_w1_prefetch_bytes': 16 * 1024,
            'bulk_w1_prefetch_on_dispatch': True},
        'w1_prefetch_2_dispatch64k': {
            'prefetch_experts': 2, 'bulk_w1_prefetch_bytes': 64 * 1024,
            'bulk_w1_prefetch_on_dispatch': True},
        'w2_tile_major_bm48_route_octet': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                           'mma_unroll': 8, 'decode_block_m': 48,
                                           'packed_route_count_group': 8, 'inline_route_scores': True,
                                           'dispatch_prefix_scan': True},
        'dispatch_w12': {'decode_dispatch_warps': 12},
        'fused_dispatch': {'decode_dispatch_warps': 12, 'fused_dispatch_compute': True},
        'fused_reuse': {'decode_dispatch_warps': 12, 'fused_worker_reuse': True},
        'fixed_w1_head': {'early_experts': 0},
        'early_w1_head1': {'early_experts': 1},
        'early_w1_head2': {'early_experts': 2},
        'split_route_head1': {'early_experts': 1, 'split_route_heads': True},
        'split_route_head2': {'early_experts': 2, 'split_route_heads': True},
        'split_route_head1_speculative': {'early_experts': 1, 'split_route_heads': True,
                                         'speculative_w1': True},
        'speculative_w1_head1': {'early_experts': 1, 'speculative_w1': True},
        'speculative_w1_head2': {'early_experts': 2, 'speculative_w1': True},
        'w2_tile_major_one_tile_lookup': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                           'mma_unroll': 8, 'single_expert_tile_lookup': True},
        'w2_tile_major_peer_mailbox': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                       'mma_unroll': 8, 'peer_decode_input': True},
        'prefetch_a_head1': {'peer_decode_input': True, 'peer_decode_token_fanout': 2,
                             'peer_prefetch_expert_heads': 1},
        'prefetch_a_head1_no_wait': {'peer_decode_input': True, 'peer_decode_token_fanout': 2,
                                     'peer_prefetch_expert_heads': 1, 'peer_prefetch_opportunistic': True},
        'prefetch_a_head2_no_wait': {'peer_decode_input': True, 'peer_decode_token_fanout': 2,
                                     'peer_prefetch_expert_heads': 2, 'peer_prefetch_opportunistic': True},
        'w2_tile_major_peer_fanout': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                      'mma_unroll': 8, 'peer_decode_input': True,
                                      'peer_decode_token_fanout': True},
        'w2_tile_major_peer_direct': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                      'mma_unroll': 8, 'peer_decode_input_direct': True},
        'w2_tile_major_peer_direct_fanout': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                             'mma_unroll': 8, 'peer_decode_input_direct': True,
                                             'peer_decode_token_fanout': True},
        'w2_tile_major_peer_direct_trim': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                           'mma_unroll': 8, 'peer_decode_input_direct': True,
                                           'trim_decode_gather_tail': True},
        'w2_tile_major_peer_direct_trim_fanout': {'w2_tile_major_input': True, 'direct_w1_store': True,
                                                  'mma_unroll': 8, 'peer_decode_input_direct': True,
                                                  'trim_decode_gather_tail': True,
                                                  'peer_decode_token_fanout': 2},
        'weight_lane': {'single_tma_lanes': 1},
        'operand_lanes': {'single_tma_lanes': 2},
        'align2': {'stage_alignment': 2},
        'dual_weight': {'stage_alignment': 2, 'dual_weight_warps': True},
        'weight_maps': {'decode_descriptor_prefetch': 'weights'},
        'no_maps': {'decode_descriptor_prefetch': 'off'},
        'scale_pair': {'stage_alignment': 2, 'pair_routed_weight_scales': True},
        'shared_w1_k24': {'shared_w1_prefetch_limit': 24},
        'shared_w1_k48': {'shared_w1_prefetch_limit': 48},
        'ordered': {'sentinel_poll': 'ordered'},
        'compute_combine': {'combine_in_compute': 1},
        'compute_combine_ordered': {'combine_in_compute': 1, 'sentinel_poll': 'ordered'},
        'compute_combine_delayed': {'combine_in_compute': 2},
        'compute_combine_delayed_ordered': {'combine_in_compute': 2, 'sentinel_poll': 'ordered'},
    }.items()
} | {
    f'sentinel_early_combine_w2_tile_major_bm48_{setting}': (
        'sentinel_early_combine',
        {'w2_tile_major_input': True, 'direct_w1_store': True,
         'mma_unroll': 8, 'decode_block_m': 48} | overrides)
    for setting, overrides in {
        'fixed_expert_slots': {'early_experts': 0},
        'packed_activations': {'packed_decode_activations': True},
        'packed_activations_scores': {'packed_decode_scores': True},
        'shared_w2_after': {'decode_shared_w2': 1},
        'shared_w2_rotated': {'decode_shared_w2': 2},
        'static_w1_prefix': {'static_w1_prefix': True},
        'shared_w2_rotated_static_w1': {'decode_shared_w2': 2, 'static_w1_prefix': True},
        'keep_weight_scales': {'keep_routed_weight_scales': True},
        'combine_warps4': {'sentinel_combine_warps': 4},
        'reuse_finalize_warps8': {'reuse_finalize_warps': True},
        'reuse_finalize_warps4': {'reuse_finalize_warps': True, 'sentinel_combine_warps': 4},
        'combine_half_vector': {'sentinel_half_vector': True},
        'combine_half_vector_reuse_warps': {'sentinel_half_vector': True, 'reuse_finalize_warps': True},
        'speculative_fixed_w1': {'early_experts': 1, 'speculative_w1': True},
        'speculative_packed_w1': {'early_experts': 1, 'speculative_w1': True,
                                  'speculative_packed_w1': True},
        'w2_a_issue2': {'w2_a_issue_window': 2},
        'w2_a_issue4': {'w2_a_issue_window': 4},
        'w2_a_issue8': {'w2_a_issue_window': 8},
        'w2_early_workers4': {'decode_w2_early_workers': 4},
        'w2_early_workers8': {'decode_w2_early_workers': 8},
        'w2_early_workers12': {'decode_w2_early_workers': 12},
        'static_w2_overlap2': {'static_decode_wave': 2, 'static_w2_overlap': True,
                               'single_expert_tile_lookup': True},
        'static_w2_overlap4': {'static_decode_wave': 4, 'static_w2_overlap': True,
                               'single_expert_tile_lookup': True},
        'static_w2_overlap6': {'static_decode_wave': 6, 'static_w2_overlap': True,
                               'single_expert_tile_lookup': True},
        'static_w2_overlap8': {'static_decode_wave': 8, 'static_w2_overlap': True,
                               'single_expert_tile_lookup': True},
        'static_w2_overlap8_reuse_finalize_warps': {
            'static_decode_wave': 8, 'static_w2_overlap': True,
            'single_expert_tile_lookup': True, 'reuse_finalize_warps': True},
        'static_waves8': {'static_decode_wave': 8, 'single_expert_tile_lookup': True},
        'task_slots4': {'decode_task_slots': 4},
        'task_slots8': {'decode_task_slots': 8},
        'task_slots4_early_w2': {'decode_task_slots': 4, 'decode_w2_early_workers': 8},
        'shared_w2_rotated_early_workers4': {'decode_shared_w2': 2, 'decode_w2_early_workers': 4},
        'shared_w2_rotated_a_issue4': {'decode_shared_w2': 2, 'w2_a_issue_window': 4},
        'w1_early_epi_inputs': {'w1_early_epi_inputs': True},
        'w1_pair_scale_rounding': {'w1_pair_scale_rounding': True},
        'w1_pair_scale_rounding_packed': {'w1_pair_scale_rounding': True,
                                          'w1_packed_scale_reduction': True},
        'w1_pair_scale_rounding_early': {'w1_pair_scale_rounding': True,
                                         'w1_early_epi_inputs': True},
        'w1_early_epi_inputs_packed_scale': {'w1_early_epi_inputs': True,
                                             'w1_packed_scale_reduction': True},
        'shared_w2_ready': {'decode_shared_w2': 4},
        'peer_prefix_k1': {'peer_activation_prefix_tiles': 1},
        'peer_prefix_k2': {'peer_activation_prefix_tiles': 2},
        'peer_prefix_k4': {'peer_activation_prefix_tiles': 4},
        'peer_tma_k1': {'peer_activation_prefix_tiles': 1, 'peer_activation_prefix_gather': True},
        'peer_tma_k2': {'peer_activation_prefix_tiles': 2, 'peer_activation_prefix_gather': True},
        'peer_tma_k4': {'peer_activation_prefix_tiles': 4, 'peer_activation_prefix_gather': True},
        'peer_tma_k1_head1': {'peer_activation_prefix_tiles': 1, 'peer_activation_prefix_gather': True,
                              'peer_activation_head_clusters': 1},
        'peer_tma_k1_head2': {'peer_activation_prefix_tiles': 1, 'peer_activation_prefix_gather': True,
                              'peer_activation_head_clusters': 2},
        'peer_tma_k1_head4': {'peer_activation_prefix_tiles': 1, 'peer_activation_prefix_gather': True,
                              'peer_activation_head_clusters': 4},
        'dispatch_pull_rows_1': {'dispatch_pull_rows_per_cta': 1},
        'dispatch_pull_rows_2': {'dispatch_pull_rows_per_cta': 2},
        'dispatch_pull_rows_4': {'dispatch_pull_rows_per_cta': 4},
        'dispatch_pull_rows_8': {'dispatch_pull_rows_per_cta': 8},
        'combine_flags': {'sentinel_chunk_major': True, 'combine_ready_flags': True},
        'combine_mask': {'sentinel_chunk_major': True, 'combine_ready_flags': True,
                         'combine_ready_mask': True},
        'combine_mask_prefix': {'sentinel_chunk_major': True, 'combine_ready_flags': True,
                                'combine_ready_mask': True, 'sentinel_poll': 'inline-prefix'},
        'native_dispatch': {'early_experts': 0, 'native_decode_dispatch': 1},
        'native_dispatch_arrival': {'early_experts': 0, 'native_decode_dispatch': 2},
        'native_dispatch_plan': {'early_experts': 0, 'native_decode_dispatch': 3},
        'align2': {'stage_alignment': 2},
        'dual_weight': {'stage_alignment': 2, 'dual_weight_warps': True},
        'weight_maps': {'decode_descriptor_prefetch': 'weights'},
        'no_maps': {'decode_descriptor_prefetch': 'off'},
        'w1_prefetch_1_tensor': {'prefetch_experts': 1},
        'w1_prefetch_1_bulk16k': {'prefetch_experts': 1, 'bulk_w1_prefetch_bytes': 16 * 1024},
        'w1_prefetch_2_bulk64k': {'prefetch_experts': 2, 'bulk_w1_prefetch_bytes': 64 * 1024},
        'w1_prefetch_2_dispatch64k': {
            'prefetch_experts': 2, 'bulk_w1_prefetch_bytes': 64 * 1024,
            'bulk_w1_prefetch_on_dispatch': True},
        'rotate_route_ranks_experts': {'rotate_route_publication': 2},
        'multicast_group4': {'multicast_route_counts': True, 'packed_route_count_group': 4},
        'prune_resets_omit_finish': {'prune_decode_resets': True, 'omit_sentinel_peer_finish': True},
        'pruned_prefix_lookup': {'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
                                 'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2,
                                 'single_expert_tile_lookup': True},
        'linear_pool': {'decode_linear_pool': True},
        'linear_pool_pruned_bulk16k': {
            'decode_linear_pool': True, 'prune_decode_resets': True,
            'omit_sentinel_peer_finish': True, 'prefetch_experts': 1,
            'bulk_w1_prefetch_bytes': 16 * 1024, 'w2_prefetch_k_tiles': 2,
            'single_expert_tile_lookup': True},
        'pruned_w1_bulk16k': {'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
                              'prefetch_experts': 1, 'bulk_w1_prefetch_bytes': 16 * 1024,
                              'w2_prefetch_k_tiles': 2, 'single_expert_tile_lookup': True},
        'pruned_w1_bulk64k': {'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
                              'prefetch_experts': 2, 'bulk_w1_prefetch_bytes': 64 * 1024,
                              'w2_prefetch_k_tiles': 2, 'single_expert_tile_lookup': True},
        'pruned_w1_bulk16k_packed_scale': {
            'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
            'prefetch_experts': 1, 'bulk_w1_prefetch_bytes': 16 * 1024,
            'w2_prefetch_k_tiles': 2, 'single_expert_tile_lookup': True,
            'w1_packed_scale_reduction': True},
        'combine_prefix': {'sentinel_poll': 'prefix'},
        'combine_inline_prefix': {'sentinel_poll': 'inline-prefix'},
        'combine_shared_first': {'sentinel_poll': 'shared-first'},
        'combine_shared_prefix': {'sentinel_poll': 'shared-prefix'},
        'combine_shared_first_delay256': {'sentinel_poll': 'shared-first', 'sentinel_poll_delay_ns': 256},
        'w2_prefix4': {'w2_prefetch_k_tiles': 4},
        'w2_from_w1_k2': {'w2_prefetch_from_w1': 2},
        'w2_from_w1_k4': {'w2_prefetch_from_w1': 4},
        'w2_from_w1_k8': {'w2_prefetch_from_w1': 8},
        'weight_k2_groups': {'grouped_routed_weights': 3, 'grouped_weight_inner_k': 2},
        'weight_k4_groups': {'grouped_routed_weights': 3, 'grouped_weight_inner_k': 4},
        'weight_k8_groups': {'grouped_routed_weights': 3, 'grouped_weight_inner_k': 8},
        'weight_k8_groups_w1': {'grouped_routed_weights': 1, 'grouped_weight_inner_k': 8},
        'weight_k8_groups_w2': {'grouped_routed_weights': 2, 'grouped_weight_inner_k': 8},
        'weight_k16_groups': {'grouped_routed_weights': 3, 'grouped_weight_inner_k': 16},
        'packed_route_count_control': {'packed_route_counts': True},
        'inline_route_indices': {'inline_route_indices': True},
        'inline_route_indices_posted': {'inline_route_indices': True, 'posted_route_sums': True},
        'inline_route_indices_posted_stripe2': {'inline_route_indices': True, 'posted_route_sums': True,
                                                'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True},
        'posted_stripe2': {'posted_route_sums': True, 'dispatch_pull_rows_per_cta': 2,
                           'dispatch_prefix_scan': True},
        'posted_indices_control': {'posted_route_sums': True},
        'early_w2_metadata': {'prefetch_w2_metadata': True},
        'packed_w2_metadata': {'packed_w2_metadata': True},
        'early_packed_w2_metadata': {'prefetch_w2_metadata': True, 'packed_w2_metadata': True},
        'heavy_expert_first': {'single_expert_tile_lookup': True, 'prioritize_expert_rows': 1},
        'light_expert_first': {'single_expert_tile_lookup': True, 'prioritize_expert_rows': -1},
        'combine_inline_prefix_delay256': {'sentinel_poll': 'inline-prefix',
                                           'sentinel_poll_delay_ns': 256},
        'prefix_bulk16k': {'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2,
                           'bulk_w1_prefetch_bytes': 16 * 1024, 'single_expert_tile_lookup': True},
        'prefix_bulk64k': {'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2,
                           'bulk_w1_prefetch_bytes': 64 * 1024, 'single_expert_tile_lookup': True},
        'prefix_bulk64k_keep': {'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2,
                                'bulk_w1_prefetch_bytes': 64 * 1024, 'keep_w1_prefix': True,
                                'single_expert_tile_lookup': True},
        'probe_control': {'decode_accumulator_stages': 3},
        'register_transpose': {'w1_register_transpose': True},
        'packed_scale': {'w1_packed_scale_reduction': True},
        'hardware_scale': {'w1_hardware_scale_reduction': True},
        'hardware_scale_transpose': {'w1_hardware_scale_reduction': True,
                                     'w1_register_transpose': True},
        'packed_scale_transpose': {'w1_packed_scale_reduction': True, 'w1_register_transpose': True},
        'prefix_lookup_transpose': {'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2,
                                    'single_expert_tile_lookup': True, 'w1_register_transpose': True},
        'probe_extra_w1_mma': {'decode_accumulator_stages': 3, 'probe_routed_mma': 1},
        'probe_extra_w2_mma': {'decode_accumulator_stages': 3, 'probe_routed_mma': 2},
        'probe_extra_w1_w2_mma': {'decode_accumulator_stages': 3, 'probe_routed_mma': 3},
        'vector_pull4': {'vector_pull': True, 'decode_vector_pull_group': 4},
        'w1_warmup1': {'decode_w1_warmup': 1},
        'w1_warmup3': {'decode_w1_warmup': 3},
        'w2_ready_k4': {'decode_w2_ready_k': 4},
        'w2_n_group2': {'decode_w2_n_group': 2},
        'w1_warmup1_bulk16k': {'decode_w1_warmup': 1, 'prefetch_experts': 1,
                              'bulk_w1_prefetch_bytes': 16 * 1024},
    }.items()
} | {
    f'sentinel_early_combine_w2_tile_major_bm48_raw_w1_k{k}_w{lanes}': (
        'sentinel_early_combine',
        {'w2_tile_major_input': True, 'direct_w1_store': True, 'decode_block_m': 48,
         'mma_unroll': 8, 'w1_prefetch_k_tiles': k, 'w2_prefetch_k_tiles': 2,
         'bulk_w1_prefetch_bytes': 64 * 1024, 'bulk_w1_prefetch_lanes': lanes,
         'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
         'single_expert_tile_lookup': True})
    for k, lanes in ((4, 1), (8, 1), (16, 1), (24, 1), (8, 4), (16, 4), (24, 8))
} | {
    f'sentinel_early_combine_w2_tile_major_bm32_{setting}': (
        'sentinel_early_combine',
        {'w2_tile_major_input': True, 'direct_w1_store': True,
         'mma_unroll': 8, 'decode_block_m': 32} | overrides)
    for setting, overrides in {
        'control': {},
        'transpose_activations': {'transpose_w2_activations': 1},
        'transpose_full_activations': {'transpose_w2_activations': 2},
        'w1_prefetch_1_bulk16k': {'prefetch_experts': 1, 'bulk_w1_prefetch_bytes': 16 * 1024},
        'w1_prefetch_2_bulk64k': {'prefetch_experts': 2, 'bulk_w1_prefetch_bytes': 64 * 1024},
        'w1_prefetch_1_dispatch16k': {
            'prefetch_experts': 1, 'bulk_w1_prefetch_bytes': 16 * 1024,
            'bulk_w1_prefetch_on_dispatch': True},
    }.items()
} | {
    'sentinel_early_w1k4_w2k2_bm32_narrow_a': (
        'sentinel_early_w1k4_w2k2_bm32', {'narrow_bm32': True}),
    'sentinel_early_w1k4_w2k2_bm32_narrow_direct_w1': (
        'sentinel_early_w1k4_w2k2_bm32', {'narrow_bm32': True, 'direct_w1_store': True}),
} | {
    f'sentinel_early_w1k4_w2k2_bm32_{setting}': (
        'sentinel_early_w1k4_w2k2_bm32',
        {'narrow_bm32': True, 'direct_w1_store': True, 'w2_tile_major_input': True} | overrides)
    for setting, overrides in {
        'narrow_w2_tile_major': {},
        'narrow_w2_tile_major_dispatch_warps12': {'decode_dispatch_warps': 12},
        'narrow_w2_tile_major_dispatch_warps16': {'decode_dispatch_warps': 16},
        'narrow_w2_tile_major_dispatch_pull_rows_2': {'dispatch_pull_rows_per_cta': 2},
        'narrow_w2_tile_major_dispatch_pull_rows_4': {'dispatch_pull_rows_per_cta': 4},
        'narrow_w2_tile_major_dispatch_pull_rows_2_prefix': {
            'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True},
        'narrow_w2_tile_major_rows2_prefix_posted': {
            'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True,
            'posted_route_sums': True},
        'narrow_w2_tile_major_rows2_prefix_scale_pair': {
            'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True,
            'stage_alignment': 2, 'pair_routed_weight_scales': True},
        'narrow_w2_tile_major_rows2_prefix_pruned': {
            'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True,
            'prune_decode_resets': True, 'omit_sentinel_peer_finish': True},
        'narrow_w2_tile_major_raw_w1_k8_w4_first2': {
            'w1_prefetch_k_tiles': 8, 'w1_prefix_expert_limit': 2,
            'bulk_w1_prefetch_bytes': 64 * 1024, 'bulk_w1_prefetch_lanes': 4},
        'narrow_w2_tile_major_linear_pool': {'decode_linear_pool': True},
        'narrow_w2_tile_major_register_transpose': {'w1_register_transpose': True},
        'narrow_w2_tile_major_packed_scale': {'w1_packed_scale_reduction': True},
        'narrow_w2_tile_major_hardware_scale': {'w1_hardware_scale_reduction': True},
        'narrow_w2_tile_major_bulk_prefix16k': {'bulk_w1_prefetch_bytes': 16 * 1024},
        'narrow_w2_tile_major_bulk_prefix64k': {'bulk_w1_prefetch_bytes': 64 * 1024},
        'narrow_w2_tile_major_bulk_prefix64k_keep': {
            'bulk_w1_prefetch_bytes': 64 * 1024, 'keep_w1_prefix': True},
        'asymmetric_a': {'asymmetric_activation_rows': 1},
        'skip_padded_a': {'asymmetric_activation_rows': 2},
        'tile_major_group2': {'decode_w2_n_group': 2},
        'tile_major_posted': {'posted_route_sums': True},
        'tile_major_shared_after': {'decode_shared_w2': 1},
        'tile_major_shared_rotated': {'decode_shared_w2': 2},
        'tile_major_ready_w2': {'decode_w2_ready_k': 16},
        'tile_major_late_combine': {'stripe_launch_delay': 2},
        'tile_major_group2_posted': {'decode_w2_n_group': 2, 'posted_route_sums': True},
        'tile_major_packed_w1_control': {'packed_decode_activations': True},
        'tile_major_both_inputs': {'packed_decode_activations': True, 'w1_tile_major_input': True},
        'tile_major_mma_u2': {'mma_unroll': 2},
        'tile_major_mma_u4': {'mma_unroll': 4},
        'tile_major_mma_u8': {'mma_unroll': 8},
        'tile_major_mma_u16': {'mma_unroll': 16},
        'tile_major_hoisted_sfb_u8': {'mma_unroll': 8, 'hoist_routed_sfb_selection': True},
        'tile_major_hoisted_sfb_u16': {'mma_unroll': 16, 'hoist_routed_sfb_selection': True},
        'tile_major_hoisted_sfb_acc3_u8': {'mma_unroll': 8, 'hoist_routed_sfb_selection': True,
                                           'decode_accumulator_stages': 3},
        'tile_major_linear_hoisted_prefetch': {
            'decode_linear_pool': True, 'prune_decode_resets': True,
            'omit_sentinel_peer_finish': True, 'mma_unroll': 8,
            'hoist_routed_sfb_selection': True, 'decode_accumulator_stages': 3,
            'bulk_w1_prefetch_bytes': 64 * 1024, 'keep_w1_prefix': True},
        'tile_major_acc3_u8': {'mma_unroll': 8, 'decode_accumulator_stages': 3},
        'tile_major_acc4_u8': {'mma_unroll': 8, 'decode_accumulator_stages': 4},
        'tile_major_cached_ranges_u8': {'mma_unroll': 8, 'cache_routed_ranges': True},
        'tile_major_peer_mailbox_u8': {'mma_unroll': 8, 'peer_decode_input': True},
        'tile_major_peer_fanout_u8': {'mma_unroll': 8, 'peer_decode_input': True,
                                      'peer_decode_token_fanout': True},
        'tile_major_peer_fanout_lanes_u8': {'mma_unroll': 8, 'peer_decode_input': True,
                                            'peer_decode_token_fanout': 2},
        'tile_major_static_wave4_u8': {'mma_unroll': 8, 'cache_routed_ranges': True,
                                       'static_decode_wave': 4},
        'tile_major_static_wave8_u8': {'mma_unroll': 8, 'cache_routed_ranges': True,
                                       'static_decode_wave': 8},
        'tile_major_static_wave16_u8': {'mma_unroll': 8, 'cache_routed_ranges': True,
                                        'static_decode_wave': 16},
        'tile_major_direct_w2_u8': {'mma_unroll': 8, 'direct_w2_store': True},
        'tile_major_chunk_scatter_u8': {'mma_unroll': 8, 'sentinel_chunk_major': True},
        'tile_major_direct_w2_chunk_u8': {'mma_unroll': 8, 'direct_w2_store': True,
                                         'sentinel_chunk_major': True},
        'tile_major_direct_w2_token_u8': {'mma_unroll': 8, 'direct_w2_store': True,
                                         'token_major_combine': True},
        'tile_major_combine_prefix_u8': {'mma_unroll': 8, 'sentinel_poll': 'prefix'},
        'tile_major_combine_shared_first_u8': {'mma_unroll': 8, 'sentinel_poll': 'shared-first'},
        'tile_major_combine_delay256_u8': {'mma_unroll': 8, 'sentinel_poll_delay_ns': 256},
        'tile_major_combine_delay128_u8': {'mma_unroll': 8, 'sentinel_poll_delay_ns': 128},
        'tile_major_combine_delay512_u8': {'mma_unroll': 8, 'sentinel_poll_delay_ns': 512},
        'tile_major_combine_delay768_u8': {'mma_unroll': 8, 'sentinel_poll_delay_ns': 768},
        'tile_major_combine_adaptive512_u8': {'mma_unroll': 8, 'sentinel_poll_delay_ns': 512,
                                               'sentinel_adaptive_poll': True},
        'tile_major_shared_last_u8': {'mma_unroll': 8, 'decode_shared_w2': 3},
        'tile_major_shared_last_delay256_u8': {'mma_unroll': 8, 'decode_shared_w2': 3,
                                                'sentinel_poll_delay_ns': 256},
        'tile_major_shared_last_hoisted_u16': {'mma_unroll': 16, 'decode_shared_w2': 3,
                                                'hoist_routed_sfb_selection': True},
        'tile_major_shared_last_hoisted_acc3_u8': {'mma_unroll': 8, 'decode_shared_w2': 3,
                                                    'hoist_routed_sfb_selection': True,
                                                    'decode_accumulator_stages': 3},
        'tile_major_combine_shared_epi_u8': {'mma_unroll': 8, 'decode_shared_w2': 3,
                                              'combine_in_compute': 3},
        'tile_major_combine_shared_epi_token_u8': {'mma_unroll': 8, 'decode_shared_w2': 3,
                                                    'combine_in_compute': 3, 'token_major_combine': True},
        'tile_major_direct_w2_token_delay256_u8': {'mma_unroll': 8, 'direct_w2_store': True,
                                                  'token_major_combine': True,
                                                  'sentinel_poll_delay_ns': 256},
        'tile_major_direct_w2_token_prefix_u8': {'mma_unroll': 8, 'direct_w2_store': True,
                                                'token_major_combine': True, 'sentinel_poll': 'prefix'},
        'tile_major_skip_padded_mma_u8': {'asymmetric_activation_rows': 2, 'mma_unroll': 8},
        'tile_major_group2_posted_mma_u8': {'decode_w2_n_group': 2, 'posted_route_sums': True,
                                            'mma_unroll': 8},
        'tile_major_single_mma_lane': {'single_mma_lane': True},
        'tile_major_align2': {'stage_alignment': 2},
        'tile_major_pair_w1': {'stage_alignment': 2, 'pair_routed_weights': 1},
        'tile_major_pair_w2': {'stage_alignment': 2, 'pair_routed_weights': 2},
        'tile_major_pair_both': {'stage_alignment': 2, 'pair_routed_weights': 3},
        'tile_major_pair_weight_scales': {'stage_alignment': 2, 'pair_routed_weights': 3,
                                         'pair_routed_weight_scales': True},
        'tile_major_packed_scales_u8': {'mma_unroll': 8, 'packed_routed_scales': True},
        'tile_major_grouped_single': {'stage_alignment': 2, 'grouped_routed_weights': 3},
        'tile_major_grouped_w1': {'stage_alignment': 2, 'grouped_routed_weights': 1,
                                  'pair_routed_weights': 1},
        'tile_major_grouped_w2': {'stage_alignment': 2, 'grouped_routed_weights': 2,
                                  'pair_routed_weights': 2},
        'tile_major_grouped_both': {'stage_alignment': 2, 'grouped_routed_weights': 3,
                                    'pair_routed_weights': 3},
    }.items()
}

RAW_W1_BM48 = 'sentinel_early_combine_w2_tile_major_bm48_raw_w1_k8_w4'
RAW_W1_MODE, RAW_W1_TUNING = BACKFILL_TUNING[RAW_W1_BM48]
BACKFILL_TUNING |= {
    f'{RAW_W1_BM48}_{name}': (RAW_W1_MODE, RAW_W1_TUNING | overrides)
    for name, overrides in {
        'dispatch_pull_rows_2': {'dispatch_pull_rows_per_cta': 2},
        'dispatch_pull_rows_4': {'dispatch_pull_rows_per_cta': 4},
        'dispatch_pull_rows_2_prefix': {'dispatch_pull_rows_per_cta': 2,
                                        'dispatch_prefix_scan': True},
        'shared_w2_rotated': {'decode_shared_w2': 2},
        'shared_w2_rotated_static_w1': {'decode_shared_w2': 2, 'static_w1_prefix': True},
        'shared_w2_rotated_keep_weight_scales': {'decode_shared_w2': 2,
                                                 'keep_routed_weight_scales': True},
        'shared_w2_rotated_stripe2': {
            'decode_shared_w2': 2, 'dispatch_pull_rows_per_cta': 2,
            'dispatch_prefix_scan': True, 'posted_route_sums': True},
        'shared_w2_rotated_stripe2_reuse_finalize_warps': {
            'decode_shared_w2': 2, 'dispatch_pull_rows_per_cta': 2,
            'dispatch_prefix_scan': True, 'posted_route_sums': True,
            'reuse_finalize_warps': True},
        'first1': {'w1_prefix_expert_limit': 1},
        'first2': {'w1_prefix_expert_limit': 2},
        'first4': {'w1_prefix_expert_limit': 4},
    }.items()
}

SPECULATIVE_PACKED_W1 = 'sentinel_early_combine_w2_tile_major_bm48_speculative_packed_w1'
SPECULATIVE_MODE, SPECULATIVE_TUNING = BACKFILL_TUNING[SPECULATIVE_PACKED_W1]
BACKFILL_TUNING |= {
    f'{SPECULATIVE_PACKED_W1}_n{tiles}': (SPECULATIVE_MODE, SPECULATIVE_TUNING | {'speculative_w1_n_tiles': tiles})
    for tiles in (1, 2, 4, 8)
}

NARROW_PRUNED = 'sentinel_early_w1k4_w2k2_bm32_narrow_w2_tile_major_rows2_prefix_pruned'
NARROW_PRUNED_MODE, NARROW_PRUNED_TUNING = BACKFILL_TUNING[NARROW_PRUNED]
for name, short_scales in (('full_a_control', False), ('short_activation_sf', True)):
    BACKFILL_TUNING[f'{NARROW_PRUNED}_{name}'] = (
        NARROW_PRUNED_MODE,
        NARROW_PRUNED_TUNING | {'narrow_bm32': False, 'short_activation_sf': short_scales,
                                'decode_descriptor_prefetch': 'off'})
BACKFILL_TUNING |= {
    f'{NARROW_PRUNED}_{name}': (NARROW_PRUNED_MODE, NARROW_PRUNED_TUNING | overrides)
    for name, overrides in {
        'keep_weight_scales': {'keep_routed_weight_scales': True},
        'static_w1_prefix': {'static_w1_prefix': True},
        'shared_w2_rotated': {'decode_shared_w2': 2},
        'shared_w2_rotated_static_w1': {'decode_shared_w2': 2, 'static_w1_prefix': True},
        'reuse_finalize_warps': {'reuse_finalize_warps': True},
        'weight_descriptors_only': {'decode_descriptor_prefetch': 'weights'},
        'skip_descriptor_prefetch': {'decode_descriptor_prefetch': 'off'},
        'skip_descriptors_static_w1': {
            'decode_descriptor_prefetch': 'off', 'static_w1_prefix': True},
        'weight_descriptors_static_w1': {
            'decode_descriptor_prefetch': 'weights', 'static_w1_prefix': True},
        'skip_descriptors_rotated_static_w1': {
            'decode_descriptor_prefetch': 'off', 'static_w1_prefix': True, 'decode_shared_w2': 2},
        'skip_descriptors_mma8': {'decode_descriptor_prefetch': 'off', 'mma_unroll': 8},
        'skip_descriptors_reuse_finalize': {
            'decode_descriptor_prefetch': 'off', 'reuse_finalize_warps': True},
        'flat_cuda_graph': {'flat_decode_graph': True},
        'skip_descriptors_flat_cuda_graph': {
            'decode_descriptor_prefetch': 'off', 'flat_decode_graph': True},
        'flat_cluster_spread': {'flat_decode_graph': True, 'decode_cluster_policy': 1},
        'skip_descriptors_flat_cluster_spread': {
            'decode_descriptor_prefetch': 'off', 'flat_decode_graph': True, 'decode_cluster_policy': 1},
        'one_weight_lane': {'single_tma_lanes': 1},
        'one_operand_lane': {'single_tma_lanes': 2},
        'weight_descriptors_one_operand_lane': {
            'decode_descriptor_prefetch': 'weights', 'single_tma_lanes': 2},
        'static_w2_overlap8': {'static_decode_wave': 8, 'static_w2_overlap': True,
                               'single_expert_tile_lookup': True},
    }.items()
}
BACKFILL_TUNING |= {
    f'{NARROW_PRUNED}_w2_early_workers{workers}': (
        NARROW_PRUNED_MODE, NARROW_PRUNED_TUNING | {'decode_w2_early_workers': workers})
    for workers in (4, 8, 12, 16)
}
# Startup experiments (branch experiment/agent-prefetch).
STARTUP_OPTIONS = {
    'prefix_warps': {'weight_prefix_warps': True},
    'bulk16k_dispatch': {'bulk_w1_prefetch_bytes': 16 * 1024,
                         'bulk_w1_prefetch_on_dispatch': True},
    'bulk64k_dispatch': {'bulk_w1_prefetch_bytes': 64 * 1024,
                         'bulk_w1_prefetch_on_dispatch': True},
    'post_init': {'post_init_prefetch': True},
    'relaxed_init': {'relaxed_init_barrier': True},
    'bulk64k_relaxed': {'bulk_w1_prefetch_bytes': 64 * 1024,
                        'bulk_w1_prefetch_on_dispatch': True,
                        'relaxed_init_barrier': True},
    'bulk64k_static_w1': {'bulk_w1_prefetch_bytes': 64 * 1024,
                          'bulk_w1_prefetch_on_dispatch': True,
                          'static_w1_prefix': True},
    'bulk64k_static_w1_relaxed': {'bulk_w1_prefetch_bytes': 64 * 1024,
                                  'bulk_w1_prefetch_on_dispatch': True,
                                  'static_w1_prefix': True,
                                  'relaxed_init_barrier': True},
    'bulk64k_static_w1_relaxed_noinit': {'bulk_w1_prefetch_bytes': 64 * 1024,
                                         'bulk_w1_prefetch_on_dispatch': True,
                                         'static_w1_prefix': True,
                                         'relaxed_init_barrier': True,
                                         'skip_init_prefetch_barrier': True},
    'bulk64k_static_w1_relaxed_post': {'bulk_w1_prefetch_bytes': 64 * 1024,
                                       'bulk_w1_prefetch_on_dispatch': True,
                                       'static_w1_prefix': True,
                                       'relaxed_init_barrier': True,
                                       'post_init_prefetch': True},
    'bulk64k_static_w1_relaxed_noinit_k2': {
        'bulk_w1_prefetch_bytes': 64 * 1024, 'bulk_w1_prefetch_on_dispatch': True,
        'static_w1_prefix': True, 'relaxed_init_barrier': True,
        'skip_init_prefetch_barrier': True, 'w1_prefetch_k_tiles': 2},
    'bulk64k_static_w1_relaxed_noinit_stripe1': {
        'bulk_w1_prefetch_bytes': 64 * 1024, 'bulk_w1_prefetch_on_dispatch': True,
        'static_w1_prefix': True, 'relaxed_init_barrier': True,
        'skip_init_prefetch_barrier': True, 'stripe_launch_delay': 1},
    'bulk64k_static_w1_relaxed_noinit_stripe2': {
        'bulk_w1_prefetch_bytes': 64 * 1024, 'bulk_w1_prefetch_on_dispatch': True,
        'static_w1_prefix': True, 'relaxed_init_barrier': True,
        'skip_init_prefetch_barrier': True, 'stripe_launch_delay': 2},
    'bulk64k_static_w1_all': {'bulk_w1_prefetch_bytes': 64 * 1024,
                              'bulk_w1_prefetch_on_dispatch': True,
                              'static_w1_prefix': True,
                              'relaxed_init_barrier': True,
                              'post_init_prefetch': True,
                              'skip_init_prefetch_barrier': True},
    'w1k8_bulk64k_static_w1_relaxed': {'w1_prefetch_k_tiles': 8,
                                       'bulk_w1_prefetch_bytes': 64 * 1024,
                                       'bulk_w1_prefetch_on_dispatch': True,
                                       'static_w1_prefix': True,
                                       'relaxed_init_barrier': True},
    'bulk64k_post_init': {'bulk_w1_prefetch_bytes': 64 * 1024,
                          'bulk_w1_prefetch_on_dispatch': True,
                          'post_init_prefetch': True},
    'bulk64k_post_relaxed': {'bulk_w1_prefetch_bytes': 64 * 1024,
                             'bulk_w1_prefetch_on_dispatch': True,
                             'post_init_prefetch': True, 'relaxed_init_barrier': True},
}
BACKFILL_TUNING |= {
    f'{NARROW_PRUNED}_{name}': (NARROW_PRUNED_MODE, NARROW_PRUNED_TUNING | overrides)
    for name, overrides in STARTUP_OPTIONS.items()
}
BACKFILL_TUNING[f'{NARROW_PRUNED}_w2_early_workers12_reuse_finalize_warps'] = (
    NARROW_PRUNED_MODE, NARROW_PRUNED_TUNING | {
        'decode_w2_early_workers': 12, 'reuse_finalize_warps': True})
for label, pipeline in {
    'stages10': {'decode_tma_stages': 10},
    'stages8': {'decode_tma_stages': 8},
    'stages6': {'decode_tma_stages': 6},
    'release2_stages10': {'release_group': 2, 'stage_alignment': 2, 'decode_tma_stages': 10},
    'release2_stages8': {'release_group': 2, 'stage_alignment': 2, 'decode_tma_stages': 8},
    'release4_stages8': {'release_group': 4, 'stage_alignment': 4, 'decode_tma_stages': 8},
    'acc3': {'decode_accumulator_stages': 3},
    'single_epilogue': {'single_epilogue_warpgroup': True},
}.items():
    BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_{label}'] = (
        NARROW_PRUNED_MODE,
        NARROW_PRUNED_TUNING | {'decode_descriptor_prefetch': 'off', 'flat_decode_graph': True} | pipeline,
    )
for stride in (3, 5, 7):
    option = f'{NARROW_PRUNED}_skip_descriptors_flat_workers{stride}'
    tuned = NARROW_PRUNED_TUNING | {
        'decode_descriptor_prefetch': 'off', 'flat_decode_graph': True, 'decode_cluster_stride': stride}
    BACKFILL_TUNING[option] = (NARROW_PRUNED_MODE, tuned)
    if stride in (5, 7):
        BACKFILL_TUNING[f'{option}_cluster_spread'] = (
            NARROW_PRUNED_MODE, tuned | {'decode_cluster_policy': 1})
for suffix, combine in {
    'combine_w2_issue': {'stripe_launch_delay': 1},
    'combine_w2_write': {'stripe_launch_delay': 2},
    'combine_lazy_poll': {'sentinel_poll_delay_ns': 512, 'sentinel_adaptive_poll': True},
    'combine_w2_issue_lazy_poll': {'stripe_launch_delay': 1, 'sentinel_poll_delay_ns': 512,
                                  'sentinel_adaptive_poll': True},
    'combine_fewer_warps': {'sentinel_combine_warps': 4, 'reuse_finalize_warps': True},
    'combine_fewer_ctas': {'sentinel_combine_sms': 32},
}.items():
    BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_{suffix}'] = (
        NARROW_PRUNED_MODE,
        NARROW_PRUNED_TUNING | {'decode_descriptor_prefetch': 'off', 'flat_decode_graph': True} | combine,
    )
for placement in ('workers5', 'cluster_spread'):
    option = f'{NARROW_PRUNED}_skip_descriptors_flat_{placement}'
    mode, tuned = BACKFILL_TUNING[option]
    BACKFILL_TUNING[f'{option}_lazy_poll'] = (
        mode, tuned | {'sentinel_poll_delay_ns': 512, 'sentinel_adaptive_poll': True})
for suffix, reset_at_entry in (('counts_ready', False), ('reset_ring_at_entry', True)):
    mode, tuned = BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_cuda_graph']
    BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_{suffix}'] = (
        mode, tuned | {'route_count_doorbell': True, 'reset_decode_ring_on_dispatch': reset_at_entry},
    )
for setting in ('activation-normal', 'activation-first', 'weights-normal', 'weights-last'):
    mode, tuned = BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_cuda_graph']
    BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_{setting.replace("-", "_")}'] = (
        mode, tuned | {'decode_operand_cache': setting})
for placement in ('skip_descriptor_prefetch', 'skip_descriptors_flat_cuda_graph'):
    option = f'{NARROW_PRUNED}_{placement}'
    mode, tuned = BACKFILL_TUNING[option]
    BACKFILL_TUNING[f'{option}_combine_at_block_start'] = (mode, tuned | {'combine_at_block_start': True})
    for suffix, roles in (('nvlink_dispatch', 1), ('nvlink_compute', 2), ('nvlink_both', 3)):
        BACKFILL_TUNING[f'{option}_{suffix}'] = (mode, tuned | {'decode_nvlink_schedule': roles})
    for suffix, scheduling in {
        'spread': {'decode_cluster_policy': 1},
        'nvlink_both_spread': {'decode_nvlink_schedule': 3, 'decode_cluster_policy': 1},
        'nvlink_both_block_start': {'decode_nvlink_schedule': 3, 'combine_at_block_start': True},
        'nvlink_both_spread_block_start': {'decode_nvlink_schedule': 3, 'decode_cluster_policy': 1,
                                         'combine_at_block_start': True},
    }.items():
        BACKFILL_TUNING[f'{option}_{suffix}'] = (mode, tuned | scheduling)
    for registers in (0, 136, 144):
        suffix = f'_combine_w4_compute_regs{registers}' if registers else '_combine_w4'
        BACKFILL_TUNING[f'{option}{suffix}'] = (
            mode, tuned | {'reuse_finalize_warps': True, 'sentinel_combine_warps': 4,
                           'decode_compute_max_registers': registers},
        )
for base in (f'{NARROW_PRUNED}_skip_descriptors_flat_cuda_graph',
             'sentinel_early_combine_w2_tile_major_bm48'):
    mode, tuned = BACKFILL_TUNING[base]
    tuned = tuned | {'flat_decode_graph': True, 'sentinel_combine_warps': 4,
                     'reuse_finalize_warps': False, 'omit_sentinel_peer_finish': False}
    for suffix, fuse in (('separate_combine', 0), ('fused_combine_join', 1), ('fused_combine_retire', 2)):
        BACKFILL_TUNING[f'{base}_{suffix}'] = (mode, tuned | {'fused_finalize_sentinel': fuse})
    BACKFILL_TUNING[f'{base}_separate_combine_cooperative'] = (mode, tuned | {'decode_cooperative_compute': True})
    for warps in (4, 8):
        control = tuned | {'sentinel_combine_warps': warps, 'reuse_finalize_warps': True,
                           'prune_decode_resets': True, 'omit_sentinel_peer_finish': True}
        BACKFILL_TUNING[f'{base}_cleanup_w{warps}'] = (mode, control)
        BACKFILL_TUNING[f'{base}_cleanup_w{warps}_single'] = (mode, control | {'single_cleanup_cta': True})
BM48_CLEANUP = 'sentinel_early_combine_w2_tile_major_bm48_cleanup_w8_single'
mode, tuned = BACKFILL_TUNING[BM48_CLEANUP]
BACKFILL_TUNING[f'{BM48_CLEANUP}_shared_route_counts'] = (
    mode, tuned | {'dispatch_shared_route_counts': True})
bm48_prefix = tuned | {'w1_prefetch_k_tiles': 4, 'w2_prefetch_k_tiles': 2,
                       'decode_descriptor_prefetch': 'off'}
BACKFILL_TUNING[f'{BM48_CLEANUP}_weight_prefix'] = (mode, bm48_prefix)
BACKFILL_TUNING[f'{BM48_CLEANUP}_weight_prefix_protect_w1'] = (
    mode, bm48_prefix | {'shared_priority_prefix': 1})
bm48_bulk_prefix = bm48_prefix | STARTUP_OPTIONS['bulk64k_static_w1_relaxed_noinit']
BACKFILL_TUNING[f'{BM48_CLEANUP}_bulk_prefix'] = (mode, bm48_bulk_prefix)
for label, overrides in {
    'short_w1': {'w1_prefetch_k_tiles': 2},
    'skip_w2_prefetch': {'w2_prefetch_k_tiles': 0},
    'bulk_w2_tail_2': {'bulk_w2_tail_experts': 2},
    'bulk_w2_tail_4': {'bulk_w2_tail_experts': 4},
    'after_route_publish': {'bulk_w1_prefetch_on_dispatch': 2},
    'shared_route_counts': {'dispatch_shared_route_counts': True},
    'posted_stripes': {'posted_route_sums': True, 'inline_route_indices': True,
                       'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True},
    'static_w2': {'static_w1_prefix': False, 'static_decode_wave': 8,
                  'static_w2_overlap': True, 'single_expert_tile_lookup': True},
}.items():
    BACKFILL_TUNING[f'{BM48_CLEANUP}_bulk_prefix_{label}'] = (mode, bm48_bulk_prefix | overrides)
for base in (f'{NARROW_PRUNED}_skip_descriptors_flat_cluster_spread',
             f'{NARROW_PRUNED}_skip_descriptors_flat_cluster_spread_lazy_poll'):
    mode, tuned = BACKFILL_TUNING[base]
    BACKFILL_TUNING[f'{base}_single_cleanup'] = (mode, tuned | {'single_cleanup_cta': True})
for base in (f'{NARROW_PRUNED}_skip_descriptor_prefetch',
             f'{NARROW_PRUNED}_skip_descriptors_flat_cluster_spread'):
    mode, tuned = BACKFILL_TUNING[base]
    for label, shared_phases in (('protect_shared_w1', 1), ('protect_shared_w1_w2', 2)):
        BACKFILL_TUNING[f'{base}_{label}'] = (
            mode, tuned | {'shared_priority_prefix': shared_phases})
FLAT_BULK_STARTUP = f'{NARROW_PRUNED}_skip_descriptors_flat_cluster_spread_bulk_startup'
mode, tuned = BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_cluster_spread']
flat_bulk_startup = tuned | STARTUP_OPTIONS['bulk64k_static_w1_relaxed_noinit']
BACKFILL_TUNING[FLAT_BULK_STARTUP] = (mode, flat_bulk_startup)
for label, overrides in {
    'protect_shared_w1': {'shared_priority_prefix': 1},
    'skip_w2_prefetch': {'w2_prefetch_k_tiles': 0},
    'focus_w1_prefix': {'w1_prefetch_k_tiles': 8, 'w1_prefix_expert_limit': 4},
    'shared_route_counts': {'dispatch_shared_route_counts': True},
    'route_count_doorbell': {'route_count_doorbell': True},
    'shared_route_counts_doorbell': {'dispatch_shared_route_counts': True,
                                     'route_count_doorbell': True},
    'after_route_publish': {'bulk_w1_prefetch_on_dispatch': 2},
    'after_route_publish_skip_w2': {'bulk_w1_prefetch_on_dispatch': 2, 'w2_prefetch_k_tiles': 0},
}.items():
    BACKFILL_TUNING[f'{FLAT_BULK_STARTUP}_{label}'] = (mode, flat_bulk_startup | overrides)
for base in (f'{NARROW_PRUNED}_skip_descriptors_flat_cuda_graph',
             f'{NARROW_PRUNED}_skip_descriptors_flat_cluster_spread'):
    mode, tuned = BACKFILL_TUNING[base]
    BACKFILL_TUNING[f'{base}_pair_shared_w1'] = (mode, tuned | {'pair_shared_w1': 1})
    BACKFILL_TUNING[f'{base}_pair_shared_w1_vector'] = (mode, tuned | {'pair_shared_w1': 2})
    for suffix, w2 in (('late', 1), ('rotated', 2)):
        BACKFILL_TUNING[f'{base}_shared_w2_{suffix}'] = (mode, tuned | {'decode_shared_w2': w2})
        BACKFILL_TUNING[f'{base}_pair_shared_w1_vector_{suffix}_w2'] = (
            mode, tuned | {'pair_shared_w1': 2, 'decode_shared_w2': w2})
for name, mask in (('even_stages', None), ('interleaved_weight_stages', 0),
                   ('paired_shared_w1', 1), ('paired_shared_w2', 2), ('paired_shared_weights', 3)):
    mode, tuned = BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_cuda_graph']
    BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_{name}'] = (
        mode, tuned | {'stage_alignment': 2, 'shared_weight_packet_mask': mask})
for suffix, group in (('four_stage_control', 0), ('decode_scale_k1', 1), ('decode_scale_k4', 4)):
    mode, tuned = BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_cuda_graph']
    BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_{suffix}'] = (
        mode, tuned | {'stage_alignment': 4, 'packed_routed_scales': group})
for suffix, helper in (('follower_scheduler', False), ('scale_load_helper', True)):
    mode, tuned = BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_cuda_graph']
    BACKFILL_TUNING[f'{NARROW_PRUNED}_skip_descriptors_flat_{suffix}'] = (
        mode, tuned | {'follower_scheduler': True, 'packed_scale_helper': helper,
                       'packed_routed_scales': 4 if helper else 0})
for release in (1, 2):
    option = f'{NARROW_PRUNED}_skip_descriptors_flat_scale_load_helper'
    mode, tuned = BACKFILL_TUNING[option]
    BACKFILL_TUNING[f'{option}_release{release}'] = (mode, tuned | {'packed_scale_release_group': release})
for base in (NARROW_PRUNED, f'{NARROW_PRUNED}_w2_early_workers12_reuse_finalize_warps'):
    mode, tuned = BACKFILL_TUNING[base]
    BACKFILL_TUNING[f'{base}_lossless_weight_scales'] = (
        mode, tuned | {'packed_routed_scales': True})
    for prefix in (2, 4, 8):
        BACKFILL_TUNING[f'{base}_w2_weight_prefix{prefix}'] = (
            mode, tuned | {'w2_b_issue_prefix': prefix})

NARROW_DIRECT = 'sentinel_early_w1k4_w2k2_bm32_narrow_direct_w1'
NARROW_DIRECT_MODE, NARROW_DIRECT_TUNING = BACKFILL_TUNING[NARROW_DIRECT]
BACKFILL_TUNING |= {
    f'{NARROW_DIRECT}_{name}': (NARROW_DIRECT_MODE, NARROW_DIRECT_TUNING | overrides)
    for name, overrides in {
        'reuse_finalize_warps': {'reuse_finalize_warps': True},
        'stripe2_pruned': {'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True,
                           'prune_decode_resets': True, 'omit_sentinel_peer_finish': True},
        'stripe2_pruned_reuse_finalize_warps': {
            'dispatch_pull_rows_per_cta': 2, 'dispatch_prefix_scan': True,
            'prune_decode_resets': True, 'omit_sentinel_peer_finish': True,
            'reuse_finalize_warps': True},
    }.items()
}
NARROW_DIRECT_REUSE = f'{NARROW_DIRECT}_stripe2_pruned_reuse_finalize_warps'
for split in (16, 24, 32):
    for rotation in (0, 2):
        pool = f'{NARROW_PRUNED}_split_w2_{split}'
        if rotation:
            pool += '_rotate_shared'
        BACKFILL_TUNING[pool] = (
            NARROW_PRUNED_MODE, NARROW_PRUNED_TUNING | {
                'decode_w2_clusters': split, 'decode_shared_w2': rotation, 'cache_routed_ranges': True})
NARROW_DIRECT_REUSE_MODE, NARROW_DIRECT_REUSE_TUNING = BACKFILL_TUNING[NARROW_DIRECT_REUSE]
BACKFILL_TUNING |= {
    f'{NARROW_DIRECT_REUSE}_w2_early_workers{workers}': (
        NARROW_DIRECT_REUSE_MODE, NARROW_DIRECT_REUSE_TUNING | {'decode_w2_early_workers': workers})
    for workers in (8, 12, 16)
}
for base in (
    'sentinel_early_combine',
    'sentinel_early_w1k4_w2k2',
    'sentinel_early_combine_w2_tile_major_bm48',
    'sentinel_early_combine_w2_tile_major_bm48_raw_w1_k8_w4_shared_w2_rotated_stripe2',
):
    mode, tuned = BACKFILL_TUNING.get(base, (base, {}))
    BACKFILL_TUNING[f'{base}_flat_cuda_graph'] = (mode, tuned | {'flat_decode_graph': True})
    BACKFILL_TUNING[f'{base}_lazy_poll'] = (
        mode, tuned | {'sentinel_poll_delay_ns': 512, 'sentinel_adaptive_poll': True})
    BACKFILL_TUNING[f'{base}_combine_w2_issue'] = (mode, tuned | {'stripe_launch_delay': 1})
    BACKFILL_TUNING[f'{base}_flat_spread_lazy_poll'] = (
        mode, tuned | {'flat_decode_graph': True, 'decode_cluster_policy': 1,
                       'sentinel_poll_delay_ns': 512, 'sentinel_adaptive_poll': True})
    BACKFILL_TUNING[f'{base}_combine_at_block_start'] = (mode, tuned | {'combine_at_block_start': True})
    BACKFILL_TUNING[f'{base}_flat_combine_at_block_start'] = (
        mode, tuned | {'flat_decode_graph': True, 'combine_at_block_start': True})
    BACKFILL_TUNING[f'{base}_spread_combine_at_block_start'] = (
        mode, tuned | {'decode_cluster_policy': 1, 'combine_at_block_start': True})
    BACKFILL_TUNING[f'{base}_nvlink_both'] = (mode, tuned | {'decode_nvlink_schedule': 3})
    BACKFILL_TUNING[f'{base}_nvlink_both_combine_at_block_start'] = (
        mode, tuned | {'decode_nvlink_schedule': 3, 'combine_at_block_start': True})
    BACKFILL_TUNING[f'{base}_single_cleanup'] = (mode, tuned | {'single_cleanup_cta': True})
    BACKFILL_TUNING[f'{base}_lazy_poll_single_cleanup'] = (
        mode, tuned | {'sentinel_poll_delay_ns': 512, 'sentinel_adaptive_poll': True,
                       'single_cleanup_cta': True})

WEIGHT_PLANE_PADDING = {
    f'{base}_plane_pad{padding}': (base, padding)
    for base in ('sentinel_early_w1k4_w2k2_bm32', 'sentinel_early_combine')
    for padding in (256, 2048, 8192)
}


def backfill_option_for(name):
    base, _, dispatch_sms = name.rpartition('_d')
    if dispatch_sms.isdecimal():
        name = base
    return max((option for option in (*BACKFILL_OPTIONS, *BACKFILL_TUNING, *WEIGHT_PLANE_PADDING)
                if name.endswith(f'_{option}')),
               key=len, default=None)


def pad_weight_plane(weight, padding):
    _, k_tiles, _, _ = weight.shape
    expert, plane, row, element = weight.stride()
    stride = (expert + k_tiles * padding, plane + padding, row, element)
    return torch.empty_strided(weight.shape, stride, dtype=weight.dtype,
                               device=weight.device).copy_(weight)


def pack_grouped_weights(weight, inner_k):
    experts, k_tiles, rows, size = weight.shape
    return (weight.reshape(experts, k_tiles // inner_k, inner_k, rows // 128, 128, size)
            .permute(0, 1, 3, 2, 4, 5).contiguous())


def pack_shared_weight_pairs(pair):
    """Pre-swizzle FP8 payload for one unswizzled TMA packet per two stages."""
    weight, scale = pair
    k_tiles, n, _ = weight.shape
    byte = torch.arange(16384, device=weight.device, dtype=torch.long)
    swizzled = byte ^ ((byte >> 3) & 112)
    tiles = weight.contiguous().view(torch.uint8).reshape(k_tiles, n // 128, 16384)
    tiles = tiles.permute(1, 0, 2).index_select(2, swizzled)
    packet = torch.zeros((n // 128, k_tiles // 2, 33280), device=weight.device, dtype=torch.uint8)
    packet[:, :, :32768] = tiles.reshape(n // 128, k_tiles // 2, 32768)
    scale_bytes = scale.T.contiguous().view(torch.uint8).reshape(k_tiles // 4, n // 128, 512)
    packet[:, ::2, 32768:] = scale_bytes.transpose(0, 1)
    return packet.reshape(-1, 256).view(torch.float8_e4m3fn)


def routed_scale_window(weights):
    device = weights[0][1].device
    shifts = torch.arange(4, device=device, dtype=torch.int32) * 8
    exponent_bounds = []
    for _, scale in weights:
        exponents = (scale.unsqueeze(-1) >> shifts) & 0xff
        exponent_bounds.append(torch.stack((exponents.amin(), -exponents.amax())))
    bounds = torch.stack(exponent_bounds).amin(dim=0)
    dist.all_reduce(bounds, op=dist.ReduceOp.MIN)
    low, negative_high = bounds.tolist()
    if -negative_high - low > 3 or low > 252:
        raise ValueError('routed scales do not fit the lossless four-exponent window')
    return low


def pack_routed_scale_tiles(scale, base, group=1):
    experts, n, k = scale.shape
    words = scale.transpose(1, 2).contiguous().reshape(experts, k, n // 128, 32, 4)
    deltas = words - base * 0x01010101
    packed = deltas[..., 0]
    for row in range(1, 4):
        packed = packed | (deltas[..., row] << (row * 2))
    if group == 4:
        return (packed.reshape(experts, k // 4, 4, n // 128, 32)
                .permute(0, 3, 1, 4, 2).contiguous())
    return packed.contiguous()


def backfill_vmm_variants(args, sms):
    variants = {}
    for mode in args.backfill_vmm_compression:
        compressed = f'compute_backfill_{sms}_vmm_{mode}'
        weights = f'vmm_compressed_{mode}'
        if not args.backfill_vmm_options_only:
            variants[compressed] = weights, 0, None
        for warps in args.backfill_vmm_dispatch_warps or (0,):
            variant = f'{compressed}_dispatch_w{warps}' if warps else compressed
            if not args.backfill_vmm_options_only:
                variants[variant] = weights, warps, None
            for option in args.backfill_vmm_options:
                name = f'{variant}_{option}'
                variants[name] = weights, warps, None
                for dispatch_sms in args.backfill_vmm_sms_sweep:
                    if sms <= dispatch_sms and dispatch_sms != args.vmm_primary_dispatch_sms:
                        variants[f'{name}_d{dispatch_sms}'] = weights, warps, dispatch_sms
    return variants


def backfill_variant_names(args):
    variants = {}
    for sms in args.compute_backfill_sms:
        base = f'compute_backfill_{sms}'
        variants[base] = sms
        for warps in args.backfill_dispatch_warps:
            variants[f'{base}_dispatch_w{warps}'] = sms
        for dispatch_sms in args.backfill_dispatch_sms:
            if sms <= dispatch_sms:
                variants[f'{base}_d{dispatch_sms}'] = sms
        variants.update(dict.fromkeys(backfill_vmm_variants(args, sms), sms))
        for option in args.backfill_options:
            variants[f'{base}_{option}'] = sms
            for dispatch_sms in args.backfill_option_sms_sweep:
                if sms <= dispatch_sms:
                    variants[f'{base}_{option}_d{dispatch_sms}'] = sms
    return variants


def percentile(samples, fraction):
    return sorted(samples)[max(0, math.ceil(len(samples) * fraction) - 1)]


def audit_routed_encoding(routed_weights):
    def summarize(counts, raw_bits):
        total = sum(counts)
        frequencies = [count / total for count in counts if count]
        return {
            'sample_count': total,
            'histogram': counts,
            'stored_bits_per_value': raw_bits,
            'entropy_bits_per_value': -sum(p * math.log2(p) for p in frequencies),
        }

    results = {}
    # Sample across tile boundaries and different packed positions without
    # materializing individual FP4 values for the full expert allocation.
    sample_stride = 31
    for phase, (weights, scales) in zip(('w1', 'w2'), routed_weights):
        packed = weights.contiguous().view(torch.uint8).flatten()[::sample_stride].long()
        symbols = torch.cat((packed & 15, packed >> 4))
        scale_bytes = scales.contiguous().view(torch.uint8).flatten()[::sample_stride].long()
        weight_counts = torch.bincount(symbols, minlength=16)
        scale_counts = torch.bincount(scale_bytes, minlength=256)
        dist.all_reduce(weight_counts)
        dist.all_reduce(scale_counts)
        histogram = weight_counts.tolist()
        weight_stats = summarize(histogram, 4)
        common = sorted(histogram, reverse=True)[:3]
        escape_fraction = 1 - sum(common) / sum(histogram)
        # Two bits per symbol plus one raw FP4 nibble for each escape.
        weight_stats['three_symbols_and_escape_bits_per_value'] = 2 + 4 * escape_fraction
        results[phase] = {
            'weights': weight_stats,
            'scale_bytes': summarize(scale_counts.tolist(), 8),
            'routed_weight_bytes_per_rank': weights.numel() * weights.element_size(),
            'routed_scale_bytes_per_rank': scales.numel() * scales.element_size(),
        }
    return results


def build_decode_tokens(args, tokens, rank, world, device):
    active = args.routing_active_experts_per_rank
    offset = args.routing_active_expert_offset
    if not active:
        if offset:
            raise ValueError('active expert offset requires a concentrated routing workload')
        return build_tokens(tokens, args.hidden, args.num_experts, args.num_topk, rank, device)
    local_experts = args.num_experts // world
    if (args.num_experts % world or not 1 <= active <= local_experts or
            not 0 <= offset <= local_experts - active or
            active * world < args.num_topk):
        raise ValueError('active expert count must fit every rank and preserve unique top-k routes')
    x, indices, scores = build_tokens(tokens, args.hidden, world * active,
                                      args.num_topk, rank, device)
    # Keep the same logical expert allocation. Choose routes inside the
    # specified subset to test how concentrated gates change weight rereads.
    indices = indices // active * local_experts + indices % active + offset
    return x, indices, scores


def audit_decode_routing(args, device):
    """Replay the benchmark's routing seeds and count logical weight rereads."""
    weight_elements = 3 * args.hidden * args.intermediate_hidden
    routed_bytes = weight_elements // 2 + weight_elements // 32
    shared_bytes = weight_elements + weight_elements // args.shared_weight_gran_k
    observations = []
    for world in args.audit_route_worlds:
        if args.num_experts % world:
            raise ValueError('audit world size must divide the expert count')
        for tokens in args.tokens:
            counts = torch.zeros(args.num_experts, device=device, dtype=torch.int64)
            for rank in range(world):
                _, ids, _ = build_decode_tokens(args, tokens, rank, world, device)
                counts += ids.flatten().bincount(minlength=args.num_experts)
            rows = counts.view(world, -1).tolist()
            active_counts = [sum(count > 0 for count in rank_rows) for rank_rows in rows]
            assert sum(map(sum, rows)) == world * tokens * args.num_topk
            tilings = {}
            for m in (16, 32, 48, 64):
                repeated = [sum(max(0, (count + m - 1) // m - 1) for count in rank_rows)
                            for rank_rows in rows]
                shared_repeated = (tokens + m - 1) // m - 1
                tilings[m] = {
                    'extra_routed_m_tiles': repeated,
                    'extra_shared_m_tiles': shared_repeated,
                    'max_logical_reissued_bytes': max(repeated) * routed_bytes + shared_repeated * shared_bytes,
                }
            observations.append({
                'world_size': world, 'tokens': tokens,
                'rows_by_rank_expert': rows,
                'inactive_experts': sum(count == 0 for rank_rows in rows for count in rank_rows),
                'max_logical_weight_bytes': max(active_counts) * routed_bytes + shared_bytes,
                'max_rows_per_expert': max(max(rank_rows) for rank_rows in rows),
                'm_tilings': tilings,
            })
    return {'routed_weight_bytes_per_expert': routed_bytes,
            'shared_weight_bytes_per_expert': shared_bytes, 'cases': observations}


def read_trace(trace_tensor):
    records = trace_tensor.cpu()
    populated = records[..., 0].nonzero(as_tuple=True)
    values = records[populated].tolist()
    begin = min(row[0] for row in values)
    events = []
    for (role, cta, slot), (stamp, tag) in zip(zip(*(v.tolist() for v in populated)), values):
        event = {'role': role + 1, 'cta': cta, 'us': (stamp - begin) / 1000}
        if role == 1 and slot >= TRACE_TASK_BASE:
            field, task = divmod(slot - TRACE_TASK_BASE, TRACE_TASKS)
            mark, phase = TRACE_TASK_FIELDS[field], TASK_PHASES[tag & 0xff]
            event.update(mark=mark, task=task, phase=phase)
            if mark in ('mma_operand_wait', 'mma_accum_credit_wait'):
                event['duration_us'] = (tag >> 8) / 1000
            elif mark == 'mma_operand_readiness':
                event['readiness'] = {name: (tag >> (8 + 12 * index)) & 0xfff
                                      for index, name in enumerate(READINESS)}
            else:
                event.update(expert=(tag >> 8) & 0xfff, n=(tag >> 20) & 0xfff,
                             pool=(tag >> 32) & 0xffffff, valid_m=(tag >> 56) & 0xff)
        elif role == 1 and slot >= 4:
            names = ('startup_pre_init', 'startup_cluster1', 'startup_tmem',
                     'startup_cluster2', 'startup_pdl', 'startup_descriptors',
                     'startup_weight_prefix', 'startup_scheduler', 'startup_entry',
                     'startup_warp2_init', 'startup_warp0_init', 'startup_warp1_init',
                     'startup_reserved')
            event.update(mark=names[slot - 4], offset_us=tag / 1000)
        elif role == 1:
            event.update(mark=('entry', 'worker_exit', 'fused_join_done', 'fused_cleanup')[slot], sm=tag)
        elif role == 0:
            names = ('entry', 'published_local_routes', 'observed_all_counts', 'pull_worker0_done')
            event.update(mark=names[slot])
        elif role == 2 and slot >= 4:
            names = ('first_chunk_wait', 'first_chunk_peers_ready', 'last_chunk_submitted')
            field, warp = divmod(slot - 4, 8)
            event.update(mark=names[field], warp=warp, chunk=tag)
        else:
            event.update(mark='entry')
        events.append(event)
    return sorted(events, key=lambda event: event['us'])


def summarize_trace(events):
    marks = {}
    tasks = {}
    intervals = {}
    active_per_sm = {}
    readiness = {}
    max_workers_per_sm = 0
    for event in events:
        if 'sm' in event:
            sm = event['sm']
            active_per_sm[sm] = active_per_sm.get(sm, 0) + (1 if event['mark'] == 'entry' else -1)
            max_workers_per_sm = max(max_workers_per_sm, active_per_sm[sm])
        name = f"r{event['role']}:{event.get('phase', '')}:{event['mark']}"
        marks.setdefault(name, []).append(event['us'])
        if 'task' in event:
            tasks.setdefault((event['cta'], event['task'], event['phase']), {})[
                event['mark']] = event['us']
        if event['mark'] in ('mma_operand_wait', 'mma_accum_credit_wait'):
            intervals.setdefault(f"{event['phase']}:{event['mark']}", []).append(
                event['duration_us'])
        if event['mark'] == 'mma_operand_readiness':
            counts = readiness.setdefault(event['phase'], dict.fromkeys(READINESS, 0))
            for name, count in event['readiness'].items():
                counts[name] += count
    pairs = {
        'wait_activation_k0': ('a_task', 'a_k0_ready'),
        'activation_loader_k0': ('a_k0_ready', 'a_k0_issued'),
        'weight_loader_k0': ('b_task', 'b_k0_issued'),
        'activation_issue_span': ('a_k0_issued', 'a_last_k_issued'),
        'weight_issue_span': ('b_k0_issued', 'b_last_k_issued'),
        'mma_to_first_operands': ('mma_task', 'mma_k0_ready'),
        'first_to_last_operands': ('mma_k0_ready', 'mma_last_k_ready'),
        'final_mma_to_accum_ready': ('mma_last_k_issued', 'epilogue_begin'),
        'epilogue': ('epilogue_begin', 'epilogue_end'),
    }
    for (_, _, phase), stamps in tasks.items():
        for name, (start, end) in pairs.items():
            if start in stamps and end in stamps:
                intervals.setdefault(f'{phase}:{name}', []).append(stamps[end] - stamps[start])
    return {
        'max_observed_workers_per_sm': max_workers_per_sm,
        'mma_operand_readiness': readiness,
        'marks': {name: {'first_us': min(rows), 'last_us': max(rows), 'n': len(rows)}
                  for name, rows in marks.items()},
        'intervals': {name: {'p50_us': median(rows), 'p95_us': percentile(rows, 0.95), 'n': len(rows)}
                      for name, rows in intervals.items()},
    }


def capture(launch, repeats, device, rank_start_flag=None):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for name in repeats:
            launch(name)
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for name in repeats[:-1]:
            launch(name)
        if rank_start_flag is not None:
            dist.all_reduce(rank_start_flag, op=dist.ReduceOp.MAX)
        start.record()
        launch(repeats[-1])
        end.record()
    return graph, start, end


def run_case(args, tokens, rank, world, device, target, decoys, recipe, colocated, padded_weights,
             vmm_weights, k_major_weights, packed_scale_base, mega_weight_sets=()):
    x, ids, scores = build_decode_tokens(args, tokens, rank, world, device)
    routing_rows = None
    if args.routing_active_experts_per_rank:
        received = ids.flatten().bincount(minlength=args.num_experts)
        dist.all_reduce(received)
        routing_rows = received.view(world, -1).tolist()
    x8, scales = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    remote_payload = None
    if args.dedup_activations or args.source_activation_owner:
        assignments, unique_rows = 0, 0
        local_experts = args.num_experts // world
        for row in ids.tolist():
            peers = [expert // local_experts for expert in row if expert // local_experts != rank]
            assignments += len(peers)
            unique_rows += len(set(peers))
        remote_payload = {
            'stock_fp8_bytes': assignments * args.hidden,
            'dedup_fp8_bytes': unique_rows * args.hidden,
        }
    variants = {'stock': 'off', 'route_reduce': 'route-reduce'}
    stock_vmm = {f'stock_vmm_{mode}': 'vmm_plain' if mode == 'plain' else f'vmm_compressed_{mode}'
                 for mode in args.vmm_stock_control}
    variants.update(dict.fromkeys(stock_vmm, 'off'))
    accumulator_stages = {f'accumulator_stages_{n}': n for n in args.accumulator_stages}
    if accumulator_stages:
        variants['route_only'] = 'route'
        variants.update(dict.fromkeys(accumulator_stages, 'route'))
    if vmm_weights:
        variants['route_only'] = 'route'
        if not args.backfill_vmm_options_only:
            variants.update((name, 'route') for name in VMM_WEIGHT_MODES if name in vmm_weights)
    if k_major_weights:
        variants['route_only'] = 'route'
        variants.update(dict.fromkeys(k_major_weights, 'route'))
    activation_prefixes = {f'activation_prefix_{n}': n for n in args.activation_prefix_bytes}
    parallel_prefixes = {f'activation_prefix_parallel_{n}': n for n in args.activation_prefix_bytes
                         if args.parallel_activation_prefix}
    activation_prefixes.update(parallel_prefixes)
    if activation_prefixes:
        variants['route_only'] = 'route'
        variants.update(dict.fromkeys(activation_prefixes, 'route'))
    dispatch_sms_sweep = {f'route_dispatch_sms_{n}': n for n in args.dispatch_sms_sweep}
    variants.update(dict.fromkeys(dispatch_sms_sweep, 'route'))
    variants.update(dict.fromkeys(padded_weights, 'off'))
    if args.routed_weight_cache_sweep:
        variants.update(dict.fromkeys(WEIGHT_CACHE_MODES, 'off'))
    w1_l2_prefetch = {f'w1_l2_prefetch_{n}': n for n in args.w1_l2_prefetch_experts}
    variants.update(dict.fromkeys(w1_l2_prefetch, 'off'))
    experts = {}
    speculative_heads = set()
    if args.early_experts:
        experts = {'fixed_slots': 0} | {
            f'early_{count}': count for count in args.early_experts
        }
        if args.speculative_w1:
            speculative = {f'speculative_{count}': count for count in args.early_experts}
            speculative_heads = set(speculative)
            experts.update(speculative)
        variants.update(dict.fromkeys(experts, 'route-reduce'))
    credits = {}
    for group in args.group_release:
        credits[f'align_{group}'] = (group, 1)
        credits[f'release_{group}'] = (group, group)
    variants.update(dict.fromkeys(credits, 'route-reduce'))
    if args.dual_weight_warps:
        credits.update(single_weight_warp_aligned=(2, 1), dual_weight_warps=(2, 1))
        variants.update(single_weight_warp_aligned='route', dual_weight_warps='route',
                        route_only='route')
    if args.local_mma:
        variants.update(local_mma='route', narrow_local_mma='route', route_only='route')
    if args.local_mma_n8:
        variants.update(narrow_local_mma='route', narrow_local_mma_n8='route',
                        narrow_local_mma_n8_short_load='route', route_only='route')
    if args.independent_compute_ctas:
        variants.update(independent_compute_ctas='route', narrow_local_mma='route', route_only='route')
    if args.independent_w1_first:
        variants.update(independent_w1_first='route', narrow_local_mma='route', route_only='route')
    direct_stores = set()
    if args.direct_w1_store:
        direct_stores.add('direct_w1')
        variants['direct_w1'] = 'route-reduce'
    narrow = set()
    if args.narrow_bm32:
        narrow.add('narrow_bm32')
        variants['narrow_bm32'] = 'route-reduce'
        if args.direct_w1_store:
            narrow.add('direct_narrow')
            direct_stores.add('direct_narrow')
            variants['direct_narrow'] = 'route-reduce'
    if args.single_mma_lane:
        variants['single_mma_lane'] = 'route-reduce'
    mma_issue = {}
    if args.mma_issue_sweep:
        mma_issue = {f'k128_u{unroll}': unroll for unroll in (2, 4, 8)}
        variants.update(dict.fromkeys(mma_issue, 'route-reduce'))
    if args.ready_tile_order:
        variants['ready_tile_order'] = 'route-reduce'
    if args.programmatic_finalize:
        variants['programmatic_finalize'] = 'route-reduce'
        if args.ready_tile_order:
            variants['ready_programmatic'] = 'route-reduce'
    if args.expert_ready_combine:
        variants.update(route_only='route', expert_combine_tma='route', expert_combine_vector='route-reduce')
    if args.recycle_dispatch_sms:
        variants['recycle_dispatch_sms'] = 'route-reduce'
    compute_backfill = backfill_variant_names(args)
    backfill_dispatch_warps = {
        f'compute_backfill_{sms}_dispatch_w{warps}': warps
        for sms in args.compute_backfill_sms for warps in args.backfill_dispatch_warps
    }
    if args.backfill_option_dispatch_warps:
        backfill_dispatch_warps.update(
            (f'compute_backfill_{sms}_{option}', args.backfill_option_dispatch_warps)
            for sms in args.compute_backfill_sms for option in args.backfill_options)
        backfill_dispatch_warps.update(
            (f'compute_backfill_{sms}_{option}_d{dispatch_sms}', args.backfill_option_dispatch_warps)
            for sms in args.compute_backfill_sms for option in args.backfill_options
            for dispatch_sms in args.backfill_option_sms_sweep if sms <= dispatch_sms)
    backfill_dispatch_sms = {
        f'compute_backfill_{sms}_d{dispatch_sms}': dispatch_sms
        for sms in args.compute_backfill_sms for dispatch_sms in args.backfill_dispatch_sms
        if sms <= dispatch_sms
    }
    backfill_dispatch_sms.update(
        (f'compute_backfill_{sms}_{option}_d{dispatch_sms}', dispatch_sms)
        for sms in args.compute_backfill_sms for option in args.backfill_options
        for dispatch_sms in args.backfill_option_sms_sweep if sms <= dispatch_sms)
    compressed = {name: config for sms in args.compute_backfill_sms
                  for name, config in backfill_vmm_variants(args, sms).items()}
    backfill_vmm = {name: mode for name, (mode, _, _) in compressed.items()}
    backfill_dispatch_warps.update(
        (name, warps) for name, (_, warps, _) in compressed.items() if warps)
    if compute_backfill:
        variants['route_only'] = 'route'
        vector_options = (SENTINEL_OPTIONS | {'vector_combine'} |
                          BACKFILL_TUNING.keys() | WEIGHT_PLANE_PADDING.keys())
        variants.update((name, 'route-reduce' if backfill_option_for(name) in vector_options else 'route')
                        for name in compute_backfill)
    if args.static_w1_prefix:
        variants['static_w1_prefix'] = 'route-reduce'
    if args.direct_w2_store:
        variants['direct_w2'] = 'route-reduce'
        variants.update(direct_w2_route='route', route_only='route')
        if args.direct_w1_store:
            variants['direct_both_route'] = 'route'
            direct_stores.add('direct_both_route')
        if args.recycle_dispatch_sms:
            variants['direct_w2_recycle'] = 'route-reduce'
    compact = {}
    if args.compact_compute:
        compact = {'compact_grid_1': 1, 'compact_grid_2': 2}
        variants.update(dict.fromkeys(compact, 'route-reduce'))
    dispatch_warps = {f'dispatch_w{w}': w for w in args.dispatch_warp_sweep}
    recycled = {'recycle_dispatch_sms'} if args.recycle_dispatch_sms else set()
    if args.recycle_dispatch_sms:
        swept_recycled = {f'{n}_recycle': w for n, w in dispatch_warps.items()}
        recycled.update(swept_recycled)
        dispatch_warps.update(swept_recycled)
    variants.update(dict.fromkeys(dispatch_warps, 'route-reduce'))
    dispatch_warps.update(backfill_dispatch_warps)
    stripe = set()
    if args.early_stripe_combine:
        stripe.add('stripe_combine')
        if args.recycle_dispatch_sms:
            stripe.add('stripe_recycle')
            recycled.add('stripe_recycle')
        variants.update(dict.fromkeys(sorted(stripe), 'route-reduce'))
    multicast_stripes = MULTICAST_STRIPE_VARIANTS if args.multicast_stripe_combine else {}
    if args.single_epilogue_warpgroup:
        variants['single_epilogue'] = 'route'
        if args.multicast_stripe_combine:
            multicast_stripes = multicast_stripes | {'single_epilogue_stripes': 0}
    grouped_stripes = {f'multicast_stripes_tiles_{n}': n for n in args.stripe_tiles}
    multicast_stripes = multicast_stripes | dict.fromkeys(grouped_stripes, 0)
    stripe.update(multicast_stripes)
    variants.update(dict.fromkeys(multicast_stripes, 'route-reduce'))
    if args.dedup_activations:
        variants['dedup_activations'] = 'route-reduce'
    if args.source_activation_owner:
        variants['dedup_source_owner'] = 'route-reduce'
    lookahead = {f'weight_lookahead_{k}': k for k in args.weight_lookahead}
    variants.update(dict.fromkeys(lookahead, 'route-reduce'))
    packed_counts = set()
    if args.packed_route_counts:
        packed_counts.add('packed_counts')
        if args.recycle_dispatch_sms:
            packed_counts.add('packed_recycle')
            recycled.add('packed_recycle')
        variants.update(dict.fromkeys(sorted(packed_counts), 'route-reduce'))
    if args.grouped_route_counts:
        variants['grouped_route_counts'] = 'route-reduce'
    if args.grouped_atomic_route_counts:
        variants['grouped_atomic_route_counts'] = 'route-reduce'
    if args.posted_route_sums:
        variants.update(posted_route_stock='off', posted_route_single='route-reduce')
        if args.grouped_atomic_route_counts:
            variants['posted_route_grouped'] = 'route-reduce'
    combine_groups = {f'combine_load_{n}': n for n in args.combine_load_groups}
    variants.update(dict.fromkeys(combine_groups, 'route-reduce'))
    prefix_scan = set()
    if args.dispatch_prefix_scan:
        prefix_scan.add('dispatch_prefix_scan')
        if args.packed_route_counts:
            prefix_scan.add('packed_prefix_scan')
            packed_counts.add('packed_prefix_scan')
        variants.update(dict.fromkeys(sorted(prefix_scan), 'route-reduce'))
    if args.decode_route_stack:
        variants.update(dict.fromkeys(ROUTE_STACK_VARIANTS, 'route'))
        prefix_scan.update(name for name in ROUTE_STACK_VARIANTS if 'prefix' in name)
        direct_stores.add('inline_route_prefix_direct')
        narrow.add('inline_route_prefix_direct')
    static_waves = {f'static_wave_{n}': n for n in args.static_decode_waves}
    variants.update(dict.fromkeys(static_waves, 'route-reduce'))
    compact_workspace = set()
    token_major = set()
    if args.compact_decode_workspace:
        compact_workspace.update(('compact_workspace_stock', 'compact_workspace_route',
                                  'compact_workspace_route_only'))
        variants.update(compact_workspace_stock='off', compact_workspace_route='route-reduce',
                        compact_workspace_route_only='route', route_only='route')
    if args.token_major_combine:
        token_major.add('token_major_combine')
        if args.compact_decode_workspace:
            token_major.add('compact_token_major_combine')
            compact_workspace.add('compact_token_major_combine')
        variants.update(dict.fromkeys(sorted(token_major), 'route-reduce'))
    receiver_routes = set()
    atomic_free_receiver_routes = set()
    warp_receiver_routes = set()
    if args.pull_route_metadata:
        receiver_routes.add('receiver_routes')
        if args.multicast_pull_routes:
            receiver_routes.add('receiver_multicast')
        if args.dispatch_prefix_scan:
            receiver_routes.add('receiver_prefix')
            prefix_scan.add('receiver_prefix')
            if args.multicast_pull_routes:
                receiver_routes.add('receiver_multicast_prefix')
                prefix_scan.add('receiver_multicast_prefix')
        variants.update(dict.fromkeys(sorted(receiver_routes), 'route-reduce'))
    if args.receiver_posted_counts:
        atomic_free_receiver_routes.update(('receiver_posted', 'receiver_multicast_posted'))
        variants.update(dict.fromkeys(sorted(atomic_free_receiver_routes), 'route'))
    if args.warp_pull_routes:
        warp_receiver_routes.update(('receiver_warp', 'receiver_multicast_warp'))
        if args.dispatch_prefix_scan:
            prefix_scan.update(('receiver_warp', 'receiver_multicast_warp'))
        variants.update(dict.fromkeys(sorted(warp_receiver_routes), 'route'))
    atomic_free_receiver_routes.update(warp_receiver_routes)
    receiver_routes.update(atomic_free_receiver_routes)
    colocated_variants = ()
    if colocated is not None:
        colocated_variants = ('colocated_scales_stock', 'colocated_scales_route')
        variants.update(colocated_scales_stock='off', colocated_scales_route='route-reduce')
    w1_schedules = W1_SCHEDULES if args.w1_schedule_sweep else {}
    variants.update(dict.fromkeys(w1_schedules, 'off'))
    if args.shared_w2_schedule_sweep:
        variants.update(dict.fromkeys(SHARED_W2_SCHEDULES, 'off'))
    if args.inline_route_scores:
        variants.update(inline_scores_stock='off', inline_scores_single='route-reduce')
    if args.packed_decode_activations:
        variants.update(packed_input_stock='off', packed_input_single='route-reduce',
                        packed_input_inline='route-reduce')
    if args.route_count_doorbell:
        variants.update(count_doorbell_stock='off', count_doorbell_single='route-reduce')
    if args.rank_route_doorbell:
        variants.update(rank_route_doorbell='route', inline_rank_route_doorbell='route')
    if args.multicast_route_counts:
        variants.update(multicast_counts='route-reduce', multicast_doorbell='route-reduce')
    if args.multicast_combine_ready:
        variants['multicast_combine'] = 'off'
        if args.multicast_route_counts:
            variants['multicast_counts_and_combine'] = 'route-reduce'
    if args.multicast_decode_input:
        variants.update(multicast_input='off', multicast_input_and_scores='off')
    if args.peer_decode_input or args.peer_decode_direct or args.inline_peer_decode_direct:
        variants.update(peer_pull_single='route', peer_push_single='route',
                        inline_peer_push_single='route')
    if args.peer_decode_direct:
        variants['peer_direct'] = 'route'
    if args.inline_peer_decode_direct:
        variants['inline_peer_direct'] = 'route'
    if args.push_decode_activations or args.push_reservation_sweep:
        variants.update(push_fixed_slots='off', push_decode_activations='off')
        experts['push_fixed_slots'] = 0
    if args.push_reservation_sweep:
        variants.update(push_overlap_reservation='off', push_reserved_spans='off')
    if args.push_vector_io or args.push_early_counts:
        variants.update(push_tma_route='route', push_tma_overlap_route='route', route_only='route')
    if args.push_vector_io:
        variants['push_vector_route'] = 'route'
    if args.push_early_counts:
        variants.update(push_reserved_spans_route='route', push_early_counts='route',
                        push_row_ready_flags='route')
    push_scale_receivers = {f'push_packed_scales_r{n}': n for n in args.push_scale_receiver_sms}
    if push_scale_receivers:
        variants.update(push_row_ready_flags='route', route_only='route')
        variants.update(dict.fromkeys(push_scale_receivers, 'route'))
    if args.compose_decode:
        variants.update(DECODE_COMBINATIONS)
        direct_stores.update(('direct_w1_stock', 'inline_direct_w1', 'inline_direct_narrow',
                              'inline_direct_narrow_multicast'))
        narrow.update(('inline_direct_narrow', 'inline_direct_narrow_multicast'))
    if args.compare_primary_context:
        variants.update(PRIMARY_CONTEXT_VARIANTS)
        direct_stores.add('primary_direct_narrow')
        narrow.add('primary_direct_narrow')
    fused_workers = set()
    if args.fused_dispatch_compute:
        fused_workers.update(('fused_dispatch_compute', 'inline_fused_dispatch_compute'))
        variants.update(dict.fromkeys(sorted(fused_workers), 'off'))
        variants['dispatch_w12_stock'] = 'off'
        dispatch_warps['dispatch_w12_stock'] = 12
    variants.update((name, FUSED_FINALIZE_VARIANTS[name]) for name in args.fused_finalize_variants)
    fused_reused = ('fused_worker_reuse', 'inline_fused_worker_reuse') if args.fused_worker_reuse else ()
    variants.update(dict.fromkeys(fused_reused, 'off'))
    tma_stages = {f'tma_stages_{n}': n for n in args.decode_tma_stages}
    variants.update(dict.fromkeys(tma_stages, 'off'))
    w2_clusters = {f'w2_startup_clusters_{n}': n for n in args.decode_w2_clusters}
    variants.update(dict.fromkeys(w2_clusters, 'off'))
    w2_ready = {f'w2_ready_k{n}': n for n in args.decode_w2_ready_k}
    vmm_sms_variants = {name: dispatch_sms for name, (_, _, dispatch_sms) in compressed.items()
                        if dispatch_sms is not None}
    vmm_aliases = backfill_vmm | stock_vmm
    grouped_variants = {}
    packed_scale_variants = {}
    for candidate, mode in backfill_vmm.items():
        option = backfill_option_for(candidate)
        if option in WEIGHT_PLANE_PADDING:
            _, padding = WEIGHT_PLANE_PADDING[option]
            vmm_aliases[candidate] = f'{mode}_plane_pad{padding}'
        tuning = BACKFILL_TUNING.get(option, (None, {}))[1]
        if tuning.get('vector_pull'):
            variants[candidate] = 'vector'
        if tuning.get('prefetch_experts'):
            w1_l2_prefetch[candidate] = tuning['prefetch_experts']
        grouped_weights = tuning.get('grouped_routed_weights')
        if grouped_weights:
            inner_k = tuning.get('grouped_weight_inner_k', 2)
            vmm_aliases[candidate] = f'{mode}_grouped{grouped_weights}_k{inner_k}'
            grouped_variants[candidate] = grouped_weights, inner_k
        packed_scale_group = int(tuning.get('packed_routed_scales', False))
        if packed_scale_group:
            vmm_aliases[candidate] = f'{mode}_packed_sf{packed_scale_group}'
            packed_scale_variants[candidate] = (packed_scale_group, tuning.get('packed_scale_helper', False),
                                                 tuning.get('packed_scale_release_group', 4))
    for mode in ('vmm_plain', 'vmm_compressed_scales'):
        for sms in args.vmm_dispatch_sweep:
            name = f'{mode}_d{sms}'
            vmm_aliases[name] = mode
            vmm_sms_variants[name] = sms
            variants[name] = 'route'
    if args.compact_vmm_sweep:
        compact_workspace.update(vmm_sms_variants)
    if args.compress_w2_ready_scales:
        vmm_aliases.update((f'w2_ready_k{n}_compressed_scales', 'vmm_compressed_scales')
                           for n in args.decode_w2_ready_k)
        w2_ready.update((f'w2_ready_k{n}_compressed_scales', n) for n in args.decode_w2_ready_k)
    if w2_ready:
        variants['route_only'] = 'route'
        variants.update(dict.fromkeys(w2_ready, 'route'))
    buffers = {}
    outputs = {}
    block_m_by_variant = {}
    decoy_names = {(0, 2): 'decoy'}
    decoy_names.update({(mask, inner_k): f'decoy_grouped{mask}_k{inner_k}'
                        for mask, inner_k in sorted(set(grouped_variants.values()))})
    decoy_weight_layouts = {name: mask for (mask, _), name in decoy_names.items()}
    packed_decoy = {f'decoy_packed_sf{group}_helper{int(helper)}_release{release}': (group, helper, release)
                    for group, helper, release in sorted(set(packed_scale_variants.values()))}
    packed_decoy_names = {mode: name for name, mode in packed_decoy.items()}
    names = (*variants, *(('trace',) if args.trace_replays else ()), *decoy_weight_layouts, *packed_decoy)
    shared_packets = None
    for name in names:
        candidate = args.trace_variant if name == 'trace' else name
        backfill_option = backfill_option_for(candidate) if candidate in compute_backfill else None
        if backfill_option in WEIGHT_PLANE_PADDING:
            backfill_option, _ = WEIGHT_PLANE_PADDING[backfill_option]
        backfill_option, tuning = BACKFILL_TUNING.get(backfill_option, (backfill_option, {}))
        fused_finalize_sentinel = tuning.get('fused_finalize_sentinel', 0)
        packet_mask = tuning.get('shared_weight_packet_mask')
        if packet_mask and shared_packets is None:
            shared_packets = tuple(pack_shared_weight_pairs(pair) for pair in target[1])
        packets = None
        if packet_mask is not None:
            packets = tuple(shared_packets[phase] if packet_mask & (1 << phase) else None for phase in range(2))
        decoy_scale_group, decoy_scale_helper, decoy_scale_release = packed_decoy.get(name, (0, False, 4))
        packed_scale_group = int(tuning.get('packed_routed_scales') or decoy_scale_group)
        packed_scale_helper = tuning.get('packed_scale_helper', decoy_scale_helper)
        stage_alignment, release_group = credits.get(candidate, (1, 1))
        if 'release_group' in tuning:
            stage_alignment = release_group = tuning['release_group']
        stage_alignment = tuning.get('stage_alignment', stage_alignment)
        if packed_scale_group == 4 and not packed_scale_helper:
            stage_alignment = 4
        use_compact_workspace = candidate in compact_workspace
        capacity = min(args.capacity, 64) if use_compact_workspace else args.capacity
        warmup, cadence = w1_schedules.get(candidate, (0, 1))
        if candidate == 'independent_w1_first' or backfill_option == 'w1_first':
            warmup = 4
        elif backfill_option == 'sentinel_early_w1k4_w2_ready_k16_warm1':
            warmup = 1
        warmup = tuning.get('decode_w1_warmup', warmup)
        buffers[name] = GCSymmBuffer(
            dist.group.WORLD, args.num_experts, capacity, args.num_topk,
            args.hidden, args.intermediate_hidden, args.num_shared_experts,
            decode_trace=name == 'trace',
            trace_stride=args.trace_stride, trace_operand_wait=args.trace_operand_wait,
            trace_operand_sources=args.trace_operand_sources,
            early_experts=tuning.get('early_experts', experts.get(candidate)),
            split_route_heads=tuning.get('split_route_heads', False),
            decode_linear_pool=tuning.get('decode_linear_pool', False),
            dispatch_pull_rows_per_cta=tuning.get('dispatch_pull_rows_per_cta', 0),
            double_buffer_dispatch_pull=tuning.get('double_buffer_dispatch_pull', False),
            native_decode_dispatch=tuning.get('native_decode_dispatch', 0),
            speculative_w1=(candidate in speculative_heads or tuning.get('speculative_w1', False)),
            speculative_packed_w1=tuning.get('speculative_packed_w1', False),
            speculative_w1_n_tiles=tuning.get('speculative_w1_n_tiles', 0),
            stage_alignment=stage_alignment, release_group=release_group,
            direct_w1_store=(tuning.get('direct_w1_store', False) or
                             candidate in direct_stores or backfill_option in
                             ('sentinel_early_direct_w1', 'sentinel_early_direct_both',
                              'sentinel_early_narrow_direct_w1')),
            narrow_bm32=(tuning.get('narrow_bm32', False) or candidate in narrow or
                         backfill_option == 'sentinel_early_narrow_direct_w1'),
            short_activation_sf=tuning.get('short_activation_sf', False),
            single_mma_lane=candidate == 'single_mma_lane' or tuning.get('single_mma_lane', False),
            reuse_k128_desc=candidate in mma_issue or 'mma_unroll' in tuning,
            mma_unroll=tuning.get('mma_unroll', mma_issue.get(candidate, 4)),
            hoist_routed_sfb_selection=tuning.get('hoist_routed_sfb_selection', False),
            ready_tile_order=candidate in ('ready_tile_order', 'ready_programmatic'),
            programmatic_finalize=(not fused_finalize_sentinel and (
                candidate in ('programmatic_finalize', 'ready_programmatic') or
                (backfill_option or '').startswith('sentinel_early_'))),
            expert_ready_combine=candidate in ('expert_combine_tma', 'expert_combine_vector'),
            recycle_dispatch_sms=candidate in recycled or candidate == 'direct_w2_recycle',
            compute_backfill_sms=compute_backfill.get(candidate, 0),
            dispatch_first=(candidate in compute_backfill and not (
                tuning.get('fused_dispatch_compute') or tuning.get('fused_worker_reuse'))),
            decode_block_m=tuning.get('decode_block_m', 32 if (backfill_option or '').endswith('_bm32') else 0),
            trim_wide_decode_activations=tuning.get('trim_wide_decode_activations', False),
            combine_token_warps=COMBINE_TOKEN_WARPS.get(backfill_option, 0),
            w1_prefetch_k_tiles=tuning.get(
                'w1_prefetch_k_tiles', WEIGHT_PREFIX_TILES.get(backfill_option, 0)),
            keep_w1_prefix=(tuning.get('keep_w1_prefix', False) or
                            backfill_option == 'sentinel_early_w1k4_w2k2_keep_w1_chunk_major'),
            prefetch_weight_scales=backfill_option in PREFETCH_WEIGHT_SCALES,
            prefetch_k_first=(backfill_option or '').endswith('_kfirst'),
            weight_prefix_warps=tuning.get('weight_prefix_warps', False),
            shared_priority_prefix=tuning.get('shared_priority_prefix', 0),
            post_init_prefetch=tuning.get('post_init_prefetch', False),
            relaxed_init_barrier=tuning.get('relaxed_init_barrier', False),
            skip_init_prefetch_barrier=tuning.get('skip_init_prefetch_barrier', False),
            scheduled_weight_tiles=tuning.get('scheduled_weight_tiles', 0),
            w2_prefetch_from_w1=tuning.get('w2_prefetch_from_w1', 0),
            combine_in_compute=tuning.get('combine_in_compute', 0),
            decode_w2_n_group=tuning.get('decode_w2_n_group', 1),
            w2_tile_major_input=tuning.get('w2_tile_major_input', False),
            transpose_w2_activations=tuning.get('transpose_w2_activations', 0),
            grouped_routed_weights=tuning.get('grouped_routed_weights',
                                              decoy_weight_layouts.get(name, 0)),
            w1_prefix_expert_limit=tuning.get('w1_prefix_expert_limit', 0),
            packed_routed_scale_base=packed_scale_base if packed_scale_group else None,
            packed_routed_scale_k_group=packed_scale_group or 1,
            packed_scale_helper=packed_scale_helper,
            packed_scale_release_group=tuning.get('packed_scale_release_group', decoy_scale_release),
            follower_scheduler=tuning.get('follower_scheduler', False),
            w1_tile_major_input=tuning.get('w1_tile_major_input', False),
            pair_routed_weights=tuning.get('pair_routed_weights', 0),
            single_tma_lanes=tuning.get('single_tma_lanes', 0),
            decode_descriptor_prefetch=tuning.get('decode_descriptor_prefetch', 'all'),
            decode_operand_cache=tuning.get('decode_operand_cache', 'auto'),
            paired_shared_weights=packets,
            probe_routed_mma=tuning.get('probe_routed_mma', 0),
            w1_register_transpose=tuning.get('w1_register_transpose', False),
            w1_packed_scale_reduction=tuning.get('w1_packed_scale_reduction', False),
            w1_hardware_scale_reduction=tuning.get('w1_hardware_scale_reduction', False),
            w1_early_epi_inputs=tuning.get('w1_early_epi_inputs', False),
            w1_pair_scale_rounding=tuning.get('w1_pair_scale_rounding', False),
            w2_a_issue_window=tuning.get('w2_a_issue_window', 0),
            w2_b_issue_prefix=tuning.get('w2_b_issue_prefix', 0),
            flat_decode_graph=tuning.get('flat_decode_graph', False),
            combine_at_block_start=tuning.get('combine_at_block_start', False),
            decode_compute_max_registers=tuning.get('decode_compute_max_registers', 0),
            decode_cooperative_compute=tuning.get('decode_cooperative_compute', False),
            decode_nvlink_schedule=tuning.get('decode_nvlink_schedule', 0),
            decode_cluster_policy=tuning.get('decode_cluster_policy', 0),
            decode_cluster_stride=tuning.get('decode_cluster_stride', 0),
            decode_w2_early_workers=tuning.get('decode_w2_early_workers', 0),
            prefetch_w2_metadata=tuning.get('prefetch_w2_metadata', False),
            packed_w2_metadata=tuning.get('packed_w2_metadata', False),
            bulk_w1_prefetch_bytes=tuning.get('bulk_w1_prefetch_bytes', 0),
            bulk_w1_prefetch_on_dispatch=tuning.get('bulk_w1_prefetch_on_dispatch', False),
            bulk_w1_prefetch_lanes=tuning.get('bulk_w1_prefetch_lanes', 1),
            pair_routed_weight_scales=tuning.get('pair_routed_weight_scales', False),
            keep_routed_weight_scales=tuning.get('keep_routed_weight_scales', False),
            shared_w1_prefetch_limit=tuning.get('shared_w1_prefetch_limit', 0),
            asymmetric_activation_rows=tuning.get('asymmetric_activation_rows', 0),
            cache_routed_ranges=(tuning.get('cache_routed_ranges', False) or
                                 (backfill_option or '').endswith('_ranges')),
            single_expert_tile_lookup=tuning.get('single_expert_tile_lookup', False),
            prioritize_expert_rows=tuning.get('prioritize_expert_rows', 0),
            w2_prefetch_k_tiles=tuning.get(
                'w2_prefetch_k_tiles', W2_PREFIX_TILES.get(backfill_option, 0)),
            bulk_w2_tail_experts=tuning.get('bulk_w2_tail_experts', 0),
            keep_w2_prefix=(backfill_option == 'sentinel_early_w1k4_w2k2_keep' or
                            '_keep_w2_' in (backfill_option or '')),
            route_publish_ctas=tuning.get(
                'route_publish_ctas',
                ROUTE_PUBLISH_CTAS.get(backfill_option, 2 if backfill_option in FRONT_STACKS else 1)),
            decode_combine_fetch=COMBINE_FETCH_MODES.get(backfill_option, 'default'),
            direct_combine_output=backfill_option in DIRECT_COMBINE_OUTPUT,
            sentinel_combine=backfill_option in SENTINEL_OPTIONS,
            reuse_finalize_warps=tuning.get('reuse_finalize_warps', False),
            single_cleanup_cta=tuning.get('single_cleanup_cta', False),
            sentinel_half_vector=tuning.get('sentinel_half_vector', False),
            sentinel_chunk_major=(tuning.get('sentinel_chunk_major', False) or
                                  (backfill_option or '').endswith('_chunk_major')),
            combine_ready_flags=('_flags_' in (backfill_option or '') or
                                 tuning.get('combine_ready_flags', False)),
            combine_ready_mask=tuning.get('combine_ready_mask', False),
            sentinel_combine_vectors=2 if (backfill_option == 'sentinel_early_wide' or
                                           backfill_option in SENTINEL_FOUR_WIDE) else 1,
            sentinel_combine_sms=tuning.get('sentinel_combine_sms', SENTINEL_SMS.get(backfill_option, 0)),
            sentinel_token_order=backfill_option == 'sentinel_early_token_order',
            sentinel_poll=tuning.get(
                'sentinel_poll', 'relaxed-sys' if '_relaxed_' in (backfill_option or '') else
                SENTINEL_POLL.get(backfill_option, 'full')),
            sentinel_poll_delay_ns=tuning.get(
                'sentinel_poll_delay_ns', SENTINEL_POLL_DELAYS.get(backfill_option, 0)),
            sentinel_adaptive_poll=tuning.get('sentinel_adaptive_poll', False),
            sentinel_combine_warps=tuning.get(
                'sentinel_combine_warps', 4 if (backfill_option == 'sentinel_early_w4' or
                                                backfill_option in SENTINEL_FOUR_WIDE) else 8),
            static_w1_prefix=(candidate == 'static_w1_prefix' or tuning.get('static_w1_prefix', False)),
            static_w2_overlap=tuning.get('static_w2_overlap', False),
            decode_task_slots=tuning.get('decode_task_slots', 2),
            direct_w2_store=(tuning.get('direct_w2_store', False) or
                             candidate in ('direct_w2', 'direct_w2_recycle',
                                           'direct_w2_route', 'direct_both_route') or
                             backfill_option in ('direct_w2', 'sentinel_early_direct_w2',
                                                 'sentinel_early_direct_both')),
            compact_compute_ctas=compact.get(candidate, 0),
            single_epilogue_warpgroup=(tuning.get('single_epilogue_warpgroup', False) or
                                       candidate in ('single_epilogue', 'single_epilogue_stripes')
                                       or backfill_option == 'single_epilogue'),
            decode_accumulator_stages=tuning.get(
                'decode_accumulator_stages', accumulator_stages.get(candidate, 2)),
            decode_dispatch_warps=tuning.get('decode_dispatch_warps', dispatch_warps.get(candidate, 0)),
            early_stripe_combine=candidate in stripe,
            stripe_active_peers=candidate == 'multicast_stripes_active',
            stripe_launch_delay=tuning.get(
                'stripe_launch_delay',
                multicast_stripes.get(candidate, SENTINEL_LAUNCH_DELAY.get(backfill_option, 0))),
            stripe_tiles=grouped_stripes.get(candidate, 1),
            dedup_activations=candidate in ('dedup_activations', 'dedup_source_owner'),
            source_elects_activation_owner=candidate == 'dedup_source_owner',
            weight_lookahead=tuning.get('weight_lookahead', lookahead.get(candidate, 0)),
            packed_route_counts=(tuning.get('packed_route_counts', False) or
                                 candidate in packed_counts or candidate in atomic_free_receiver_routes),
            inline_route_indices=tuning.get('inline_route_indices', False),
            packed_route_count_group=tuning.get(
                'packed_route_count_group', {'grouped_route_counts': 4, 'grouped_atomic_route_counts': 3,
                                             'posted_route_grouped': 3, 'rank_route_doorbell': 4,
                                             'inline_rank_route_doorbell': 4}.get(candidate, 1)),
            rank_route_doorbell=candidate.endswith('rank_route_doorbell'),
            posted_route_sums=(tuning.get('posted_route_sums', False) or
                               candidate.startswith('posted_route_') or
                               backfill_option == 'posted_counts' or backfill_option in FRONT_STACKS),
            combine_load_group=combine_groups.get(candidate, 1),
            dispatch_prefix_scan=(candidate in prefix_scan or tuning.get('dispatch_prefix_scan', False)),
            dispatch_shared_route_counts=tuning.get('dispatch_shared_route_counts', False),
            rotate_route_publication=tuning.get('rotate_route_publication', 0),
            prune_decode_resets=tuning.get('prune_decode_resets', False),
            reset_decode_ring_on_dispatch=tuning.get('reset_decode_ring_on_dispatch', False),
            omit_sentinel_peer_finish=tuning.get('omit_sentinel_peer_finish', False),
            decode_vector_pull_group=tuning.get('decode_vector_pull_group', 1),
            static_decode_wave=tuning.get('static_decode_wave', static_waves.get(candidate, 0)),
            compact_decode_workspace=use_compact_workspace,
            token_major_combine=(tuning.get('token_major_combine', False) or
                                 candidate in token_major or (backfill_option or '').endswith('_token_major')),
            pull_route_metadata=candidate in receiver_routes,
            multicast_pull_routes=candidate.startswith('receiver_multicast'),
            warp_pull_routes=candidate in warp_receiver_routes,
            decode_w1_warmup=warmup, decode_w1_cadence=cadence,
            decode_shared_w2=tuning.get('decode_shared_w2', SHARED_W2_SCHEDULES.get(candidate, 0)),
            pair_shared_w1=tuning.get('pair_shared_w1', 0),
            inline_route_scores=(tuning.get('inline_route_scores', False) or
                                 backfill_option in ('inline_scores', 'front_k8_inline') or
                                 candidate.startswith('inline_') or candidate in (
                'packed_input_inline', 'multicast_input_and_scores', 'primary_inline_scores')),
            packed_decode_activations=(tuning.get('packed_decode_activations', False) or
                                       backfill_option == 'packed_input' or candidate.startswith('packed_input_') or
                                       candidate in ('push_fixed_slots', 'peer_pull_single')),
            packed_decode_scores=tuning.get('packed_decode_scores', False),
            activation_prefix_bytes=activation_prefixes.get(candidate, 0),
            peer_activation_prefix_tiles=tuning.get('peer_activation_prefix_tiles', 0),
            peer_activation_prefix_gather=tuning.get('peer_activation_prefix_gather', False),
            peer_activation_head_clusters=tuning.get('peer_activation_head_clusters', 0),
            parallel_activation_prefix=candidate in parallel_prefixes,
            route_count_doorbell=(tuning.get('route_count_doorbell', False) or
                                  candidate.startswith('count_doorbell_') or
                                  candidate == 'multicast_doorbell'),
            multicast_route_counts=(tuning.get('multicast_route_counts', False) or
                                    candidate in ('multicast_counts', 'multicast_doorbell',
                                                  'multicast_counts_and_combine')),
            multicast_combine_ready=backfill_option == 'multicast_ready' or candidate in multicast_stripes or candidate in (
                'multicast_combine', 'multicast_counts_and_combine', 'inline_multicast_combine',
                'inline_direct_narrow_multicast', 'inline_reduce_multicast',
                'inline_route_prefix_multicast'),
            multicast_decode_input=candidate in ('multicast_input', 'multicast_input_and_scores'),
            peer_decode_input=(tuning.get('peer_decode_input', False) or
                               candidate.endswith('peer_push_single')),
            peer_decode_input_direct=(tuning.get('peer_decode_input_direct', False) or
                                      candidate in ('peer_direct', 'inline_peer_direct')),
            peer_decode_token_fanout=tuning.get('peer_decode_token_fanout', 0),
            peer_prefetch_expert_heads=tuning.get('peer_prefetch_expert_heads', 0),
            peer_prefetch_opportunistic=tuning.get('peer_prefetch_opportunistic', False),
            trim_decode_gather_tail=tuning.get('trim_decode_gather_tail', False),
            push_decode_activations=candidate in ('push_decode_activations', 'push_tma_route'),
            push_overlap_reservation=candidate in ('push_overlap_reservation', 'push_tma_overlap_route'),
            push_reserve_spans=candidate in ('push_reserved_spans', 'push_reserved_spans_route'),
            push_vector_io=candidate == 'push_vector_route',
            push_early_counts=candidate == 'push_early_counts',
            push_row_ready_flags=candidate == 'push_row_ready_flags',
            push_scale_receiver_sms=push_scale_receivers.get(candidate, 0),
            decode_tma_stages=tuning.get('decode_tma_stages', tma_stages.get(candidate, 0)),
            decode_w2_clusters=tuning.get('decode_w2_clusters', w2_clusters.get(candidate, 0)),
            decode_w2_ready_k=tuning.get(
                'decode_w2_ready_k', w2_ready.get(candidate, BACKFILL_W2_READY_K.get(backfill_option, 0))),
            dual_weight_warps=(candidate == 'dual_weight_warps' or tuning.get('dual_weight_warps', False)),
            local_mma=candidate == 'local_mma',
            narrow_local_mma=candidate in ('narrow_local_mma', 'narrow_local_mma_n8',
                                           'narrow_local_mma_n8_short_load'),
            local_mma_n8={'narrow_local_mma_n8': 1, 'narrow_local_mma_n8_short_load': 2}.get(candidate, 0),
            independent_compute_ctas=candidate in ('independent_compute_ctas', 'independent_w1_first'),
            fused_dispatch_compute=(candidate in fused_workers or tuning.get('fused_dispatch_compute', False)),
            fused_compute_finalize=bool(fused_finalize_sentinel) or candidate in FUSED_FINALIZE_VARIANTS,
            fused_finalize_flags=bool(fused_finalize_sentinel) or candidate in (
                'fused_compute_finalize_flags', 'fused_compute_finalize_flags_tma'),
            fused_finalize_sentinel=fused_finalize_sentinel,
            fused_worker_reuse=(candidate in fused_reused or tuning.get('fused_worker_reuse', False)),
        )
        outputs[name] = torch.empty_like(x)
        packed, block_m_by_variant[name] = prepare_shared_sf(args, scales, buffers[name], tokens, world, 'gc')
        stage_inputs(buffers[name], x8, scales, ids, scores, packed)
    block_m = block_m_by_variant['stock']
    mega_variants = {}
    if mega_weight_sets:
        import deep_gemm
        mega_op = deep_gemm.fp8_fp4_mega_moe
        mega_variants['megamoe'] = None
        if args.sweep_megamoe_pdl:
            default_pdl = deep_gemm.get_pdl()
            mega_variants[f'megamoe_pdl_{"off" if default_pdl else "on"}'] = not default_pdl
        for name in mega_variants:
            buffers[name] = allocate_buffer(args, args.capacity, 'base')
            packed, block_m_by_variant[name] = prepare_shared_sf(
                args, scales, buffers[name], tokens, world, 'base')
            stage_inputs(buffers[name], x8, scales, ids, scores, packed)
            outputs[name] = torch.empty_like(x)
            variants[name] = 'off'

    def launch(name, graph_variant):
        if graph_variant in mega_variants:
            # Legacy MegaMoE uses K32 shared weights and the default SwiGLU alpha.
            routed, shared = mega_weight_sets[name + 1 if isinstance(name, int) else 0]
            pdl = mega_variants[graph_variant]
            if pdl is not None:
                deep_gemm.set_pdl(pdl)
            try:
                mega_op(outputs[graph_variant], *routed, buffers[graph_variant],
                        shared_l1_weights=shared[0], shared_l2_weights=shared[1],
                        recipe=ROUTED_RECIPE, activation_clamp=10.0)
            finally:
                if pdl is not None:
                    deep_gemm.set_pdl(default_pdl)
            return
        vmm_set = vmm_weights.get(vmm_aliases.get(graph_variant, graph_variant))
        alternate_weights = vmm_set or k_major_weights.get(graph_variant)
        intervening = isinstance(name, int)
        if intervening:
            weights = decoys[name] if alternate_weights is None else alternate_weights[name + 1]
            name = (packed_decoy_names[packed_scale_variants[graph_variant]] if graph_variant in packed_scale_variants else
                    decoy_names[grouped_variants.get(graph_variant, (0, 2))])
        else:
            candidate = args.trace_variant if name == 'trace' else name
            weights = colocated if candidate in colocated_variants else padded_weights.get(candidate, target)
            if alternate_weights is not None:
                weights = alternate_weights[0]
        routed, shared = weights
        variant = args.trace_variant if name == 'trace' else name
        # All intervening layers use the same cache policy as their target.
        cache_variant = graph_variant if intervening else variant
        weight_reuse = ('alternate' if cache_variant == 'inline_route_prefix_weight_order'
                        else WEIGHT_CACHE_MODES.get(cache_variant, 'off'))
        vmm_dispatch_sms = (vmm_sms_variants.get(graph_variant, args.vmm_primary_dispatch_sms)
                            if vmm_set is not None and graph_variant not in stock_vmm else None)
        explicit_primary = variant in args.primary_decode_variants
        override = (variant in PRIMARY_CONTEXT_VARIANTS or vmm_dispatch_sms is not None or
                    explicit_primary or variant in backfill_dispatch_sms)
        mode = os.environ.get('GC_MOE_PRIMARY_CTX')
        global_dispatch_sms = os.environ.get('GC_MOE_DISPATCH_SMS')
        dispatch_sms = (args.primary_decode_sms if explicit_primary else
                        backfill_dispatch_sms.get(variant, dispatch_sms_sweep.get(variant, vmm_dispatch_sms)))
        if override:
            os.environ['GC_MOE_PRIMARY_CTX'] = '1'
        if dispatch_sms is not None:
            os.environ['GC_MOE_DISPATCH_SMS'] = str(dispatch_sms)
        try:
            gc_moe.fp8_fp4_gc_moe(
                outputs[name], *routed, buffers[name], shared_l1_weights=shared[0],
                shared_l2_weights=shared[1], decode_mode=variants.get(variant, 'off'),
                weight_reuse=weight_reuse,
                prefetch_experts=w1_l2_prefetch.get(cache_variant, 0),
                recipe=ROUTED_RECIPE, shared_recipe=recipe, activation_clamp=10.0,
            )
        finally:
            if override:
                if mode is None:
                    del os.environ['GC_MOE_PRIMARY_CTX']
                else:
                    os.environ['GC_MOE_PRIMARY_CTX'] = mode
            if dispatch_sms is not None:
                if global_dispatch_sms is None:
                    del os.environ['GC_MOE_DISPATCH_SMS']
                else:
                    os.environ['GC_MOE_DISPATCH_SMS'] = global_dispatch_sms

    setup = tuple(range(len(decoys)))
    graphs = {}
    timing_start_flag = torch.zeros(1, device=device, dtype=torch.int32) if args.align_timing_ranks else None
    for name in variants:
        if rank == 0:
            print('CAPTURE', tokens, name, flush=True)
        graphs[name] = capture(lambda item, variant=name: launch(item, variant),
                               (*setup, name), device, timing_start_flag)
    local = {name: [] for name in variants}
    for window in range(args.windows + 3):
        order = list(variants) if window % 2 == 0 else list(reversed(variants))
        for name in order:
            dist.barrier()
            graph, start, end = graphs[name]
            label_window = (args.nsys_label_windows and rank == 0 and window >= 3
                            and name not in mega_variants)
            if label_window:
                torch.cuda.nvtx.range_push(f'GC_MOE|{tokens}|{window - 3}|{name}')
            try:
                graph.replay()
                torch.cuda.synchronize(device)
            finally:
                if label_window:
                    torch.cuda.nvtx.range_pop()
            if window >= 3:
                local[name].append(start.elapsed_time(end) * 1000)
    del graphs, graph
    trace_results = []
    if args.trace_replays:
        buffer = buffers['trace']
        trace_tensor = buffer.buffer.narrow(0, buffer.decode_trace_offset, TRACE_BYTES)
        trace_tensor = trace_tensor.view(torch.int64).view(3, TRACE_CTAS, TRACE_CTA_FIELDS, 2)
        trace_start_flag = torch.zeros(1, device=device, dtype=torch.int32) if args.trace_align_ranks else None
        trace_graph, start, end = capture(lambda item: launch(item, args.trace_variant),
                                          (*setup, 'trace'), device, trace_start_flag)
        if args.trace_align_ranks:
            # Warm both the captured collective and the instrumented kernel.
            dist.barrier()
            trace_graph.replay()
            torch.cuda.synchronize(device)
        for sample in range(args.trace_replays):
            trace_tensor.zero_()
            dist.barrier()
            trace_graph.replay()
            torch.cuda.synchronize(device)
            events = read_trace(trace_tensor)
            summary = summarize_trace(events)
            if args.trace_operand_sources:
                print('OPERANDS', tokens, rank, sample, summary['mma_operand_readiness'], flush=True)
            trace_results.append(summary | {'traced_us': start.elapsed_time(end) * 1000})
            trace_path = args.output_json.with_name(
                f'{args.output_json.stem}-t{tokens}-r{rank}-s{sample}.trace.json')
            trace_path.write_text(json.dumps(events, separators=(',', ':')))
        del trace_graph, trace_tensor, buffer
    diffs = {name: (outputs[name].float() - outputs['stock'].float()).abs().max().item()
             for name in outputs if name not in decoy_weight_layouts and name not in packed_decoy}
    stock_bytes = outputs['stock'].view(torch.uint8)
    bitwise_equal = {name: torch.equal(outputs[name].view(torch.uint8), stock_bytes)
                     for name in diffs}
    workspace_bytes = {name: buffers[name].buffer.numel() for name in variants}
    rank_results = [None] * world
    dist.all_gather_object(rank_results, {'times_us': local, 'diffs': diffs, 'trace': trace_results,
                                         'bitwise_equal': bitwise_equal,
                                         'remote_activation_payload': remote_payload})
    for buffer in buffers.values():
        buffer.destroy()
    assert all(max(row['diffs'].values()) == 0 for row in rank_results), rank_results
    assert all(all(row['bitwise_equal'].values()) for row in rank_results), rank_results
    latencies = {}
    for name in variants:
        slowest = [max(samples) for samples in zip(*(row['times_us'][name] for row in rank_results))]
        latencies[name] = {'p50_us': median(slowest), 'p95_us': percentile(slowest, 0.95)}
    if rank == 0:
        print('COMPARE', tokens, 'BM', block_m, 'distinct intervening layers', len(decoys),
              latencies, flush=True)
        if remote_payload is not None:
            print('AUDIT', tokens, {k: sum(r['remote_activation_payload'][k] for r in rank_results)
                                    for k in remote_payload}, flush=True)
        if compact_workspace:
            print('WORKSPACE', workspace_bytes, flush=True)
    return {'tokens': tokens, 'block_m': block_m, 'block_m_by_variant': block_m_by_variant,
            'latency': latencies, 'per_rank': rank_results,
            'workspace_bytes': workspace_bytes,
            **({'rows_by_rank_expert': routing_rows} if routing_rows is not None else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--compare-megamoe', action='store_true',
                        help='add fused MegaMoE to the same timing windows using shared K32')
    parser.add_argument('--sweep-megamoe-pdl', action='store_true',
                        help='also time MegaMoE with its programmatic launch dependency flipped')
    parser.add_argument('--routing-active-experts-per-rank', type=int, default=0,
                        help='restrict synthetic routes to this many eligible experts per rank')
    parser.add_argument('--routing-active-expert-offset', type=int, default=0,
                        help='start the eligible expert range at this local expert')
    parser.add_argument('--tokens', type=int, nargs='+', choices=(16, 32), default=(16, 32))
    parser.add_argument('--capacity', type=int, default=256)
    parser.add_argument('--intervening-layers', type=int, choices=range(5), default=3)
    parser.add_argument('--windows', type=int, default=60)
    parser.add_argument('--align-timing-ranks', action='store_true',
                        help='rendezvous after intervening layers and before each timed layer')
    parser.add_argument('--trace-replays', type=int, default=2)
    parser.add_argument('--nsys-label-windows', action='store_true')
    parser.add_argument('--early-experts', type=int, nargs='+', choices=(1, 2, 3), default=())
    parser.add_argument('--speculative-w1', action='store_true')
    parser.add_argument('--group-release', type=int, nargs='+', choices=(2, 4), default=())
    parser.add_argument('--dual-weight-warps', action='store_true')
    parser.add_argument('--local-mma', action='store_true')
    parser.add_argument('--local-mma-n8', action='store_true')
    parser.add_argument('--independent-compute-ctas', action='store_true')
    parser.add_argument('--independent-w1-first', action='store_true')
    parser.add_argument('--direct-w1-store', action='store_true')
    parser.add_argument('--narrow-bm32', action='store_true')
    parser.add_argument('--single-mma-lane', action='store_true')
    parser.add_argument('--mma-issue-sweep', action='store_true')
    parser.add_argument('--ready-tile-order', action='store_true')
    parser.add_argument('--programmatic-finalize', action='store_true')
    parser.add_argument('--expert-ready-combine', action='store_true')
    parser.add_argument('--recycle-dispatch-sms', action='store_true')
    parser.add_argument('--compute-backfill-sms', type=int, nargs='+', choices=range(0, 25, 4), default=())
    parser.add_argument('--backfill-dispatch-warps', type=int, nargs='+', choices=(8, 12, 16, 24), default=())
    parser.add_argument('--backfill-option-dispatch-warps', type=int, choices=(8, 12, 16, 24), default=0)
    parser.add_argument('--backfill-dispatch-sms', type=int, nargs='+', choices=range(4, 33, 4), default=())
    parser.add_argument('--backfill-options', nargs='+',
                        choices=(*BACKFILL_OPTIONS, *BACKFILL_TUNING), default=())
    parser.add_argument('--backfill-option-sms-sweep', nargs='+', type=int,
                        choices=range(4, 33, 4), default=())
    parser.add_argument('--backfill-vmm-compression', nargs='+', choices=('scales', 'all'), default=())
    parser.add_argument('--backfill-vmm-dispatch-warps', type=int, nargs='+',
                        choices=(8, 12, 16, 24), default=())
    parser.add_argument('--backfill-vmm-options', nargs='+',
                        choices=(*BACKFILL_OPTIONS, *BACKFILL_TUNING, *WEIGHT_PLANE_PADDING),
                        default=())
    parser.add_argument('--backfill-vmm-options-only', action='store_true',
                        help='only allocate the requested compressed backfill graphs')
    parser.add_argument('--backfill-vmm-sms-sweep', type=int, nargs='+',
                        choices=range(4, 49, 4), default=())
    parser.add_argument('--static-w1-prefix', action='store_true')
    parser.add_argument('--direct-w2-store', action='store_true')
    parser.add_argument('--compact-compute', action='store_true')
    parser.add_argument('--single-epilogue-warpgroup', action='store_true')
    parser.add_argument('--accumulator-stages', type=int, nargs='+', choices=(3, 4, 6), default=())
    parser.add_argument('--dispatch-warp-sweep', type=int, nargs='+', choices=(8, 12, 16, 24), default=())
    parser.add_argument('--dispatch-sms-sweep', type=int, nargs='+', choices=range(4, 33, 4), default=())
    parser.add_argument('--early-stripe-combine', action='store_true')
    parser.add_argument('--multicast-stripe-combine', action='store_true')
    parser.add_argument('--stripe-tiles', type=int, nargs='+', choices=(2, 4, 6), default=())
    parser.add_argument('--dedup-activations', action='store_true')
    parser.add_argument('--source-activation-owner', action='store_true')
    parser.add_argument('--weight-lookahead', type=int, nargs='+', choices=range(1, 33), default=())
    parser.add_argument('--packed-route-counts', action='store_true')
    parser.add_argument('--grouped-route-counts', action='store_true')
    parser.add_argument('--grouped-atomic-route-counts', action='store_true')
    parser.add_argument('--posted-route-sums', action='store_true')
    parser.add_argument('--combine-load-groups', type=int, nargs='+', choices=(2, 4, 8), default=())
    parser.add_argument('--dispatch-prefix-scan', action='store_true')
    parser.add_argument('--decode-route-stack', action='store_true')
    parser.add_argument('--static-decode-waves', type=int, nargs='+', choices=(2, 4, 8, 16), default=())
    parser.add_argument('--compact-decode-workspace', action='store_true')
    parser.add_argument('--token-major-combine', action='store_true')
    parser.add_argument('--pull-route-metadata', action='store_true')
    parser.add_argument('--multicast-pull-routes', action='store_true')
    parser.add_argument('--receiver-posted-counts', action='store_true')
    parser.add_argument('--warp-pull-routes', action='store_true')
    parser.add_argument('--colocate-routed-scales', action='store_true')
    parser.add_argument('--expert-weight-pad-bytes', type=int, nargs='+',
                        choices=(0, 128, 512, 2048, 8192, 65536), default=())
    parser.add_argument('--routed-weight-cache-sweep', action='store_true')
    parser.add_argument('--routed-weight-k-major', action='store_true')
    parser.add_argument('--audit-routed-encoding', action='store_true')
    parser.add_argument('--audit-route-worlds', type=int, nargs='+', choices=(16, 32), default=())
    parser.add_argument('--vmm-weight-compression', action='store_true')
    parser.add_argument('--vmm-stock-control', nargs='+', choices=('plain', 'scales', 'all'), default=())
    parser.add_argument('--vmm-primary-dispatch-sms', type=int, choices=range(4, 33, 4))
    parser.add_argument('--vmm-dispatch-sweep', type=int, nargs='+', choices=range(4, 33, 4), default=())
    parser.add_argument('--compact-vmm-sweep', action='store_true')
    parser.add_argument('--w1-l2-prefetch-experts', type=int, nargs='+', choices=(1, 2, 3, 4), default=())
    parser.add_argument('--w1-schedule-sweep', action='store_true')
    parser.add_argument('--shared-w2-schedule-sweep', action='store_true')
    parser.add_argument('--inline-route-scores', action='store_true')
    parser.add_argument('--packed-decode-activations', action='store_true')
    parser.add_argument('--activation-prefix-bytes', type=int, nargs='+',
                        choices=(512, 1024, 2048, 4096), default=())
    parser.add_argument('--parallel-activation-prefix', action='store_true')
    parser.add_argument('--route-count-doorbell', action='store_true')
    parser.add_argument('--rank-route-doorbell', action='store_true')
    parser.add_argument('--multicast-route-counts', action='store_true')
    parser.add_argument('--multicast-combine-ready', action='store_true')
    parser.add_argument('--multicast-decode-input', action='store_true')
    parser.add_argument('--peer-decode-input', action='store_true')
    parser.add_argument('--peer-decode-direct', action='store_true')
    parser.add_argument('--inline-peer-decode-direct', action='store_true')
    parser.add_argument('--push-vector-io', action='store_true')
    parser.add_argument('--push-early-counts', action='store_true')
    parser.add_argument('--push-scale-receiver-sms', type=int, nargs='+', choices=(4, 8, 12), default=())
    parser.add_argument('--push-decode-activations', action='store_true')
    parser.add_argument('--push-reservation-sweep', action='store_true')
    parser.add_argument('--compose-decode', action='store_true')
    parser.add_argument('--compare-primary-context', action='store_true')
    parser.add_argument('--decode-tma-stages', type=int, nargs='+', choices=(4, 6, 8, 10), default=())
    parser.add_argument('--decode-w2-clusters', type=int, nargs='+', choices=(8, 16, 24, 32), default=())
    parser.add_argument('--decode-w2-ready-k', type=int, nargs='+', choices=(1, 4, 8, 16), default=())
    parser.add_argument('--compress-w2-ready-scales', action='store_true')
    parser.add_argument('--primary-decode-variants', nargs='+', default=())
    parser.add_argument('--primary-decode-sms', type=int, choices=range(4, 33, 4), default=24)
    parser.add_argument('--fused-dispatch-compute', action='store_true')
    parser.add_argument('--fused-compute-finalize', action='store_true')
    parser.add_argument('--fused-finalize-variants', nargs='+', choices=tuple(FUSED_FINALIZE_VARIANTS),
                        default=())
    parser.add_argument('--fused-worker-reuse', action='store_true')
    parser.add_argument('--trace-variant', default='stock')
    parser.add_argument('--trace-stride', type=int, choices=(0, 1, 2, 4, 8, 16), default=1,
                        help='trace CTA boundaries only when set to zero')
    parser.add_argument('--trace-align-ranks', action='store_true')
    parser.add_argument('--trace-operand-wait', action='store_true')
    parser.add_argument('--trace-operand-sources', action='store_true')
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if any(BACKFILL_TUNING.get(option, (None, {}))[1].keys() &
           {'grouped_routed_weights', 'packed_routed_scales'}
           for option in args.backfill_options):
        parser.error('alternate weight allocation requires --backfill-vmm-options')
    if args.fused_compute_finalize and not args.fused_finalize_variants:
        args.fused_finalize_variants = ('fused_compute_finalize',)
    if args.multicast_pull_routes:
        args.pull_route_metadata = True
    if (args.vmm_primary_dispatch_sms is not None or args.vmm_dispatch_sweep) and not args.vmm_weight_compression:
        parser.error('VMM Dispatch settings require --vmm-weight-compression')
    if args.compact_vmm_sweep and not args.vmm_dispatch_sweep:
        parser.error('--compact-vmm-sweep requires a VMM Dispatch sweep')
    if args.backfill_vmm_compression and not (args.compute_backfill_sms and args.vmm_primary_dispatch_sms):
        parser.error('--backfill-vmm-compression needs backfill and a primary VMM Dispatch size')
    if (args.backfill_vmm_dispatch_warps or args.backfill_vmm_options) and not args.backfill_vmm_compression:
        parser.error('widened or composite VMM backfill requires a compression mode')
    if args.backfill_vmm_options_only and (
            not args.backfill_vmm_options or args.vmm_dispatch_sweep or args.compress_w2_ready_scales):
        parser.error('compressed options-only sweeps need backfill options and no other VMM comparisons')
    if args.backfill_vmm_sms_sweep and not args.backfill_vmm_options:
        parser.error('--backfill-vmm-sms-sweep requires composed VMM backfill options')
    if args.vmm_stock_control and not args.vmm_weight_compression:
        parser.error('--vmm-stock-control requires --vmm-weight-compression')
    if args.backfill_option_sms_sweep and not args.backfill_options:
        parser.error('Dispatch SM sweep requires backfill options')
    if ((args.backfill_dispatch_warps or args.backfill_dispatch_sms or args.backfill_options)
            and not args.compute_backfill_sms):
        parser.error('backfill variations need at least one backfill size')
    if args.backfill_option_dispatch_warps and not args.backfill_options:
        parser.error('--backfill-option-dispatch-warps requires backfill options')
    if args.compress_w2_ready_scales and not (args.decode_w2_ready_k and args.vmm_weight_compression):
        parser.error('--compress-w2-ready-scales needs a W2 readiness prefix and VMM allocations')
    if args.windows < 1 or args.trace_replays < 0:
        parser.error('expected at least one window and a nonnegative trace replay count')
    trace_candidates = {'stock', 'route_reduce', *(f'weight_lookahead_{k}' for k in args.weight_lookahead)}
    trace_candidates.update(f'stock_vmm_{mode}' for mode in args.vmm_stock_control)
    if args.dual_weight_warps:
        trace_candidates.update(('single_weight_warp_aligned', 'dual_weight_warps', 'route_only'))
    if args.local_mma:
        trace_candidates.update(('local_mma', 'narrow_local_mma', 'route_only'))
    if args.local_mma_n8:
        trace_candidates.update(('narrow_local_mma', 'narrow_local_mma_n8',
                                 'narrow_local_mma_n8_short_load', 'route_only'))
    if args.independent_compute_ctas:
        trace_candidates.update(('independent_compute_ctas', 'narrow_local_mma', 'route_only'))
    if args.independent_w1_first:
        trace_candidates.update(('independent_w1_first', 'narrow_local_mma', 'route_only'))
    if args.early_experts:
        trace_candidates |= {'fixed_slots', *(f'early_{count}' for count in args.early_experts)}
        if args.speculative_w1:
            trace_candidates.update(f'speculative_{count}' for count in args.early_experts)
    elif args.speculative_w1:
        parser.error('--speculative-w1 requires --early-experts')
    for group in args.group_release:
        trace_candidates.update((f'align_{group}', f'release_{group}'))
    if args.direct_w1_store:
        trace_candidates.add('direct_w1')
    if args.narrow_bm32:
        trace_candidates.add('narrow_bm32')
        if args.direct_w1_store:
            trace_candidates.add('direct_narrow')
    if args.single_mma_lane:
        trace_candidates.add('single_mma_lane')
    if args.mma_issue_sweep:
        trace_candidates.update(f'k128_u{u}' for u in (2, 4, 8))
    if args.ready_tile_order:
        trace_candidates.add('ready_tile_order')
    if args.programmatic_finalize:
        trace_candidates.add('programmatic_finalize')
        if args.ready_tile_order:
            trace_candidates.add('ready_programmatic')
    if args.expert_ready_combine:
        trace_candidates.update(('route_only', 'expert_combine_tma', 'expert_combine_vector'))
    if args.recycle_dispatch_sms:
        trace_candidates.add('recycle_dispatch_sms')
    if args.compute_backfill_sms:
        trace_candidates.add('route_only')
        trace_candidates.update(backfill_variant_names(args))
    if args.static_w1_prefix:
        trace_candidates.add('static_w1_prefix')
    if args.direct_w2_store:
        trace_candidates.add('direct_w2')
        trace_candidates.update(('route_only', 'direct_w2_route'))
        if args.direct_w1_store:
            trace_candidates.add('direct_both_route')
        if args.recycle_dispatch_sms:
            trace_candidates.add('direct_w2_recycle')
    if args.compact_compute:
        trace_candidates.update(('compact_grid_1', 'compact_grid_2'))
    if args.single_epilogue_warpgroup:
        trace_candidates.add('single_epilogue')
        if args.multicast_stripe_combine:
            trace_candidates.add('single_epilogue_stripes')
    if args.accumulator_stages:
        trace_candidates.add('route_only')
        trace_candidates.update(f'accumulator_stages_{n}' for n in args.accumulator_stages)
    trace_candidates.update(f'dispatch_w{w}' for w in args.dispatch_warp_sweep)
    trace_candidates.update(f'route_dispatch_sms_{n}' for n in args.dispatch_sms_sweep)
    if args.recycle_dispatch_sms:
        trace_candidates.update(f'dispatch_w{w}_recycle' for w in args.dispatch_warp_sweep)
    if args.early_stripe_combine:
        trace_candidates.add('stripe_combine')
        if args.recycle_dispatch_sms:
            trace_candidates.add('stripe_recycle')
    if args.multicast_stripe_combine:
        trace_candidates.update(MULTICAST_STRIPE_VARIANTS)
    trace_candidates.update(f'multicast_stripes_tiles_{n}' for n in args.stripe_tiles)
    if args.dedup_activations:
        trace_candidates.add('dedup_activations')
    if args.source_activation_owner:
        trace_candidates.add('dedup_source_owner')
    if args.packed_route_counts:
        trace_candidates.add('packed_counts')
        if args.recycle_dispatch_sms:
            trace_candidates.add('packed_recycle')
    if args.grouped_route_counts:
        trace_candidates.add('grouped_route_counts')
    if args.grouped_atomic_route_counts:
        trace_candidates.add('grouped_atomic_route_counts')
    if args.posted_route_sums:
        trace_candidates.update(('posted_route_stock', 'posted_route_single'))
        if args.grouped_atomic_route_counts:
            trace_candidates.add('posted_route_grouped')
    trace_candidates.update(f'combine_load_{n}' for n in args.combine_load_groups)
    if args.dispatch_prefix_scan:
        trace_candidates.add('dispatch_prefix_scan')
        if args.packed_route_counts:
            trace_candidates.add('packed_prefix_scan')
    if args.decode_route_stack:
        trace_candidates.update(ROUTE_STACK_VARIANTS)
    trace_candidates.update(f'static_wave_{n}' for n in args.static_decode_waves)
    if args.compact_decode_workspace:
        trace_candidates.update(('compact_workspace_stock', 'compact_workspace_route',
                                 'compact_workspace_route_only', 'route_only'))
    if args.token_major_combine:
        trace_candidates.add('token_major_combine')
        if args.compact_decode_workspace:
            trace_candidates.add('compact_token_major_combine')
    if args.pull_route_metadata:
        trace_candidates.add('receiver_routes')
        if args.multicast_pull_routes:
            trace_candidates.add('receiver_multicast')
        if args.dispatch_prefix_scan:
            trace_candidates.add('receiver_prefix')
            if args.multicast_pull_routes:
                trace_candidates.add('receiver_multicast_prefix')
    if args.receiver_posted_counts:
        trace_candidates.update(('receiver_posted', 'receiver_multicast_posted'))
    if args.warp_pull_routes:
        trace_candidates.update(('receiver_warp', 'receiver_multicast_warp'))
    if args.colocate_routed_scales:
        trace_candidates.update(('colocated_scales_stock', 'colocated_scales_route'))
    trace_candidates.update(f'expert_weight_pad_{n}' for n in args.expert_weight_pad_bytes)
    if args.routed_weight_cache_sweep:
        trace_candidates.update(WEIGHT_CACHE_MODES)
    if args.routed_weight_k_major:
        trace_candidates.add('route_only')
        trace_candidates.update(K_MAJOR_MODES)
    if args.vmm_weight_compression:
        if not args.backfill_vmm_options_only:
            trace_candidates.update(VMM_WEIGHT_MODES)
        trace_candidates.add('route_only')
        trace_candidates.update(f'{mode}_d{sms}' for mode in ('vmm_plain', 'vmm_compressed_scales')
                                for sms in args.vmm_dispatch_sweep)
    trace_candidates.update(f'w1_l2_prefetch_{n}' for n in args.w1_l2_prefetch_experts)
    if args.w1_schedule_sweep:
        trace_candidates.update(W1_SCHEDULES)
    if args.shared_w2_schedule_sweep:
        trace_candidates.update(SHARED_W2_SCHEDULES)
    if args.inline_route_scores:
        trace_candidates.update(('inline_scores_stock', 'inline_scores_single'))
    if args.packed_decode_activations:
        trace_candidates.update(('packed_input_stock', 'packed_input_single', 'packed_input_inline'))
    if args.activation_prefix_bytes:
        trace_candidates.add('route_only')
        trace_candidates.update(f'activation_prefix_{n}' for n in args.activation_prefix_bytes)
        if args.parallel_activation_prefix:
            trace_candidates.update(f'activation_prefix_parallel_{n}' for n in args.activation_prefix_bytes)
    if args.route_count_doorbell:
        trace_candidates.update(('count_doorbell_stock', 'count_doorbell_single'))
    if args.rank_route_doorbell:
        trace_candidates.update(('rank_route_doorbell', 'inline_rank_route_doorbell'))
    if args.multicast_route_counts:
        trace_candidates.update(('multicast_counts', 'multicast_doorbell'))
    if args.multicast_combine_ready:
        trace_candidates.add('multicast_combine')
        if args.multicast_route_counts:
            trace_candidates.add('multicast_counts_and_combine')
    if args.multicast_decode_input:
        trace_candidates.update(('multicast_input', 'multicast_input_and_scores'))
    if args.peer_decode_input or args.peer_decode_direct or args.inline_peer_decode_direct:
        trace_candidates.update(('peer_pull_single', 'peer_push_single', 'inline_peer_push_single'))
    if args.peer_decode_direct:
        trace_candidates.add('peer_direct')
    if args.inline_peer_decode_direct:
        trace_candidates.add('inline_peer_direct')
    if args.push_decode_activations or args.push_reservation_sweep:
        trace_candidates.update(('push_fixed_slots', 'push_decode_activations'))
    if args.push_reservation_sweep:
        trace_candidates.update(('push_overlap_reservation', 'push_reserved_spans'))
    if args.push_vector_io or args.push_early_counts:
        trace_candidates.update(('push_tma_route', 'push_tma_overlap_route', 'route_only'))
    if args.push_vector_io:
        trace_candidates.add('push_vector_route')
    if args.push_early_counts:
        trace_candidates.update(('push_reserved_spans_route', 'push_early_counts',
                                 'push_row_ready_flags'))
    if args.push_scale_receiver_sms:
        trace_candidates.update(('push_row_ready_flags', 'route_only'))
        trace_candidates.update(f'push_packed_scales_r{n}' for n in args.push_scale_receiver_sms)
    if args.compose_decode:
        trace_candidates.update(DECODE_COMBINATIONS)
    if args.compare_primary_context:
        if int(os.environ.get('GC_MOE_PRIMARY_CTX', '0')):
            parser.error('context comparison needs a Green Context baseline')
        trace_candidates.update(PRIMARY_CONTEXT_VARIANTS)
    trace_candidates.update(f'tma_stages_{n}' for n in args.decode_tma_stages)
    trace_candidates.update(f'w2_startup_clusters_{n}' for n in args.decode_w2_clusters)
    if args.decode_w2_ready_k:
        trace_candidates.add('route_only')
        trace_candidates.update(f'w2_ready_k{n}' for n in args.decode_w2_ready_k)
        if args.compress_w2_ready_scales:
            trace_candidates.update(f'w2_ready_k{n}_compressed_scales' for n in args.decode_w2_ready_k)
    if not set(args.primary_decode_variants).issubset(trace_candidates):
        parser.error('--primary-decode-variants requires active benchmark variants')
    if args.fused_dispatch_compute:
        trace_candidates.update(('fused_dispatch_compute', 'inline_fused_dispatch_compute',
                                 'dispatch_w12_stock'))
    trace_candidates.update(args.fused_finalize_variants)
    if args.fused_worker_reuse:
        trace_candidates.update(('fused_worker_reuse', 'inline_fused_worker_reuse'))
    if args.trace_variant not in trace_candidates:
        parser.error('trace variant is not included in this comparison')
    args.hidden, args.intermediate_hidden, args.num_topk = 6144, 2048, 8
    args.num_shared_experts, args.activation = 1, 'swiglu'
    if args.sweep_megamoe_pdl and not args.compare_megamoe:
        parser.error('--sweep-megamoe-pdl needs --compare-megamoe')
    args.shared_weight_gran_k = 32 if args.compare_megamoe else 128
    device_index = 0 if args.audit_route_worlds else int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', device_index)
    torch.cuda.set_device(device)
    if args.audit_route_worlds:
        routing = audit_decode_routing(args, device)
        args.output_json.write_text(json.dumps(routing, indent=2))
        for case in routing['cases']:
            print('AUDIT', case['world_size'], case['tokens'], case['inactive_experts'],
                  case['max_rows_per_expert'],
                  {m: tiles['max_logical_reissued_bytes'] for m, tiles in case['m_tilings'].items()},
                  flush=True)
        return
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    mega_routed, routed, _, mega_shared, shared, recipe = build_weights(
        args, rank, world, device, with_mega=args.compare_megamoe)
    if args.audit_routed_encoding:
        stats = audit_routed_encoding(routed)
        if rank == 0:
            args.output_json.write_text(json.dumps({
                'world_size': world, 'weights': 'synthetic FP4 with UE8M0 scales',
                'routed': stats,
            }, indent=2))
            print('AUDIT', {phase: (data['weights']['entropy_bits_per_value'],
                                   data['weights']['three_symbols_and_escape_bits_per_value'],
                                   data['scale_bytes']['entropy_bits_per_value'])
                            for phase, data in stats.items()}, flush=True)
        dist.destroy_process_group()
        return
    target = routed, shared
    packed_scale_groups = {int(BACKFILL_TUNING.get(option, (None, {}))[1].get('packed_routed_scales', False))
                           for option in args.backfill_vmm_options} - {0}
    packed_scale_base = routed_scale_window(routed) if packed_scale_groups else None
    colocated = ((tuple(colocate_routed_scales(*pair) for pair in routed), shared)
                 if args.colocate_routed_scales else None)
    padded_weights = {}
    for padding in args.expert_weight_pad_bytes:
        payload = tuple((torch.empty_strided(w.shape, (w.stride(0) + padding, *w.stride()[1:]),
                                             dtype=w.dtype, device=w.device).copy_(w), sf)
                        for w, sf in routed)
        padded_weights[f'expert_weight_pad_{padding}'] = payload, shared
    decoys = [tuple(tuple(tuple(t.clone() for t in pair) for pair in group) for group in target)
              for _ in range(args.intervening_layers)]
    mega_weight_sets = ()
    if args.compare_megamoe:
        mega_target = mega_routed, mega_shared
        mega_weight_sets = (mega_target, *(
            tuple(tuple(tuple(t.clone() for t in pair) for pair in group) for group in mega_target)
            for _ in range(args.intervening_layers)))
    k_major_weights = {}
    if args.routed_weight_k_major:
        k_major_weights = {mode: [] for mode in K_MAJOR_MODES}
        for payloads, shared_set in (target, *decoys):
            transposed = tuple((weight.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3), scale)
                               for weight, scale in payloads)
            for mode, phases in K_MAJOR_MODES.items():
                weights = tuple(transposed[p] if p in phases else payloads[p] for p in range(2))
                k_major_weights[mode].append((weights, shared_set))
    with WeightVmm() as arena:
        vmm_modes = VMM_WEIGHT_MODES
        if args.backfill_vmm_options_only:
            vmm_modes = {name: VMM_WEIGHT_MODES[name]
                         for mode in args.backfill_vmm_compression
                         for name in (f'vmm_compressed_{mode}',)}
            for mode in args.vmm_stock_control:
                name = 'vmm_plain' if mode == 'plain' else f'vmm_compressed_{mode}'
                vmm_modes[name] = VMM_WEIGHT_MODES[name]
        vmm_weights = {
            mode: [tuple(tuple((arena.copy(values, payload), arena.copy(scales, scale))
                               for values, scales in group) for group in weight_set)
                   for weight_set in (target, *decoys)]
            for mode, (payload, scale) in vmm_modes.items()
        } if args.vmm_weight_compression else {}
        plane_padding = {WEIGHT_PLANE_PADDING[option][1] for option in args.backfill_vmm_options
                         if option in WEIGHT_PLANE_PADDING}
        grouped_weights = set()
        for option in args.backfill_vmm_options:
            tuning = BACKFILL_TUNING.get(option, (None, {}))[1]
            if tuning.get('grouped_routed_weights'):
                grouped_weights.add((tuning['grouped_routed_weights'],
                                     tuning.get('grouped_weight_inner_k', 2)))
        for mode in args.backfill_vmm_compression:
            name = f'vmm_compressed_{mode}'
            payload, scale = VMM_WEIGHT_MODES[name]
            for padding in sorted(plane_padding):
                vmm_weights[f'{name}_plane_pad{padding}'] = [
                    (tuple((arena.copy(pad_weight_plane(weight, padding), payload), copied_scale)
                           for (weight, _), (_, copied_scale) in zip(weight_set[0], copied[0])),
                     copied[1])
                    for weight_set, copied in zip((target, *decoys), vmm_weights[name])
                ]
            for grouped, inner_k in sorted(grouped_weights):
                vmm_weights[f'{name}_grouped{grouped}_k{inner_k}'] = [
                    (tuple((arena.copy(pack_grouped_weights(weight, inner_k), payload), copied[0][p][1])
                           if grouped & (1 << p) else copied[0][p]
                           for p, (weight, _) in enumerate(weight_set[0])),
                     copied[1])
                    for weight_set, copied in zip((target, *decoys), vmm_weights[name])
                ]
            for scale_group in sorted(packed_scale_groups):
                vmm_weights[f'{name}_packed_sf{scale_group}'] = [
                    (tuple((weight, arena.copy(pack_routed_scale_tiles(sf, packed_scale_base, scale_group), scale))
                           for (_, sf), (weight, _) in zip(weight_set[0], copied[0])), copied[1])
                    for weight_set, copied in zip((target, *decoys), vmm_weights[name])
                ]
        results = []
        for tokens in args.tokens:
            results.append(run_case(args, tokens, rank, world, device, target, decoys, recipe,
                                    colocated, padded_weights, vmm_weights, k_major_weights,
                                    packed_scale_base, mega_weight_sets))
            if rank == 0:
                args.output_json.write_text(json.dumps({
                    'world_size': world, 'config': vars(args) | {'output_json': str(args.output_json),
                                                                 'packed_scale_base': packed_scale_base},
                    'cases': results,
                }, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
