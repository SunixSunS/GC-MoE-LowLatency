"""Public GC MoE allocation and weight-layout entrypoints."""

from __future__ import annotations

from typing import Optional, Tuple
import types

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ..jit_kernels.heuristics.gc_moe import get_block_m_for_gc_moe, get_ring_tokens
from ..layout.gc_moe import GCBufferLayout, TOKEN_ALIGNMENT, align


class GCSymmBuffer:
    """GC-owned symmetric workspace; serialize calls sharing one allocation.

    Stage shared input scales in ``shared_l1_acts_sf`` at row
    ``(t // BM) * align(BM, 128) + (u // 128) * 128 + (u % 32) * 4
    + (u % 128) // 32``, where ``u = t % BM``. Query BM using this rank's
    actual token count. Shared activations always use K32 scales; ``x_sf``
    remains K-major while shared scale views are MN-major.

    ``base`` must be a live ``GCSymmBuffer`` with the same group and exact
    GC layout. Foreign workspaces (including DeepGEMM) are incompatible.
    Serialize calls across all wrappers sharing the allocation; retire captured
    graphs before destroying an allocation used by them.
    """

    _views = ('x', 'x_sf', 'topk_idx', 'topk_weights', 'shared_l1_acts',
              'shared_l1_acts_sf', 'shared_l2_acts', 'shared_l2_acts_sf',
              'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf')

    def __init__(self, group: dist.ProcessGroup, num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int, hidden: int,
                 intermediate_hidden: int, num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4', activation: str = 'swiglu', base=None,
                 *, decode_trace: bool = False, early_experts: int | None = None,
                 stage_alignment: int = 1, release_group: int = 1,
                 direct_w1_store: bool = False, narrow_bm32: bool = False,
                 short_activation_sf: bool = False,
                 single_mma_lane: bool = False, trace_stride: int = 1,
                 trace_operand_wait: bool = False, reuse_k128_desc: bool = False,
                 mma_unroll: int = 4, hoist_routed_sfb_selection: bool = False,
                 ready_tile_order: bool = False,
                 programmatic_finalize: bool = False, recycle_dispatch_sms: bool = False,
                 static_w1_prefix: bool = False, direct_w2_store: bool = False,
                 speculative_w1: bool = False, compact_compute_ctas: int = 0,
                 speculative_packed_w1: bool = False, speculative_w1_n_tiles: int = 0,
                 decode_dispatch_warps: int = 0, early_stripe_combine: bool = False,
                 dispatch_pull_rows_per_cta: int = 0,
                 double_buffer_dispatch_pull: bool = False,
                 dedup_activations: bool = False, source_elects_activation_owner: bool = False,
                 weight_lookahead: int = 0, trace_operand_sources: bool = False,
                 packed_route_counts: bool = False, combine_load_group: int = 1,
                 inline_route_indices: bool = False,
                 dispatch_prefix_scan: bool = False, static_decode_wave: int = 0,
                 static_w2_overlap: bool = False,
                 compact_decode_workspace: bool = False, token_major_combine: bool = False,
                 pull_route_metadata: bool = False, decode_w1_warmup: int = 0,
                 decode_w1_cadence: int = 1, packed_route_count_group: int = 1,
                 posted_route_sums: bool = False, decode_shared_w2: int = 0,
                 pair_shared_w1: int = 0,
                 inline_route_scores: bool = False, packed_decode_activations: bool = False,
                 packed_decode_scores: bool = False,
                 rotate_route_publication: int = 0,
                 prune_decode_resets: bool = False, omit_sentinel_peer_finish: bool = False,
                 reset_decode_ring_on_dispatch: bool = False,
                 decode_linear_pool: bool = False,
                 native_decode_dispatch: int = 0,
                 decode_vector_pull_group: int = 1,
                 route_count_doorbell: bool = False, dispatch_shared_route_counts: bool = False,
                 multicast_route_counts: bool = False,
                 split_route_heads: bool = False,
                 multicast_combine_ready: bool = False, decode_tma_stages: int = 0,
                 decode_task_slots: int = 2,
                 fused_dispatch_compute: bool = False, multicast_decode_input: bool = False,
                 push_decode_activations: bool = False, fused_worker_reuse: bool = False,
                 push_overlap_reservation: bool = False, push_reserve_spans: bool = False,
                 decode_w2_clusters: int = 0, rank_route_doorbell: bool = False,
                 w2_a_issue_window: int = 0, decode_w2_early_workers: int = 0,
                 w2_b_issue_prefix: int = 0,
                 flat_decode_graph: bool = False, decode_cluster_policy: int = 0,
                 combine_at_block_start: bool = False, decode_compute_max_registers: int = 0,
                 decode_cooperative_compute: bool = False,
                 decode_nvlink_schedule: int = 0,
                 decode_cluster_stride: int = 0,
                 peer_decode_input: bool = False, stripe_active_peers: bool = False,
                 stripe_launch_delay: int = 0, stripe_tiles: int = 1,
                 activation_prefix_bytes: int = 0, parallel_activation_prefix: bool = False,
                 peer_activation_prefix_tiles: int = 0, peer_activation_prefix_gather: bool = False,
                 peer_activation_head_clusters: int = 0,
                 single_epilogue_warpgroup: bool = False, decode_accumulator_stages: int = 2,
                 fused_compute_finalize: bool = False, peer_decode_input_direct: bool = False,
                 peer_decode_token_fanout: int = 0,
                 peer_prefetch_expert_heads: int = 0, peer_prefetch_opportunistic: bool = False,
                 trim_decode_gather_tail: bool = False,
                 fused_finalize_flags: bool = False, fused_finalize_sentinel: int = 0,
                 multicast_pull_routes: bool = False,
                 push_vector_io: bool = False, warp_pull_routes: bool = False,
                 decode_w2_ready_k: int = 0, push_early_counts: bool = False,
                 push_row_ready_flags: bool = False, dual_weight_warps: bool = False,
                 push_scale_receiver_sms: int = 0, local_mma: bool = False,
                 narrow_local_mma: bool = False, independent_compute_ctas: bool = False,
                 local_mma_n8: int = 0,
                 expert_ready_combine: bool = False, compute_backfill_sms: int = 0,
                 dispatch_first: bool = False, combine_token_warps: int = 0,
                 w1_prefetch_k_tiles: int = 0, route_publish_ctas: int = 1,
                 w1_prefix_expert_limit: int = 0,
                 decode_combine_fetch: str = 'default', direct_combine_output: bool = False,
                 sentinel_combine: bool = False, w2_prefetch_k_tiles: int = 0,
                 bulk_w2_tail_experts: int = 0,
                 sentinel_combine_vectors: int = 1, sentinel_combine_sms: int = 0,
                 reuse_finalize_warps: bool = False,
                 single_cleanup_cta: bool = False,
                 sentinel_half_vector: bool = False,
                 sentinel_token_order: bool = False, sentinel_poll: str = 'full',
                 sentinel_combine_warps: int = 8, keep_w2_prefix: bool = False,
                 prefetch_weight_scales: bool = False, prefetch_k_first: bool = False,
                 weight_prefix_warps: bool = False,
                 shared_priority_prefix: int = 0,
                 post_init_prefetch: bool = False,
                 relaxed_init_barrier: bool = False,
                 skip_init_prefetch_barrier: bool = False,
                 scheduled_weight_tiles: int = 0,
                 w2_prefetch_from_w1: int = 0,
                 combine_in_compute: int = 0,
                 decode_w2_n_group: int = 1,
                 w2_tile_major_input: bool = False,
                 transpose_w2_activations: int = 0,
                 grouped_routed_weights: int = 0,
                 packed_routed_scale_base: int | None = None, packed_routed_scale_k_group: int = 1,
                 packed_scale_helper: bool = False, follower_scheduler: bool = False,
                 packed_scale_release_group: int = 4,
                 single_tma_lanes: int = 0, decode_descriptor_prefetch: str = 'all',
                 decode_operand_cache: str = 'auto', paired_shared_weights: tuple | None = None,
                 bulk_w1_prefetch_bytes: int = 0, bulk_w1_prefetch_on_dispatch: int = 0,
                 bulk_w1_prefetch_lanes: int = 1,
                 probe_routed_mma: int = 0,
                 w1_register_transpose: bool = False,
                 w1_packed_scale_reduction: bool = False,
                 w1_hardware_scale_reduction: bool = False,
                 w1_early_epi_inputs: bool = False, w1_pair_scale_rounding: bool = False,
                 prefetch_w2_metadata: bool = False,
                 packed_w2_metadata: bool = False,
                 pair_routed_weight_scales: bool = False,
                 keep_routed_weight_scales: bool = False,
                 shared_w1_prefetch_limit: int = 0,
                 asymmetric_activation_rows: int = 0,
                 w1_tile_major_input: bool = False,
                 pair_routed_weights: int = 0,
                 cache_routed_ranges: bool = False, sentinel_chunk_major: bool = False,
                 keep_w1_prefix: bool = False, decode_block_m: int = 0,
                 trim_wide_decode_activations: bool = False,
                 single_expert_tile_lookup: bool = False,
                 prioritize_expert_rows: int = 0,
                 sentinel_poll_delay_ns: int = 0, combine_ready_flags: bool = False,
                 combine_ready_mask: bool = False,
                 sentinel_adaptive_poll: bool = False):
        if mma_type != 'fp8xfp4' or activation != 'swiglu':
            raise ValueError('GC MoE requires fp8xfp4/swiglu')
        if base is not None and not isinstance(base, GCSymmBuffer):
            raise ValueError('base must be a GC-compatible GCSymmBuffer; foreign workspaces are unsupported')
        self.group = group
        self.rank_idx = dist.get_rank(group=group)
        self.num_experts = num_experts
        if compact_decode_workspace and (base is not None or not 1 <= num_max_tokens_per_rank <= 64):
            raise ValueError('compact decode storage needs an owned buffer of at most 64 tokens per rank')
        # The tested decode M sizes divide 64. Keep the independent routed
        # ring alignment and its scale layout unchanged.
        self.num_max_tokens_per_rank = align(
            num_max_tokens_per_rank, 64 if compact_decode_workspace else TOKEN_ALIGNMENT)
        self.compact_decode_workspace = compact_decode_workspace
        if push_scale_receiver_sms not in (0, 4, 8, 12) or (push_scale_receiver_sms and hidden % 512):
            raise ValueError('packed push scales need 4, 8 or 12 receiver SMs and vector-aligned scales')
        self.push_scale_receiver_sms = push_scale_receiver_sms
        push_row_ready_flags = push_row_ready_flags or bool(push_scale_receiver_sms)
        push_early_counts = push_early_counts or push_row_ready_flags
        push_reserve_spans = push_reserve_spans or push_early_counts
        push_decode_activations = (push_decode_activations or push_overlap_reservation or
                                   push_reserve_spans or push_vector_io)
        if push_vector_io and (push_overlap_reservation or push_reserve_spans or hidden % 512):
            raise ValueError('vector pushes require whole 512-byte chunks and per-route reservation')
        self.push_vector_io = push_vector_io
        peer_decode_input = peer_decode_input or peer_decode_input_direct
        if peer_decode_token_fanout not in (0, 1, 2):
            raise ValueError('decode token fanout uses one or multiple destination lanes')
        if peer_decode_token_fanout and not peer_decode_input:
            raise ValueError('decode token fanout requires the destination mailbox')
        self.peer_decode_token_fanout = peer_decode_token_fanout
        if peer_prefetch_expert_heads not in (0, 1, 2) or (
                peer_prefetch_expert_heads and (not peer_decode_input or peer_decode_input_direct)):
            raise ValueError('selective activation prefetch needs the destination mailbox')
        if peer_prefetch_opportunistic and not peer_prefetch_expert_heads:
            raise ValueError('opportunistic activation prefetch requires selected expert heads')
        self.peer_prefetch_expert_heads = peer_prefetch_expert_heads
        self.peer_prefetch_opportunistic = peer_prefetch_opportunistic
        if packed_decode_scores and (inline_route_scores or multicast_decode_input or
                                     push_decode_activations or peer_decode_input):
            raise ValueError('packed decode scores require ordinary receiver-side TMA pulls')
        packed_decode_activations = (packed_decode_activations or multicast_decode_input or
                                     push_decode_activations or peer_decode_input or packed_decode_scores)
        self.packed_decode_scores = packed_decode_scores
        if peer_decode_input and (multicast_decode_input or not 2 <= group.size() <= 32):
            raise ValueError('destination-only decode input needs 2 to 32 unicast peers')
        if packed_decode_activations and (base is not None or dedup_activations):
            raise ValueError('packed decode activations need owned storage and one transfer per route')
        self.packed_decode_activations = packed_decode_activations
        self.peer_decode_input_direct = peer_decode_input_direct
        if trim_decode_gather_tail and not peer_decode_input_direct:
            raise ValueError('gather tail trimming requires in-place mailbox activations')
        self.trim_decode_gather_tail = trim_decode_gather_tail
        packed_row_bytes = hidden + hidden // 32
        self.peer_decode_input_stride = (align(packed_row_bytes, 128)
                                         if peer_decode_input_direct else packed_row_bytes)
        self.token_major_combine = token_major_combine
        self.num_topk = num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type, self.activation = mma_type, activation
        # Opt-in experiment. Every rank must use the same mode and contribute
        # at most 32 distinct tokens, including in asymmetric batches.
        if push_decode_activations and early_experts is None:
            early_experts = 0
        if early_experts is not None:
            if base is not None or not 0 <= early_experts <= num_experts // group.size():
                raise ValueError('invalid expert-first decode workspace')
        self.early_experts = early_experts
        if split_route_heads and early_experts not in (1, 2):
            raise ValueError('split route publication requires one or two early experts')
        self.split_route_heads = split_route_heads
        if speculative_w1 and not early_experts:
            raise ValueError('speculative W1 requires an early expert')
        if speculative_packed_w1 and (not speculative_w1 or early_experts != 1 or split_route_heads):
            raise ValueError('speculative packed W1 requires one early expert and ordinary route publication')
        if (speculative_w1_n_tiles not in (0, 1, 2, 4, 8) or
                (speculative_w1_n_tiles and not speculative_packed_w1)):
            raise ValueError('partial speculative W1 requires one packed expert head')
        self.speculative_w1 = speculative_w1
        self.speculative_packed_w1 = speculative_packed_w1
        self.speculative_w1_n_tiles = speculative_w1_n_tiles
        if dual_weight_warps and stage_alignment == 1:
            stage_alignment = 2
        if (stage_alignment not in (1, 2, 4) or release_group not in (1, 2, 4) or
                stage_alignment % release_group):
            raise ValueError('invalid grouping of decode pipeline credits')
        self.stage_alignment, self.release_group = stage_alignment, release_group
        self.direct_w1_store = direct_w1_store
        self.narrow_bm32 = narrow_bm32
        self.short_activation_sf = short_activation_sf
        self.single_mma_lane = single_mma_lane
        if trace_stride not in (0, 1, 2, 4, 8, 16):
            raise ValueError('invalid decode trace sample stride')
        if decode_trace and trace_stride == 0 and (trace_operand_wait or trace_operand_sources):
            raise ValueError('CTA-only trace does not collect MMA operand timestamps')
        self.trace_stride = trace_stride if decode_trace else 1
        self.trace_operand_wait = decode_trace and (trace_operand_wait or trace_operand_sources)
        self.trace_operand_sources = decode_trace and trace_operand_sources
        if mma_unroll not in (2, 4, 8, 16):
            raise ValueError('invalid MMA loop unroll')
        self.reuse_k128_desc, self.mma_unroll = reuse_k128_desc, mma_unroll
        self.hoist_routed_sfb_selection = hoist_routed_sfb_selection
        self.programmatic_finalize = programmatic_finalize
        self.recycle_dispatch_sms = recycle_dispatch_sms
        self.static_w1_prefix = static_w1_prefix
        self.direct_w2_store = direct_w2_store
        if compact_compute_ctas not in (0, 1, 2):
            raise ValueError('compact compute CTA multiplier must be 0, 1, or 2')
        self.compact_compute_ctas = compact_compute_ctas
        if decode_dispatch_warps not in (0, 8, 12, 16, 24):
            raise ValueError('invalid decode Dispatch warp count')
        self.decode_dispatch_warps = decode_dispatch_warps
        if dispatch_pull_rows_per_cta not in (0, 1, 2, 4, 8):
            raise ValueError('Dispatch pull striping uses 1, 2, 4 or 8 rows per CTA')
        self.dispatch_pull_rows_per_cta = dispatch_pull_rows_per_cta
        self.double_buffer_dispatch_pull = double_buffer_dispatch_pull
        if type(weight_lookahead) is not int or not 0 <= weight_lookahead <= 32:
            raise ValueError('weight lookahead must be an integer from 0 to 32 K128 tiles')
        self.weight_lookahead = weight_lookahead
        if packed_route_count_group not in (1, 3, 4, 8):
            raise ValueError('packed route count groups must contain one, three, four or eight experts')
        self.packed_route_counts = (packed_route_counts or packed_route_count_group > 1 or
                                    (inline_route_indices and not posted_route_sums))
        self.packed_route_count_group = packed_route_count_group
        self.inline_route_indices = inline_route_indices
        self.rank_route_doorbell = rank_route_doorbell
        self.posted_route_sums = posted_route_sums
        self.inline_route_scores = inline_route_scores
        if rotate_route_publication not in (0, 1, 2):
            raise ValueError('route publication rotation uses ranks or ranks and experts')
        self.rotate_route_publication = rotate_route_publication
        self.prune_decode_resets = prune_decode_resets
        self.reset_decode_ring_on_dispatch = reset_decode_ring_on_dispatch
        self.decode_linear_pool = decode_linear_pool
        if native_decode_dispatch not in (0, 1, 2, 3):
            raise ValueError('native decode Dispatch uses disabled, prefix, arrival or planned pulls')
        self.native_decode_dispatch = native_decode_dispatch
        self.omit_sentinel_peer_finish = omit_sentinel_peer_finish
        if decode_vector_pull_group not in (1, 2, 4):
            raise ValueError('vector activation pull supports groups of one, two or four')
        self.decode_vector_pull_group = decode_vector_pull_group
        self.route_count_doorbell = route_count_doorbell
        self.dispatch_shared_route_counts = dispatch_shared_route_counts
        multicast_combine_ready = multicast_combine_ready or expert_ready_combine
        multicast_signals = (multicast_route_counts or multicast_combine_ready or
                             multicast_decode_input or multicast_pull_routes)
        if multicast_signals and (
                base is not None or not 2 <= group.size() <= 32):
            raise ValueError('multicast signaling requires owned storage across 2 to 32 ranks')
        if multicast_pull_routes and (not pull_route_metadata or num_topk % 2):
            raise ValueError('multicast route metadata needs receiver routing and even top-k')
        self.multicast_pull_routes = multicast_pull_routes
        if warp_pull_routes and (not pull_route_metadata or not self.packed_route_counts):
            raise ValueError('warp-local route scans need per-source count publication')
        self.warp_pull_routes = warp_pull_routes
        self.multicast_route_counts = multicast_route_counts
        self.multicast_combine_ready = multicast_combine_ready
        if stripe_active_peers and not early_stripe_combine:
            raise ValueError('per-token peer readiness requires striped combine')
        self.stripe_active_peers = stripe_active_peers
        early_sentinel = sentinel_combine and programmatic_finalize
        if (stripe_launch_delay not in (0, 1, 2) or
                (stripe_launch_delay and not (early_stripe_combine or early_sentinel))):
            raise ValueError('delayed finalize launch requires early Combine')
        self.stripe_launch_delay = stripe_launch_delay
        if stripe_tiles not in (1, 2, 4, 6) or (stripe_tiles != 1 and not early_stripe_combine):
            raise ValueError('grouped readiness requires striped combine')
        self.stripe_tiles = stripe_tiles
        if activation_prefix_bytes not in (0, 512, 1024, 2048, 4096) or (
                activation_prefix_bytes and (base is not None or activation_prefix_bytes >= hidden)):
            raise ValueError('activation prefixes require owned storage and a nonempty remainder')
        self.activation_prefix_bytes = activation_prefix_bytes
        if parallel_activation_prefix and not activation_prefix_bytes:
            raise ValueError('parallel activation pulls require an activation prefix')
        self.parallel_activation_prefix = parallel_activation_prefix
        if peer_activation_prefix_tiles not in (0, 1, 2, 4) or (peer_activation_prefix_tiles and (
                base is not None or group.size() < 2 or activation_prefix_bytes)):
            raise ValueError('direct peer prefixes need owned buffers and one, two or four K128 tiles')
        self.peer_activation_prefix_tiles = peer_activation_prefix_tiles
        if peer_activation_prefix_gather and not peer_activation_prefix_tiles:
            raise ValueError('peer activation gather needs a nonempty direct prefix')
        if peer_activation_head_clusters not in (0, 1, 2, 4, 8) or (
                peer_activation_head_clusters and not peer_activation_prefix_gather):
            raise ValueError('selective peer prefixes need asynchronous gathers')
        self.peer_activation_head_clusters = peer_activation_head_clusters
        self.peer_activation_window = None
        self.single_epilogue_warpgroup = single_epilogue_warpgroup
        if decode_accumulator_stages not in (2, 3, 4, 6):
            raise ValueError('invalid decode accumulator depth')
        self.decode_accumulator_stages = decode_accumulator_stages
        if decode_tma_stages not in (0, 4, 6, 8, 10):
            raise ValueError('decode TMA stage limit must be 0, 4, 6, 8 or 10')
        self.decode_tma_stages = decode_tma_stages
        if decode_task_slots not in (2, 4, 8):
            raise ValueError('task queue needs two, four, or eight slots')
        self.decode_task_slots = decode_task_slots
        self.fused_dispatch_compute = fused_dispatch_compute or fused_worker_reuse
        self.fused_compute_finalize = fused_compute_finalize
        self.decode_cooperative_compute = decode_cooperative_compute
        if fused_finalize_flags and (not fused_compute_finalize or base is not None):
            raise ValueError('compute completion flags require owned fused-finalize storage')
        if fused_finalize_sentinel not in (0, 1, 2) or (fused_finalize_sentinel and (
                not fused_finalize_flags or not sentinel_combine or programmatic_finalize or
                sentinel_combine_warps != 4 or combine_in_compute or combine_ready_flags)):
            raise ValueError('fused sentinel Combine requires four warps and compute completion flags')
        self.fused_finalize_sentinel = fused_finalize_sentinel
        self.fused_worker_reuse = fused_worker_reuse
        self.push_decode_activations = push_decode_activations
        self.push_overlap_reservation = push_overlap_reservation or push_reserve_spans
        self.push_reserve_spans = push_reserve_spans
        self.dispatch_prefix_scan = dispatch_prefix_scan
        if static_decode_wave not in (0, 2, 4, 6, 8, 16):
            raise ValueError('static compute waves need 2, 4, 8 or 16 M blocks')
        self.static_decode_wave = static_decode_wave
        if static_w2_overlap and not static_decode_wave:
            raise ValueError('static W2 overlap requires a W1 prefix length')
        self.static_w2_overlap = static_w2_overlap
        if decode_w1_warmup not in (0, 1, 2, 3, 4) or decode_w1_cadence not in (1, 2, 3):
            raise ValueError('invalid decode W1 warmup or interleave size')
        self.decode_w1_warmup = decode_w1_warmup
        self.decode_w1_cadence = decode_w1_cadence
        if decode_shared_w2 not in (0, 1, 2, 3, 4):
            raise ValueError('invalid shared W2 schedule')
        self.decode_shared_w2 = decode_shared_w2
        if decode_w2_clusters not in (0, 8, 16, 24, 32):
            raise ValueError('invalid number of decode clusters assigned to W2 at startup')
        self.decode_w2_clusters = decode_w2_clusters
        if decode_w2_ready_k not in (0, 1, 4, 8, 16):
            raise ValueError('unsupported W2 readiness prefix')
        self.decode_w2_ready_k = decode_w2_ready_k
        if w2_a_issue_window not in (0, 2, 4, 8):
            raise ValueError('W2 activation issue window needs two, four or eight K tiles')
        self.w2_a_issue_window = w2_a_issue_window
        if w2_b_issue_prefix not in (0, 2, 4, 8):
            raise ValueError('W2 weight issue prefix needs two, four or eight K tiles')
        self.w2_b_issue_prefix = w2_b_issue_prefix
        self.flat_decode_graph = flat_decode_graph
        self.combine_at_block_start = combine_at_block_start
        if decode_nvlink_schedule not in (0, 1, 2, 3):
            raise ValueError('NVLink scheduling can target Dispatch, compute, or both')
        self.decode_nvlink_schedule = decode_nvlink_schedule
        if decode_compute_max_registers not in (0, 136, 144):
            raise ValueError('decode compute register budget must be 136 or 144')
        self.decode_compute_max_registers = decode_compute_max_registers
        if decode_cluster_policy not in (0, 1):
            raise ValueError('decode cluster policy must be default or spread')
        self.decode_cluster_policy = decode_cluster_policy
        if decode_cluster_stride not in (0, 3, 5, 7):
            raise ValueError('decode cluster stride must be 3, 5, or 7')
        self.decode_cluster_stride = decode_cluster_stride
        self.keep_routed_weight_scales = keep_routed_weight_scales
        if decode_w2_early_workers not in (0, 2, 4, 8, 12, 16):
            raise ValueError('early W2 workers must be one of the supported cluster counts')
        self.decode_w2_early_workers = decode_w2_early_workers
        if compute_backfill_sms not in range(0, 25, 4):
            raise ValueError('decode compute backfill uses multiples of four SMs up to 24')
        self.compute_backfill_sms = compute_backfill_sms
        self.dispatch_first = dispatch_first or bool(compute_backfill_sms)
        if combine_token_warps not in (0, 2, 4, 8):
            raise ValueError('token-local Combine uses 2, 4 or 8 warps per token group')
        self.combine_token_warps = combine_token_warps
        if (w1_prefetch_k_tiles not in (0, 2, 4, 8, 16, 24) or
                (w1_prefetch_k_tiles > 8 and not bulk_w1_prefetch_bytes) or
                w1_prefetch_k_tiles * 128 > hidden):
            raise ValueError('invalid routed W1 cache prefetch prefix')
        self.w1_prefetch_k_tiles = w1_prefetch_k_tiles
        if (not 0 <= w1_prefix_expert_limit <= num_experts // group.size() or
                (w1_prefix_expert_limit and not w1_prefetch_k_tiles)):
            raise ValueError('the W1 prefetch limit must cover existing expert prefixes')
        self.w1_prefix_expert_limit = w1_prefix_expert_limit
        if keep_w1_prefix and not w1_prefetch_k_tiles:
            raise ValueError('preserving a W1 prefix requires W1 weight prefetch')
        self.keep_w1_prefix = keep_w1_prefix
        if w2_prefetch_k_tiles not in (0, 2, 4, 8) or w2_prefetch_k_tiles * 128 > intermediate_hidden:
            raise ValueError('invalid routed W2 cache prefetch prefix')
        self.w2_prefetch_k_tiles = w2_prefetch_k_tiles
        if not 0 <= bulk_w2_tail_experts <= num_experts // group.size():
            raise ValueError('Dispatch-tail W2 prefetch range must fit local experts')
        if bulk_w2_tail_experts and (not w2_prefetch_k_tiles or prefetch_weight_scales):
            raise ValueError('Dispatch-tail W2 prefetch needs a weight-only K prefix')
        self.bulk_w2_tail_experts = bulk_w2_tail_experts
        if prefetch_weight_scales and not (w1_prefetch_k_tiles or w2_prefetch_k_tiles):
            raise ValueError('scale prefetch requires a routed weight prefix')
        self.prefetch_weight_scales = prefetch_weight_scales
        if prefetch_k_first and not (w1_prefetch_k_tiles or w2_prefetch_k_tiles):
            raise ValueError('K-first order requires a routed weight prefix')
        self.prefetch_k_first = prefetch_k_first
        if weight_prefix_warps and not (w1_prefetch_k_tiles or w2_prefetch_k_tiles):
            raise ValueError('dedicated prefetch warps require a routed weight prefix')
        self.weight_prefix_warps = weight_prefix_warps
        if shared_priority_prefix not in (0, 1, 2):
            raise ValueError('select no Shared protection, W1, or W1 and W2')
        self.shared_priority_prefix = shared_priority_prefix
        if post_init_prefetch and not (w1_prefetch_k_tiles or w2_prefetch_k_tiles):
            raise ValueError('post-init prefetch requires a routed weight prefix')
        self.post_init_prefetch = post_init_prefetch
        self.relaxed_init_barrier = relaxed_init_barrier
        self.skip_init_prefetch_barrier = skip_init_prefetch_barrier
        if scheduled_weight_tiles not in (0, 8, 16):
            raise ValueError('scheduled prefetch uses up to 8 or 16 K128 tiles per task')
        self.scheduled_weight_tiles = scheduled_weight_tiles
        if w2_prefetch_from_w1 not in (0, 2, 4, 8) or (w2_prefetch_from_w1 and (
                scheduled_weight_tiles or w2_prefetch_k_tiles or
                w2_prefetch_from_w1 * 128 > intermediate_hidden)):
            raise ValueError('select one W2 weight prefetch schedule')
        self.w2_prefetch_from_w1 = w2_prefetch_from_w1
        if combine_in_compute not in (0, 1, 2, 3):
            raise ValueError('unsupported compute-side Combine placement')
        if combine_in_compute == 3 and decode_shared_w2 != 3:
            raise ValueError('shared epilogue Combine requires shared W2 after routed work')
        self.combine_in_compute = combine_in_compute
        if decode_w2_n_group not in (1, 2, 4):
            raise ValueError('W2 N-first scheduling groups one, two, or four M blocks')
        self.decode_w2_n_group = decode_w2_n_group
        self.w2_tile_major_input = w2_tile_major_input
        if transpose_w2_activations not in (0, 1, 2) or (
                transpose_w2_activations and not w2_tile_major_input):
            raise ValueError('transpose all W2 tiles or only tiles with more than 16 rows')
        self.transpose_w2_activations = transpose_w2_activations
        if grouped_routed_weights not in (0, 1, 2, 3):
            raise ValueError('grouped K pairs can be enabled for routed W1, W2, or both')
        self.grouped_routed_weights = grouped_routed_weights
        if packed_routed_scale_base is not None and (
                type(packed_routed_scale_base) is not int or
                not 0 <= packed_routed_scale_base <= 252 or group.size() > 32):
            raise ValueError('packed routed scales require a four-exponent UE8M0 window and EP at most 32')
        self.packed_routed_scale_base = packed_routed_scale_base
        if packed_routed_scale_k_group not in (1, 4) or (
                packed_routed_scale_k_group != 1 and packed_routed_scale_base is None):
            raise ValueError('grouped packed scales require four K tiles and a valid exponent base')
        self.packed_routed_scale_k_group = packed_routed_scale_k_group
        if packed_scale_helper and packed_routed_scale_k_group != 4:
            raise ValueError('scale helper requires four-tile scale packets')
        self.packed_scale_helper = packed_scale_helper
        self.follower_scheduler = follower_scheduler or packed_scale_helper
        if packed_scale_release_group not in (1, 2, 4) or (packed_scale_release_group != 4 and not packed_scale_helper):
            raise ValueError('packed scale groups are released in one, two or four slots by a helper')
        self.packed_scale_release_group = packed_scale_release_group
        if single_tma_lanes not in (0, 1, 2):
            raise ValueError('single-lane TMA can drive weights or both operands')
        self.single_tma_lanes = single_tma_lanes
        if decode_descriptor_prefetch not in ('all', 'weights', 'off'):
            raise ValueError('invalid decode tensor-map prefetch mode')
        self.decode_descriptor_prefetch = decode_descriptor_prefetch
        if decode_operand_cache not in ('auto', 'activation-normal', 'activation-first',
                                        'weights-normal', 'weights-last'):
            raise ValueError('invalid decode operand cache policy')
        self.decode_operand_cache = decode_operand_cache
        self.paired_shared_weights = paired_shared_weights
        if bulk_w1_prefetch_bytes not in (0, 16 * 1024, 64 * 1024):
            raise ValueError('routed W1 bulk prefetch needs 16 KiB or 64 KiB requests')
        if bulk_w1_prefetch_on_dispatch not in (0, 1, 2):
            raise ValueError('Dispatch W1 prefetch must be off, at startup or after route publication')
        if bulk_w1_prefetch_on_dispatch and not bulk_w1_prefetch_bytes:
            raise ValueError('Dispatch W1 prefetch needs a bulk request size')
        self.bulk_w1_prefetch_bytes = bulk_w1_prefetch_bytes
        self.bulk_w1_prefetch_on_dispatch = bulk_w1_prefetch_on_dispatch
        if bulk_w1_prefetch_lanes not in (1, 4, 8) or (bulk_w1_prefetch_lanes > 1 and
                (not bulk_w1_prefetch_bytes or bulk_w1_prefetch_on_dispatch)):
            raise ValueError('parallel bulk W1 prefetch needs the SM100 compute grid')
        self.bulk_w1_prefetch_lanes = bulk_w1_prefetch_lanes
        if probe_routed_mma not in (0, 1, 2, 3):
            raise ValueError('routed MMA probe uses a W1/W2 bit mask')
        self.probe_routed_mma = probe_routed_mma
        self.w1_register_transpose = w1_register_transpose
        self.w1_packed_scale_reduction = w1_packed_scale_reduction
        self.w1_hardware_scale_reduction = w1_hardware_scale_reduction
        self.w1_early_epi_inputs = w1_early_epi_inputs
        self.w1_pair_scale_rounding = w1_pair_scale_rounding
        self.prefetch_w2_metadata = prefetch_w2_metadata
        self.packed_w2_metadata = packed_w2_metadata
        self.pair_routed_weight_scales = pair_routed_weight_scales
        if shared_w1_prefetch_limit not in (0, 24, 48):
            raise ValueError('shared W1 lookahead ends at K tile 24 or 48')
        self.shared_w1_prefetch_limit = shared_w1_prefetch_limit
        if asymmetric_activation_rows not in (0, 1, 2):
            raise ValueError('asymmetric activation loads can narrow or skip padded rows')
        self.asymmetric_activation_rows = asymmetric_activation_rows
        self.w1_tile_major_input = w1_tile_major_input
        if pair_routed_weights not in (0, 1, 2, 3):
            raise ValueError('paired FP4 TMA can target W1, W2, or both')
        self.pair_routed_weights = pair_routed_weights
        self.cache_routed_ranges = cache_routed_ranges
        self.single_expert_tile_lookup = single_expert_tile_lookup
        if prioritize_expert_rows not in (-1, 0, 1) or (prioritize_expert_rows and
                not single_expert_tile_lookup):
            raise ValueError('expert row priority requires single-tile lookup')
        self.prioritize_expert_rows = prioritize_expert_rows
        if decode_block_m not in (0, 32, 48, 64, 96, 128):
            raise ValueError('decode block M override must use 32, 48, 64, 96 or 128 rows')
        self.decode_block_m = decode_block_m
        self.trim_wide_decode_activations = trim_wide_decode_activations
        if keep_w2_prefix and not w2_prefetch_k_tiles:
            raise ValueError('preserving a W2 prefix requires W2 weight prefetch')
        self.keep_w2_prefix = keep_w2_prefix
        if route_publish_ctas not in (1, 2, 4, 8):
            raise ValueError('route publication allows 1, 2, 4 or 8 CTAs')
        self.route_publish_ctas = route_publish_ctas
        if decode_combine_fetch not in ('default', 'small', 'contiguous', 'contiguous-wide',
                                        'streaming', 'streaming-small'):
            raise ValueError('invalid decode Combine fetch layout')
        self.decode_combine_fetch = decode_combine_fetch
        self.direct_combine_output = direct_combine_output
        if base is not None and (sentinel_combine or base.sentinel_combine):
            raise ValueError('sentinel Combine requires an owned workspace')
        self.sentinel_combine = sentinel_combine
        if reuse_finalize_warps and (not sentinel_combine or not programmatic_finalize or combine_in_compute):
            raise ValueError('finalize warp reuse requires a separate early sentinel Combine')
        self.reuse_finalize_warps = reuse_finalize_warps
        if single_cleanup_cta and (not sentinel_combine or not programmatic_finalize or
                                   combine_in_compute or early_stripe_combine or expert_ready_combine):
            raise ValueError('single cleanup CTA requires standalone early sentinel Combine')
        self.single_cleanup_cta = single_cleanup_cta
        if combine_ready_mask and (not combine_ready_flags or sentinel_combine_vectors != 1):
            raise ValueError('ready masks need single-vector Combine and an owned flag allocation')
        if combine_ready_flags and (not sentinel_combine or not sentinel_chunk_major or
                                    (sentinel_poll != 'full' and
                                     not (combine_ready_mask and sentinel_poll == 'inline-prefix'))):
            raise ValueError('ready flags require ordinary chunk-major sentinel Combine')
        self.combine_ready_mask = combine_ready_mask
        if sentinel_poll_delay_ns not in (0, 64, 128, 256, 512, 768, 1024) or (
                sentinel_poll_delay_ns and not sentinel_combine):
            raise ValueError('invalid sentinel polling delay')
        if sentinel_adaptive_poll and not sentinel_poll_delay_ns:
            raise ValueError('adaptive polling needs an initial delay')
        self.sentinel_poll_delay_ns = sentinel_poll_delay_ns
        self.sentinel_adaptive_poll = sentinel_adaptive_poll
        if sentinel_chunk_major and (not sentinel_combine or token_major_combine):
            raise ValueError('chunk-major storage requires sentinel Combine')
        self.sentinel_chunk_major = sentinel_chunk_major
        if (sentinel_combine_vectors not in (1, 2) or sentinel_combine_sms not in (0, 32, 48, 64)
                or sentinel_combine_warps not in (4, 8)):
            raise ValueError('invalid sentinel Combine CTA geometry')
        if sentinel_poll not in ('full', 'probe', 'defer-reset', 'warp-probe',
                                 'relaxed-sys', 'ordered', 'prefix', 'inline-prefix',
                                 'shared-first', 'shared-prefix'):
            raise ValueError('invalid sentinel Combine polling mode')
        if (sentinel_poll in ('ordered', 'prefix', 'inline-prefix', 'shared-prefix') and
                sentinel_combine_vectors != 1):
            raise ValueError('ordered sentinel reduction uses one vector per lane')
        if sentinel_poll in ('shared-first', 'shared-prefix') and not num_shared_experts:
            raise ValueError('shared-first Combine depends on an early shared expert')
        if not sentinel_combine and (
                sentinel_combine_vectors != 1 or sentinel_combine_sms or sentinel_token_order
                or sentinel_poll != 'full' or sentinel_combine_warps != 8):
            raise ValueError('sentinel Combine tuning requires sentinel Combine')
        self.sentinel_combine_vectors = sentinel_combine_vectors
        if sentinel_half_vector and (not sentinel_combine or sentinel_combine_vectors != 1 or
                                     sentinel_poll != 'full' or combine_ready_flags or
                                     sentinel_chunk_major or combine_in_compute):
            raise ValueError('half-vector sentinel Combine requires full polling of the ordinary mailboxes')
        self.sentinel_half_vector = sentinel_half_vector
        self.sentinel_combine_sms = sentinel_combine_sms
        self.sentinel_token_order = sentinel_token_order
        self.sentinel_poll = sentinel_poll
        self.sentinel_combine_warps = sentinel_combine_warps
        self.dual_weight_warps = dual_weight_warps
        self.independent_compute_ctas = independent_compute_ctas
        self.narrow_local_mma = narrow_local_mma or independent_compute_ctas
        self.local_mma = local_mma or self.narrow_local_mma
        if local_mma_n8 not in (0, 1, 2) or (local_mma_n8 and not narrow_local_mma):
            raise ValueError('N8 tiles require narrow single-CTA MMA')
        self.local_mma_n8 = local_mma_n8
        if combine_load_group not in (1, 2, 4, 8):
            raise ValueError('Combine load group must be 1, 2, 4 or 8 slots')
        self.combine_load_group = combine_load_group
        num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        ring_tokens, sf_ring_tokens = get_ring_tokens(
            group.size(), num_experts, self.num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, num_sms)
        self.layout = GCBufferLayout(group.size(), num_experts,
                                     self.num_max_tokens_per_rank, num_topk,
                                     hidden, intermediate_hidden, ring_tokens,
                                     sf_ring_tokens, num_shared_experts, packed_decode_activations,
                                     packed_decode_scores)
        # Keep all operational views byte-identical. The profiler appends a
        # rank-local log to its allocation and clears it before each sample.
        self.decode_trace_offset = 0
        self.ready_tile_offset = 0
        self.stripe_combine_offset = 0
        self.expert_combine_offset = 0
        self.dedup_activation_offset = 0
        self.pull_route_metadata_offset = 0
        self.multicast_decode_input_offset = 0
        self.peer_decode_input_offset = 0
        self.activation_prefix_offset = 0
        self.compute_completion_offset = 0
        self.pair_shared_w1_offset = 0
        if pair_shared_w1 not in (0, 1, 2):
            raise ValueError('paired shared W1 supports individual or 16-byte output stores')
        self.pair_shared_w1_vector = pair_shared_w1 == 2
        self.push_early_pool_offset = 0
        self.push_row_ready_offset = 0
        self.push_packed_scales_offset = 0
        if source_elects_activation_owner and not dedup_activations:
            raise ValueError('source election requires deduplicated activations')
        self.source_elects_activation_owner = source_elects_activation_owner
        nbytes = self.layout.nbytes
        self.combine_ready_flags_offset = 0
        if combine_ready_flags:
            self.combine_ready_flags_offset = align(nbytes, 16)
            words_per_segment = 1 if combine_ready_mask else num_topk + 1
            num_flags = self.num_max_tokens_per_rank * (hidden // 128) * words_per_segment
            nbytes = align(self.combine_ready_flags_offset + num_flags * 4, 16)
        if expert_ready_combine:
            if early_stripe_combine:
                raise ValueError('expert and stripe Combine readiness use different queues')
            self.expert_combine_offset = align(nbytes, 16)
            local_experts = num_experts // group.size()
            nbytes = align(self.expert_combine_offset +
                           (local_experts + 1 + num_experts + group.size()) * 8, 16)
        if push_early_counts:
            self.push_early_pool_offset = align(nbytes, 16)
            nbytes = self.push_early_pool_offset + 16
        if push_row_ready_flags:
            self.push_row_ready_offset = nbytes
            nbytes = align(nbytes + ring_tokens * 4, 16)
        if push_scale_receiver_sms:
            self.push_packed_scales_offset = nbytes
            nbytes = align(nbytes + ring_tokens * (hidden // 32), 16)
        if fused_finalize_flags:
            self.compute_completion_offset = align(nbytes, 16)
            nbytes = align(self.compute_completion_offset + (num_sms + 1) * 4, 16)
        if pair_shared_w1:
            if base is not None or group.size() < 2 or group.size() % 2 or num_shared_experts != 1:
                raise ValueError('paired shared W1 requires owned storage and one shared expert on rank pairs')
            self.pair_shared_w1_offset = align(nbytes, 16)
            nbytes = self.pair_shared_w1_offset + 16
        if activation_prefix_bytes or peer_activation_prefix_tiles:
            self.activation_prefix_offset = align(nbytes, 16)
            nbytes = self.activation_prefix_offset + (ring_tokens // 16) * 4
        if multicast_decode_input or peer_decode_input:
            inbox = align(nbytes, 128)
            rows = 2 * group.size() * 32
            nbytes = inbox + rows * (self.peer_decode_input_stride + 8)
            if multicast_decode_input:
                self.multicast_decode_input_offset = inbox
            else:
                self.peer_decode_input_offset = inbox
        if pull_route_metadata:
            if base is not None or group.size() > 32:
                raise ValueError('receiver-driven routing needs owned storage and at most 32 ranks')
            self.pull_route_metadata_offset = align(nbytes, 16)
            # Each multicast receiver gets a distinct compact row for each source.
            sources = group.size() if multicast_pull_routes else 1
            nbytes = (self.pull_route_metadata_offset + 2 * group.size() * 8 +
                      2 * sources * 32 * num_topk * 4)
        if dedup_activations:
            if base is not None or group.size() * 32 > ring_tokens:
                raise ValueError('deduplicated activation storage does not fit this owned ring')
            self.dedup_activation_offset = align(nbytes, 16)
            nbytes = align(self.dedup_activation_offset + group.size() * 32 * 4, 16)
        if early_stripe_combine:
            if base is not None or early_experts is not None or hidden % (256 * stripe_tiles):
                raise ValueError('striped combine needs contiguous decode slots and whole channel groups')
            self.stripe_combine_offset = align(nbytes, 16)
            # One local task count and one readiness epoch per peer for each
            # channel group. Only local counters are reset on each replay.
            nbytes = align(self.stripe_combine_offset + hidden // (256 * stripe_tiles) *
                           (group.size() + 1) * 8, 16)
        if ready_tile_order:
            if base is not None or early_experts is not None:
                raise ValueError('ready tile ordering needs unaliased, contiguous routed slots')
            self.ready_tile_offset = align(nbytes, 16)
            # One counter and a publication slot per routed M block. The
            # supported call geometry is checked before all three roles launch.
            nbytes = align(self.ready_tile_offset + (1 + ring_tokens // 16) * 4, 16)
        if decode_trace:
            from ..common.decode_trace import TRACE_BYTES

            if base is not None:
                raise ValueError('decode tracing requires an owned allocation')
            self.decode_trace_offset = align(nbytes, 16)
            nbytes = self.decode_trace_offset + TRACE_BYTES
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(nbytes, dtype=torch.int8, device='cuda')
            self.handle = (types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                           if group.size() == 1 else symm_mem.rendezvous(self.buffer, group=group))
            peer_addresses = tuple(int(ptr) for ptr in self.handle.buffer_ptrs)
            if (len(peer_addresses) != group.size() or
                    any(ptr <= 0 or ptr % 16 for ptr in peer_addresses) or
                    peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
                raise RuntimeError('symmetric rendezvous returned invalid peer addresses')
            self.peer_addresses = peer_addresses
            self.buffer.zero_()
            if sentinel_combine and not combine_ready_flags:
                combine = self.layout.views['combine_tokens']
                self.buffer.narrow(0, combine.offset, combine.nbytes).fill_(-1)
                torch.cuda.current_stream().synchronize()
            group.barrier().wait()
            torch.cuda.synchronize()
        else:
            if base.buffer is None or base.handle is None or base.group is not group:
                raise ValueError('base must be a live allocation in the same group')
            for name in ('num_experts', 'num_max_tokens_per_rank', 'num_topk', 'hidden',
                         'intermediate_hidden', 'num_shared_experts', 'mma_type', 'activation'):
                if getattr(base, name) != getattr(self, name):
                    raise ValueError(f"incompatible base: {name}")
            if self.layout != base.layout:
                raise ValueError('incompatible base: GC workspace layout')
            if self.layout.nbytes > base.buffer.nbytes:
                raise ValueError('base allocation too small')
            self.buffer, self.handle = base.buffer, base.handle
            peer_addresses = tuple(int(ptr) for ptr in self.handle.buffer_ptrs)
            if (len(peer_addresses) != group.size() or
                    any(ptr <= 0 or ptr % 16 for ptr in peer_addresses) or
                    peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
                raise RuntimeError('base rendezvous returned invalid peer addresses')
            # A compatible base may own the allocation. The 592-byte host
            # SymBuffer value is assembled per call from these rendezvous UVA
            # addresses and retained by each child Graph argument vector.
            self.peer_addresses = peer_addresses
        if peer_activation_prefix_gather:
            from ..runtime.peer_activation import PeerActivationWindow

            self.peer_activation_window = PeerActivationWindow(
                self.peer_addresses, self.buffer.device, self.layout.views['x'].offset, hidden, 32)
        if multicast_signals:
            multicast_ptr = int(getattr(self.handle, 'multicast_ptr', 0) or 0)
            if not multicast_ptr:
                raise ValueError('this symmetric allocation has no NVLink multicast mapping')
        for name in self._views:
            spec = self.layout.views[name]
            if spec.nbytes == 0:
                view = None
            elif name == 'shared_l1_acts':
                view = self.x
            else:
                dtype = getattr(torch, spec.dtype)
                elem_size = torch.empty((), dtype=dtype).element_size()
                raw = self.buffer.narrow(0, spec.offset, spec.nbytes).view(dtype)
                view = raw.as_strided(spec.shape, spec.stride) if spec.stride else raw.view(spec.shape)
                if spec.nbytes % elem_size:
                    raise AssertionError(f"unaligned view {name}")
            setattr(self, name, view)

    def _for_call(self, num_shared_experts: int):
        """Re-slice the raw allocation for the shared count of this call."""
        if self.buffer is None or self.handle is None:
            raise ValueError('symmetric buffer has been destroyed')
        if num_shared_experts == self.num_shared_experts:
            return self
        layout = GCBufferLayout(
            self.group.size(), self.num_experts,
            self.num_max_tokens_per_rank, self.num_topk,
            self.hidden, self.intermediate_hidden,
            self.layout.ring_tokens, self.layout.sf_ring_tokens,
            num_shared_experts, self.packed_decode_activations, self.packed_decode_scores)
        if layout.nbytes > self.layout.nbytes:
            raise ValueError("symmetric buffer is too small for call's shared experts")
        return _CallGCSymmBuffer(self, layout)

    def destroy(self):
        if self.peer_activation_window is not None:
            self.peer_activation_window.close()
            self.peer_activation_window = None
        for name in self._views:
            setattr(self, name, None)
        self.peer_addresses = self.handle = self.buffer = self.group = None


class _CallGCSymmBuffer:
    """Reference C++ call-time slice over the caller-owned symmetric bytes."""

    def __init__(self, base: GCSymmBuffer, layout: GCBufferLayout):
        self._base = base
        self.layout = layout
        self.num_shared_experts = layout.num_shared_experts
        for name in base._views:
            spec = layout.views[name]
            if spec.nbytes == 0:
                view = None
            elif name == 'shared_l1_acts':
                view = self.x
            else:
                dtype = getattr(torch, spec.dtype)
                elem_size = torch.empty((), dtype=dtype).element_size()
                raw = base.buffer.narrow(0, spec.offset, spec.nbytes).view(dtype)
                view = raw.as_strided(spec.shape, spec.stride) if spec.stride else raw.view(spec.shape)
                if spec.nbytes % elem_size:
                    raise AssertionError(f"unaligned call view {name}")
            setattr(self, name, view)

    def __getattr__(self, name):
        return getattr(self._base, name)


def get_symm_buffer_for_gc_moe(group: dist.ProcessGroup, num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 mma_type: str = 'fp8xfp4', activation: str = 'swiglu') -> GCSymmBuffer:
    return GCSymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                        hidden, intermediate_hidden, num_shared_experts, mma_type, activation)


def _gc_interleave_rows(t: torch.Tensor) -> torch.Tensor:
    if t.ndim not in (2, 3) or t.shape[-2] % 16:
        raise ValueError('W1 rows must be a multiple of 16')
    grouped = t if t.ndim == 3 else t.unsqueeze(0)
    groups, rows = grouped.shape[:2]
    half = rows // 2
    gate = grouped[:, :half].reshape(groups, half // 8, 8, *grouped.shape[2:])
    up = grouped[:, half:].reshape(groups, half // 8, 8, *grouped.shape[2:])
    result = torch.empty_like(grouped).copy_(torch.stack((gate, up), dim=2).reshape(grouped.shape))
    return result if t.ndim == 3 else result.squeeze(0)


def _gc_transpose_sf(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype != torch.int32 or sf.ndim not in (2, 3) or sf.shape[-2] % 128:
        raise ValueError('SF must be packed int32 with 128-aligned rows')
    grouped = sf if sf.ndim == 3 else sf.unsqueeze(0)
    groups, rows, packs = grouped.shape
    result = torch.empty_like(grouped).copy_(
        grouped.reshape(groups, -1, 4, 32, packs).transpose(2, 3).reshape(grouped.shape))
    return result if sf.ndim == 3 else result.squeeze(0)


def transform_weights_for_gc_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor], activation: str = 'swiglu',
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Convert original quantized weights to owned K128 (payload, SF) pairs.

    Shared K128 SF must already be expanded to one packed row per N, as for
    the original transform. Call outside capture with ordered source writes.
    Copies run asynchronously on the current CUDA stream; raw inputs may die
    on return.
    """
    from ..jit_kernels.impls.runtime_utils import repack_fp4_weights, repack_fp8_weights
    if activation != 'swiglu':
        raise ValueError('GC MoE supports swiglu')
    pairs = (l1_weights, l2_weights)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 or
           any(not isinstance(t, torch.Tensor) for t in pair) for pair in pairs):
        raise ValueError('transform needs two original quantized (payload, SF) pairs')
    if any(not t.is_cuda or t.device.index != torch.cuda.current_device()
           for pair in pairs for t in pair):
        raise ValueError('transform inputs must use the current CUDA device')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('transform weights outside CUDA Graph capture')
    stream = torch.cuda.current_stream()
    for pair in pairs:
        for tensor in pair:
            tensor.record_stream(stream)
    source_storage = {pair[0].untyped_storage().data_ptr() for pair in pairs}
    repack = repack_fp4_weights if l1_weights[0].dtype == torch.int8 else repack_fp8_weights
    packed = [repack(_gc_interleave_rows(l1_weights[0])), repack(l2_weights[0])]
    # A degenerate K128 permutation may return a view of its input.
    packed = [t.clone() if t.untyped_storage().data_ptr() in source_storage else t
              for t in packed]
    sf1 = _gc_transpose_sf(_gc_interleave_rows(l1_weights[1]))
    sf2 = _gc_transpose_sf(l2_weights[1])
    return (packed[0], sf1), (packed[1], sf2)


def fp8_fp4_gc_moe(y: torch.Tensor,
                    l1_weights: Tuple[torch.Tensor, torch.Tensor],
                    l2_weights: Tuple[torch.Tensor, torch.Tensor],
                    sym_buffer: GCSymmBuffer,
                    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                    recipe: Tuple[int, int, int] = (1, 1, 32),
                    activation: str = 'swiglu', activation_clamp: Optional[float] = None,
                    fast_math: bool = True,
                    shared_recipe: Optional[Tuple[int, int, int]] = None,
                    activation_alpha: Optional[float] = None, *,
                    decode_mode: str = 'off', weight_reuse: str = 'off',
                    prefetch_experts: int = 0) -> None:
    if hasattr(sym_buffer, 'num_shared_experts'):
        assert (sym_buffer.num_shared_experts > 0) == (shared_l1_weights is not None)
    from ..apis.gc_moe import fp8_fp4_gc_moe as launch
    launch(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights,
           shared_l2_weights, cumulative_local_expert_recv_stats, recipe,
           activation, activation_clamp, fast_math, shared_recipe, activation_alpha,
           decode_mode=decode_mode, weight_reuse=weight_reuse,
           prefetch_experts=prefetch_experts)
