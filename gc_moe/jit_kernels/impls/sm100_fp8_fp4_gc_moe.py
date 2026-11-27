"""Prepare the three CuTe GC role cubins and live TensorMaps for a call.

The supported H/I and token ranges are checked before graph launch. Each
role's PTX/cubin export is owned by its compile, including concurrent ranks.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from math import gcd, inf
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from cutlass import Float32, Uint32
from cutlass.cute.runtime import from_dlpack
import torch
from cuda.bindings import driver as cu

from ...comm.gc_barrier import DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS
from ...common.tma_copy import use_small_tile_path
from ...common.decode_trace import TRACE_CTA_FIELDS
from ...impls.sm100_decode import WARP_BYTES
from ...impls.sm100_fp8_fp4_gc_moe import GCMoERoleKernel
from ...layout.gc_moe import (
    PACKED_ROUTED_SCALE_OFFSETS,
    ROUTED_W1_PREFETCH_OFFSET,
    ROUTED_W2_PREFETCH_OFFSET,
    MAX_EXPERTS,
    MAX_RANKS,
    SIGNAL_OFFSETS,
    align,
    ceil_div,
)
from ...runtime.gc_moe import RoleBinary, get_context
from ...runtime.jit import (
    COMPILE_LOCK,
    PTXAS_OPTIONS,
    CompilerIdentity,
    assemble_ptx,
    compile_kernel,
    compiler_identity,
    device_compile_target,
    ptx_target,
)
from ..heuristics.gc_moe import (
    get_combine_geometry,
    get_role_config,
    num_max_live_pool_blocks,
)
from .runtime_utils import (
    TmaDescriptorSpec,
    is_routed_weight_k_major,
    make_tma_2d_desc,
    make_tma_sf_desc,
    make_tma_weights_2d_desc,
    make_tma_weights_3d_desc,
)


@dataclass(frozen=True)
class GCTmaPlan:
    """Original eighteen maps, optionally followed by four fused-SE prefix maps."""

    l1_acts: TmaDescriptorSpec
    l1_acts_sf: TmaDescriptorSpec
    l1_weights: TmaDescriptorSpec
    l1_weights_sf: TmaDescriptorSpec
    l1_output: TmaDescriptorSpec
    l2_acts: TmaDescriptorSpec
    l2_acts_sf: TmaDescriptorSpec
    l2_weights: TmaDescriptorSpec
    l2_weights_sf: TmaDescriptorSpec
    shared_l1_acts: TmaDescriptorSpec
    shared_l1_acts_sf: TmaDescriptorSpec
    shared_l1_weights: TmaDescriptorSpec
    shared_l1_weights_sf: TmaDescriptorSpec
    shared_l1_output: TmaDescriptorSpec
    shared_l2_acts: TmaDescriptorSpec
    shared_l2_acts_sf: TmaDescriptorSpec
    shared_l2_weights: TmaDescriptorSpec
    shared_l2_weights_sf: TmaDescriptorSpec

    l1_acts_16: TmaDescriptorSpec | None = None
    l2_acts_16: TmaDescriptorSpec | None = None
    l1_acts_24: TmaDescriptorSpec | None = None
    l2_acts_24: TmaDescriptorSpec | None = None

    def ordered(self) -> tuple[TmaDescriptorSpec, ...]:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        optional = values[18:]
        if any(value is None for value in optional):
            if not all(value is None for value in optional):
                raise ValueError('prefix maps must be all absent or all present')
            return values[:18]
        return values


def build_gc_tma_plan(config, sym_buffer, l1_weights, l2_weights,
                      shared_l1_weights=None, shared_l2_weights=None,
                      shared_weight_gran_k: int = 32, *,
                      narrow_bm32: bool = False, local_mma: bool = False,
                      narrow_local_mma: bool = False, trim_wide_tokens: int = 0) -> GCTmaPlan:
    """Describe routed/shared tensors; encode with live pointers at launch.

    Each weight tuple's [0] tensor is the tile-repacked payload prepared by
    the caller; scale factors and activations use their original storage.
    """
    bk, bn = config.block_k, config.block_n
    h, i = sym_buffer.hidden, sym_buffer.intermediate_hidden
    sf_m = config.sf_block_m
    ring, sf_ring = sym_buffer.layout.ring_tokens, sym_buffer.layout.sf_ring_tokens
    shared_count = sym_buffer.num_shared_experts
    shared_i = shared_count * i
    if shared_weight_gran_k not in (32, 128):
        raise ValueError('shared UE8M0 K granularity must be 32 or 128')
    if bool(shared_l1_weights) != bool(shared_l2_weights) or \
            bool(shared_l1_weights) != bool(shared_count):
        raise ValueError('shared tensors and expert count disagree')
    def acts(t, inner, outer, block_inner, block_outer, swizzle):
        return make_tma_2d_desc(t, inner, outer, block_inner, block_outer,
                         t.stride(-2), swizzle)

    def tile_acts(t, width, rows, load_m, tile_major):
        if tile_major:
            return make_tma_2d_desc(t, bk, rows * (width // bk), bk, load_m,
                                    bk, config.swizzle_acts)
        return acts(t, width, rows, bk, load_m, config.swizzle_acts)

    def w1_acts(t, rows, load_m):
        return tile_acts(t, h, rows, load_m, sym_buffer.w1_tile_major_input)

    def w2_acts(t, width, rows, load_m):
        return tile_acts(t, width, rows, load_m, sym_buffer.w2_tile_major_input)

    def w1_output(t, width, rows):
        if sym_buffer.transpose_w2_activations:
            # Direct W1 writes leave this map unused as a store descriptor.
            # Use its fixed ABI slot for the alternate W2 load instead.
            return make_tma_2d_desc(t, 32, rows * (width // 32), 16, bk, 32, 0)
        return acts(t, width, rows, bn // 2, config.store_m_l1, config.swizzle_output)

    def scales(t, mn, k, block_mn, gran, groups, pair=False):
        return make_tma_sf_desc(t, mn, k, block_mn, gran, groups,
                     2 if pair else (bk + gran * 4 - 1) // (gran * 4), t.stride(-1))

    def routed_scales(t, mn, k):
        if sym_buffer.packed_routed_scale_base is not None:
            # Routed scales load via LD/ST. Retain a valid descriptor for
            # the fixed role ABI and keep the shared-phase maps separate.
            return make_tma_2d_desc(t, 32, t.numel() // 32, 32, 1, 32, 0)
        return scales(t, mn, k, bn, 32, sym_buffer.num_experts // sym_buffer.group.size(),
                      pair=sym_buffer.pair_routed_weight_scales)

    routed_l1_acts, routed_l1_rows = sym_buffer.l1_acts, ring
    if sym_buffer.peer_decode_input_direct:
        routed_l1_rows = 2 * sym_buffer.group.size() * 32
        stride = sym_buffer.peer_decode_input_stride
        buffer = sym_buffer.buffer.narrow(0, sym_buffer.peer_decode_input_offset,
                                          routed_l1_rows * stride)
        routed_l1_acts = buffer.view(sym_buffer.l1_acts.dtype).as_strided(
            (routed_l1_rows, h), (stride, 1))
    routed_w1_output = w1_output(sym_buffer.l2_acts, i, ring)
    window = sym_buffer.peer_activation_window
    if window is not None:
        # Direct W1 stores leave this ABI slot available for the peer input map.
        routed_w1_output = TmaDescriptorSpec(window.address, torch.float8_e4m3fn,
                                             (h, window.rows), (h,), (bk, 1), config.swizzle_acts)
    routed = (
        w1_acts(routed_l1_acts, routed_l1_rows,
                1 if sym_buffer.dedup_activation_offset or sym_buffer.peer_decode_input_direct
                else config.load_m),
        scales(sym_buffer.l1_acts_sf, sf_ring, h, sf_m, 32, 1),
        make_tma_weights_3d_desc(l1_weights[0], bk, bn, swizzle=config.swizzle_weights,
                                 pair_k=bool(sym_buffer.pair_routed_weights & 1)),
        routed_scales(l1_weights[1], 2 * i, h),
        routed_w1_output,
        w2_acts(sym_buffer.l2_acts, i, ring, config.load_m),
        scales(sym_buffer.l2_acts_sf, sf_ring, i, sf_m, 32, 1),
        make_tma_weights_3d_desc(l2_weights[0], bk, bn, swizzle=config.swizzle_weights,
                                 pair_k=bool(sym_buffer.pair_routed_weights & 2)),
        routed_scales(l2_weights[1], h, i),
    )
    if shared_count:
        shared_sf_tokens = sym_buffer.shared_l1_acts_sf.shape[0]
        def shared_weights(phase, tensor):
            if (sym_buffer.paired_shared_weights is not None and
                    sym_buffer.paired_shared_weights[phase] is not None):
                packed = sym_buffer.paired_shared_weights[phase]
                return make_tma_2d_desc(packed, 256, packed.shape[0], 256, 130, 256, 0)
            return make_tma_weights_2d_desc(tensor, bk, bn, swizzle=config.swizzle_weights)

        shared = (
            acts(sym_buffer.shared_l1_acts, h,
                 sym_buffer.num_max_tokens_per_rank, bk, config.load_m, config.swizzle_acts),
            scales(sym_buffer.shared_l1_acts_sf, shared_sf_tokens,
                   h, sf_m, 32, 1),
            shared_weights(0, shared_l1_weights[0]),
            scales(shared_l1_weights[1], 2 * shared_i, h, bn,
                   shared_weight_gran_k, 1),
            w1_output(sym_buffer.shared_l2_acts, shared_i, sym_buffer.num_max_tokens_per_rank),
            w2_acts(sym_buffer.shared_l2_acts, shared_i,
                    sym_buffer.num_max_tokens_per_rank, config.load_m),
            scales(sym_buffer.shared_l2_acts_sf,
                   sym_buffer.shared_l2_acts_sf.shape[0],
                   shared_i, sf_m, 32, 1),
            shared_weights(1, shared_l2_weights[0]),
            scales(shared_l2_weights[1], h, shared_i, bn,
                   shared_weight_gran_k, 1),
        )
    else:
        # Supply routed maps in unused shared positions.
        shared = routed
    narrow = tuple(spec
                   for rows in (16, 24)
                   for spec in (w1_acts(sym_buffer.l1_acts, ring, min(rows, config.load_m)),
                                w2_acts(sym_buffer.l2_acts, i, ring, min(rows, config.load_m)))) \
        if not local_mma and use_small_tile_path(shared_count, config.load_m * 2) else ()
    if narrow_bm32 or narrow_local_mma:
        # One descriptor per phase with only the rows needed by a small tile.
        rows = 16 if narrow_local_mma and sym_buffer.local_mma_n8 != 2 else 8
        capacity = sym_buffer.num_max_tokens_per_rank
        narrow = (
            w1_acts(sym_buffer.l1_acts, ring, rows),
            w2_acts(sym_buffer.l2_acts, i, ring, rows),
            acts(sym_buffer.shared_l1_acts, h, capacity, bk, rows, config.swizzle_acts),
            w2_acts(sym_buffer.shared_l2_acts, shared_i, capacity, rows),
        )
    if trim_wide_tokens:
        capacity = sym_buffer.num_max_tokens_per_rank
        shared_rows = ceil_div(trim_wide_tokens, 16) * 8
        narrow = (
            acts(sym_buffer.shared_l1_acts, h, capacity, bk, shared_rows, config.swizzle_acts),
            w2_acts(sym_buffer.shared_l2_acts, shared_i, capacity, shared_rows),
            w1_acts(sym_buffer.l1_acts, ring, 24),
            w2_acts(sym_buffer.l2_acts, i, ring, 24),
        )
    if sym_buffer.short_activation_sf:
        # For aligned N=16 only the first 64 packed words are consumed.
        # The global coordinate and SMEM pitch remain one full SF block.
        narrow = (
            scales(sym_buffer.l1_acts_sf, sf_ring, h, 64, 32, 1),
            scales(sym_buffer.l2_acts_sf, sf_ring, i, 64, 32, 1),
            scales(sym_buffer.shared_l1_acts_sf, shared_sf_tokens, h, 64, 32, 1),
            scales(sym_buffer.shared_l2_acts_sf,
                   sym_buffer.shared_l2_acts_sf.shape[0], shared_i, 64, 32, 1),
        )
    return GCTmaPlan(*routed, *shared, *narrow)

def _get_gc_moe_nvlink_barrier_timeout_secs() -> int:
    """Read the positive signed-decimal int32 barrier timeout."""
    name = 'GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS'
    value = os.getenv(name)
    if value is None or value == '':
        return DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS
    if re.fullmatch(r"[ \t\n\v\f\r]*[+-]?[0-9]+", value, re.ASCII) is None:
        raise ValueError(f"Invalid {name}: {value}")
    try:
        seconds = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc
    if not 0 < seconds <= 2_147_483_647:
        raise ValueError(f"Invalid {name}: {value}")
    return seconds


def _routed_weight_linger(config, num_topk: int) -> bool:
    """Linger when the uniform-routing estimate gives an expert two or more m-blocks.

    The gate compares average tokens per expert against the tile, so skewed
    routing can leave some experts below one m-block and others above it.
    """
    tokens_per_expert = ceil_div(config.tokens * num_topk, config.local_experts)
    return tokens_per_expert >= 2 * config.block_m


def _compiler_identity(target: str) -> CompilerIdentity:
    return compiler_identity(target, GCMoERoleKernel.__call__,
                             Path(__file__).resolve().parents[2], 'gc_moe_role')


@dataclass
class _Prepared:
    maps: bytes
    binaries: tuple[RoleBinary, ...]
    # Hold both payload and SF storage for every graph using these maps.
    keepalive: tuple = ()


# Lifetime owner of loaded weights used by TensorMaps and child/outer graphs.
# Entries remain alive; no cache clear may release in-flight graph resources.
_CACHE: dict[tuple, _Prepared] = {}
_BINARY_CACHE: dict[tuple, tuple[RoleBinary, ...]] = {}


def _get_or_compile_binaries(key: tuple, identity: CompilerIdentity,
                             target: str, compile_roles) -> tuple[RoleBinary, ...]:
    """Publish a complete cubin set once, after compiler identity is stable."""
    with COMPILE_LOCK:
        binaries = _BINARY_CACHE.get(key)
        if binaries is None:
            binaries = tuple(compile_roles())
            if tuple(item.role for item in binaries) not in ((1, 2, 3), (1, 2, 2, 3), (2, 3), (1, 2)):
                raise RuntimeError('GC role compilation did not produce a complete role sequence')
            if _compiler_identity(target) != identity:
                raise RuntimeError('compiler identity changed while compiling role binaries')
            _BINARY_CACHE[key] = binaries
        return binaries


class _SymBufferValue(ctypes.Structure):
    """592-byte by-value symmetric peer-buffer launch argument."""
    _fields_ = (('base', ctypes.c_int64),
                ('offsets', ctypes.c_int64 * 72),
                ('rank_idx', ctypes.c_uint32))


@lru_cache(maxsize=32)
def _compile_native_dispatch(defines: tuple[tuple[str, int], ...], source: bytes) -> bytes:
    nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    with tempfile.TemporaryDirectory(prefix='gc-native-dispatch-') as directory:
        code = Path(directory) / 'dispatch.cu'
        cubin = Path(directory) / 'dispatch.cubin'
        code.write_bytes(source)
        command = [nvcc, '-std=c++17', '-arch=sm_100a', '--cubin',
                   *[f'-DGCMOE_{name}={value}' for name, value in defines],
                   '-o', str(cubin), str(code)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        return cubin.read_bytes()


def _native_dispatch_binary(layout, role_kernel, role_config, fixed_expert_span,
                            dispatch_first, timeout_secs, grid, mode) -> RoleBinary:
    views = layout.views
    workspace = layout.workspace_offsets
    defines = {
        'RANKS': layout.num_ranks,
        'LOCAL_EXPERTS': role_kernel.local_experts,
        'BLOCK_M': role_kernel.block_m,
        'EXPERT_SPAN': fixed_expert_span,
        'THREADS': role_config.threads,
        'PROGRAMMATIC': int(dispatch_first),
        'ARRIVAL_ORDER': int(mode == 2),
        'PULL_PLAN': int(mode == 3),
        'TRACE_OFFSET': role_kernel.decode_trace_offset,
        'TRACE_FIELDS': TRACE_CTA_FIELDS,
        'CAPACITY': layout.capacity,
        'TIMEOUT_CLOCKS': timeout_secs * 2000000000,
        'RECV_PARITY': SIGNAL_OFFSETS['recv_parity'],
        'ROLE3_EPOCH': SIGNAL_OFFSETS['role3_epoch'],
        'PACKETS': SIGNAL_OFFSETS['expert_recv_count'],
        'TOTALS': SIGNAL_OFFSETS['expert_recv_count_sum'],
        'L1_FULL': SIGNAL_OFFSETS['l1_full_count'],
        'SOURCE_INDICES': workspace['src_token_topk_idx'],
        'METADATA': workspace['token_src_metadata'],
        'TOPK': views['topk_idx'].offset,
        'SCORES': views['topk_weights'].offset,
        'INPUT_X': views['x'].offset,
        'INPUT_SF': views['x_sf'].offset,
        'L1_X': views['l1_acts'].offset,
        'L1_SF': views['l1_acts_sf'].offset,
        'L1_WEIGHT': views['l1_topk_weights'].offset,
        'SF_ROWS': layout.sf_ring_tokens,
    }
    source = Path(__file__).with_name('sm100_decode_dispatch.cu').read_bytes()
    try:
        cubin = _compile_native_dispatch(tuple(defines.items()), source)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Native Dispatch compilation failed\n{error.stderr}') from error
    shared_bytes = role_config.threads // 32 * layout.hidden
    return RoleBinary(1, cubin, shared_bytes, role_config.threads, grid)


def _sym_buffer_value(sym_buffer, output: int = 0, statistics: int = 0,
                      packed_scale_addresses: tuple[int, ...] = (),
                      routed_w1_prefetch: int = 0, routed_w2_prefetch: int = 0) -> _SymBufferValue:
    addresses = tuple(int(ptr) for ptr in sym_buffer.peer_addresses)
    rank = int(sym_buffer.rank_idx)
    if (not 1 <= len(addresses) <= 72 or not 0 <= rank < len(addresses) or
            any(ptr <= 0 or ptr >= 1 << 64 or ptr % 16 for ptr in addresses)):
        raise ValueError('invalid symmetric peer address set')
    base = int(sym_buffer.buffer.data_ptr())
    if addresses[rank] != base:
        raise ValueError('local symmetric buffer differs from rendezvous base')
    value = _SymBufferValue()
    value.base = base if base < 1 << 63 else base - (1 << 64)
    for i, address in enumerate(addresses):
        delta = address - base
        if not -(1 << 63) <= delta < (1 << 63):
            raise ValueError('symmetric peer offset exceeds signed int64')
        value.offsets[i] = delta
    if (sym_buffer.multicast_route_counts or sym_buffer.multicast_combine_ready or
            sym_buffer.multicast_pull_routes or
            sym_buffer.multicast_decode_input_offset):
        # This mode has at most 32 ranks, leaving the last offset field free.
        # The device reads it as an absolute address, not a peer delta.
        multicast = int(sym_buffer.handle.multicast_ptr)
        if not 0 < multicast < 1 << 64 or multicast % 16:
            raise ValueError('invalid multicast address')
        value.offsets[MAX_RANKS - 1] = multicast if multicast < 1 << 63 else multicast - (1 << 64)
    if output:
        # In small EP the final two unused offsets before multicast carry
        # absolute output pointers for the compute/finalize grid.
        value.offsets[MAX_RANKS - 2] = output
        value.offsets[MAX_RANKS - 3] = statistics
    if packed_scale_addresses:
        if len(addresses) > 32 or len(packed_scale_addresses) != 2:
            raise ValueError('packed routed scale pointers require small EP')
        for index, address in zip(PACKED_ROUTED_SCALE_OFFSETS, packed_scale_addresses):
            value.offsets[index] = address
    if routed_w1_prefetch:
        if len(addresses) > ROUTED_W1_PREFETCH_OFFSET or routed_w1_prefetch % 16:
            raise ValueError('bulk W1 prefetch needs an aligned pointer and a free peer slot')
        value.offsets[ROUTED_W1_PREFETCH_OFFSET] = routed_w1_prefetch
    if routed_w2_prefetch:
        if len(addresses) > ROUTED_W2_PREFETCH_OFFSET or routed_w2_prefetch % 16:
            raise ValueError('bulk W2 prefetch needs an aligned pointer and a free peer slot')
        value.offsets[ROUTED_W2_PREFETCH_OFFSET] = routed_w2_prefetch
    value.rank_idx = rank
    if (ctypes.sizeof(value) != 592 or
            (_SymBufferValue.base.offset, _SymBufferValue.offsets.offset,
             _SymBufferValue.rank_idx.offset) != (0, 8, 584)):
        raise AssertionError('SymBuffer by-value ABI changed')
    return value


def _clean_cute_ptx(ptx: str) -> str:
    if ptx.endswith("\x00"):
        ptx = ptx[:-1]
    if "\x00" in ptx:
        raise RuntimeError('unexpected NUL inside CuTe PTX')
    return ptx


def _rewrite_value_formal(ptx: str, ordinal: int, value_bytes: int,
                          alignment: int, *, raw_param_address: bool = False,
                          allow_unused: bool = False) -> str:
    """Replace one audited 8B CuTe pointer formal with one value formal."""
    declaration = re.compile(
        r"(?m)^([ \t]*\.param[ \t]+\.align[ \t]+8[ \t]+\.b8[ \t]+)"
        rf"([A-Za-z0-9_.$]+_param_{ordinal})\[8\](,?)[ \t]*$")
    matches = list(declaration.finditer(ptx))
    if len(matches) != 1:
        raise RuntimeError(f"CuTe value formal {ordinal} changed")
    name = matches[0].group(2)
    uses = re.findall(r"(?<![A-Za-z0-9_.$])" + re.escape(name) +
                      r"(?![A-Za-z0-9_.$])", ptx)
    unused = allow_unused and len(uses) == 1
    if not unused and len(uses) != 2:
        raise RuntimeError(f"CuTe value formal {ordinal} has unexpected uses")
    load = re.compile(
        r"(?m)^([ \t]*)ld\.param\.(?:u64|b64)[ \t]+(%rd[0-9]+),[ \t]*\[" +
        re.escape(name) + r"\];[ \t]*$")
    if len(list(load.finditer(ptx))) != int(not unused):
        raise RuntimeError(f"CuTe value formal {ordinal} pointer load changed")
    ptx = declaration.sub(
        lambda m: re.sub(r"\.align[ \t]+8\b", f".align {alignment}",
                         m.group(1)) + f"{name}[{value_bytes}]{m.group(3)}",
        ptx, count=1)
    # The reference keeps by-value TensorMap addresses in param space across
    # task-phase selection. The peer value is accessed through a generic
    # address, so only the role2 maps use the raw form here.
    address_op = 'mov.b64' if raw_param_address else 'cvta.param::entry.u64'
    ptx = load.sub(
        lambda m: f"{m.group(1)}{address_op} {m.group(2)}, {name};",
        ptx, count=1)
    return ptx


def _convert_role2_map_uses(ptx: str, map_count: int, direct_w1_store: bool = False,
                            dedup_activations: bool = False, dual_weight_warps: bool = False,
                            alternate_weight_load_sites: bool = False,
                            prefetch_maps: tuple[int, ...] | None = None, *,
                            peer_gather: bool = False, guarded_weight_loop: bool = False,
                            paired_shared_weights: bool = False) -> str:
    """Convert early prefetches and W1 output stores at PTX instruction uses.

    CuTe exposes map parameters as pointers. After the by-value formal rewrite
    those registers contain *param-space* addresses, as in the reference.
    Loader warps convert their two selected maps once per task before the K
    loop. This bridge converts only the earlier prefetches and the per-store
    W1 output map, which the reference also selects inside its elected store.
    Every conversion is scalar; no elected lane executes a warp collective.

    Fail closed on changes to CuTe's generated descriptor instruction forms.
    This bridge deliberately does not depend on PTX register IDs or labels.
    """
    map_ordinals = (4, *range(7, map_count + 6))
    raw_map_regs = []
    for ordinal in map_ordinals:
        raw = re.findall(
            r"(?m)^[ \t]*mov\.b64[ \t]+(%rd[0-9]+),[ \t]*"
            r"[A-Za-z0-9_.$]+_param_" + str(ordinal) + r";[ \t]*$", ptx)
        if len(raw) == 0 and prefetch_maps is not None:
            raw_map_regs.append(None)
            continue
        if len(raw) != 1:
            raise RuntimeError(f"role2 raw TensorMap {ordinal} changed")
        raw_map_regs.append(raw[0])
    used_map_regs = [reg for reg in raw_map_regs if reg is not None]
    if len(set(used_map_regs)) != len(used_map_regs):
        raise RuntimeError('role2 TensorMap address registers alias')

    register_decl = re.compile(
        r"(?m)^([ \t]*\.reg[ \t]+\.b64[ \t]+%rd<)([0-9]+)(>;[ \t]*)$")
    declarations = list(register_decl.finditer(ptx))
    if len(declarations) != 1:
        raise RuntimeError('role2 64-bit PTX register declaration changed')
    next_reg = int(declarations[0].group(2))
    prefetch = re.compile(
        r"(?m)^([ \t]*prefetch\.tensormap[ \t]+\[)(%rd[0-9]+)"
        r"(\];[ \t]*)$")
    data_prefetch = re.compile(
        r"(?m)^([ \t]*cp\.async\.bulk\.prefetch\.tensor\.(?:2d|4d)\.L2\.global\.tile"
        r"(?:\.L2::cache_hint)?"
        r"[ \t]+\[)(%rd[0-9]+)(,[ \t]*\{[^\n]+)$")
    g2s = re.compile(
        r"(?m)^([ \t]*cp\.async\.bulk\.tensor\.(?:2d|4d)(?:\.cta_group::2)?"
        r"[^\n]*?\[[^\n]*?\],[ \t]*\[)(%rd[0-9]+)"
        r"(,[ \t]*\{[^\n]+)$")
    s2g = re.compile(
        r"(?m)^([ \t]*cp\.async\.bulk\.tensor\.2d\.global\.shared"
        r"[^\n]*?\[)(%rd[0-9]+)(,[ \t]*\{[^\n]+)$")
    prefetch_regs = [m.group(2) for m in prefetch.finditer(ptx)]
    expected_prefetch_regs = (raw_map_regs if prefetch_maps is None else
                              [raw_map_regs[index] for index in prefetch_maps])
    if len(prefetch_regs) != len(expected_prefetch_regs) or set(prefetch_regs) != set(expected_prefetch_regs):
        raise RuntimeError('role2 TensorMap prefetch set changed')
    data_prefetch_regs = [m.group(2) for m in data_prefetch.finditer(ptx)]
    tensor_op_count = len(re.findall(
        r"(?m)^[ \t]*cp\.async\.bulk\.tensor\.", ptx))
    g2s_ops = list(g2s.finditer(ptx))
    g2s_count = len(g2s_ops)
    gathers = sum('.tile::gather4.' in op.group(0) for op in g2s_ops)
    s2g_count = len(list(s2g.finditer(ptx)))
    expected_stores = s2g_count == 0 if direct_w1_store else s2g_count >= 1
    # A different producer loop stride can change the number of unrolled
    # copies. The selected descriptor topology still checks all operations.
    load_sites = g2s_count - gathers
    expected_loads = (load_sites >= 4 if dual_weight_warps else
                      load_sites in (8, 16) if alternate_weight_load_sites else load_sites == 16)
    if guarded_weight_loop:
        # CuTe can lower the guarded K loop without replicating its four load sites.
        expected_loads = load_sites in (4, 8, 16)
    if paired_shared_weights:
        # The selected weight descriptor serves either a routed 4D tile or
        # a 2D shared packet. Their source parameter and converted register
        # must be the same after the task's weight map selection.
        tensor_2d = {op.group(2) for op in g2s_ops if '.tensor.2d.' in op.group(1)}
        tensor_4d = {op.group(2) for op in g2s_ops if '.tensor.4d.' in op.group(1)}
        expected_loads = load_sites >= 5 and len(tensor_2d & tensor_4d) == 1
    if (bool(gathers) != (dedup_activations or peer_gather) or not expected_loads or
            not expected_stores or tensor_op_count != g2s_count + s2g_count):
        raise RuntimeError(f'role2 TensorMap forms changed (G2S={g2s_count}, S2G={s2g_count})')
    selected_conversions = re.findall(
        r"(?m)^[ \t]*cvta\.param\.u64[ \t]+(%rd[0-9]+),[ \t]*"
        r"(%rd[0-9]+);[ \t]*$", ptx)
    source_cvta_count = len(selected_conversions)
    if peer_gather:
        gather_regs = {op.group(2) for op in g2s_ops if '.tile::gather4.' in op.group(0)}
        direct_conversions = [(dst, src) for dst, src in selected_conversions if src == raw_map_regs[4]]
        if len(direct_conversions) != 1 or gather_regs != {direct_conversions[0][0]}:
            raise RuntimeError('role2 peer gather lost its dedicated TensorMap')
        selected_conversions.remove(direct_conversions[0])
        g2s_ops = [op for op in g2s_ops if '.tile::gather4.' not in op.group(0)]
        if any(op.group(2) in gather_regs for op in g2s_ops):
            raise RuntimeError('role2 peer TensorMap also used for ordinary loads')
    loader_regs = {m.group(2) for m in g2s_ops}
    if (len(selected_conversions) != 4 or
            len(loader_regs) != 4 or
            loader_regs != {dst for dst, _ in selected_conversions} or
            len({src for _, src in selected_conversions}) != 4 or
            any(src in used_map_regs for _, src in selected_conversions)):
        raise RuntimeError('role2 loader selected-map conversions changed')
    if set(data_prefetch_regs) - set(used_map_regs) - loader_regs:
        raise RuntimeError('role2 data prefetch descriptor is neither raw nor converted')
    raw_prefetch_count = sum(reg in used_map_regs for reg in data_prefetch_regs)

    def converted_use(match: re.Match[str]) -> str:
        nonlocal next_reg
        prefix, raw_addr, suffix = match.groups()
        generic = f"%rd{next_reg}"
        next_reg += 1
        indent = re.match(r"[ \t]*", prefix).group(0)
        return (f"{indent}cvta.param.u64 {generic}, {raw_addr};\n"
                f"{prefix}{generic}{suffix}")

    ptx = prefetch.sub(converted_use, ptx)
    ptx = data_prefetch.sub(
        lambda m: converted_use(m) if m.group(2) in used_map_regs else m.group(0), ptx)
    ptx = s2g.sub(converted_use, ptx)
    ptx = register_decl.sub(
        lambda m: m.group(1) + str(next_reg) + m.group(3), ptx, count=1)
    if len(re.findall(r"\bcvta\.param\.u64\b", ptx)) != \
            source_cvta_count + len(prefetch_regs) + raw_prefetch_count + s2g_count:
        raise RuntimeError('role2 TensorMap conversion count changed')
    return ptx


def _device_compile_target(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise NotImplementedError('current three-role device path requires compute capability 10.0 or 10.3')
    return device_compile_target((major, minor))


def _entry_formals(ptx: str) -> list[str]:
    """Read only the launch ABI, excluding function and call-local params."""
    entries = list(re.finditer(
        r"(?m)^[ \t]*\.visible[ \t]+\.entry[ \t]+[A-Za-z0-9_.$]+\([ \t]*$",
        ptx))
    if len(entries) != 1:
        raise RuntimeError('CuTe kernel entry count changed')
    start = entries[0].end()
    closing = re.search(r"(?m)^\)[ \t]*$", ptx[start:])
    if closing is None:
        raise RuntimeError('CuTe kernel entry signature is unterminated')
    lines = [line for line in ptx[start:start + closing.start()].splitlines()
             if line.strip()]
    if not lines or any(re.fullmatch(r"[ \t]*\.param\b[^\n]*", line) is None
                        for line in lines):
        raise RuntimeError('CuTe kernel entry formals changed')
    return lines


def _order_role2_value_formals(ptx: str, map_count: int) -> str:
    """Keep ordinary scalars ahead of the configuration-selected value maps."""
    formals = _entry_formals(ptx)
    if len(formals) != map_count + 6:
        raise RuntimeError('role2 value-formal count changed')
    for index, line in enumerate(formals):
        if re.search(r"\b[A-Za-z0-9_.$]+_param_" + str(index) +
                     r"(?:\[[0-9]+\])?,?[ \t]*$", line) is None:
            raise RuntimeError('role2 value-formal order changed')
    order = (0, 1, 2, 3, 5, 6, 4, *range(7, map_count + 6))
    ordered = [formals[i].rstrip().rstrip(',') for i in order]
    original = '\n'.join(formals)
    if ptx.count(original) != 1:
        raise RuntimeError('role2 value-formal signature changed')
    return ptx.replace(original, ',\n'.join(ordered), 1)


def _assemble_role_value_params(ptx: str, role: int,
                                dump_dir: str, *, target: str,
                                use_small_tile_path: bool,
                                direct_w1_store: bool = False,
                                dedup_activations: bool = False,
                                dual_weight_warps: bool = False,
                                alternate_weight_load_sites: bool = False,
                                prefetch_maps: tuple[int, ...] | None = None,
                                omit_peer_finish: bool = False, peer_gather: bool = False,
                                guarded_weight_loop: bool = False, paired_shared_weights: bool = False,
                                ptxas_options: tuple[str, ...] = PTXAS_OPTIONS) -> tuple[str, bytes, str]:
    """Build each role with relative peer value, plus role2's value maps."""
    map_count = 22 if use_small_tile_path else 18
    ptx = _clean_cute_ptx(ptx)
    if ptx_target(ptx) != target:
        raise RuntimeError('CuTe generated a different device target')
    expected = {1: ('ptr', 'ptr', 'u32', 'u32', 'f32', 'f32'),
                2: (('ptr', 'ptr', 'u32', 'u32', 'ptr', 'f32', 'f32') +
                    ('ptr',) * (map_count - 1)),
                3: ('ptr', 'ptr', 'u32', 'u32', 'ptr', 'ptr', 'ptr',
                    'f32', 'f32')}[role]
    formals = _entry_formals(ptx)
    if len(formals) != len(expected):
        raise RuntimeError(f"role{role} CuTe parameter count changed")
    for index, (line, kind) in enumerate(zip(formals, expected)):
        form = (r"\.align[ \t]+8[ \t]+\.b8[ \t]+\S+_param_" +
                str(index) + r"\[8\]" if kind == 'ptr' else
                r"\." + kind + r"[ \t]+\S+_param_" + str(index))
        if re.search(form + r",?[ \t]*$", line) is None:
            raise RuntimeError(f"role{role} CuTe formal {index} changed")
    unused_peer_value = role == 3 and omit_peer_finish
    if not unused_peer_value and ('cvta.to.param::entry.u64' not in ptx or 'ld.param.u64' not in ptx):
        raise RuntimeError(f"role{role} peer parameter load path is missing")
    ptx = _rewrite_value_formal(ptx, 1, 592, 8, allow_unused=unused_peer_value)
    if role == 2:
        for map_ordinal in (4, *range(7, map_count + 6)):
            ptx = _rewrite_value_formal(
                ptx, map_ordinal, 128, 128, raw_param_address=True,
                allow_unused=prefetch_maps is not None)
        ptx = _convert_role2_map_uses(ptx, map_count, direct_w1_store, dedup_activations,
                                      dual_weight_warps, alternate_weight_load_sites,
                                      prefetch_maps, peer_gather=peer_gather,
                                      guarded_weight_loop=guarded_weight_loop,
                                      paired_shared_weights=paired_shared_weights)
        ptx = _order_role2_value_formals(ptx, map_count)
    return assemble_ptx(ptx, dump_dir, f"role{role}_value_params",
                                target=target, ptxas_options=ptxas_options)


def _capture_key(config, sym_buffer, green_context, y, l1_weights, l2_weights,
                 shared_l1_weights, shared_l2_weights, stats,
                 nvlink_timeout_secs: int, routed_weight_linger: bool,
                 role_resources: tuple,
                 combine_resources: tuple, compiler_identity: CompilerIdentity) -> tuple:
    status, cuda_context = cu.cuCtxGetCurrent()
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot query CUDA context: {status}")
    layout = sym_buffer.layout
    weights = (*l1_weights, *l2_weights,
               *(shared_l1_weights or ()), *(shared_l2_weights or ()))
    return (int(cuda_context), compiler_identity, green_context.dispatch_sms,
            green_context.compute_sms,
            sym_buffer.buffer.data_ptr(),
            tuple(sym_buffer.peer_addresses), sym_buffer.rank_idx,
            sym_buffer.group.size(), sym_buffer.num_experts,
            sym_buffer.num_max_tokens_per_rank,
            config.tokens, config.block_m, config.hidden, config.intermediate,
            config.local_experts, config.shared_experts,
            config.shared_weight_gran_k, config.clamp != inf,
            config.fast_math, sym_buffer.num_topk,
            nvlink_timeout_secs,
            routed_weight_linger,
            stats is not None,
            role_resources, combine_resources,
            layout.nbytes, layout.ring_tokens, layout.sf_ring_tokens,
            _tensor_metadata(y),
            _tensor_metadata(stats) if stats is not None else None,
            *(item for tensor in weights
              for item in (tensor.data_ptr(), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype))))


def _tensor_metadata(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))


def _binary_key(compiler_identity: CompilerIdentity, role_kernels: tuple,
                role_resources: tuple, combine_resources: tuple,
                sym_buffer, y: torch.Tensor, weights: tuple,
                stats: torch.Tensor | None) -> tuple:
    """Identify code generation without retaining any live CUDA address.

    Every role kernel attribute is included so new constexpr or derived layout
    fields cannot silently reuse an older cubin. Tensor metadata covers the
    CuTe compile argument types; actual addresses remain in _capture_key.
    """
    key = (compiler_identity,
           tuple(tuple(sorted(vars(kernel).items())) for kernel in role_kernels),
           role_resources, combine_resources,
           sym_buffer.layout.nbytes,
           _tensor_metadata(sym_buffer.buffer), _tensor_metadata(y),
           tuple(_tensor_metadata(tensor) for tensor in weights),
           _tensor_metadata(stats) if stats is not None else None)
    return key


def _check_supported(config, sym_buffer) -> None:
    if (not 1 <= sym_buffer.group.size() <= MAX_RANKS or
            not 1 <= config.local_experts <= 512 or
            sym_buffer.group.size() * config.local_experts > 2048 or
            not 0 < sym_buffer.num_topk or
            sym_buffer.num_topk + bool(config.shared_experts) > 32 or
            config.shared_experts < 0):
        raise NotImplementedError('current CuTe device path needs 1..512 experts/rank, <=2048 total experts, topk>=1 and topk+shared slot<=32')
    if (config.hidden <= 0 or config.intermediate <= 0 or
            config.hidden % 256 != 0 or config.intermediate % 128 != 0 or
            (2 * config.intermediate) // 128 > 64 or
            config.block_m not in (16, 32, 48, 64, 96, 128, 192, 240) or
            config.tokens < 0):
        raise NotImplementedError('current CuTe device math slice needs 2CTA H/I geometry, <=64 W1 N blocks, supported BM and nonnegative T')


def _check_ring_capacity(config, sym_buffer, compute_sms: int) -> None:
    """Recheck the full-device ring against the actual Green compute partition."""
    layout = sym_buffer.layout
    block_m = config.block_m
    if layout.ring_tokens % block_m:
        raise ValueError('GC ring capacity must contain whole M blocks')
    max_routed = (sym_buffer.num_max_tokens_per_rank * sym_buffer.group.size() *
                  min(sym_buffer.num_topk, config.local_experts))
    max_pool_blocks = ceil_div(max_routed, block_m) + config.local_experts
    live_blocks = num_max_live_pool_blocks(
        max_pool_blocks, compute_sms, config.hidden, config.intermediate)
    ring_blocks = layout.ring_tokens // block_m
    if live_blocks > ring_blocks:
        raise ValueError('GC ring capacity is insufficient for Green compute SMs')
    if layout.sf_ring_tokens < ring_blocks * align(block_m, 128):
        raise ValueError('GC SF ring capacity is insufficient for block M')


def prepare_role_binaries(config, y: torch.Tensor,
                          l1_weights: tuple[torch.Tensor, torch.Tensor],
                          l2_weights: tuple[torch.Tensor, torch.Tensor],
                          sym_buffer, shared_l1_weights=None,
                          shared_l2_weights=None, stats=None,
                          context=None, *, decode_mode: str = 'off', weight_reuse: str = 'off',
                          prefetch_experts: int = 0):
    """Encode live maps once per pointer set and compile each role once/shape."""
    _check_supported(config, sym_buffer)
    if sym_buffer.compact_decode_workspace and (config.tokens > 32 or config.block_m > 64):
        raise ValueError('compact workspace requires a small decode M tile')
    narrow_bm32 = (sym_buffer.narrow_bm32 and config.tokens <= 32 and
                   config.shared_experts > 0 and config.block_m == 32)
    dedup_activation_offset = sym_buffer.dedup_activation_offset
    source_elects_activation_owner = sym_buffer.source_elects_activation_owner
    if dedup_activation_offset:
        max_routes = 32 * sym_buffer.group.size() * min(config.local_experts, sym_buffer.num_topk)
        if (config.tokens > 32 or config.block_m > 32 or narrow_bm32 or
                ceil_div(max_routes, config.block_m) + config.local_experts >
                sym_buffer.layout.ring_tokens // config.block_m):
            raise ValueError('activation deduplication requires small, non-wrapping decode tiles')
    local_mma = sym_buffer.local_mma
    narrow_local_mma = sym_buffer.narrow_local_mma
    independent_compute_ctas = sym_buffer.independent_compute_ctas
    trim_wide_activations = sym_buffer.trim_wide_decode_activations
    if trim_wide_activations and (
            not 1 <= config.tokens <= 32 or config.block_m not in (96, 128) or
            not config.shared_experts or local_mma):
        raise ValueError('wide activation trimming needs two-CTA decode and a shared expert')
    short_activation_sf = sym_buffer.short_activation_sf
    if short_activation_sf and (
            config.block_m != 32 or not 1 <= config.tokens <= 32 or
            not config.shared_experts or
            narrow_bm32 or local_mma or narrow_local_mma or sym_buffer.transpose_w2_activations):
        raise ValueError('short activation scales need the full-A two-CTA BM32 decode path')
    has_prefix_maps = (short_activation_sf or narrow_bm32 or narrow_local_mma or trim_wide_activations or
                       (not local_mma and use_small_tile_path(config.shared_experts, config.block_m)))
    map_count = 22 if has_prefix_maps else 18
    if config.shared_experts != sym_buffer.num_shared_experts:
        raise ValueError('configuration and symmetric-buffer shared experts disagree')
    if context is None:
        context = get_context(num_ranks=sym_buffer.group.size())
    device = sym_buffer.buffer.device
    weight_pairs = (l1_weights, l2_weights)
    routed_k_major = tuple(is_routed_weight_k_major(pair[0]) for pair in weight_pairs)
    grouped_routed_weights = sym_buffer.grouped_routed_weights
    grouped_weight_inner_k = tuple(
        values.shape[3] if grouped_routed_weights & (1 << phase) else 2
        for phase, (values, _) in enumerate(weight_pairs))
    w2_prefetch_from_w1 = sym_buffer.w2_prefetch_from_w1
    packed_routed_scale_base = sym_buffer.packed_routed_scale_base
    packed_routed_scale_k_group = sym_buffer.packed_routed_scale_k_group
    packed_scale_helper = sym_buffer.packed_scale_helper
    packed_scale_release_group = sym_buffer.packed_scale_release_group
    follower_scheduler = sym_buffer.follower_scheduler
    if follower_scheduler and (config.tokens > 32 or local_mma or independent_compute_ctas):
        raise ValueError('follower scheduler requires a small two-CTA compute grid')
    if packed_routed_scale_base is not None and (
            config.tokens > 32 or config.shared_experts != 1 or sym_buffer.local_mma or
            sym_buffer.dual_weight_warps or sym_buffer.single_tma_lanes or
            sym_buffer.pair_routed_weight_scales or sym_buffer.prefetch_weight_scales):
        raise ValueError('packed scales require full weight warps and an independent shared TMA path')
    if grouped_routed_weights and (
            config.tokens > 32 or sym_buffer.local_mma or sym_buffer.dual_weight_warps or
            sym_buffer.weight_lookahead or sym_buffer.scheduled_weight_tiles or
            w2_prefetch_from_w1 or prefetch_experts):
        raise ValueError('grouped routed K pairs need the ordinary two-CTA decode weight pipeline')
    if shared_l1_weights is not None:
        weight_pairs += (shared_l1_weights, shared_l2_weights)
    weights = tuple(tensor for pair in weight_pairs for tensor in pair)
    shared_packets = sym_buffer.paired_shared_weights
    shared_packet_mask = 0
    if shared_packets is not None:
        expected_rows = tuple((n // 128) * (k // 256) * 130 for n, k in
                              ((2 * config.intermediate, config.hidden), (config.hidden, config.intermediate)))
        if (config.shared_experts != 1 or config.shared_weight_gran_k != 128 or
                not isinstance(shared_packets, tuple) or len(shared_packets) != 2 or
                any(t is not None and (not isinstance(t, torch.Tensor) or t.dtype != torch.float8_e4m3fn or
                    tuple(t.shape) != (rows, 256) or not t.is_contiguous() or t.data_ptr() % 16)
                    for t, rows in zip(shared_packets, expected_rows))):
            raise ValueError('invalid packet tensors for paired FP8 shared weights')
        weights += tuple(packet for packet in shared_packets if packet is not None)
        shared_packet_mask = sum(1 << phase for phase, packet in enumerate(shared_packets) if packet is not None)
    tensors = (y, *weights,
               *((stats,) if stats is not None else ()))
    if (device.type != 'cuda' or torch.cuda.current_device() != device.index or
            int(context.device) != device.index or any(tensor.device != device for tensor in tensors)):
        raise ValueError('role tensors and Green contexts must use the current CUDA device')
    _check_ring_capacity(config, sym_buffer, context.compute_sms)
    compiler_target = _device_compile_target(sym_buffer.buffer.device)
    compiler_identity = _compiler_identity(compiler_target)
    nvlink_timeout_secs = _get_gc_moe_nvlink_barrier_timeout_secs()
    routed_weight_linger = _routed_weight_linger(config, sym_buffer.num_topk)
    stage_alignment, release_group = sym_buffer.stage_alignment, sym_buffer.release_group
    if packed_routed_scale_k_group == 4 and (
            (not packed_scale_helper and stage_alignment != 4) or release_group != 1 or
            config.hidden % 512 or config.intermediate % 512):
        raise ValueError('grouped routed scales need contiguous groups of four pipeline slots')
    if packed_scale_helper and (shared_packets is not None or sym_buffer.trace_operand_sources or
                                sym_buffer.shared_w1_prefetch_limit or sym_buffer.weight_prefix_warps or
                                sym_buffer.scheduled_weight_tiles or sym_buffer.w2_prefetch_from_w1 or
                                sym_buffer.combine_in_compute):
        raise ValueError('packed scales need a free helper warp in each CTA and ordinary operand barriers')
    if shared_packets is not None and (
            config.tokens > 32 or config.block_m > 64 or stage_alignment != 2 or release_group != 1 or
            config.hidden % 512 or config.intermediate % 512 or local_mma or independent_compute_ctas or
            sym_buffer.dual_weight_warps or sym_buffer.pair_routed_weights or sym_buffer.pair_routed_weight_scales or
            sym_buffer.shared_w1_prefetch_limit or sym_buffer.trace_operand_sources or
            sym_buffer.decode_descriptor_prefetch != 'off'):
        raise ValueError('paired shared weights require a two-CTA decode pipeline with even stage pairs')
    if local_mma and (config.tokens > 32 or config.block_m > 64 or release_group != 1 or
                      narrow_bm32 or dedup_activation_offset or sym_buffer.peer_decode_input_direct or
                      sym_buffer.dual_weight_warps):
        raise ValueError('local MMA requires small decode and independent producer credits')
    if narrow_local_mma and config.shared_experts == 0:
        raise ValueError('narrow local MMA requires the shared descriptor set')
    if independent_compute_ctas and (sym_buffer.early_experts is not None or
                                     sym_buffer.static_w1_prefix or sym_buffer.static_decode_wave or
                                     sym_buffer.decode_w2_clusters or sym_buffer.stripe_combine_offset or
                                     sym_buffer.fused_dispatch_compute or sym_buffer.fused_compute_finalize):
        raise ValueError('the independent grid uses the ordinary routed task queue')
    direct_w1_store = sym_buffer.direct_w1_store
    w1_register_transpose = sym_buffer.w1_register_transpose
    w1_packed_scale_reduction = sym_buffer.w1_packed_scale_reduction and config.tokens <= 32
    w1_hardware_scale_reduction = sym_buffer.w1_hardware_scale_reduction
    w1_early_epi_inputs = sym_buffer.w1_early_epi_inputs
    w1_pair_scale_rounding = sym_buffer.w1_pair_scale_rounding
    if w1_pair_scale_rounding and (config.tokens > 32 or config.block_m > 64):
        raise ValueError('paired W1 scale rounding needs a small decode epilogue')
    if w1_early_epi_inputs and (config.tokens > 32 or config.block_m > 64 or
                                sym_buffer.single_epilogue_warpgroup):
        raise ValueError('early W1 epilogue inputs need at most 32 rows per warpgroup')
    if w1_hardware_scale_reduction and (w1_packed_scale_reduction or config.tokens > 32):
        raise ValueError('hardware scale reduction needs the FP32 small-decode epilogue')
    if w1_register_transpose and (not direct_w1_store or config.tokens > 32 or config.block_m > 64):
        raise ValueError('W1 register transpose requires direct decode stores')
    direct_w2_store = sym_buffer.direct_w2_store
    single_mma_lane = sym_buffer.single_mma_lane and config.tokens <= 32
    reuse_k128_desc = use_small_tile_path(config.shared_experts, config.block_m) or (
        sym_buffer.reuse_k128_desc and config.tokens <= 32)
    mma_unroll = sym_buffer.mma_unroll if config.tokens <= 32 else 4
    hoist_routed_sfb_selection = sym_buffer.hoist_routed_sfb_selection and config.tokens <= 32
    descriptor_prefetch = sym_buffer.decode_descriptor_prefetch if config.tokens <= 32 else 'all'
    decode_operand_cache = sym_buffer.decode_operand_cache if config.tokens <= 32 else 'auto'
    probe_routed_mma = sym_buffer.probe_routed_mma
    if probe_routed_mma and (config.tokens > 32 or config.block_m > 64 or local_mma):
        raise ValueError('MMA probe requires clustered small decode tiles')
    compact_compute_ctas = sym_buffer.compact_compute_ctas
    if compact_compute_ctas:
        max_routes = 32 * sym_buffer.group.size() * min(config.local_experts, sym_buffer.num_topk)
        if (context.primary_ctx_roles or config.tokens > 32 or config.block_m > 64 or
                sym_buffer.early_experts is not None or sym_buffer.ready_tile_offset or
                sym_buffer.recycle_dispatch_sms or sym_buffer.static_w1_prefix or
                ceil_div(max_routes, config.block_m) + config.local_experts >
                sym_buffer.layout.ring_tokens // config.block_m):
            raise ValueError('compact compute needs Green contexts and small, non-wrapping tiles')
    if direct_w1_store and config.tokens > 32:
        raise ValueError('direct W1 stores are available only for small decode')
    w2_tile_major_input = sym_buffer.w2_tile_major_input
    if w2_tile_major_input and (not direct_w1_store or config.block_m > 64):
        raise ValueError('tile-major W2 activations need direct small-decode W1 stores')
    transpose_w2_activations = sym_buffer.transpose_w2_activations
    if transpose_w2_activations and (
            config.tokens > 32 or config.block_m != 32 or not direct_w1_store or
            narrow_bm32 or local_mma or narrow_local_mma or
            sym_buffer.num_max_tokens_per_rank % 32):
        raise ValueError('transposed W2 activations need the full BM32 two-CTA path')
    if direct_w2_store and (config.tokens > 32 or config.block_m > 64):
        raise ValueError('direct W2 stores need small decode tiles')
    if release_group > 1 and ((config.hidden // 128) % release_group or
                              (config.intermediate // 128) % release_group or
                              config.tokens > 32):
        raise ValueError('grouped decode credits require complete K-stage groups and at most 32 tokens')
    fused_dispatch_compute = sym_buffer.fused_dispatch_compute
    single_epilogue_warpgroup = sym_buffer.single_epilogue_warpgroup
    accumulator_stages = sym_buffer.decode_accumulator_stages
    if accumulator_stages != 2 and (
            config.tokens > 32 or config.block_m > 64 or compact_compute_ctas):
        raise ValueError('alternate accumulator queues require small noncompact compute tiles')
    if single_epilogue_warpgroup and (
            config.tokens > 32 or config.block_m > 64 or fused_dispatch_compute):
        raise ValueError('one epilogue warpgroup requires an independent small-batch compute grid')
    fused_worker_reuse = sym_buffer.fused_worker_reuse
    decode_dispatch_warps = sym_buffer.decode_dispatch_warps
    if fused_dispatch_compute:
        if (not context.primary_ctx_roles or config.tokens > 32 or config.block_m > 64 or
                context.dispatch_sms % 2 or context.compute_sms < 16 or
                decode_dispatch_warps not in (0, 12) or
                (sym_buffer.programmatic_finalize and not sym_buffer.sentinel_combine) or
                sym_buffer.compute_backfill_sms or sym_buffer.recycle_dispatch_sms or compact_compute_ctas):
            raise ValueError('fused dispatch/compute needs small clustered decode in the primary context')
        decode_dispatch_warps = 12
    if decode_dispatch_warps and config.tokens > 32:
        raise ValueError('custom Dispatch warps support at most 32 tokens per rank')
    weight_lookahead = sym_buffer.weight_lookahead
    if weight_lookahead and (config.tokens > 32 or config.block_m > 64):
        raise ValueError('rolling weight lookahead requires small decode tiles')
    packed_decode_activations = sym_buffer.packed_decode_activations
    packed_decode_scores = sym_buffer.packed_decode_scores
    activation_prefix_bytes = sym_buffer.activation_prefix_bytes
    peer_activation_prefix_tiles = sym_buffer.peer_activation_prefix_tiles
    peer_activation_window = sym_buffer.peer_activation_window
    peer_activation_rank_rows = peer_activation_window.rank_rows if peer_activation_window else 0
    if peer_activation_rank_rows and (not direct_w1_store or transpose_w2_activations):
        raise ValueError('direct peer TMA needs the unused routed W1 store descriptor')
    activation_prefix_offset = sym_buffer.activation_prefix_offset
    parallel_activation_prefix = sym_buffer.parallel_activation_prefix
    if activation_prefix_bytes and (
            config.tokens > 32 or config.block_m > 64 or config.hidden > 8192 or
            packed_decode_activations or decode_mode not in ('route', 'route-reduce') or
            dedup_activation_offset or fused_dispatch_compute or sym_buffer.early_experts is not None or
            sym_buffer.ready_tile_offset):
        raise ValueError('activation prefixes require the ordinary small-batch TMA ring')
    multicast_decode_input_offset = sym_buffer.multicast_decode_input_offset
    peer_decode_input_offset = sym_buffer.peer_decode_input_offset
    peer_decode_input_direct = sym_buffer.peer_decode_input_direct
    peer_decode_token_fanout = sym_buffer.peer_decode_token_fanout
    peer_prefetch_expert_heads = sym_buffer.peer_prefetch_expert_heads
    peer_prefetch_opportunistic = sym_buffer.peer_prefetch_opportunistic
    trim_decode_gather_tail = sym_buffer.trim_decode_gather_tail
    asymmetric_activation_rows = sym_buffer.asymmetric_activation_rows
    if asymmetric_activation_rows and (not narrow_bm32 or local_mma or
                                       dedup_activation_offset or peer_decode_input_direct):
        raise ValueError('asymmetric activation rows need the two-CTA BM32 prefix maps')
    if peer_decode_input_direct and narrow_bm32:
        raise ValueError('mailbox gather uses its own routed W1 tensor map')
    push_decode_activations = sym_buffer.push_decode_activations
    prefetch_w2_metadata = sym_buffer.prefetch_w2_metadata
    packed_w2_metadata = sym_buffer.packed_w2_metadata
    if packed_w2_metadata and (
            config.tokens > 32 or config.block_m > 64 or
            decode_mode not in ('route', 'route-reduce') or
            push_decode_activations or dedup_activation_offset or peer_decode_input_direct or
            sym_buffer.num_max_tokens_per_rank > 65536):
        raise ValueError('packed W2 metadata requires ordinary small-decode Dispatch')
    if prefetch_w2_metadata and (
            config.tokens > 32 or config.block_m > 64 or direct_w2_store or
            decode_mode not in ('route', 'route-reduce') or
            push_decode_activations or sym_buffer.speculative_w1 or
            sym_buffer.group.size() > 256 or sym_buffer.num_topk > 255):
        raise ValueError('early W2 metadata requires ordinary small-decode dispatch and scatter')
    w1_tile_major_input = sym_buffer.w1_tile_major_input
    if w1_tile_major_input and (
            not packed_decode_activations or config.tokens > 32 or config.block_m > 64 or
            config.hidden > 8192 or config.hidden % 512 or
            decode_mode not in ('route', 'route-reduce') or dedup_activation_offset or
            peer_decode_input_direct or push_decode_activations or activation_prefix_bytes):
        raise ValueError('tile-major W1 activations need the ordinary packed TMA pull')
    push_scale_receiver_sms = sym_buffer.push_scale_receiver_sms
    if push_scale_receiver_sms >= context.dispatch_sms:
        raise ValueError('packed scales require both sender and receiver Dispatch SMs')
    if push_decode_activations and (
            config.tokens > 32 or config.block_m > 64 or sym_buffer.group.size() > 32 or
            config.local_experts > 16 or sym_buffer.early_experts != 0 or
            multicast_decode_input_offset or peer_decode_input_offset or sym_buffer.pull_route_metadata_offset or
            sym_buffer.packed_route_counts or sym_buffer.route_count_doorbell or
            sym_buffer.multicast_route_counts or sym_buffer.inline_route_scores or
            fused_dispatch_compute):
        raise ValueError('source-push activations require fixed small decode slots and an independent Dispatch grid')
    decode_tma_stages = sym_buffer.decode_tma_stages
    decode_task_slots = sym_buffer.decode_task_slots
    if decode_task_slots != 2 and (config.tokens > 32 or config.block_m > 64):
        raise ValueError('alternate task queue depth requires small decode')
    keep_routed_weight_scales = sym_buffer.keep_routed_weight_scales
    if keep_routed_weight_scales and (config.tokens > 32 or sym_buffer.packed_routed_scale_base is not None):
        raise ValueError('retaining routed weight scales requires small decode with separate scale maps')
    dual_weight_warps = sym_buffer.dual_weight_warps
    pair_routed_weight_scales = sym_buffer.pair_routed_weight_scales
    pair_routed_weights = sym_buffer.pair_routed_weights
    if (pair_routed_weight_scales or pair_routed_weights) and (
            config.tokens > 32 or config.block_m > 64 or
            config.hidden % 256 or config.intermediate % 256 or
            stage_alignment != 2 or release_group != 1 or local_mma or dual_weight_warps or
            sym_buffer.prefetch_weight_scales):
        raise ValueError('paired weight transfers need even K stages and their individual TMA credits')
    if pair_routed_weights and (weight_lookahead or sym_buffer.scheduled_weight_tiles or w2_prefetch_from_w1):
        raise ValueError('paired weights need lookahead to issue complete K pairs')
    single_tma_lanes = sym_buffer.single_tma_lanes
    if single_tma_lanes and (config.tokens > 32 or config.block_m > 64 or
                             weight_lookahead or dual_weight_warps):
        raise ValueError('single-lane TMA needs the ordinary two small-decode loader warps')
    if single_tma_lanes == 2 and (dedup_activation_offset or peer_decode_input_direct or
                                  push_decode_activations or sym_buffer.speculative_w1):
        raise ValueError('single-lane activation loads cannot gather or wait for individual rows')
    task_weight_prefetch = bool(sym_buffer.scheduled_weight_tiles or w2_prefetch_from_w1)
    if (sym_buffer.weight_prefix_warps or sym_buffer.post_init_prefetch or
            task_weight_prefetch) and (
            dual_weight_warps or local_mma or independent_compute_ctas or fused_dispatch_compute):
        raise ValueError('dedicated prefetch warps need the two-CTA compute follower')
    shared_w1_prefetch_limit = sym_buffer.shared_w1_prefetch_limit
    if shared_w1_prefetch_limit and (
            config.tokens > 32 or not config.shared_experts or dual_weight_warps or
            local_mma or independent_compute_ctas or fused_dispatch_compute or
            sym_buffer.weight_prefix_warps or sym_buffer.post_init_prefetch or
            task_weight_prefetch or
            sym_buffer.combine_in_compute):
        raise ValueError('shared W1 prefetch needs the two unused follower warps in small decode')
    if task_weight_prefetch and (config.tokens > 32 or config.block_m > 64):
        raise ValueError('scheduled weight prefetch needs small decode tiles')
    if dual_weight_warps and (config.tokens > 32 or config.block_m > 64 or release_group != 1):
        raise ValueError('dual weight producers require small decode with independent TMA credits')
    if decode_tma_stages and (config.tokens > 32 or config.block_m > 64):
        raise ValueError('custom TMA stages need small decode tiles')
    double_buffer_dispatch_pull = sym_buffer.double_buffer_dispatch_pull
    if double_buffer_dispatch_pull:
        max_rows = (min(32, sym_buffer.layout.capacity) *
                    sym_buffer.group.size() * sym_buffer.num_topk)
        max_blocks = ceil_div(max_rows, config.block_m) + config.local_experts
        if (config.tokens > 32 or config.hidden > 8192 or
                max_blocks * config.block_m > sym_buffer.layout.ring_tokens or
                decode_mode not in ('route', 'route-reduce') or
                any((packed_decode_activations, activation_prefix_bytes, peer_decode_input_direct,
                     dedup_activation_offset, push_decode_activations, fused_dispatch_compute,
                     sym_buffer.ready_tile_offset, sym_buffer.native_decode_dispatch))):
            raise ValueError('double-buffered pulls need non-wrapping small-decode TMA rows')
    if peer_activation_prefix_tiles and (
            config.tokens > 32 or config.block_m not in (32, 48) or config.hidden != 6144 or
            decode_mode not in ('route', 'route-reduce') or any((
                narrow_bm32, local_mma, packed_decode_activations, dedup_activation_offset,
                push_decode_activations, sym_buffer.speculative_w1, w1_tile_major_input,
                double_buffer_dispatch_pull, fused_dispatch_compute,
                sym_buffer.native_decode_dispatch, sym_buffer.early_experts is not None))):
        raise ValueError('direct peer prefixes need the ordinary two-CTA BM32 or BM48 decode path')
    role_configs = tuple(get_role_config(role, config.block_m,
                                         sym_buffer.num_experts,
                                         config.hidden,
                                         local_experts=config.local_experts,
                                         stage_alignment=stage_alignment if role == 2 else 1,
                                         compact_compute=role == 2 and bool(compact_compute_ctas),
                                         decode_dispatch_warps=decode_dispatch_warps if role == 1 else 0,
                                         separate_weight_barrier=role == 2 and sym_buffer.trace_operand_sources,
                                         packed_decode_activations=role == 1 and packed_decode_activations,
                                         packed_decode_scores=role == 1 and packed_decode_scores,
                                         decode_tma_stages=decode_tma_stages if role == 2 else 0,
                                         parallel_activation_prefix=role == 1 and parallel_activation_prefix,
                                         double_buffer_dispatch_pull=role == 1 and double_buffer_dispatch_pull,
                                         single_epilogue_warpgroup=role == 2 and single_epilogue_warpgroup,
                                         accumulator_stages=accumulator_stages if role == 2 else 2,
                                         local_mma=role == 2 and local_mma,
                                         task_slots=decode_task_slots if role == 2 else 2,
                                         paired_shared_weights=role == 2 and shared_packets is not None)
                         for role in (1, 2, 3))
    if packed_scale_helper and role_configs[1].num_stages < 4:
        raise ValueError('scale helper needs at least four live K128 slots')
    if config.tokens > 32:
        decode_mode = 'off'
        weight_reuse = 'off'
        prefetch_experts = 0
    w1_prefetch_k_tiles = sym_buffer.w1_prefetch_k_tiles if config.tokens <= 32 else 0
    w2_prefetch_k_tiles = sym_buffer.w2_prefetch_k_tiles if config.tokens <= 32 else 0
    if w1_prefetch_k_tiles and prefetch_experts:
        raise ValueError('choose a routed W1 K prefix or full-expert prefetch')
    bulk_w1_prefetch_bytes = sym_buffer.bulk_w1_prefetch_bytes if config.tokens <= 32 else 0
    bulk_w1_prefetch_on_dispatch = (
        int(sym_buffer.bulk_w1_prefetch_on_dispatch) if bulk_w1_prefetch_bytes else 0)
    bulk_w1_prefetch_lanes = sym_buffer.bulk_w1_prefetch_lanes if bulk_w1_prefetch_bytes else 1
    if bulk_w1_prefetch_bytes:
        w1 = l1_weights[0]
        prefix_bytes = ((w1_prefetch_k_tiles * 128 or config.hidden) * config.intermediate)
        if (not (prefetch_experts or w1_prefetch_k_tiles) or prefix_bytes % bulk_w1_prefetch_bytes or
                not w1.is_contiguous() or w1.ndim != 4 or w1.dtype != torch.int8 or
                w1.stride(0) != config.hidden * config.intermediate or weight_reuse != 'off' or
                sym_buffer.group.size() > 32 or fused_dispatch_compute or sym_buffer.recycle_dispatch_sms):
            raise ValueError('bulk W1 prefetch needs contiguous small-decode expert weights')
    if bulk_w1_prefetch_on_dispatch and (context.dispatch_sms < 2 or push_decode_activations):
        raise ValueError('Dispatch W1 prefetch needs a separate route publisher and pull workers')
    bulk_w2_tail_experts = sym_buffer.bulk_w2_tail_experts
    if bulk_w2_tail_experts:
        w2 = l2_weights[0]
        prefix_bytes = w2_prefetch_k_tiles * 128 * config.hidden // 2
        if (not prefix_bytes or prefix_bytes % (64 * 1024) or
                not w2.is_contiguous() or w2.ndim != 4 or w2.dtype != torch.int8 or
                w2.stride(0) != config.hidden * config.intermediate // 2 or
                weight_reuse != 'off' or sym_buffer.group.size() > 32 or
                push_decode_activations or fused_dispatch_compute or sym_buffer.recycle_dispatch_sms or
                sym_buffer.native_decode_dispatch):
            raise ValueError('Dispatch-tail bulk W2 prefetch needs contiguous small-decode expert weights')
    single_publisher = (decode_mode in ('route', 'route-reduce', 'vector') and
                        config.tokens <= role_configs[0].routing_warps *
                        (32 // sym_buffer.num_topk))
    if peer_decode_input_offset and (
            not single_publisher or context.dispatch_sms < 2 or
            decode_mode == 'vector' or fused_dispatch_compute or sym_buffer.pull_route_metadata_offset or
            (sym_buffer.hidden // 128) % 4 != 0 or
            sym_buffer.hidden + sym_buffer.hidden // 32 > 8192):
        raise ValueError('destination-only decode input needs an independent single-publisher TMA dispatch')
    packed_route_counts = sym_buffer.packed_route_counts
    packed_route_count_group = sym_buffer.packed_route_count_group
    rank_route_doorbell = sym_buffer.rank_route_doorbell
    pull_route_metadata_offset = sym_buffer.pull_route_metadata_offset
    inline_route_scores = sym_buffer.inline_route_scores
    inline_route_indices = sym_buffer.inline_route_indices
    if inline_route_indices and (
            not single_publisher or config.tokens > 32 or sym_buffer.num_topk > 8 or
            sym_buffer.group.size() > 32 or packed_route_count_group != 1 or inline_route_scores or
            pull_route_metadata_offset or source_elects_activation_owner or push_decode_activations or
            sym_buffer.native_decode_dispatch or sym_buffer.split_route_heads or fused_dispatch_compute):
        raise ValueError('inline route indices need one small-batch count packet per source and expert')
    rotate_route_publication = sym_buffer.rotate_route_publication
    prune_decode_resets = sym_buffer.prune_decode_resets and single_publisher
    route_count_doorbell = sym_buffer.route_count_doorbell
    dispatch_shared_route_counts = sym_buffer.dispatch_shared_route_counts
    multicast_route_counts = sym_buffer.multicast_route_counts
    route_publish_ctas = sym_buffer.route_publish_ctas
    dispatch_pull_rows_per_cta = sym_buffer.dispatch_pull_rows_per_cta
    if dispatch_pull_rows_per_cta and (
            config.tokens > 32 or sym_buffer.push_decode_activations or
            sym_buffer.native_decode_dispatch or
            role_configs[0].threads // 32 % dispatch_pull_rows_per_cta):
        raise ValueError('Dispatch pull striping needs evenly grouped small-batch TMA workers')
    split_route_heads = sym_buffer.split_route_heads
    if split_route_heads and (
            not single_publisher or context.dispatch_sms < 4 or config.tokens > 32 or
            not 1 <= sym_buffer.early_experts < config.local_experts or
            packed_route_counts or pull_route_metadata_offset or route_publish_ctas != 1 or
            multicast_route_counts or sym_buffer.route_count_doorbell or
            source_elects_activation_owner or rotate_route_publication or
            peer_decode_input_offset or multicast_decode_input_offset or fused_dispatch_compute or
            sym_buffer.decode_shared_w2 or sym_buffer.recycle_dispatch_sms):
        raise ValueError('split route heads require ordinary decode with two independent publishers')
    if prune_decode_resets and (push_decode_activations or pull_route_metadata_offset):
        raise ValueError('decode reset pruning requires ordinary route publication')
    if rotate_route_publication and (
            not single_publisher or config.tokens > 32 or packed_route_counts or
            pull_route_metadata_offset or multicast_route_counts or
            push_decode_activations or route_publish_ctas != 1):
        raise ValueError('rotated route publication needs one ordinary unicast publisher')
    if route_publish_ctas > 1 and (
            not single_publisher or context.dispatch_sms < route_publish_ctas or
            sym_buffer.group.size() % route_publish_ctas or
            packed_route_counts or packed_route_count_group != 1 or pull_route_metadata_offset or
            multicast_route_counts or sym_buffer.multicast_decode_input_offset or
            peer_decode_input_offset or push_decode_activations):
        raise ValueError('partitioned routing needs small batches and regular per-expert receive counts')
    if multicast_route_counts and (
            not single_publisher or config.tokens > 32 or config.local_experts > 32 or
            (packed_route_counts and packed_route_count_group != 4) or
            pull_route_metadata_offset or sym_buffer.early_experts is not None or
            3 * sym_buffer.num_experts > MAX_EXPERTS):
        raise ValueError('multicast routing requires one publisher and room for two local epoch banks')
    if multicast_route_counts and packed_route_count_group == 4 and (
            config.local_experts % 4 or sym_buffer.num_topk > 8 or rank_route_doorbell):
        raise ValueError('packed multicast routing needs full four-expert groups and top-k at most eight')
    if route_count_doorbell and (config.tokens > 32 or config.local_experts > 32 or
                                packed_route_counts or pull_route_metadata_offset or
                                sym_buffer.early_experts is not None):
        raise ValueError('one route count coordinator requires ordinary small decode and at most 32 local experts')
    if dispatch_shared_route_counts and (
            not single_publisher or config.tokens > 32 or config.local_experts > 32 or
            sym_buffer.early_experts is not None or sym_buffer.native_decode_dispatch or
            push_decode_activations or fused_dispatch_compute):
        raise ValueError('per-CTA Dispatch counts need ordinary small decode and at most 32 local experts')
    if inline_route_scores and (config.tokens > 32 or pull_route_metadata_offset or
                                sym_buffer.num_max_tokens_per_rank // 2 < 32 * sym_buffer.num_topk):
        raise ValueError('inline route scores need small decode and room for packed source records')
    if packed_decode_activations and (config.tokens > 32 or decode_mode == 'vector'):
        raise ValueError('packed decode activations need small batches and TMA dispatch')
    if packed_decode_scores and (not single_publisher or fused_dispatch_compute or
                                 decode_mode not in ('route', 'route-reduce')):
        raise ValueError('packed decode scores need the specialized routed TMA dispatch kernel')
    if pull_route_metadata_offset:
        route_scan_warps = role_configs[0].routing_warps
        route_scan_workers = context.dispatch_sms * (route_scan_warps if sym_buffer.warp_pull_routes else 1)
        if (not single_publisher or config.tokens > 32 or packed_route_count_group != 1 or
                source_elects_activation_owner or route_scan_workers < sym_buffer.group.size() or
                config.local_experts > (32 if sym_buffer.warp_pull_routes else route_scan_warps * 32) or
                (sym_buffer.warp_pull_routes and route_scan_warps > sym_buffer.group.size())):
            raise ValueError('receiver-driven routing requires sufficient Dispatch scan workers')
    dispatch_prefix_scan = sym_buffer.dispatch_prefix_scan
    if dispatch_prefix_scan and (not single_publisher or sym_buffer.group.size() > 32):
        raise ValueError('parallel source intervals need small decode and at most 32 ranks')
    if packed_route_counts and (not single_publisher or sym_buffer.group.size() > 32 or
                                config.local_experts > 32 or config.local_experts >
                                context.dispatch_sms * role_configs[0].routing_warps):
        raise ValueError('packed counts need a single route publisher and at most 32 ranks and local experts')
    if packed_route_count_group > 1 and config.tokens > 32:
        raise ValueError('grouped route count packets need at most 32 tokens per rank')
    if packed_route_count_group == 8 and (config.local_experts != 8 or sym_buffer.num_topk > 8):
        raise ValueError('eight-expert route packets require eight local experts and top-k at most eight')
    if rank_route_doorbell and (
            packed_route_count_group != 4 or
            role_configs[0].routing_warps < ceil_div(config.local_experts, 4)):
        raise ValueError('rank route doorbell requires one CTA with four-expert count groups')
    early_experts = sym_buffer.early_experts
    speculative_w1 = sym_buffer.speculative_w1
    speculative_packed_w1 = sym_buffer.speculative_packed_w1
    speculative_w1_n_tiles = sym_buffer.speculative_w1_n_tiles
    if speculative_w1 and single_mma_lane:
        raise ValueError('speculative W1 requires a full MMA warp for the count acquire')
    ready_tile_offset = sym_buffer.ready_tile_offset
    if ready_tile_offset:
        routed_rows = 32 * sym_buffer.group.size() * min(
            config.local_experts, sym_buffer.num_topk)
        block_limit = ceil_div(routed_rows, config.block_m) + config.local_experts
        if (config.tokens > 32 or block_limit > sym_buffer.layout.ring_tokens // config.block_m):
            raise ValueError('ready tile ordering requires a small decode batch with no ring wrap')
    fixed_expert_span = 0
    if early_experts is not None:
        if config.tokens > 32 or weight_reuse != 'off':
            raise ValueError('expert-first decode requires at most 32 tokens per rank')
        if speculative_packed_w1:
            # If expert zero is active, its first tile always starts the packed pool.
            if (ready_tile_offset or sym_buffer.static_w1_prefix or sym_buffer.static_decode_wave or
                    sym_buffer.decode_w2_clusters or sym_buffer.prioritize_expert_rows or
                    push_decode_activations or fused_dispatch_compute or compact_compute_ctas or
                    decode_mode not in ('route', 'route-reduce')):
                raise ValueError('speculative packed W1 requires the ordinary small-decode queues')
        else:
            fixed_expert_span = ceil_div(32 * sym_buffer.group.size(), config.block_m)
        total_tokens = fixed_expert_span * config.local_experts * config.block_m
        head_clusters = speculative_w1_n_tiles or (2 * config.intermediate // 256)
        if (total_tokens > min(sym_buffer.layout.ring_tokens,
                              sym_buffer.layout.num_max_pool_tokens) or
                context.dispatch_sms * role_configs[0].routing_warps < config.local_experts or
                head_clusters > 2 * config.intermediate // 256 or
                early_experts * head_clusters > context.compute_sms // 2):
            raise ValueError('expert-first decode does not fit this launch geometry')
    decode_linear_pool = sym_buffer.decode_linear_pool
    if decode_linear_pool:
        # All ranks using this decode path contribute at most 32 rows per
        # top-k slot. Include an extra padded tile for every local expert.
        rows = min(32, sym_buffer.num_max_tokens_per_rank) * sym_buffer.group.size() * sym_buffer.num_topk
        blocks = ceil_div(rows, config.block_m) + config.local_experts
        if (config.tokens > 32 or fixed_expert_span or push_decode_activations or
                decode_mode not in ('route', 'route-reduce') or
                blocks * config.block_m > min(sym_buffer.layout.ring_tokens,
                                              sym_buffer.layout.num_max_pool_tokens)):
            raise ValueError('linear decode requires an ordinary pool large enough to avoid wrapping')
    vector_pull = decode_mode == 'vector' and config.hidden % WARP_BYTES == 0
    vector_pull_group = sym_buffer.decode_vector_pull_group
    if vector_pull_group > 1 and (not vector_pull or config.hidden % (WARP_BYTES * vector_pull_group)):
        raise ValueError('vector activation pull groups must divide the FP8 row')
    vector_combine = decode_mode in ('reduce', 'route-reduce', 'vector')
    token_major_combine = sym_buffer.token_major_combine
    if token_major_combine and (not vector_combine or config.tokens > 32 or
                                sym_buffer.stripe_combine_offset):
        raise ValueError('token-major storage requires small vector Combine')
    combine_load_group = sym_buffer.combine_load_group
    if combine_load_group > 1 and (not vector_combine or config.tokens > 32 or
                                   sym_buffer.stripe_combine_offset):
        raise ValueError('grouped loads require the regular small vector Combine')
    stripe_combine_offset = sym_buffer.stripe_combine_offset
    if stripe_combine_offset and (
            config.tokens > 32 or config.shared_experts != 1 or not vector_combine):
        raise ValueError('striped combine requires small vector decode with one shared expert')
    expert_combine_offset = sym_buffer.expert_combine_offset
    if expert_combine_offset and (config.tokens > 32 or config.shared_experts == 0):
        raise ValueError('expert readiness expects small decode with a shared expert')
    sentinel_combine = sym_buffer.sentinel_combine
    fused_finalize_sentinel = sym_buffer.fused_finalize_sentinel
    reuse_finalize_warps = sym_buffer.reuse_finalize_warps
    sentinel_half_vector = sym_buffer.sentinel_half_vector
    if sentinel_combine and (
            not vector_combine or not config.shared_experts or config.tokens > 32 or
            config.hidden * 2 % (WARP_BYTES * sym_buffer.sentinel_combine_vectors) or
            stripe_combine_offset or expert_combine_offset or
            (sym_buffer.fused_compute_finalize and not fused_finalize_sentinel)):
        raise ValueError('sentinel Combine needs a small W2 scatter and vector reduction')
    combine_in_compute = sym_buffer.combine_in_compute
    if combine_in_compute and (
            not sentinel_combine or not sym_buffer.programmatic_finalize or
            sym_buffer.sentinel_combine_vectors != 1 or sym_buffer.group.size() > 32 or
            dual_weight_warps or local_mma or independent_compute_ctas or compact_compute_ctas or
            fused_dispatch_compute or sym_buffer.recycle_dispatch_sms or sym_buffer.weight_prefix_warps or
            task_weight_prefetch or sym_buffer.combine_ready_flags_offset):
        raise ValueError('compute-side Combine needs ordinary two-CTA workers and 512-byte early sentinel chunks')
    if combine_in_compute == 3 and (
            config.shared_experts != 1 or direct_w2_store or sym_buffer.sentinel_chunk_major):
        raise ValueError('shared epilogue Combine requires one shared expert and its ordinary W2 scatter')
    programmatic_finalize = (sym_buffer.programmatic_finalize or
                             bool(stripe_combine_offset or expert_combine_offset))
    combine_at_block_start = sym_buffer.combine_at_block_start
    if combine_at_block_start and (not programmatic_finalize or sym_buffer.stripe_launch_delay):
        raise ValueError('block-start Combine needs an undelayed programmatic dependency')
    decode_compute_max_registers = sym_buffer.decode_compute_max_registers
    decode_nvlink_schedule = sym_buffer.decode_nvlink_schedule
    if decode_nvlink_schedule and (
            config.tokens > 32 or sym_buffer.group.size() < 2 or
            not context.primary_ctx_roles or not sym_buffer.dispatch_first or sym_buffer.native_decode_dispatch):
        raise ValueError('NVLink-centric scheduling requires primary-context small decode backfill')
    if decode_compute_max_registers and (
            config.tokens > 32 or config.block_m > 64 or not sentinel_combine or
            not programmatic_finalize or not reuse_finalize_warps or
            sym_buffer.sentinel_combine_warps != 4 or combine_in_compute or
            sym_buffer.fused_compute_finalize or fused_dispatch_compute or
            compact_compute_ctas or single_epilogue_warpgroup):
        raise ValueError('decode register cap needs separate compute and four-warp early Combine')
    omit_sentinel_peer_finish = sym_buffer.omit_sentinel_peer_finish
    if omit_sentinel_peer_finish and (
            not sentinel_combine or not programmatic_finalize or combine_in_compute or
            sym_buffer.multicast_combine_ready):
        raise ValueError('only standalone early sentinel Combine can omit peer finish notifications')
    if programmatic_finalize and (not context.primary_ctx_roles or
                                  (not vector_combine and not expert_combine_offset)):
        raise ValueError('programmatic finalize needs the primary context and an early Combine path')
    recycle_dispatch_sms = sym_buffer.recycle_dispatch_sms
    fused_compute_finalize = sym_buffer.fused_compute_finalize
    compute_completion_offset = sym_buffer.compute_completion_offset
    if fused_compute_finalize and (
            not context.primary_ctx_roles or config.tokens > 32 or config.block_m > 64 or
            sym_buffer.group.size() > 32 or fused_dispatch_compute or
            programmatic_finalize or recycle_dispatch_sms or single_epilogue_warpgroup or
            compact_compute_ctas):
        raise ValueError('fused compute/finalize requires a cooperative small decode grid')
    static_w1_prefix = sym_buffer.static_w1_prefix
    static_decode_wave = sym_buffer.static_decode_wave
    static_w2_overlap = sym_buffer.static_w2_overlap
    reset_decode_ring_on_dispatch = sym_buffer.reset_decode_ring_on_dispatch
    if reset_decode_ring_on_dispatch and (
            not (prune_decode_resets and route_count_doorbell and programmatic_finalize) or
            fixed_expert_span or ready_tile_offset or activation_prefix_offset or
            push_decode_activations or fused_dispatch_compute or fused_compute_finalize or
            static_w1_prefix or static_decode_wave or sym_buffer.decode_w1_warmup):
        raise ValueError('Dispatch ring reset requires the ordinary small-decode counts barrier')
    single_expert_tile_lookup = (sym_buffer.single_expert_tile_lookup and
                                 config.tokens <= 32 and config.local_experts <= 32)
    prioritize_expert_rows = sym_buffer.prioritize_expert_rows
    if prioritize_expert_rows and (
            not single_expert_tile_lookup or fixed_expert_span or ready_tile_offset or
            static_decode_wave or static_w1_prefix or sym_buffer.decode_w2_clusters or
            sym_buffer.decode_w2_ready_k or sym_buffer.decode_w2_n_group != 1):
        raise ValueError('expert row priority requires ordinary interleaved single-tile scheduling')
    decode_w1_warmup = sym_buffer.decode_w1_warmup
    decode_w1_cadence = sym_buffer.decode_w1_cadence
    decode_shared_w2 = sym_buffer.decode_shared_w2
    pair_shared_w1_offset = sym_buffer.pair_shared_w1_offset
    if pair_shared_w1_offset and (
            not context.primary_ctx_roles or not sym_buffer.dispatch_first or context.dispatch_sms < 16 or
            config.tokens != 16 or config.block_m != 32 or config.hidden != 6144 or
            config.intermediate != 2048 or sym_buffer.num_max_tokens_per_rank < 32 or
            not direct_w1_store or not w2_tile_major_input or not sentinel_combine or
            sym_buffer.packed_decode_activations or sym_buffer.native_decode_dispatch or
            push_decode_activations or local_mma or independent_compute_ctas or
            fused_dispatch_compute or fused_compute_finalize or recycle_dispatch_sms or
            decode_shared_w2 not in (0, 1, 2) or sym_buffer.decode_w2_clusters):
        raise ValueError('paired shared W1 requires the 16-token clustered decode path with direct W1 stores')
    if decode_shared_w2 and (config.tokens > 32 or not config.shared_experts or
                            recycle_dispatch_sms or static_decode_wave or
                            early_experts is not None):
        raise ValueError('custom shared W2 placement requires ordinary small decode on one compute grid')
    if decode_shared_w2 == 4 and (
            not 0 < config.tokens <= config.block_m or config.shared_experts != 1 or
            local_mma or independent_compute_ctas or fused_dispatch_compute or
            fused_compute_finalize or sym_buffer.decode_w2_clusters):
        raise ValueError('ready shared W2 scheduling needs one shared M tile on the clustered decode grid')
    if decode_w1_warmup or decode_w1_cadence > 1:
        max_routes = 32 * sym_buffer.group.size() * min(config.local_experts, sym_buffer.num_topk)
        if (config.tokens > 32 or fixed_expert_span or ready_tile_offset or
                static_w1_prefix or static_decode_wave or recycle_dispatch_sms or
                2 * config.intermediate > config.hidden or
                ceil_div(max_routes, config.block_m) + config.local_experts >
                sym_buffer.layout.ring_tokens // config.block_m):
            raise ValueError('custom W1 waves require normal queues and small non-wrapping tiles')
    if static_decode_wave:
        max_routes = 32 * sym_buffer.group.size() * min(config.local_experts, sym_buffer.num_topk)
        if (config.tokens > 32 or fixed_expert_span or ready_tile_offset or
                static_w1_prefix or recycle_dispatch_sms or compact_compute_ctas or
                ceil_div(max_routes, config.block_m) + config.local_experts >
                sym_buffer.layout.ring_tokens // config.block_m):
            raise ValueError('static waves need one compute grid and a small non-wrapping input ring')
    if static_w1_prefix and (config.tokens > 32 or fixed_expert_span or ready_tile_offset or
                             recycle_dispatch_sms):
        raise ValueError('static W1 prefix needs a single grid with regular decode slots')
    compute_grid = context.compute_sms * max(1, compact_compute_ctas)
    compute_backfill_sms = sym_buffer.compute_backfill_sms
    dispatch_first = sym_buffer.dispatch_first
    if dispatch_first:
        if (not context.primary_ctx_roles or config.tokens > 32 or
                compute_backfill_sms > context.dispatch_sms or
                recycle_dispatch_sms or fused_dispatch_compute or compact_compute_ctas or
                (fused_compute_finalize and (not fused_finalize_sentinel or compute_backfill_sms))):
            raise ValueError('compute backfill requires an independent primary-context small decode grid')
        compute_grid += compute_backfill_sms
        _check_ring_capacity(config, sym_buffer, compute_grid)
    if combine_in_compute and compute_backfill_sms > context.dispatch_sms - 2:
        raise ValueError('compute-side Combine reserves a cluster slot for the early cleanup grid')
    if fused_finalize_sentinel and (context.dispatch_sms != 24 or compute_grid != 128 or
                                    not dispatch_first):
        raise ValueError('fused sentinel Combine requires the verified 24/128 Dispatch/compute placement')
    cooperative_compute = sym_buffer.decode_cooperative_compute
    if cooperative_compute and (
            not (programmatic_finalize and dispatch_first and sentinel_combine) or
            context.dispatch_sms != 24 or compute_grid != 128 or
            independent_compute_ctas or compact_compute_ctas or fused_compute_finalize):
        raise ValueError('cooperative compute diagnostic requires the 24/128 Dispatch/compute placement')
    if fused_dispatch_compute or (fused_compute_finalize and not fused_finalize_sentinel):
        # Leave room for two-CTA placement on GPCs with unequal SM counts.
        compute_grid -= 8
    total_compute_sms = compute_grid
    if fused_worker_reuse:
        shared_blocks = ceil_div(config.tokens, config.block_m)
        shared_clusters = max(config.hidden, 2 * config.intermediate * config.shared_experts) // 256
        if config.shared_experts and shared_blocks * shared_clusters > compute_grid // 2:
            raise ValueError('shared expert work must fit the early fused compute clusters')
        total_compute_sms += context.dispatch_sms
    decode_w2_clusters = sym_buffer.decode_w2_clusters
    decode_cluster_stride = sym_buffer.decode_cluster_stride
    shared_priority_prefix = sym_buffer.shared_priority_prefix
    routed_in_grid_prefix = (w2_prefetch_k_tiles and not bulk_w2_tail_experts) or (
        sym_buffer.w1_prefetch_k_tiles and not sym_buffer.bulk_w1_prefetch_bytes)
    if shared_priority_prefix and (
            not config.shared_experts or config.tokens > min(32, config.block_m) or
            local_mma or independent_compute_ctas or decode_cluster_stride or
            recycle_dispatch_sms or fused_dispatch_compute or fused_compute_finalize or
            sym_buffer.decode_shared_w2 or decode_w2_clusters or sym_buffer.pair_shared_w1_offset or
            sym_buffer.weight_prefix_warps or sym_buffer.post_init_prefetch or
            (sym_buffer.bulk_w1_prefetch_bytes and not sym_buffer.bulk_w1_prefetch_on_dispatch) or
            not routed_in_grid_prefix):
        raise ValueError('Shared priority needs one static token tile and in-grid Routed prefix prefetch')
    decode_resident_clusters = min(context.compute_sms, compute_grid) // 2
    if decode_cluster_stride and (
            config.tokens > 32 or config.block_m > 64 or local_mma or independent_compute_ctas or
            fused_dispatch_compute or fused_compute_finalize or recycle_dispatch_sms or
            gcd(decode_cluster_stride, decode_resident_clusters) != 1):
        raise ValueError('cluster stride needs a permutation of the initial two-CTA decode workers')
    decode_w2_ready_k = sym_buffer.decode_w2_ready_k
    decode_w2_early_workers = sym_buffer.decode_w2_early_workers
    if decode_w2_early_workers and (
            config.tokens > 32 or config.block_m > 64 or
            decode_w2_early_workers * 4 >= total_compute_sms or
            local_mma or independent_compute_ctas or static_decode_wave or
            static_w1_prefix or decode_w2_clusters or decode_w1_warmup or
            decode_w2_ready_k or recycle_dispatch_sms or early_experts is not None or
            fused_dispatch_compute or fused_compute_finalize):
        raise ValueError('early W2 workers need the ordinary clustered decode queue')
    w2_a_issue_window = sym_buffer.w2_a_issue_window
    if w2_a_issue_window and (
            config.tokens > 32 or config.block_m > 64 or local_mma or release_group != 1 or
            w2_a_issue_window > role_configs[1].num_stages or single_tma_lanes == 2 or
            config.intermediate % (128 * w2_a_issue_window)):
        raise ValueError('out-of-order W2 A loads need complete windows with one credit per stage')
    w2_b_issue_prefix = sym_buffer.w2_b_issue_prefix
    if w2_b_issue_prefix and (
            config.tokens > 32 or config.block_m > 64 or local_mma or dual_weight_warps or
            independent_compute_ctas or sym_buffer.pair_routed_weights or
            w2_b_issue_prefix * 128 >= config.intermediate):
        raise ValueError('W2 weight issue prefix needs a single clustered decode weight producer')
    decode_w2_n_group = sym_buffer.decode_w2_n_group
    if decode_w2_n_group > 1 and (
            config.tokens > 32 or 2 * config.intermediate > config.hidden or
            fixed_expert_span or ready_tile_offset or static_decode_wave or static_w1_prefix or
            early_experts is not None or decode_w2_clusters or decode_w2_ready_k or
            decode_w1_warmup or decode_w1_cadence != 1 or independent_compute_ctas):
        raise ValueError('N-first W2 needs the ordinary interleaved small-decode scheduler')
    if decode_w2_ready_k and (
            config.tokens > 32 or fixed_expert_span or ready_tile_offset or static_decode_wave or
            static_w1_prefix or decode_w2_clusters or decode_w2_ready_k > config.intermediate // 128):
        raise ValueError('W2 readiness priority requires contiguous small decode tasks')
    if decode_w2_clusters:
        max_rows = 32 * sym_buffer.group.size() * min(config.local_experts, sym_buffer.num_topk)
        if (config.tokens > 32 or fixed_expert_span or ready_tile_offset or static_decode_wave or
                static_w1_prefix or decode_w1_warmup or decode_w1_cadence > 1 or decode_shared_w2 not in (0, 2) or
                recycle_dispatch_sms or fused_dispatch_compute or compact_compute_ctas or
                decode_w2_clusters >= compute_grid // 2 or
                ceil_div(max_rows, config.block_m) + config.local_experts >
                sym_buffer.layout.ring_tokens // config.block_m):
            raise ValueError('startup W2 clusters require one non-wrapping small decode compute grid')
    role_sequence = ((1, 0), (2, 0), (3, 0))
    if recycle_dispatch_sms:
        shared_blocks = ceil_div(config.tokens, config.block_m)
        shared_clusters = max(config.hidden // 256,
                              2 * config.intermediate * config.shared_experts // 256)
        if (not context.primary_ctx_roles or config.tokens > 32 or early_experts is not None or
                context.dispatch_sms % 2 or
                shared_blocks * shared_clusters > context.compute_sms // 2):
            raise ValueError('dispatch-SM recycling requires an independent, paired decode helper grid')
        total_compute_sms += context.dispatch_sms
        role_sequence = ((1, 0), (2, 0), (2, context.compute_sms), (3, 0))
    role_resources = tuple((item.num_stages, item.dynamic_smem, item.threads)
                           for item in role_configs)
    role3_config = role_configs[2]
    role3_layout = role3_config.shared_layout
    combine_reusable_bytes = role3_layout.reusable_bytes
    if role3_layout.field_end > role3_config.dynamic_smem:
        raise ValueError('role3 shared storage exceeds its launch allocation')
    if vector_combine:
        combine_vectors = sym_buffer.sentinel_combine_vectors
        warp_chunk = combine_vectors * (WARP_BYTES // 2 if sentinel_half_vector else WARP_BYTES)
        combine_geometry = (config.hidden * 2 // warp_chunk, warp_chunk, combine_vectors)
    else:
        small_fetch = sym_buffer.decode_combine_fetch in ('small', 'contiguous', 'streaming-small')
        combine_geometry = get_combine_geometry(
            config.hidden, sym_buffer.num_topk +
            (1 if sym_buffer.num_shared_experts else 0),
            combine_reusable_bytes, role3_config.dynamic_smem,
            max_fetch_bytes=16 * 1024 if small_fetch else None)
    combine_chunks, combine_chunk_bytes, _ = combine_geometry
    direct_combine_output = sym_buffer.direct_combine_output
    if direct_combine_output and (
            config.tokens > 32 or vector_combine or stripe_combine_offset or
            y.data_ptr() % 16 or config.hidden * 2 % 16):
        raise ValueError('direct Combine output needs aligned small-batch TMA input')
    if sym_buffer.decode_combine_fetch != 'default' and (
            config.tokens > 32 or config.block_m > 64 or vector_combine or direct_w2_store or
            expert_combine_offset or token_major_combine or fused_compute_finalize):
        raise ValueError('alternate combine fetch needs normal small-batch W2 stores and TMA Combine')
    combine_interleaved_bytes = (combine_chunk_bytes if
                                 sym_buffer.decode_combine_fetch.startswith('contiguous') or
                                 sym_buffer.sentinel_chunk_major else 0)
    streaming_combine = sym_buffer.decode_combine_fetch.startswith('streaming')
    combine_token_warps = sym_buffer.combine_token_warps
    if combine_token_warps and (
            not config.shared_experts or config.tokens > 32 or
            vector_combine or stripe_combine_offset or expert_combine_offset or streaming_combine or
            combine_chunks % combine_token_warps or
            config.tokens * combine_chunks // combine_token_warps > total_compute_sms):
        raise ValueError('token-local Combine needs a full small-batch TMA group per CTA')
    decode_trace_offset = sym_buffer.decode_trace_offset
    native_decode_dispatch = sym_buffer.native_decode_dispatch
    if native_decode_dispatch:
        peers = sym_buffer.group.size()
        if (not 1 <= peers <= 32 or config.local_experts not in (8, 16) or
                config.hidden != 6144 or config.block_m not in (32, 48, 64) or
                sym_buffer.num_topk != 8 or sym_buffer.layout.capacity < 256 or
                config.tokens > 32 or sym_buffer.early_experts != 0 or
                context.dispatch_sms < config.local_experts or
                decode_mode not in ('route', 'route-reduce') or
                any((fused_dispatch_compute, fused_compute_finalize, recycle_dispatch_sms,
                     bulk_w1_prefetch_on_dispatch, route_count_doorbell, push_decode_activations,
                     dedup_activation_offset, pull_route_metadata_offset,
                     activation_prefix_bytes, ready_tile_offset,
                     peer_decode_input_offset, multicast_decode_input_offset,
                     packed_decode_activations, packed_w2_metadata, sym_buffer.split_route_heads,
                     prefetch_w2_metadata,
                     stripe_combine_offset, speculative_w1))):
            raise ValueError('native Dispatch needs fixed decode slots with 8 or 16 local experts')
    combine_resources = (combine_reusable_bytes, combine_chunks, combine_chunk_bytes,
                         single_publisher, vector_pull, vector_pull_group, vector_combine, weight_reuse,
                         prefetch_experts, decode_trace_offset, fixed_expert_span,
                         early_experts, release_group, direct_w1_store, w1_register_transpose,
                         w1_packed_scale_reduction, w1_hardware_scale_reduction,
                         w1_early_epi_inputs, w1_pair_scale_rounding, narrow_bm32, short_activation_sf,
                         trim_wide_activations,
                         single_mma_lane, sym_buffer.trace_stride,
                         sym_buffer.trace_operand_wait, reuse_k128_desc, mma_unroll,
                         hoist_routed_sfb_selection, descriptor_prefetch, probe_routed_mma,
                         bulk_w1_prefetch_bytes, bulk_w1_prefetch_on_dispatch, bulk_w1_prefetch_lanes,
                         ready_tile_offset, programmatic_finalize, recycle_dispatch_sms,
                         static_w1_prefix, static_w2_overlap, decode_task_slots, direct_w2_store, prefetch_w2_metadata,
                         packed_w2_metadata, speculative_w1, speculative_w1_n_tiles,
                         compact_compute_ctas, stripe_combine_offset, dedup_activation_offset,
                         source_elects_activation_owner, weight_lookahead,
                         sym_buffer.trace_operand_sources, packed_route_counts, combine_load_group,
                         dispatch_prefix_scan, static_decode_wave, token_major_combine, inline_route_indices,
                         pull_route_metadata_offset, decode_w1_warmup, decode_w1_cadence,
                         packed_route_count_group, sym_buffer.posted_route_sums, decode_shared_w2,
                         pair_shared_w1_offset, sym_buffer.pair_shared_w1_vector,
                         inline_route_scores, rotate_route_publication,
                         prune_decode_resets, omit_sentinel_peer_finish, reset_decode_ring_on_dispatch,
                         packed_decode_activations,
                         packed_decode_scores,
                         route_count_doorbell, dispatch_shared_route_counts,
                         multicast_route_counts, sym_buffer.multicast_combine_ready,
                         fused_dispatch_compute, multicast_decode_input_offset, push_decode_activations,
                         fused_worker_reuse, sym_buffer.push_overlap_reservation, sym_buffer.push_reserve_spans,
                         decode_w2_clusters, rank_route_doorbell, peer_decode_input_offset,
                         sym_buffer.stripe_active_peers, sym_buffer.stripe_launch_delay,
                         sym_buffer.stripe_tiles, activation_prefix_bytes, activation_prefix_offset,
                         peer_activation_prefix_tiles, peer_activation_rank_rows,
                         sym_buffer.peer_activation_head_clusters,
                         parallel_activation_prefix, single_epilogue_warpgroup, accumulator_stages,
                         fused_compute_finalize, compute_completion_offset, fused_finalize_sentinel,
                         cooperative_compute,
                         peer_decode_input_direct,
                         peer_decode_token_fanout, peer_prefetch_expert_heads,
                         peer_prefetch_opportunistic, trim_decode_gather_tail,
                         sym_buffer.multicast_pull_routes, sym_buffer.push_vector_io,
                         sym_buffer.warp_pull_routes, decode_w2_ready_k, w2_a_issue_window,
                         decode_w2_early_workers, w2_b_issue_prefix, keep_routed_weight_scales,
                         decode_cluster_stride, decode_resident_clusters,
                         sym_buffer.push_early_pool_offset, sym_buffer.push_row_ready_offset,
                         dual_weight_warps, push_scale_receiver_sms, sym_buffer.push_packed_scales_offset,
                         local_mma, narrow_local_mma, sym_buffer.local_mma_n8,
                         independent_compute_ctas, routed_k_major,
                         grouped_routed_weights, grouped_weight_inner_k, packed_routed_scale_base,
                         packed_routed_scale_k_group, packed_scale_helper, follower_scheduler,
                         packed_scale_release_group,
                         expert_combine_offset, compute_backfill_sms, dispatch_first,
                         combine_token_warps, w1_prefetch_k_tiles, sym_buffer.w1_prefix_expert_limit,
                         w2_prefetch_k_tiles, bulk_w2_tail_experts, route_publish_ctas,
                         split_route_heads, decode_linear_pool, native_decode_dispatch,
                         dispatch_pull_rows_per_cta, double_buffer_dispatch_pull,
                         combine_interleaved_bytes, streaming_combine, direct_combine_output,
                         sentinel_combine, sym_buffer.sentinel_combine_sms, reuse_finalize_warps,
                         sentinel_half_vector,
                         sym_buffer.sentinel_token_order, sym_buffer.sentinel_poll,
                         sym_buffer.sentinel_poll_delay_ns, sym_buffer.sentinel_adaptive_poll,
                         sym_buffer.combine_ready_flags_offset,
                         sym_buffer.combine_ready_mask,
                         sym_buffer.sentinel_combine_warps, sym_buffer.keep_w2_prefix,
                         sym_buffer.prefetch_weight_scales, sym_buffer.prefetch_k_first,
                         sym_buffer.weight_prefix_warps, shared_priority_prefix,
                         sym_buffer.post_init_prefetch, sym_buffer.relaxed_init_barrier,
                         sym_buffer.skip_init_prefetch_barrier,
                         sym_buffer.scheduled_weight_tiles, w2_prefetch_from_w1,
                         combine_in_compute,
                         sym_buffer.cache_routed_ranges, sym_buffer.keep_w1_prefix,
                         single_expert_tile_lookup, prioritize_expert_rows,
                         decode_w2_n_group, w2_tile_major_input, transpose_w2_activations,
                         single_tma_lanes, decode_operand_cache, shared_packets is not None, shared_packet_mask,
                         combine_at_block_start, decode_compute_max_registers, decode_nvlink_schedule,
                         pair_routed_weight_scales, shared_w1_prefetch_limit,
                         asymmetric_activation_rows, w1_tile_major_input, pair_routed_weights)
    key = _capture_key(config, sym_buffer, context, y, l1_weights, l2_weights,
                       shared_l1_weights, shared_l2_weights, stats,
                       nvlink_timeout_secs, routed_weight_linger,
                       role_resources, combine_resources,
                       compiler_identity)
    if peer_activation_window is not None:
        key = (*key, peer_activation_window.address)
    if shared_packets is not None:
        key = (*key, *(packet.data_ptr() if packet is not None else 0 for packet in shared_packets))
    prepared = _CACHE.get(key)
    if prepared is None:
        with COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                layout = sym_buffer.layout
                tma_plan = build_gc_tma_plan(role_configs[1], sym_buffer,
                                             l1_weights, l2_weights,
                                             shared_l1_weights, shared_l2_weights,
                                             config.shared_weight_gran_k,
                                             narrow_bm32=narrow_bm32, local_mma=local_mma,
                                             narrow_local_mma=narrow_local_mma,
                                             trim_wide_tokens=config.tokens if trim_wide_activations else 0)
                payloads = []
                for i, spec in enumerate(tma_plan.ordered()):
                    descriptor = spec.encode()
                    payload = ctypes.string_at(descriptor.getPtr(), 128)
                    if len(payload) != 128:
                        raise RuntimeError(f"TensorMap {i} is not 128 bytes")
                    payloads.append(payload)
                if len(payloads) != map_count:
                    raise RuntimeError('GC role2 TensorMap count disagrees with its configuration')
                maps = b"".join(payloads)
                role_kernels = []
                for role, block_offset in role_sequence:
                    role_kernel = GCMoERoleKernel(
                        role, context.dispatch_sms, total_compute_sms,
                        config.block_m, config.hidden, config.intermediate,
                        config.tokens, sym_buffer.group.size(), config.local_experts,
                        config.shared_experts,
                        sym_buffer.num_max_tokens_per_rank,
                        finalize_output=role == 3,
                        ring_tokens=layout.ring_tokens,
                        sf_ring_tokens=layout.sf_ring_tokens,
                        topk=sym_buffer.num_topk,
                        shared_weight_gran_k=config.shared_weight_gran_k,
                        # Only the clamp-presence flag specializes the binary. The
                        # actual FP32 value is a kernel argument on every replay.
                        activation_clamp=0.0 if config.clamp != inf else inf,
                        fast_math=config.fast_math,
                        has_stats=stats is not None,
                        nvlink_timeout_secs=nvlink_timeout_secs,
                        routed_weight_linger=routed_weight_linger,
                        decode_single_publisher=single_publisher,
                        route_publish_ctas=route_publish_ctas if role == 1 else 1,
                        split_route_heads=role in (1, 2) and split_route_heads,
                        inline_route_indices=role == 1 and inline_route_indices,
                        decode_linear_pool=decode_linear_pool,
                        dispatch_pull_rows_per_cta=dispatch_pull_rows_per_cta if role == 1 else 0,
                        double_buffer_dispatch_pull=role == 1 and double_buffer_dispatch_pull,
                        decode_vector_pull=vector_pull,
                        decode_vector_pull_group=vector_pull_group if role == 1 else 1,
                        decode_vector_combine=vector_combine,
                        alternate_experts=weight_reuse in ('alternate', 'alternate-w2'),
                        retain_routed_w2=weight_reuse in ('w2', 'alternate-w2'),
                        prefetch_experts=prefetch_experts,
                        w1_prefetch_k_tiles=(w1_prefetch_k_tiles if role == 2 or
                                             (role == 1 and bulk_w1_prefetch_on_dispatch) else 0),
                        w2_prefetch_k_tiles=(
                            w2_prefetch_k_tiles if role == (1 if bulk_w2_tail_experts else 2) else 0),
                        bulk_w2_tail_experts=bulk_w2_tail_experts if role == 1 else 0,
                        keep_w2_prefix=sym_buffer.keep_w2_prefix and role == 2,
                        decode_trace_offset=decode_trace_offset,
                        fixed_expert_span=fixed_expert_span,
                        early_experts=early_experts or 0,
                        speculative_w1=role == 2 and speculative_w1,
                        speculative_w1_n_tiles=speculative_w1_n_tiles if role == 2 else 0,
                        release_group=release_group if role == 2 else 1,
                        direct_w1_store=role == 2 and direct_w1_store,
                        direct_w2_store=role == 2 and direct_w2_store,
                        prefetch_w2_metadata=role == 2 and prefetch_w2_metadata,
                        packed_w2_metadata=role in (1, 2) and packed_w2_metadata,
                        narrow_bm32=role == 2 and narrow_bm32,
                        short_activation_sf=role == 2 and short_activation_sf,
                        trim_wide_activations=role == 2 and trim_wide_activations,
                        single_mma_lane=role == 2 and single_mma_lane,
                        trace_stride=sym_buffer.trace_stride if role == 2 else 1,
                        trace_operand_wait=role == 2 and sym_buffer.trace_operand_wait,
                        trace_operand_sources=role == 2 and sym_buffer.trace_operand_sources,
                        reuse_k128_desc=reuse_k128_desc if role == 2 else None,
                        mma_unroll=mma_unroll if role == 2 else 4,
                        hoist_routed_sfb_selection=hoist_routed_sfb_selection and role == 2,
                        descriptor_prefetch=descriptor_prefetch if role == 2 else 'all',
                        probe_routed_mma=probe_routed_mma if role == 2 else 0,
                        w1_register_transpose=role == 2 and w1_register_transpose,
                        w1_packed_scale_reduction=role == 2 and w1_packed_scale_reduction,
                        w1_hardware_scale_reduction=role == 2 and w1_hardware_scale_reduction,
                        w1_early_epi_inputs=role == 2 and w1_early_epi_inputs,
                        w1_pair_scale_rounding=role == 2 and w1_pair_scale_rounding,
                        bulk_w1_prefetch_bytes=(bulk_w1_prefetch_bytes if role == 2 or
                                                (role == 1 and bulk_w1_prefetch_on_dispatch) else 0),
                        bulk_w1_prefetch_on_dispatch=bulk_w1_prefetch_on_dispatch if role in (1, 2) else 0,
                        bulk_w1_prefetch_lanes=bulk_w1_prefetch_lanes if role == 2 else 1,
                        ready_tile_offset=ready_tile_offset,
                        static_w1_prefix=static_w1_prefix,
                        static_w2_overlap=role == 2 and static_w2_overlap,
                        task_slots=decode_task_slots if role == 2 else 2,
                        compact_compute=role == 2 and bool(compact_compute_ctas),
                        programmatic_finalize=role in (2, 3) and programmatic_finalize,
                        stripe_combine_offset=stripe_combine_offset,
                        expert_combine_offset=expert_combine_offset if role in (2, 3) else 0,
                        dispatch_backfill=role in (1, 2) and dispatch_first,
                        dedup_activation_offset=dedup_activation_offset,
                        source_elects_activation_owner=source_elects_activation_owner,
                        weight_lookahead=weight_lookahead if role == 2 else 0,
                        packed_route_counts=role == 1 and packed_route_counts,
                        packed_route_count_group=packed_route_count_group if role == 1 else 1,
                        rank_route_doorbell=role == 1 and rank_route_doorbell,
                        posted_route_sums=role == 1 and sym_buffer.posted_route_sums,
                        inline_route_scores=role == 1 and inline_route_scores,
                        rotate_route_publication=rotate_route_publication if role == 1 else 0,
                        prune_decode_resets=role in (1, 3) and prune_decode_resets,
                        reset_decode_ring_on_dispatch=role in (1, 3) and reset_decode_ring_on_dispatch,
                        omit_sentinel_peer_finish=role == 3 and omit_sentinel_peer_finish,
                        packed_decode_activations=packed_decode_activations,
                        packed_decode_scores=packed_decode_scores,
                        route_count_doorbell=role in (1, 2) and route_count_doorbell,
                        dispatch_shared_route_counts=role == 1 and dispatch_shared_route_counts,
                        multicast_route_counts=role == 1 and multicast_route_counts,
                        multicast_combine_ready=((role == 3 or stripe_combine_offset != 0) and
                                                 sym_buffer.multicast_combine_ready),
                        stripe_active_peers=role == 3 and sym_buffer.stripe_active_peers,
                        stripe_launch_delay=sym_buffer.stripe_launch_delay if role == 2 else 0,
                        stripe_tiles=sym_buffer.stripe_tiles,
                        activation_prefix_bytes=activation_prefix_bytes,
                        activation_prefix_offset=activation_prefix_offset,
                        peer_activation_prefix_tiles=peer_activation_prefix_tiles,
                        peer_activation_rank_rows=peer_activation_rank_rows,
                        peer_activation_head_clusters=sym_buffer.peer_activation_head_clusters,
                        parallel_activation_prefix=role == 1 and parallel_activation_prefix,
                        accumulator_stages=accumulator_stages if role == 2 else 2,
                        compute_completion_offset=compute_completion_offset if role == 2 else 0,
                        multicast_decode_input_offset=multicast_decode_input_offset if role == 1 else 0,
                        peer_decode_input_offset=peer_decode_input_offset if role == 1 else 0,
                        peer_decode_input_direct=peer_decode_input_direct and role in (1, 2),
                        peer_decode_token_fanout=peer_decode_token_fanout if role == 1 else 0,
                        peer_prefetch_expert_heads=peer_prefetch_expert_heads if role == 1 else 0,
                        peer_prefetch_opportunistic=role == 1 and peer_prefetch_opportunistic,
                        trim_decode_gather_tail=trim_decode_gather_tail and role == 2,
                        decode_input_stride=sym_buffer.peer_decode_input_stride,
                        push_decode_activations=role == 1 and push_decode_activations,
                        push_vector_io=role == 1 and sym_buffer.push_vector_io,
                        push_overlap_reservation=role == 1 and sym_buffer.push_overlap_reservation,
                        push_reserve_spans=role == 1 and sym_buffer.push_reserve_spans,
                        push_early_pool_offset=sym_buffer.push_early_pool_offset if role in (1, 2) else 0,
                        push_row_ready_offset=sym_buffer.push_row_ready_offset if role in (1, 2) else 0,
                        push_scale_receiver_sms=push_scale_receiver_sms if role in (1, 2) else 0,
                        push_packed_scales_offset=sym_buffer.push_packed_scales_offset if role == 1 else 0,
                        pull_route_metadata_offset=pull_route_metadata_offset if role == 1 else 0,
                        multicast_pull_routes=role == 1 and sym_buffer.multicast_pull_routes,
                        warp_pull_routes=role == 1 and sym_buffer.warp_pull_routes,
                        dispatch_prefix_scan=role == 1 and dispatch_prefix_scan,
                        static_decode_wave=static_decode_wave if role == 2 else 0,
                        decode_w1_warmup=decode_w1_warmup if role == 2 else 0,
                        decode_w1_cadence=decode_w1_cadence if role == 2 else 1,
                        decode_shared_w2=decode_shared_w2 if role == 2 else 0,
                        pair_shared_w1_offset=pair_shared_w1_offset if role in (1, 2) else 0,
                        pair_shared_w1_vector=role == 2 and sym_buffer.pair_shared_w1_vector,
                        decode_w2_clusters=decode_w2_clusters if role == 2 else 0,
                        decode_w2_ready_k=decode_w2_ready_k if role == 2 else 0,
                        w2_a_issue_window=w2_a_issue_window if role == 2 else 0,
                        w2_b_issue_prefix=w2_b_issue_prefix if role == 2 else 0,
                        decode_cluster_stride=decode_cluster_stride if role == 2 else 0,
                        decode_resident_clusters=decode_resident_clusters if role == 2 and decode_cluster_stride else 0,
                        decode_w2_early_workers=decode_w2_early_workers if role == 2 else 0,
                        keep_routed_weight_scales=role == 2 and keep_routed_weight_scales,
                        decode_w2_n_group=decode_w2_n_group if role == 2 else 1,
                        w2_tile_major_input=w2_tile_major_input and role == 2,
                        transpose_w2_activations=transpose_w2_activations if role == 2 else 0,
                        single_tma_lanes=single_tma_lanes if role == 2 else 0,
                        decode_operand_cache=decode_operand_cache if role == 2 else 'auto',
                        paired_shared_weights=role == 2 and shared_packets is not None,
                        shared_weight_packet_mask=shared_packet_mask if role == 2 else 0,
                        decode_compute_max_registers=decode_compute_max_registers if role == 2 else 0,
                        pair_routed_weight_scales=pair_routed_weight_scales and role == 2,
                        shared_w1_prefetch_limit=shared_w1_prefetch_limit if role == 2 else 0,
                        asymmetric_activation_rows=asymmetric_activation_rows if role == 2 else 0,
                        w1_tile_major_input=w1_tile_major_input and role in (1, 2),
                        pair_routed_weights=pair_routed_weights if role == 2 else 0,
                        dual_weight_warps=role == 2 and dual_weight_warps,
                        local_mma=role == 2 and local_mma,
                        narrow_local_mma=role == 2 and narrow_local_mma,
                        local_mma_n8=sym_buffer.local_mma_n8 if role == 2 else 0,
                        independent_compute_ctas=role == 2 and independent_compute_ctas,
                        routed_k_major=routed_k_major if role == 2 else (False, False),
                        grouped_routed_weights=grouped_routed_weights if role == 2 else 0,
                        grouped_weight_inner_k=grouped_weight_inner_k if role == 2 else (2, 2),
                        packed_routed_scale_base=packed_routed_scale_base if role == 2 else None,
                        packed_routed_scale_k_group=packed_routed_scale_k_group if role == 2 else 1,
                        packed_scale_helper=role == 2 and packed_scale_helper,
                        packed_scale_release_group=packed_scale_release_group if role == 2 else 4,
                        follower_scheduler=role == 2 and follower_scheduler,
                        token_major_combine=role in (2, 3) and token_major_combine,
                        combine_load_group=combine_load_group if role == 3 else 1,
                        combine_token_warps=combine_token_warps if role == 3 else 0,
                        combine_interleaved_bytes=combine_interleaved_bytes if role in (2, 3) else 0,
                        streaming_combine=streaming_combine and role == 3,
                        direct_combine_output=direct_combine_output and role == 3,
                        sentinel_combine=sentinel_combine and role in (2, 3),
                        sentinel_combine_sms=sym_buffer.sentinel_combine_sms if role == 3 else 0,
                        reuse_finalize_warps=role == 3 and reuse_finalize_warps,
                        single_cleanup_cta=role == 3 and sym_buffer.single_cleanup_cta,
                        sentinel_half_vector=role == 3 and sentinel_half_vector,
                        sentinel_token_order=sym_buffer.sentinel_token_order and role == 3,
                        sentinel_poll=(sym_buffer.sentinel_poll
                                       if role == 3 or (combine_in_compute and role == 2) else 'full'),
                        sentinel_poll_delay_ns=sym_buffer.sentinel_poll_delay_ns if role == 3 else 0,
                        sentinel_adaptive_poll=sym_buffer.sentinel_adaptive_poll and role == 3,
                        combine_ready_flags_offset=sym_buffer.combine_ready_flags_offset if role in (2, 3) else 0,
                        combine_ready_mask=role in (2, 3) and sym_buffer.combine_ready_mask,
                        sentinel_combine_warps=sym_buffer.sentinel_combine_warps if role == 3 else 8,
                        prefetch_weight_scales=sym_buffer.prefetch_weight_scales and role == 2,
                        prefetch_k_first=sym_buffer.prefetch_k_first and role == 2,
                        weight_prefix_warps=sym_buffer.weight_prefix_warps and role == 2,
                        shared_priority_prefix=shared_priority_prefix if role == 2 else 0,
                        post_init_prefetch=sym_buffer.post_init_prefetch and role == 2,
                        relaxed_init_barrier=sym_buffer.relaxed_init_barrier and role == 2,
                        skip_init_prefetch_barrier=(sym_buffer.skip_init_prefetch_barrier and
                                                    role == 2),
                        scheduled_weight_tiles=sym_buffer.scheduled_weight_tiles if role == 2 else 0,
                        w2_prefetch_from_w1=w2_prefetch_from_w1 if role == 2 else 0,
                        combine_in_compute=combine_in_compute if role in (2, 3) else 0,
                        cache_routed_ranges=sym_buffer.cache_routed_ranges and role == 2,
                        single_expert_tile_lookup=single_expert_tile_lookup and role == 2,
                        prioritize_expert_rows=prioritize_expert_rows if role == 2 else 0,
                        keep_w1_prefix=(sym_buffer.keep_w1_prefix and (role == 2 or
                                        (role == 1 and bulk_w1_prefetch_on_dispatch))),
                        compute_block_offset=block_offset,
                        w1_prefix_expert_limit=sym_buffer.w1_prefix_expert_limit,
                        role_config=role_configs[role - 1],
                        combine_geometry=combine_geometry if role == 3 else None)
                    role_kernels.append(role_kernel)
                binary_key = _binary_key(
                    compiler_identity, tuple(role_kernels), role_resources,
                    combine_resources, sym_buffer, y, weights, stats)
                if fused_dispatch_compute:
                    dispatch_kernel, compute_kernel, finalize_kernel = role_kernels
                    compute_kernel.fused_dispatch_role = dispatch_kernel
                    compute_kernel.fused_worker_reuse = fused_worker_reuse
                    launch_kernels = (compute_kernel, finalize_kernel)
                elif fused_compute_finalize:
                    dispatch_kernel, compute_kernel, finalize_kernel = role_kernels
                    compute_kernel.fused_finalize_role = finalize_kernel
                    compute_kernel.fused_finalize_sentinel = fused_finalize_sentinel
                    launch_kernels = (dispatch_kernel, compute_kernel)
                else:
                    launch_kernels = role_kernels
                def compile_roles():
                    signal_view = from_dlpack(sym_buffer.buffer.view(torch.int32))
                    peer_view = from_dlpack(sym_buffer.buffer.view(torch.int64).flatten()[:74])
                    # These views supply CuTe's pointer type/shape only. Every
                    # TensorMap pointer load becomes a host-supplied value param.
                    map_storage = sym_buffer.buffer.view(torch.uint8).flatten()
                    map_views = tuple(from_dlpack(map_storage[i * 128:(i + 1) * 128])
                                      for i in range(map_count))
                    role3_map_view = from_dlpack(map_storage[:2304])
                    y_view = from_dlpack(y)
                    stats_view = from_dlpack(stats) if stats is not None else signal_view
                    rank_view, token_view = Uint32(0), Uint32(0)
                    clamp_view, alpha_view = Float32(0.0), Float32(1.0)
                    compiled_binaries = []
                    for role_kernel in launch_kernels:
                        role = role_kernel.role
                        if role == 1 and native_decode_dispatch:
                            compiled_binaries.append(_native_dispatch_binary(
                                layout, role_kernel, role_configs[0], fixed_expert_span,
                                dispatch_first, nvlink_timeout_secs, context.dispatch_sms,
                                native_decode_dispatch))
                            continue
                        is_helper = role_kernel.compute_block_offset != 0
                        prefetch_maps = None
                        if role_kernel.descriptor_prefetch == 'weights':
                            prefetch_maps = (2, 3, 7, 8)
                            if role_kernel.shared_experts:
                                prefetch_maps += (11, 12, 16, 17)
                        elif role_kernel.descriptor_prefetch == 'off':
                            prefetch_maps = ()
                        compile_args = ((signal_view, peer_view, rank_view, token_view,
                                         map_views[0], None, None,
                                         clamp_view, alpha_view, *map_views[1:])
                                        if role == 2 else
                                        (signal_view, peer_view, rank_view, token_view,
                                         role3_map_view, y_view, stats_view,
                                         clamp_view, alpha_view)
                                        if role == 3 else
                                        (signal_view, peer_view, rank_view, token_view,
                                         None, None, None,
                                         clamp_view, alpha_view))
                        ptxas_options = compiler_identity.ptxas_options
                        if role_kernel.decode_compute_max_registers:
                            ptxas_options += (f'--maxrregcount={role_kernel.decode_compute_max_registers}',)
                        cubin = compile_kernel(
                            role_kernel, compile_args, compiler_identity,
                            lambda ptx, dump_dir: _assemble_role_value_params(
                                ptx, role, dump_dir, target=compiler_target,
                                use_small_tile_path=has_prefix_maps,
                                direct_w1_store=direct_w1_store,
                                dedup_activations=bool(dedup_activation_offset or peer_decode_input_direct),
                                dual_weight_warps=role_kernel.dual_weight_warps,
                                alternate_weight_load_sites=(
                                    bool(role_kernel.grouped_routed_weights & role_kernel.pair_routed_weights)
                                    or role_kernel.packed_routed_scale_base is not None),
                                prefetch_maps=prefetch_maps,
                                peer_gather=bool(peer_activation_rank_rows),
                                guarded_weight_loop=(bool(role_kernel.w2_b_issue_prefix) or
                                                     role_kernel.packed_routed_scale_k_group == 4),
                                paired_shared_weights=bool(role_kernel.shared_weight_packet_mask),
                                omit_peer_finish=role_kernel.omit_sentinel_peer_finish,
                                ptxas_options=ptxas_options))
                        role_config = role_configs[role - 1]
                        grid = (context.dispatch_sms if role == 1 or is_helper else
                                compute_grid if role == 2 else role_kernel.finalize_sms)
                        smem = role_config.dynamic_smem
                        if fused_dispatch_compute and role == 2:
                            grid += context.dispatch_sms
                            smem = max(smem, role_configs[0].dynamic_smem)
                        if fused_compute_finalize and role == 2 and not vector_combine:
                            smem = max(smem, role_configs[2].dynamic_smem)
                        threads = role_kernel.finalize_threads if role == 3 else role_config.threads
                        compiled_binaries.append(RoleBinary(
                            role, cubin, 0 if role_kernel.decode_vector_combine
                            else smem, threads, grid,
                            programmatic_input=role == 3 and programmatic_finalize,
                            wait_on_dispatch=is_helper,
                            cooperative=role == 2 and (fused_compute_finalize or cooperative_compute),
                            independent_compute_ctas=role == 2 and independent_compute_ctas,
                            programmatic_at_block_start=role == 3 and combine_at_block_start,
                            nvlink_centric_scheduling=bool(decode_nvlink_schedule & (1 << (role - 1))),
                            programmatic_dispatch_input=role == 2 and dispatch_first))
                    return compiled_binaries

                binaries = _get_or_compile_binaries(
                    binary_key, compiler_identity, compiler_target, compile_roles)
                prepared = _Prepared(maps, binaries, weights)
                if _compiler_identity(compiler_target) != compiler_identity:
                    raise RuntimeError('compiler identity changed while preparing role binaries')
                _CACHE[key] = prepared
    if len(prepared.maps) != map_count * 128:
        raise RuntimeError('cached TensorMap payload disagrees with its configuration')
    signal_arg = ctypes.c_void_p(sym_buffer.buffer.data_ptr())
    stats_pointer = stats.data_ptr() if stats is not None else sym_buffer.buffer.data_ptr()
    packed_scale_addresses = ((l1_weights[1].data_ptr(), l2_weights[1].data_ptr())
                              if packed_routed_scale_base is not None else ())
    needs_output = fused_compute_finalize or combine_in_compute
    peer_arg = _sym_buffer_value(sym_buffer, y.data_ptr() if needs_output else 0,
                                 stats_pointer if needs_output else 0, packed_scale_addresses,
                                 l1_weights[0].data_ptr() if bulk_w1_prefetch_bytes else 0,
                                 l2_weights[0].data_ptr() if bulk_w2_tail_experts else 0)
    rank_arg = ctypes.c_uint32(sym_buffer.rank_idx)
    token_arg = ctypes.c_uint32(config.tokens)
    clamp_arg = ctypes.c_float(config.clamp)
    alpha_arg = ctypes.c_float(config.alpha)
    map_args = tuple((ctypes.c_uint8 * 128).from_buffer_copy(
        prepared.maps[i * 128:(i + 1) * 128]) for i in range(map_count))
    # Role3 retains its original 8-byte unused CuTe pointer formal. Role2
    # carries the configuration-selected 128-byte descriptor values.
    role3_map_arg = ctypes.c_void_p(sym_buffer.buffer.data_ptr())
    y_arg = ctypes.c_void_p(y.data_ptr())
    stats_arg = ctypes.c_void_p(stats_pointer)
    arguments = ((signal_arg, peer_arg, rank_arg, token_arg,
                  clamp_arg, alpha_arg),
                 (signal_arg, peer_arg, rank_arg, token_arg,
                  clamp_arg, alpha_arg, *map_args),
                 (signal_arg, peer_arg, rank_arg, token_arg, role3_map_arg,
                  y_arg, stats_arg, clamp_arg, alpha_arg))
    if recycle_dispatch_sms:
        arguments = (*arguments[:2], arguments[1], arguments[2])
    elif fused_dispatch_compute:
        arguments = arguments[1:]
    elif fused_compute_finalize:
        arguments = arguments[:2]
    return list(prepared.binaries), arguments

def launch_gc_moe(config, y, l1_weights, l2_weights, sym_buffer,
                  shared_l1_weights, shared_l2_weights, stats, *,
                  decode_mode: str = 'off', weight_reuse: str = 'off',
                  prefetch_experts: int = 0) -> None:
    context = get_context(num_ranks=sym_buffer.group.size())
    binaries, arguments = prepare_role_binaries(
        config, y, l1_weights, l2_weights, sym_buffer,
        shared_l1_weights, shared_l2_weights, stats, context,
        decode_mode=decode_mode, weight_reuse=weight_reuse,
        prefetch_experts=prefetch_experts)
    context.launch(binaries, arguments, flatten_capture=sym_buffer.flat_decode_graph,
                   cluster_policy=sym_buffer.decode_cluster_policy)
