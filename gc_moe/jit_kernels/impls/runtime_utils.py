"""Host TensorMap construction and CUDA encoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from cuda.bindings import driver as cu

_DTYPE = {
    torch.int32: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.float32: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.bfloat16: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int8: cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
}
_SWIZZLE = {
    0: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


@dataclass(frozen=True)
class TmaDescriptorSpec:
    address: int
    dtype: torch.dtype
    dims: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    box: tuple[int, ...]
    swizzle_bytes: int

    def __post_init__(self) -> None:
        rank = len(self.dims)
        if self.dtype not in _DTYPE or rank not in (2, 3, 4, 5):
            raise ValueError('unsupported GC TMA descriptor type/rank')
        if len(self.strides_bytes) != rank - 1 or len(self.box) != rank:
            raise ValueError('invalid GC TMA rank')
        if self.swizzle_bytes not in _SWIZZLE or self.address % 16:
            raise ValueError('unaligned GC TMA base or swizzle')
        if any(x <= 0 for x in self.dims + self.box):
            raise ValueError('GC TMA dimensions must be positive')
        if any(x % 16 for x in self.strides_bytes):
            raise ValueError('GC TMA strides must be 16-byte aligned')

    def encode(self) -> cu.CUtensorMap:
        """Encode only at launch preparation, with a CUDA context current."""
        result = cu.cuTensorMapEncodeTiled(
            _DTYPE[self.dtype], len(self.dims), self.address,
            tuple(cu.cuuint64_t(x) for x in self.dims),
            tuple(cu.cuuint64_t(x) for x in self.strides_bytes),
            tuple(cu.cuuint32_t(x) for x in self.box),
            tuple(cu.cuuint32_t(1) for _ in self.dims),
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            _SWIZZLE[self.swizzle_bytes],
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result[0]}")
        return result[1]


def _base(tensor: torch.Tensor, dims: Sequence[int], strides_elements: Sequence[int],
          box: Sequence[int], swizzle: int) -> TmaDescriptorSpec:
    element_bytes = tensor.element_size()
    box = tuple(box)
    if swizzle:
        box = (swizzle // element_bytes, *box[1:])
    return TmaDescriptorSpec(
        tensor.data_ptr(), tensor.dtype, tuple(dims),
        tuple(int(s) * element_bytes for s in strides_elements), box, swizzle)


def make_tma_2d_desc(tensor: torch.Tensor, inner: int, outer: int,
              block_inner: int, block_outer: int,
              outer_stride: int, swizzle: int) -> TmaDescriptorSpec:
    """Construct a 2D map, including FP4 unpack to 8-bit shared memory."""
    if tensor.dtype == torch.int8 and inner % 128:
        raise ValueError('FP4 unpack requires a 128-element aligned K axis')
    return _base(tensor, (inner, outer), (outer_stride,),
                 (block_inner, block_outer), swizzle)


def repack_fp4_weights(tensor: torch.Tensor, block_k: int = 128) -> torch.Tensor:
    """Repack routed FP4 weights into tile-contiguous 64-byte-row segments.

    Input (experts, rows, K/2) int8 becomes (experts, kb, rows, block_k/2):
    every 64-byte K segment of every row stays byte-identical, but segments
    of the same K block across all rows of an expert become contiguous. The
    weight TensorMap then bursts one 8KB contiguous global range per op
    instead of 128 segments spaced a full weight row apart.
    """
    if tensor.ndim != 3 or tensor.dtype != torch.int8:
        raise ValueError('routed FP4 weights must be packed int8 3D')
    if not tensor.is_contiguous():
        raise ValueError('routed FP4 weight dense storage must be contiguous')
    segment = block_k // 2
    experts, rows, cols = tensor.shape
    if cols % segment:
        raise ValueError('FP4 weight K axis must contain whole tile segments')
    return (tensor.view(experts, rows, cols // segment, segment)
                 .permute(0, 2, 1, 3).contiguous())


def repack_fp8_weights(tensor: torch.Tensor, block_k: int = 128) -> torch.Tensor:
    """Repack shared FP8 weights into tile-contiguous 128-byte-row segments."""
    if tensor.ndim != 2 or tensor.dtype != torch.float8_e4m3fn:
        raise ValueError('shared FP8 weights must be 2D FP8')
    if not tensor.is_contiguous():
        raise ValueError('shared FP8 weight dense storage must be contiguous')
    segment = block_k
    rows, cols = tensor.shape
    if cols % segment:
        raise ValueError('FP8 weight K axis must contain whole tile segments')
    return (tensor.view(rows, cols // segment, segment)
                 .permute(1, 0, 2).contiguous())


def is_routed_weight_tile_layout(tensor: torch.Tensor) -> bool:
    if tensor.ndim != 4 or tensor.dtype != torch.int8:
        return False
    _, tiles, rows, row_bytes = tensor.shape
    expert_stride, tile_stride, row_stride, element_stride = tensor.stride()
    return (element_stride == 1 and row_stride == row_bytes and
            ((tile_stride >= rows * row_bytes and expert_stride >= tiles * tile_stride) or
             is_routed_weight_k_major(tensor)))


def is_routed_weight_k_major(tensor: torch.Tensor) -> bool:
    """Recognize K planes that physically contain all local experts."""
    if tensor.ndim != 4 or tensor.dtype != torch.int8:
        return False
    experts, _, rows, row_bytes = tensor.shape
    return experts > 1 and tensor.stride() == (rows * row_bytes, experts * rows * row_bytes,
                                               row_bytes, 1)


def colocate_routed_scales(payload: torch.Tensor, scales: torch.Tensor):
    """Place each routed K128 plane beside its original packed row scales."""
    if not is_routed_weight_tile_layout(payload):
        raise ValueError('expected K128 routed weight tiles')
    experts, tiles, rows, row_bytes = payload.shape
    if (scales.dtype != torch.int32 or tuple(scales.shape) != (experts, rows, tiles) or
            scales.device != payload.device or rows % 128):
        raise ValueError('expected matching packed routed scales and 128-aligned N')
    weight_bytes = rows * row_bytes
    plane_bytes = weight_bytes + rows * 4
    storage = torch.empty(experts * tiles * plane_bytes, dtype=torch.int8, device=payload.device)
    weight_view = storage.as_strided(payload.shape, (tiles * plane_bytes, plane_bytes, row_bytes, 1))
    scale_view = storage.view(torch.int32).as_strided(
        scales.shape, (tiles * plane_bytes // 4, 1, plane_bytes // 4), weight_bytes // 4)
    weight_view.copy_(payload)
    scale_view.copy_(scales)
    return weight_view, scale_view


def make_tma_weights_3d_desc(tensor: torch.Tensor, block_k: int, load_n: int,
                     swizzle: int = 128, *, pair_k: bool = False) -> TmaDescriptorSpec:
    """Map a tile-repacked FP4 weight tensor to a 4D (K-tile,rows,kb,expert) map.

    The tensor is (experts, kb, rows, block_k/2) int8 from repack_fp4_weights.
    One box (block_k, load_n, 1, 1) then reads a single contiguous global
    range: row stride is block_k/2 bytes and rows are adjacent.
    """
    if tensor.ndim == 6 and tensor.dtype == torch.int8:
        experts, groups, n_tiles, inner_k, rows, segment = tensor.shape
        if (not tensor.is_contiguous() or inner_k not in (2, 4, 8, 16) or
                (rows, segment * 2) != (load_n, block_k)):
            raise ValueError('grouped routed weights must have contiguous N128 K groups')
        return _base(tensor, (block_k, rows, inner_k, experts * groups * n_tiles),
                     (tensor.stride(4), tensor.stride(3), tensor.stride(2)),
                     (block_k, load_n, 2 if pair_k else 1, 1), swizzle)
    if tensor.ndim != 4 or tensor.dtype != torch.int8:
        raise ValueError('routed FP4 weights must be tile-repacked int8 4D')
    if not is_routed_weight_tile_layout(tensor):
        raise ValueError('routed FP4 repacked storage must have contiguous rows within each K tile')
    experts, kb, rows, segment = tensor.shape
    if segment * 2 != block_k or rows % load_n or kb <= 0:
        raise ValueError('FP4 weight tiles must match the descriptor K block')
    k_box = 2 if pair_k else 1
    if kb % k_box:
        raise ValueError('paired FP4 TMA needs an even number of K128 planes')
    if is_routed_weight_k_major(tensor):
        return _base(tensor, (block_k, rows, experts, kb),
                     (tensor.stride(2), tensor.stride(0), tensor.stride(1)),
                     (block_k, load_n, 1, k_box), swizzle)
    return _base(tensor, (block_k, rows, kb, experts),
                 (tensor.stride(2), tensor.stride(1), tensor.stride(0)),
                 (block_k, load_n, k_box, 1), swizzle)


def make_tma_weights_2d_desc(tensor: torch.Tensor, block_k: int, load_n: int,
                     swizzle: int = 128) -> TmaDescriptorSpec:
    """Map a tile-repacked FP8 weight tensor to a 4D (K-tile,rows,kb,1) map.

    The tensor is (kb, rows, block_k) FP8 from repack_fp8_weights; one box
    (block_k, load_n, 1, 1) reads a contiguous 16KB global range.
    """
    if tensor.ndim != 3 or tensor.dtype != torch.float8_e4m3fn:
        raise ValueError('shared FP8 weights must be tile-repacked FP8 3D')
    if not tensor.is_contiguous():
        raise ValueError('shared FP8 repacked storage must be contiguous')
    kb, rows, segment = tensor.shape
    if segment != block_k or rows % load_n or kb <= 0:
        raise ValueError('FP8 weight tiles must match the descriptor K block')
    # The trailing size-1 expert axis takes the full-tensor byte stride,
    # mirroring the dense-2D map it replaces.
    return _base(tensor, (block_k, rows, kb, 1),
                 (tensor.stride(1), tensor.stride(0), rows * segment * kb),
                 (block_k, load_n, 1, 1), swizzle)


def make_tma_sf_desc(tensor: torch.Tensor, mn: int, k: int, block_mn: int,
          gran_k: int, groups: int, smem_outer: int,
          sf_k_stride: int = 0) -> TmaDescriptorSpec:
    """Construct a K-major packed-int32 UE8M0 map with the MN axis first."""
    if tensor.dtype != torch.int32 or gran_k not in (32, 128):
        raise ValueError('GC SF is packed int32 at K32 or K128')
    aligned_mn = (mn + 3) // 4 * 4
    outer_stride = sf_k_stride or aligned_mn
    if outer_stride < aligned_mn:
        raise ValueError('SF K stride shorter than aligned MN axis')
    k_packs = (k + gran_k * 4 - 1) // (gran_k * 4)
    return make_tma_2d_desc(tensor, aligned_mn, k_packs * groups,
                     block_mn, smem_outer, outer_stride, 0)
