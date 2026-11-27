"""A rank-strided CUDA TensorMap view of existing symmetric input allocations."""

from contextlib import ExitStack
from math import lcm

from cuda.bindings import driver as cu
import torch

from ..layout.gc_moe import align
from .gc_moe import _ok


class PeerActivationWindow:
    def __init__(self, peer_addresses, device, input_offset, hidden, max_tokens):
        self.device = device
        with ExitStack() as resources:
            handles = []
            lengths = []
            granularities = []
            offsets = []
            for peer in peer_addresses:
                handle = _ok(cu.cuMemRetainAllocationHandle(peer))
                resources.callback(lambda allocation=handle: _ok(cu.cuMemRelease(allocation)))
                handles.append(handle)
                properties = _ok(cu.cuMemGetAllocationPropertiesFromHandle(handle))
                granularities.append(_ok(cu.cuMemGetAllocationGranularity(
                    properties, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)))
                pointer = cu.CUdeviceptr(peer)
                start = _ok(cu.cuPointerGetAttribute(
                    cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, pointer))
                offsets.append(peer - int(start) + input_offset)
                lengths.append(int(_ok(cu.cuPointerGetAttribute(
                    cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE, pointer))))
            if len(set(offsets)) != 1 or any(
                    offset < 0 or offset + max_tokens * hidden > length
                    for offset, length in zip(offsets, lengths)):
                raise ValueError('symmetric activation views have incompatible allocation boundaries')
            stride = align(max(lengths), lcm(hidden, *granularities))
            length = len(handles) * stride
            base = _ok(cu.cuMemAddressReserve(length, 0, 0, 0))
            resources.callback(lambda start=base, size=length: _ok(cu.cuMemAddressFree(start, size)))
            access = cu.CUmemAccessDesc()
            access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = device.index
            access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
            for rank, (handle, length) in enumerate(zip(handles, lengths)):
                segment = cu.CUdeviceptr(int(base) + rank * stride)
                _ok(cu.cuMemMap(segment, length, 0, handle, 0))
                resources.callback(lambda start=segment, size=length: _ok(cu.cuMemUnmap(start, size)))
                _ok(cu.cuMemSetAccess(segment, length, [access], 1))
            self.address = int(base) + offsets[0]
            self.rank_rows = stride // hidden
            self.rows = (len(handles) - 1) * self.rank_rows + max_tokens
            self.resources = resources.pop_all()

    def close(self):
        if self.resources is not None:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self.resources.close()
            self.resources = None
