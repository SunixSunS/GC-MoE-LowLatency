"""Allocate benchmark weights in plain or hardware-compressible CUDA VMM."""

import torch
from cuda.bindings import driver as cu

from gc_moe.runtime.gc_moe import _ok


class WeightVmm:
    def __init__(self):
        self.mappings = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.mappings:
            torch.cuda.synchronize()
        for address, size in reversed(self.mappings):
            _ok(cu.cuMemUnmap(address, size))
            _ok(cu.cuMemAddressFree(address, size))

    def copy(self, source: torch.Tensor, compress: bool) -> torch.Tensor:
        elements = 1 + sum((dimension - 1) * stride
                           for dimension, stride in zip(source.shape, source.stride()))
        nbytes = elements * source.element_size()
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = source.get_device()
        prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        prop.allocFlags.compressionType = (
            cu.CUmemAllocationCompType.CU_MEM_ALLOCATION_COMP_GENERIC if compress
            else cu.CUmemAllocationCompType.CU_MEM_ALLOCATION_COMP_NONE)
        granularity = _ok(cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        size = (nbytes + granularity - 1) // granularity * granularity
        address = _ok(cu.cuMemAddressReserve(size, 0, 0, 0))
        mapped = False
        try:
            handle = _ok(cu.cuMemCreate(size, prop, 0))
            try:
                actual = _ok(cu.cuMemGetAllocationPropertiesFromHandle(handle))
                if actual.allocFlags.compressionType != prop.allocFlags.compressionType:
                    raise RuntimeError('CUDA did not grant the requested VMM compression mode')
                _ok(cu.cuMemMap(address, size, 0, handle, 0))
                mapped = True
            finally:
                _ok(cu.cuMemRelease(handle))
            access = cu.CUmemAccessDesc()
            access.location = prop.location
            access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _ok(cu.cuMemSetAccess(address, size, [access], 1))
            storage = torch._C._construct_storage_from_data_pointer(int(address), source.device, nbytes)
            result = torch.empty(0, dtype=source.dtype, device=source.device)
            result.set_(storage, 0, source.shape, source.stride()).copy_(source)
        except BaseException:
            if mapped:
                _ok(cu.cuMemUnmap(address, size))
            _ok(cu.cuMemAddressFree(address, size))
            raise
        self.mappings.append((address, size))
        return result
