"""Check whether live GC symmetric activations can back a rank-strided TMA view."""

import argparse
import json
from math import lcm
import os
from pathlib import Path
import traceback

from cuda.bindings import driver as cu
import torch
import torch.distributed as dist

from gc_moe.gc import GCSymmBuffer
from gc_moe.layout.gc_moe import align
from gc_moe.runtime.gc_moe import _ok


TOKENS = 32
HIDDEN = 6144


def check_rank(buffer, rank, world, device):
    access = cu.CUmemAccessDesc()
    access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device.index
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
    granularity_flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    x_offset = buffer.layout.views['x'].offset
    handles = []
    mapped = []
    address = None
    try:
        granularities = []
        lengths = []
        input_offsets = []
        for peer in buffer.peer_addresses:
            handle = _ok(cu.cuMemRetainAllocationHandle(peer))
            handles.append(handle)
            properties = _ok(cu.cuMemGetAllocationPropertiesFromHandle(handle))
            granularities.append(_ok(cu.cuMemGetAllocationGranularity(properties, granularity_flag)))
            peer_pointer = cu.CUdeviceptr(peer)
            start = _ok(cu.cuPointerGetAttribute(
                cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, peer_pointer))
            length = _ok(cu.cuPointerGetAttribute(
                cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE, peer_pointer))
            input_offsets.append(peer - int(start) + x_offset)
            lengths.append(int(length))
        if len(set(input_offsets)) != 1 or any(offset + TOKENS * HIDDEN > length
                                               for offset, length in zip(input_offsets, lengths)):
            raise RuntimeError('mapped peers have inconsistent allocation boundaries')
        x_offset = input_offsets[0]
        stride = align(max(lengths), lcm(HIDDEN, *granularities))
        address = _ok(cu.cuMemAddressReserve(world * stride, 0, 0, 0))
        for peer, (handle, length) in enumerate(zip(handles, lengths)):
            segment = cu.CUdeviceptr(int(address) + peer * stride)
            _ok(cu.cuMemMap(segment, length, 0, handle, 0))
            mapped.append((segment, length))
            _ok(cu.cuMemSetAccess(segment, length, [access], 1))
        matches = []
        for peer in range(world):
            data_ptr = int(address) + x_offset + peer * stride
            storage = torch._C._construct_storage_from_data_pointer(data_ptr, device, TOKENS * HIDDEN)
            values = torch.empty(0, dtype=torch.uint8, device=device)
            values.set_(storage, 0, (TOKENS, HIDDEN), (HIDDEN, 1))
            expected = (torch.arange(TOKENS, device=device).unsqueeze(1) * 13 +
                        torch.arange(HIDDEN, device=device) * 7 + peer * 37).to(torch.uint8)
            matches.append(torch.equal(values, expected))
            del values, storage
        return {'rank': rank, 'bytewise_equal': all(matches), 'matches': matches,
                'x_offset': x_offset, 'segment_bytes': stride, 'segment_rows': stride // HIDDEN,
                'mapped_bytes': lengths}
    finally:
        if mapped:
            torch.cuda.synchronize()
        for segment, length in reversed(mapped):
            _ok(cu.cuMemUnmap(segment, length))
        if address is not None:
            _ok(cu.cuMemAddressFree(address, world * stride))
        for handle in reversed(handles):
            _ok(cu.cuMemRelease(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    buffer = GCSymmBuffer(dist.group.WORLD, world * 8, 256, 8, HIDDEN, 2048, 1)
    pattern = (torch.arange(TOKENS, device=device).unsqueeze(1) * 13 +
               torch.arange(HIDDEN, device=device) * 7 + rank * 37).to(torch.uint8)
    buffer.x[:TOKENS].view(torch.uint8).copy_(pattern)
    torch.cuda.synchronize()
    dist.barrier()
    try:
        result = check_rank(buffer, rank, world, device)
    except RuntimeError as error:
        result = {'rank': rank, 'bytewise_equal': False, 'error': str(error),
                  'traceback': traceback.format_exc()}
    results = [None] * world
    dist.all_gather_object(results, result)
    if rank == 0:
        args.output_json.write_text(json.dumps({'world_size': world, 'per_rank': results}, indent=2))
        print(results, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
