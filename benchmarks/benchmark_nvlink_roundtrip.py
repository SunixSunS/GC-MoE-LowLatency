"""Time FlashInfer dispatch and combine with an exact integer-valued oracle."""

import argparse
import json
import os
from math import ceil
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from flashinfer.moe_ep import (
    BootstrapConfig,
    MoEEpCommParams,
    NVLinkOneSidedConfig,
    create_communication,
)


@triton.jit
def _contribution_kernel(activation, expert_ids, output, hidden: tl.constexpr,
                         activation_row_stride: tl.constexpr, activation_col_stride: tl.constexpr,
                         ids_row_stride: tl.constexpr, ids_col_stride: tl.constexpr,
                         topk: tl.constexpr, rank: tl.constexpr, local_experts: tl.constexpr,
                         block_k: tl.constexpr, block_h: tl.constexpr):
    row = tl.program_id(0)
    choice = tl.arange(0, block_k)
    expert = tl.load(expert_ids + row * ids_row_stride + choice * ids_col_stride,
                     choice < topk, other=-1)
    local = (expert >= rank * local_experts) & (expert < (rank + 1) * local_experts)
    coefficient = tl.sum(tl.where(local, expert % 3 + 1, 0), axis=0)
    col = tl.arange(0, block_h)
    value = tl.load(activation + row * activation_row_stride + col * activation_col_stride,
                    col < hidden, other=0).to(tl.float32)
    tl.store(output + row * hidden + col, tl.where(coefficient != 0, value * coefficient, 0),
             col < hidden)


def measure(args, bootstrap, tokens, sentinel, device):
    rank, world = bootstrap.rank, bootstrap.world_size
    local_experts = args.num_experts // world
    generator = torch.Generator(device=device).manual_seed(20261127 + rank)
    ids = torch.rand((tokens, args.num_experts), generator=generator, device=device)
    ids = ids.topk(args.topk, dim=1).indices.to(torch.int32)
    x = torch.randint(-4, 5, (tokens, args.hidden), generator=generator, device=device)
    x = x.to(torch.bfloat16)
    reference = x * (ids.remainder(3) + 1).sum(1, keepdim=True).to(torch.bfloat16)
    params = MoEEpCommParams(args.num_experts, args.topk, tokens, args.hidden, dtype=torch.bfloat16)
    protocol = 'sentinel' if sentinel else 'fence'
    config = NVLinkOneSidedConfig(sync_protocol=protocol, timeout_sec=15)
    communication = create_communication(bootstrap, params, backend=config)
    output = torch.empty_like(x)

    def dispatch():
        return communication.dispatch(x, ids, runtime_max_tokens_per_rank=tokens)

    def local_contribution(received):
        routed = received.topk_ids
        if args.oracle_kernel == 'triton':
            activation = received.hidden_states
            contribution = torch.empty(activation.shape, dtype=activation.dtype, device=device)
            _contribution_kernel[(activation.shape[0],)](
                activation, routed, contribution, args.hidden,
                *activation.stride(), *routed.stride(), args.topk, rank, local_experts,
                triton.next_power_of_2(args.topk), triton.next_power_of_2(args.hidden),
                num_warps=4)
            return contribution
        local = routed.div(local_experts, rounding_mode='floor') == rank
        coeff = torch.where(local, routed.remainder(3) + 1, 0).sum(1, keepdim=True)
        value = received.hidden_states * coeff.to(torch.bfloat16)
        return torch.where(coeff != 0, value, 0)

    try:
        if sentinel and not communication.sentinel_enabled:
            raise RuntimeError('FlashInfer did not enable the sentinel protocol on this group')
        for _ in range(3):
            communication.combine(local_contribution(dispatch()), output=output)
        torch.cuda.synchronize(device)
        assert torch.equal(output, reference), (rank, tokens, sentinel)
        events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            events[0].record()
            received = dispatch()
            events[1].record()
            contribution = local_contribution(received)
            events[2].record()
            communication.combine(contribution, output=output)
            events[3].record()
        samples = {key: [] for key in ('dispatch', 'local_oracle', 'combine', 'roundtrip')}
        for window in range(args.windows + 3):
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            if window >= 3:
                for key, start, end in (('dispatch', 0, 1), ('local_oracle', 1, 2),
                                        ('combine', 2, 3), ('roundtrip', 0, 3)):
                    samples[key].append(events[start].elapsed_time(events[end]) * 1000)
        assert torch.equal(output, reference), (rank, tokens, sentinel, 'graph')
        del graph
        all_ranks = [None] * world
        dist.all_gather_object(all_ranks, samples)
    finally:
        communication.destroy()
    stats = {}
    for key in samples:
        slowest = sorted(max(values) for values in zip(*(row[key] for row in all_ranks)))
        stats[key] = {'p50_us': median(slowest), 'p95_us': slowest[ceil(len(slowest) * .95) - 1]}
    return {'tokens': tokens, 'protocol': protocol,
            'latency': stats, 'per_rank': all_ranks, 'exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, nargs='+', default=(16, 32))
    parser.add_argument('--hidden', type=int, default=6144)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--windows', type=int, default=40)
    parser.add_argument('--oracle-kernel', choices=('torch', 'triton'), default='torch')
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.num_experts % world:
        raise ValueError('num_experts must be divisible by the rank count')
    bootstrap = BootstrapConfig(world, rank, process_group=dist.group.WORLD)
    data = []
    for tokens in args.tokens:
        for sentinel in (True, False):
            result = measure(args, bootstrap, tokens, sentinel, device)
            data.append(result)
            if rank == 0:
                print(tokens, result['protocol'], result['latency'], flush=True)
    if rank == 0:
        args.output_json.write_text(json.dumps(
            {'world_size': world, 'hidden': args.hidden, 'num_experts': args.num_experts,
             'topk': args.topk, 'windows': args.windows, 'payload': 'BF16',
             'oracle_kernel': args.oracle_kernel, 'cases': data}, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

