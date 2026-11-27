"""Measure N-sharded shared-expert FP8 GEMMs and their NVLink exchanges."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem


HIDDEN = 6144
INTERMEDIATE = 2048
FP8 = torch.float8_e4m3fn
PEER_W1_STAGES = ('gather', 'w1_and_gate', 'exchange', 'w2')
PEER_KERNEL_STAGES = ('gather_grid', 'exchange_grid', 'wait_for_reuse',
                      'transfers', 'wait_for_peer_payload')


def fp8_input(shape, scale, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, device=device, generator=rng) * scale).to(FP8)


def make_weights(layers, device):
    return [
        (
            fp8_input((INTERMEDIATE * 2, HIDDEN), 1 / 32, 100 + 2 * layer, device),
            fp8_input((HIDDEN, INTERMEDIATE), 1 / 32, 101 + 2 * layer, device),
        )
        for layer in range(layers)
    ]


def gemm(a, weight, one):
    return torch._scaled_mm(a, weight.T, scale_a=one, scale_b=one,
                            out_dtype=torch.bfloat16)


def gate_up(projected):
    gate, up = projected.float().chunk(2, dim=1)
    return (F.silu(gate) * up).to(FP8)


def replicated(x, weights, one):
    w1, w2 = weights
    return gemm(gate_up(gemm(x, w1, one)), w2, one)


def shard_weights(weights, group_rank, group_size):
    width = INTERMEDIATE // group_size
    channels = HIDDEN // group_size
    start = group_rank * width
    w1, w2 = weights
    gate = w1.narrow(0, start, width)
    up = w1.narrow(0, INTERMEDIATE + start, width)
    return (
        torch.cat((gate.view(torch.uint8), up.view(torch.uint8)), dim=0).view(FP8),
        w2.narrow(0, group_rank * channels, channels).clone(),
    )


class ShardedSharedExpert:
    def __init__(self, x, group, one):
        self.x = x
        self.one = one
        self.group = group
        self.group_size = dist.get_world_size(group)
        self.tokens = x.shape[0]
        rows = self.tokens * self.group_size
        shard_width = INTERMEDIATE // self.group_size
        output_width = HIDDEN // self.group_size
        self.inputs = torch.empty((rows, HIDDEN), device=x.device, dtype=FP8)
        self.activations = torch.empty((self.group_size * rows, shard_width),
                                       device=x.device, dtype=FP8)
        self.returned = torch.empty((self.group_size * self.tokens, output_width),
                                    device=x.device, dtype=torch.bfloat16)

    def __call__(self, weights):
        """Keep W2's K dimension intact and shard only its output channels."""
        w1, w2 = weights
        dist.all_gather_into_tensor(self.inputs.view(torch.uint8), self.x.view(torch.uint8),
                                    group=self.group)
        activation = gate_up(gemm(self.inputs, w1, self.one))
        dist.all_gather_into_tensor(self.activations.view(torch.uint8), activation.view(torch.uint8),
                                    group=self.group)
        rows = self.tokens * self.group_size
        w2_input = (self.activations.view(torch.uint8).view(self.group_size, rows, -1)
                    .transpose(0, 1).contiguous().view(rows, INTERMEDIATE).view(FP8))
        output_shard = gemm(w2_input, w2, self.one)
        dist.all_to_all_single(self.returned, output_shard, group=self.group)
        return (self.returned.view(self.group_size, self.tokens, -1)
                .transpose(0, 1).contiguous().view(self.tokens, HIDDEN))


class PeerShardedSharedW1:
    """Compute half of shared W1 for both ranks and exchange FP8 channels."""

    def __init__(self, x, group, one, library, exchange_name, banks, profile_kernels):
        self.one = one
        self.tokens, hidden = x.shape
        self.device = x.device
        self.shard = dist.get_rank(group)
        self.gather = library.gc_shared_gather_pair
        self.exchange = getattr(library, exchange_name)
        self.bank = 0
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        self.gather.argtypes = [pointer, pointer, pointer, integer, pointer, pointer]
        self.gather.restype = integer
        self.exchange.argtypes = [pointer, pointer, pointer, integer, integer, integer, pointer, pointer]
        self.exchange.restype = integer

        # This benchmark holds each rank's input fixed across its weight sets.
        self.local = symm_mem.empty(x.numel(), dtype=torch.uint8, device=x.device)
        self.local.copy_(x.view(torch.uint8).reshape(-1))
        self.input_handle = symm_mem.rendezvous(self.local, group=group)
        self.peer_input = self.input_handle.buffer_ptrs[self.shard ^ 1]
        self.paired = torch.empty((2 * self.tokens, hidden), dtype=FP8, device=x.device)
        payload_bytes = self.tokens * INTERMEDIATE
        flag_bytes = (1 + banks) * self.tokens * 4
        self.mailbox = symm_mem.empty(banks * payload_bytes + flag_bytes,
                                      dtype=torch.uint8, device=x.device)
        self.mailbox.zero_()
        self.mailbox_handle = symm_mem.rendezvous(self.mailbox, group=group)
        self.peer_mailbox = self.mailbox_handle.buffer_ptrs[self.shard ^ 1]
        self.w2_input = self.mailbox[:banks * payload_bytes].view(FP8).view(
            banks, self.tokens, INTERMEDIATE)
        if profile_kernels:
            gather_blocks = math.ceil(x.numel() / (16 * 128))
            self.gather_trace = torch.empty((gather_blocks, 2), dtype=torch.int64, device=x.device)
            self.exchange_trace = torch.empty((self.tokens, 4), dtype=torch.int64, device=x.device)
        torch.cuda.synchronize(x.device)
        dist.barrier(group=group)

    def __call__(self, weights, stage_events=(), trace_kernels=False):
        w1, w2 = weights
        stream = torch.cuda.current_stream(self.device).cuda_stream
        status = self.gather(self.local.data_ptr(), self.peer_input, self.paired.data_ptr(),
                             self.local.numel(),
                             self.gather_trace.data_ptr() if trace_kernels else 0, stream)
        if status:
            raise RuntimeError(f'pair input gather failed with CUDA status {status}')
        if stage_events:
            stage_events[0].record()
        activation = gate_up(gemm(self.paired, w1, self.one))
        if stage_events:
            stage_events[1].record()
        status = self.exchange(activation.data_ptr(), self.mailbox.data_ptr(), self.peer_mailbox,
                               self.tokens, INTERMEDIATE // 2, self.shard,
                               self.exchange_trace.data_ptr() if trace_kernels else 0, stream)
        if status:
            raise RuntimeError(f'W1 exchange failed with CUDA status {status}')
        if stage_events:
            stage_events[2].record()
        self.bank = (self.bank + 1) % len(self.w2_input)
        return gemm(self.w2_input[self.bank], w2, self.one)

    def kernel_times(self):
        gather = self.gather_trace.tolist()
        exchange = self.exchange_trace.tolist()
        gather_span = max(row[-1] for row in gather) - min(row[0] for row in gather)
        exchange_span = max(row[-1] for row in exchange) - min(row[0] for row in exchange)
        cta_max = tuple(max(row[stage + 1] - row[stage] for row in exchange) for stage in range(3))
        return tuple(ticks / 1000 for ticks in (gather_span, exchange_span, *cta_max))


class PeerPairSharedOwner:
    """Run the whole shared expert for both ranks on the designated owner."""

    def __init__(self, x, group, one, library):
        self.one = one
        self.device = x.device
        self.tokens, hidden = x.shape
        rank = dist.get_rank(group)
        self.owner = rank == 0
        self.bank = 0
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        self.gather = library.gc_shared_gather_pair
        self.gather.argtypes = [pointer, pointer, pointer, integer, pointer, pointer]
        self.gather.restype = integer
        self.return_rows = library.gc_shared_return_owned_rows
        self.return_rows.argtypes = [pointer, pointer, pointer, integer, integer, integer, pointer]
        self.return_rows.restype = integer

        self.local = symm_mem.empty(x.numel(), dtype=torch.uint8, device=x.device)
        self.local.copy_(x.view(torch.uint8).reshape(-1))
        self.input_handle = symm_mem.rendezvous(self.local, group=group)
        self.peer_input = self.input_handle.buffer_ptrs[rank ^ 1]
        self.paired = (torch.empty((2 * self.tokens, hidden), dtype=FP8, device=x.device)
                       if self.owner else None)
        data_bytes = self.tokens * hidden * 2
        self.mailbox = symm_mem.empty(2 * data_bytes + 3 * self.tokens * 4,
                                      dtype=torch.uint8, device=x.device)
        self.mailbox.zero_()
        self.mailbox_handle = symm_mem.rendezvous(self.mailbox, group=group)
        self.peer_mailbox = self.mailbox_handle.buffer_ptrs[rank ^ 1]
        self.returned = self.mailbox[:2 * data_bytes].view(torch.bfloat16).view(
            2, self.tokens, hidden)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def __call__(self, weights):
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if self.owner:
            status = self.gather(self.local.data_ptr(), self.peer_input, self.paired.data_ptr(),
                                 self.local.numel(), 0, stream)
            if status:
                raise RuntimeError(f'shared owner input gather failed with CUDA status {status}')
            outputs = replicated(self.paired, weights, self.one)
        status = self.return_rows(outputs.data_ptr() if self.owner else 0,
                                  self.mailbox.data_ptr(), self.peer_mailbox,
                                  self.tokens, HIDDEN * 2, self.owner, stream)
        if status:
            raise RuntimeError(f'shared owner return failed with CUDA status {status}')
        self.bank ^= 1
        return outputs[:self.tokens] if self.owner else self.returned[self.bank]


def measure(launch, weights, windows, device, stage_names=(), trace_kernels=False):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for weight in weights:
            launch(weight)
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    stages = tuple(torch.cuda.Event(enable_timing=True, external=True)
                   for _ in range(len(stage_names) - 1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for weight in weights[:-1]:
            launch(weight)
        start.record()
        if stages:
            launch(weights[-1], stages)
        elif trace_kernels:
            launch(weights[-1], trace_kernels=True)
        else:
            launch(weights[-1])
        end.record()
    dist.barrier()
    stamps = (start, *stages, end)
    stage_pairs = tuple(zip(stamps, stamps[1:])) if stages else ()
    names = PEER_KERNEL_STAGES if trace_kernels else stage_names
    readings = [[] for _ in range(1 + len(names))]
    for _ in range(windows):
        graph.replay()
        end.synchronize()
        readings[0].append(start.elapsed_time(end) * 1000)
        for samples, (first, last) in zip(readings[1:], stage_pairs):
            samples.append(first.elapsed_time(last) * 1000)
        if trace_kernels:
            for samples, span in zip(readings[1:], launch.kernel_times()):
                samples.append(span)
    across_ranks = torch.tensor(readings, dtype=torch.float32, device=device)
    dist.all_reduce(across_ranks, op=dist.ReduceOp.MAX)
    def summary(worst):
        return {
            'p50_us': median(worst),
            'p95_us': sorted(worst)[math.ceil(0.95 * len(worst)) - 1],
            'window_rank_max_us': worst,
        }

    total, *stage_times = across_ranks.tolist()
    timing = summary(total)
    if stage_times:
        timing['stages'] = {name: summary(samples) for name, samples in zip(names, stage_times)}
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='+', type=int, choices=(16, 32), default=(16, 32))
    parser.add_argument('--windows', type=int, default=100)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--peer-w1-library', type=Path)
    parser.add_argument('--profile-peer-stages', action='store_true')
    parser.add_argument('--profile-peer-kernels', action='store_true')
    parser.add_argument('--peer-double-buffered', action='store_true')
    parser.add_argument('--peer-pair-owner', action='store_true')
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.windows < 1 or args.layers < 2:
        parser.error('positive windows and at least two distinct weight sets are required')
    if (args.peer_double_buffered or args.peer_pair_owner) and (
            not args.peer_w1_library or args.layers % 2):
        parser.error('the double-buffered graph needs a peer library and an even layer count')

    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError('run on four GPUs in one NVLink domain')
    pairs = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    groups = {2: pairs[rank // 2], 4: dist.group.WORLD}
    one = torch.ones((), device=device)
    weights = make_weights(args.layers, device)
    peer_library = ctypes.CDLL(str(args.peer_w1_library)) if args.peer_w1_library else None
    peer_modes = [
        ('peer_sharded_w1', 'gc_shared_exchange_halves', 1),
        ('peer_sharded_w1_bulk', 'gc_shared_exchange_bulk', 1),
    ]
    if args.peer_double_buffered:
        peer_modes.append(('peer_sharded_w1_double_buffered', 'gc_shared_exchange_double_buffered', 2))
    tests = []
    for tokens in args.tokens:
        x = fp8_input((tokens, HIDDEN), 1 / 4, 20 + rank, device)
        expected = replicated(x, weights[-1], one)
        latency = {'replicated': measure(lambda w: replicated(x, w, one), weights, args.windows, device)}
        accuracy = {}
        traces = {}
        kernel_traces = {}
        if peer_library:
            pair = groups[2]
            shards = [(shard_weights(w, dist.get_rank(pair), 2)[0], w[1]) for w in weights]
            for name, exchange, banks in peer_modes:
                peer_w1 = PeerShardedSharedW1(
                    x, pair, one, peer_library, exchange, banks, args.profile_peer_kernels)
                actual = peer_w1(shards[-1])
                max_error = (actual.float() - expected.float()).abs().amax()
                exact = torch.tensor(int(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))),
                                     dtype=torch.int32, device=device)
                dist.all_reduce(exact, op=dist.ReduceOp.MIN)
                dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
                if not exact.item():
                    raise AssertionError(f'{name} changed BF16 output, max error {max_error.item()}')
                accuracy[name] = max_error.item()
                latency[name] = measure(peer_w1, shards, args.windows, device)
                if args.profile_peer_stages:
                    traces[name] = measure(peer_w1, shards, min(args.windows, 40), device,
                                           PEER_W1_STAGES)
                if args.profile_peer_kernels:
                    kernel_traces[name] = measure(peer_w1, shards, min(args.windows, 40), device,
                                                  trace_kernels=True)
            if args.peer_pair_owner:
                owner = PeerPairSharedOwner(x, pair, one, peer_library)
                actual = owner(weights[-1])
                exact = torch.tensor(int(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))),
                                     dtype=torch.int32, device=device)
                dist.all_reduce(exact, op=dist.ReduceOp.MIN)
                if not exact.item():
                    raise AssertionError('shared pair ownership changed BF16 output')
                accuracy['peer_pair_owner'] = 0
                latency['peer_pair_owner'] = measure(owner, weights, args.windows, device)
        for size, group in groups.items():
            name = f'sharded_{size}'
            launch = ShardedSharedExpert(x, group, one)
            shards = [shard_weights(w, dist.get_rank(group), size) for w in weights]
            actual = launch(shards[-1])
            torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.002)
            max_error = (actual.float() - expected.float()).abs().amax()
            dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
            accuracy[name] = max_error.item()
            latency[name] = measure(launch, shards, args.windows, device)
        tests.append({'tokens': tokens, 'latency': latency,
                      'max_abs_error': accuracy, 'phase_traces': traces,
                      'kernel_traces': kernel_traces})
        if rank == 0:
            print(tokens, {n: (v['p50_us'], v['p95_us']) for n, v in latency.items()},
                  accuracy, flush=True)
    if rank == 0:
        bytes_full = 3 * HIDDEN * INTERMEDIATE
        args.output_json.write_text(json.dumps({
            'world_size': world_size, 'layers': args.layers,
            'logical_fp8_weight_bytes': {'replicated': bytes_full, 'sharded_2': bytes_full // 2,
                                         'sharded_4': bytes_full // 4} | (
                                             dict.fromkeys((name for name, _, _ in peer_modes),
                                                           2 * bytes_full // 3) if peer_library else {}),
            'cases': tests,
            **({'pair_owner_weight_bytes': {'owner': bytes_full, 'partner': 0}}
               if args.peer_pair_owner else {}),
        }, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
