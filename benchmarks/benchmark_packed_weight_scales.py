"""Measure cold synthetic FP4 tiles with raw or losslessly packed W scales."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl


WEIGHT_WORDS = 128 * 64 // 4
RAW_SCALE_WORDS = 128
PACKED_SCALE_WORDS = 32
SCALE_BASE = 119
SCALE_BASE_WORD = SCALE_BASE * 0x01010101


@triton.jit
def stream_weight_tiles(weights, scales, sums, tile_rows: tl.constexpr,
                        packed: tl.constexpr, weight_words: tl.constexpr,
                        scale_base: tl.constexpr):
    tile = tl.program_id(0) * tile_rows + tl.arange(0, tile_rows)
    columns = tl.arange(0, weight_words)
    weight_words_read = tl.load(weights + tile[:, None] * weight_words + columns[None, :])
    weight_sum = tl.sum(weight_words_read.to(tl.uint32), axis=1)
    if packed:
        sf_columns = tl.arange(0, 32)
        bits = tl.load(scales + tile[:, None] * 32 + sf_columns[None, :]).to(tl.uint32)
        group_sum = tl.full((tile_rows, 32), 0, tl.uint32)
        for word in tl.static_range(4):
            decoded = ((bits >> (2 * word)) & 0x03030303) + scale_base
            group_sum += decoded
        sf_sum = tl.sum(group_sum, axis=1)
    else:
        sf_columns = tl.arange(0, 128)
        words = tl.load(scales + tile[:, None] * 128 + sf_columns[None, :]).to(tl.uint32)
        sf_sum = tl.sum(words, axis=1)
    tl.store(sums + tile, weight_sum + sf_sum)


def build_scale_storage(tiles, device):
    # Restrict the synthetic scale window to four exponents. No real model
    # checkpoint is assumed to fit this window without a fallback.
    exponents = torch.randint(0, 4, (tiles, 32, 4, 4), device=device, dtype=torch.int32)
    words = exponents[..., 0]
    for byte in range(1, 4):
        words = words | (exponents[..., byte] << (8 * byte))
    raw = (words + SCALE_BASE_WORD).reshape(tiles, 128)
    packed = words[..., 0]
    for row in range(1, 4):
        packed = packed | (words[..., row] << (2 * row))
    return raw.contiguous(), packed.contiguous()


def measure(sets, packed, tile_rows, windows, device):
    tile_count = sets[0][0].shape[0]
    result = torch.empty(tile_count, dtype=torch.int32, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()

    def launch(weight_set):
        weight, raw, compressed = weight_set
        stream_weight_tiles[(tile_count // tile_rows,)](
            weight, compressed if packed else raw, result, tile_rows, packed,
            WEIGHT_WORDS, SCALE_BASE_WORD, num_warps=4)

    with torch.cuda.stream(stream):
        launch(sets[-1])
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        for weight_set in sets[:-1]:
            launch(weight_set)
        start.record()
        launch(sets[-1])
        end.record()
    durations = []
    for i in range(windows + 4):
        graph.replay()
        end.synchronize()
        if i >= 4:
            durations.append(start.elapsed_time(end) * 1000)
    return result, {
        'p50_us': median(durations),
        'p95_us': sorted(durations)[math.ceil(windows * 0.95) - 1],
        'samples_us': durations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-experts', nargs='+', type=int, choices=(8, 16), default=(8, 16))
    parser.add_argument('--windows', type=int, default=90)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--output-json', required=True, type=Path)
    args = parser.parse_args()
    if args.windows < 1 or args.layers < 2:
        parser.error('positive windows and at least two weight sets are required')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.manual_seed(1408)
    results = []
    for experts in args.local_experts:
        tiles = experts * ((4096 // 128) * (6144 // 128) + (6144 // 128) * (2048 // 128))
        sets = [(torch.randint(-2**31, 2**31 - 1, (tiles, WEIGHT_WORDS), device=device,
                               dtype=torch.int32), *build_scale_storage(tiles, device))
                for _ in range(args.layers)]
        cases = {}
        for tile_rows in (1, 2, 4):
            raw_sum, baseline = measure(sets, False, tile_rows, args.windows, device)
            packed_sum, optimized = measure(sets, True, tile_rows, args.windows, device)
            if not torch.equal(raw_sum, packed_sum):
                raise ValueError(f'packed scale checksum changed for {experts} experts')
            cases[tile_rows] = {'raw': baseline, 'packed': optimized}
            print('CASE', experts, tile_rows, baseline['p50_us'], optimized['p50_us'], flush=True)
        results.append({
            'local_experts': experts,
            'fp4_weight_bytes': tiles * WEIGHT_WORDS * 4,
            'raw_scale_bytes': tiles * RAW_SCALE_WORDS * 4,
            'packed_scale_bytes': tiles * PACKED_SCALE_WORDS * 4,
            'cta_tiles': cases,
        })
    args.output_json.write_text(json.dumps({'gpu': torch.cuda.get_device_name(device),
                                           'layers': args.layers, 'cases': results}, indent=2))


if __name__ == '__main__':
    main()
