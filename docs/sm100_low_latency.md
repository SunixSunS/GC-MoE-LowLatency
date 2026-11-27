# SM100 small-batch GC MoE

GLM-5.2 and GLM-5.3 have the same MoE geometry for this operator. The hidden width is 6144, the expert intermediate width is 2048, and each token selects 8 of 256 routed experts. There is one shared expert. The operator benchmark uses synthetic FP8 activations, FP4 routed weights, and FP8 shared weights with K128 scales. The published model checkpoints and their routing traces are outside this benchmark.

[GLM-5.2 model configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) · [GLM-5.3 model configuration](https://huggingface.co/zai-org/GLM-5.3/blob/e0b07fd2751b42d5efa199cc02c2b271deadc516/config.json)

The `decode_mode` argument to `gc_moe.fp8_fp4_gc_moe` selects these paths for batches of up to 32 tokens per rank.

| Mode | Behavior |
| --- | --- |
| `off` | Original three-role graph. |
| `route` | One CTA scatters and publishes the rank's routes when the whole batch fits. The other dispatch CTAs pull activations after the usual receive-count acquire. |
| `reduce` | Each warp reduces a 512-byte output slice from global BF16 slots into FP32 registers and stores BF16 output directly. |
| `route-reduce` | Single-CTA routing and register reduction. |
| `vector` | Single-CTA routing, register reduction, and 16-byte vector loads and stores for the peer activation pull when the hidden width is a multiple of 512. |

The single-CTA publisher eliminates the grid-wide dispatch barrier and per-CTA global prefix atomics. The reduction keeps the shared-first addition order. Separate dispatch and compute CTAs exchange packed activation rows through the global input ring. The default return protocol publishes completion per rank. The small-batch profiles below also exercise a tile-level sentinel protocol that allows Combine to consume W2 output as it is published.

## Reproduce on GB200

Use an environment with the repo's CUDA/CuTe dependencies and DeepGEMM installed. The following single-node command preserves the EP16 local expert count of 16 per card on four GPUs. Set `--num-experts 32` to preserve the EP32 local count of 8 per card. Native EP16/EP32 runs use `--num-experts 256` with the corresponding distributed world size.

```bash
export GC_MOE_PRIMARY_CTX=1 GC_MOE_DISPATCH_SMS=16
args=(--baseline-gc --gc-decode-mode route-reduce --hidden 6144 \
      --intermediate-hidden 2048 --num-experts 64 --num-topk 8 \
      --num-shared-experts 1 --shared-weight-gran-k 128 \
      --num-max-tokens-per-rank 256 --num-tokens-per-rank-list 16,32)
torchrun --standalone --nproc-per-node=4 tests/test_gc_moe_se.py "${args[@]}" \
  --per-rank-tokens-list '16,16,16,16;32,32,32,32;0,16,32,16' \
  --case-category asymmetric --output-json results/decode-accuracy.json
torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_gc_moe_se.py "${args[@]}" \
  --output-json results/decode-benchmark.json
```

The accuracy test checks eager execution and CUDA Graph replays against stock GC and independent Torch math. The benchmark reports the slowest rank's median CUDA Graph time. Omitting `--baseline-gc` selects MegaMoE and requires a DeepGEMM build with the matching fused shared-expert recipe.

## Validated EP16 and EP32 profiles

Use four GB200 nodes for EP16 or eight for EP32, with four GPUs per node and CUDA symmetric memory over the NVLink fabric. The captured benchmarks used PyTorch 2.11.0+cu130 and CuTe DSL 4.8.1. Install the same revision on every node and run each command on every node. Set `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` for the distributed job.

The following shell setup keeps the Green Context stock baseline in every run. The benchmark applies the requested primary-context SM profile to each candidate.

```bash
export GC_MOE_PRIMARY_CTX=0 GC_MOE_DISPATCH_SMS=16
mkdir -p results
launch=(torchrun --nnodes "$NNODES" --nproc-per-node 4 --node-rank "$NODE_RANK" \
          --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
bench=benchmarks/benchmark_decode_critical_path.py
common=(--num-experts 256 --capacity 256 --intervening-layers 3 --trace-replays 0)
```

### EP16

D24 with narrow BM32 and cluster spread gave the lowest measured T32 latency. For T16, the lowest measured case used D32, W1 bulk prefetch and CTA-local route counts. In both cases the routed weight candidate uses VMM compression and 16 Dispatch warps, with 24 compute CTAs backfilled after Dispatch.

```bash
narrow=sentinel_early_w1k4_w2k2_bm32_narrow_w2_tile_major_rows2_prefix_pruned_skip_descriptors_flat_cluster_spread
"${launch[@]}" "$bench" "${common[@]}" --tokens 16 32 --windows 64 \
  --compute-backfill-sms 24 --vmm-weight-compression \
  --vmm-primary-dispatch-sms 24 --backfill-vmm-compression all \
  --backfill-vmm-dispatch-warps 16 --backfill-vmm-options-only \
  --backfill-vmm-options "$narrow" "${narrow}_bulk_startup_shared_route_counts" \
  --backfill-vmm-sms-sweep 32 --primary-decode-sms 24 \
  --primary-decode-variants route_only compute_backfill_24 \
  --output-json results/ep16-decode.json
```

Select the unsuffixed cluster-spread variant for the T32 D24 point and the `_bulk_startup_shared_route_counts_d32` variant for the T16 D32 point.

### EP32

The T16 reference profile uses early Combine and zero compute backfill. The lowest measured T32 profile uses BM48, eight Combine warps, a single cleanup CTA, W1 bulk prefetch and CTA-local route counts with ordinary weight allocation.

```bash
"${launch[@]}" "$bench" "${common[@]}" --tokens 16 --windows 110 \
  --compute-backfill-sms 0 --backfill-options sentinel_early_combine \
  --primary-decode-sms 24 \
  --primary-decode-variants route_only compute_backfill_0 compute_backfill_0_sentinel_early_combine \
  --output-json results/ep32-t16-decode.json

bm48=sentinel_early_combine_w2_tile_major_bm48_cleanup_w8_single
"${launch[@]}" "$bench" "${common[@]}" --tokens 32 --windows 60 \
  --compute-backfill-sms 16 --primary-decode-sms 24 \
  --backfill-options "$bm48" "${bm48}_bulk_prefix" "${bm48}_bulk_prefix_shared_route_counts" \
  --primary-decode-variants route_only compute_backfill_16 \
    "compute_backfill_16_${bm48}" "compute_backfill_16_${bm48}_bulk_prefix" \
    "compute_backfill_16_${bm48}_bulk_prefix_shared_route_counts" \
  --output-json results/ep32-t32-decode.json
```

The EP32 T32 configuration with the largest regular paired reduction used VMM compression with W1K4/W2K2 prefetch. Use `--compute-backfill-sms 16 --vmm-weight-compression --vmm-primary-dispatch-sms 24 --backfill-vmm-compression all --backfill-vmm-dispatch-warps 16 --backfill-vmm-options-only --backfill-vmm-options sentinel_early_w1k4_w2k2` with the common launch flags.

### Why the profiles differ

- **Activation arrival.** Routed W1 waits for all valid activation rows for its M tile. The Shared W1 CTA can start from local input while Dispatch is still active. W1 bulk prefetch loads a weight prefix into L2 during startup. CTA-local route counts let one warp acquire counts and share them through SMEM.
- **Weight traffic.** EP16 uses narrow BM32 for this geometry. For EP32 with 32 tokens/rank, routing can assign more than 32 rows to an expert. BM48 covers up to 48 rows with a single M tile and can avoid a repeated weight request for a second tile.
- **Return traffic.** A sentinel lets Combine consume the ready W2 tiles while other tiles are still computing. In the single-cleanup variant, completed Combine CTAs can retire while CTA 0 waits for computation to finish and resets the shared workspace.

The JSON records the per-rank timings and bitwise comparisons against `stock`. Take the maximum rank latency in each window before calculating aggregate p50 and p95. A four-card diagnostic can use 64 experts for the EP16 local expert count or 32 for the EP32 count. Use the physical rank counts for the fabric and tail-latency comparison. [Paired measurements and trace findings](gb200-low-latency-results-2026-11-27.md)
