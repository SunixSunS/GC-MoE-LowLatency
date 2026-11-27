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

The single-CTA publisher eliminates the grid-wide dispatch barrier and per-CTA global prefix atomics. The reduction keeps the shared-first addition order. Separate dispatch and compute CTAs still exchange packed activation rows through the global input ring. CTA-local shared memory cannot serve as a general cross-kernel exchange buffer. The current return protocol publishes completion per rank. Overlapping a separate Split-K merge with return traffic would require a finer readiness protocol that releases each tile when all its partial accumulators are visible.

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
